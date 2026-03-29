"""
Learning Engine — Self-Learning AI via Backtesting Analysis
===========================================================
Mines historical ai_signals + market_data to measure which indicators
actually predict profitable trades, then:

1. Computes per-indicator success rates → stored in `indicator_stats`
2. Derives optimal score/probability thresholds
3. Adjusts model weights (RSI, EMA, MACD, Volume, News) by ≤10% per cycle
4. Persists learned weights to `model_weights` (with rollback backup)
5. Standalone runner + scheduler-friendly entry point

Collections written:
    indicator_stats     — per-indicator win rates & analysis
    model_weights       — current adaptive weights (with backup)

Safety:
    - Max ±10% weight change per cycle
    - Rollback backup always saved before writing new weights
    - Graceful degradation if data is sparse (keeps current weights)

Deliverables (per task spec):
    learning_engine.py        ← this file
    indicator_stats coll      ← written here
    model_weights coll        ← written here
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pymongo

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Collection names ───────────────────────────────────────────
COLL_INDICATOR_STATS = 'indicator_stats'
COLL_MODEL_WEIGHTS   = 'model_weights'

# ── Default baseline weights (sum=1.0) ────────────────────────
DEFAULT_WEIGHTS = {
    'rsi':         0.20,
    'ema':         0.20,
    'macd':        0.15,
    'volume':      0.15,
    'news':        0.15,
    'probability': 0.15,
}

# ── Safety limits ─────────────────────────────────────────────
MAX_WEIGHT_CHANGE_PCT = 0.10   # ±10% max change per learning cycle
MIN_TRADES_REQUIRED   = 10     # Minimum trades before trusting a stat
LEARNING_LOOKBACK_DAYS = 30    # Analyse last N days of signals


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_learning_cycle(db=None, lookback_days: int = LEARNING_LOOKBACK_DAYS) -> Dict:
    """
    Run one full learning cycle:
      1. Load signal history + simulated trade outcomes
      2. Analyse per-indicator win rates
      3. Compute new weights (capped at ±10%)
      4. Save indicator_stats + model_weights to MongoDB

    Parameters
    ----------
    db           : pymongo database object (or None — opens own connection)
    lookback_days: days of history to analyse

    Returns
    -------
    dict with keys: weights, indicator_stats, improvements, cycle_ts
    """
    own_conn = (db is None)
    if own_conn:
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db     = client[settings.DATABASE_NAME]

    try:
        logger.info("=" * 60)
        logger.info("[Learning] Starting learning cycle")
        logger.info("=" * 60)

        since = datetime.utcnow() - timedelta(days=lookback_days)

        # ── Step 1: Load historical signals ──
        signals = _load_signals(db, since)
        if not signals:
            logger.warning("[Learning] No signals found — skipping cycle")
            return _empty_result()

        # ── Step 2: Simulate trade outcomes ──
        trade_outcomes = _simulate_outcomes(db, signals, since)
        logger.info(f"[Learning] Simulated {len(trade_outcomes)} trade outcomes")

        if len(trade_outcomes) < MIN_TRADES_REQUIRED:
            logger.warning(f"[Learning] Too few trades ({len(trade_outcomes)}) — keeping current weights")
            return _empty_result()

        # ── Step 3: Analyse per-indicator win rates ──
        stats = _analyse_indicators(trade_outcomes)
        logger.info(f"[Learning] Indicator analysis complete: {len(stats)} indicators")

        # ── Step 4: Compute threshold analysis ──
        thresholds = _analyse_thresholds(trade_outcomes)

        # ── Step 5: Load current weights (or defaults) ──
        current   = _load_current_weights(db)
        old_backup = {k: round(v, 6) for k, v in current.items()}

        # ── Step 6: Derive new weights — XGBoost first, statistical fallback ──
        new_weights  = _compute_new_weights_xgb(trade_outcomes, current)
        improvements = _describe_improvements(current, new_weights, stats)

        # ── Step 7: Per-coin learning (top 10 most-traded coins) ──
        try:
            run_per_coin_learning(db, trade_outcomes, top_n=10)
        except Exception as _pce:
            logger.warning(f"[Learning] Per-coin learning error: {_pce}")

        # ── Step 7: Persist ──
        cycle_ts = datetime.utcnow()
        _save_indicator_stats(db, stats, cycle_ts)
        _save_model_weights(db, new_weights, old_backup, stats, thresholds, cycle_ts)

        logger.info(f"[Learning] Cycle complete. New weights: {new_weights}")
        logger.info(f"[Learning] Changes: {improvements}")

        return {
            'weights':         new_weights,
            'indicator_stats': stats,
            'thresholds':      thresholds,
            'improvements':    improvements,
            'trade_count':     len(trade_outcomes),
            'cycle_ts':        cycle_ts,
            'lookback_days':   lookback_days,
        }

    except Exception as e:
        logger.error(f"[Learning] Cycle error: {e}")
        return _empty_result()
    finally:
        if own_conn:
            client.close()


def get_current_weights(db=None) -> Dict:
    """Return current model weights from DB (or defaults if none saved)."""
    own_conn = (db is None)
    if own_conn:
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db     = client[settings.DATABASE_NAME]
    try:
        return _load_current_weights(db)
    finally:
        if own_conn:
            client.close()


def get_indicator_stats(db=None, limit: int = 10) -> List[Dict]:
    """Return latest per-indicator stats for dashboard display."""
    own_conn = (db is None)
    if own_conn:
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db     = client[settings.DATABASE_NAME]
    try:
        docs = list(db[COLL_INDICATOR_STATS].find({}, {'_id': 0})
                                            .sort('win_rate', -1)
                                            .limit(limit))
        return docs
    finally:
        if own_conn:
            client.close()


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def _load_signals(db, since: datetime) -> List[Dict]:
    """Load recent ai_signals with required indicator fields."""
    try:
        cursor = db[settings.COLLECTION_AI_SIGNALS].find(
            {'timestamp': {'$gte': since}},
            {
                '_id': 0, 'symbol': 1, 'timestamp': 1,
                'rsi': 1, 'ema20': 1, 'ema50': 1,
                'macd': 1, 'macd_signal': 1, 'macd_histogram': 1,
                'volume_spike': 1, 'volatility': 1,
                'final_score': 1, 'final_signal': 1,
                'probability_up': 1, 'news_score': 1,
                'profit_score': 1, 'technical_score': 1,
            }
        ).sort('timestamp', 1)
        signals = list(cursor)
        logger.info(f"[Learning] Loaded {len(signals)} signals since {since.strftime('%Y-%m-%d')}")
        return signals
    except Exception as e:
        logger.error(f"[Learning] Signal load error: {e}")
        return []


def _simulate_outcomes(db, signals: List[Dict], since: datetime) -> List[Dict]:
    """
    For each BUY/HOLD signal, check if market went up by TP% before SL%.
    Uses the same TP=12% / SL=5% as the backtester's optimal params.

    Returns list of outcome dicts:
      { symbol, timestamp, rsi, ema_cross, macd_bull, vol_spike_flag,
        news_positive, prob, score, outcome: WIN|LOSS|UNKNOWN }
    """
    TP = 0.12
    SL = 0.05
    SIGNAL_WINDOW_HOURS = 24   # match candle data forward-look window

    # Pre-load market data for symbols in signals
    symbols = list({s.get('symbol') for s in signals if s.get('symbol')})
    market_map = {}   # symbol -> sorted list of {open_time, close, high, low}

    try:
        cursor = db[settings.COLLECTION_MARKET_DATA].find(
            {'symbol': {'$in': symbols},
             'open_time': {'$gte': since}},
            {'_id': 0, 'symbol': 1, 'open_time': 1, 'close': 1, 'high': 1, 'low': 1}
        ).sort('open_time', 1)
        for c in cursor:
            sym = c['symbol']
            if sym not in market_map:
                market_map[sym] = []
            market_map[sym].append(c)
    except Exception as e:
        logger.warning(f"[Learning] Market data load error: {e}")

    outcomes = []
    for sig in signals:
        sym       = sig.get('symbol', '')
        sig_ts    = sig.get('timestamp')
        score     = float(sig.get('final_score', 0) or 0)
        sig_label = sig.get('final_signal', '')

        # Only evaluate entry signals
        if sig_label not in ('BUY', 'STRONG BUY', 'HOLD'):
            continue
        if not sig_ts or not sym:
            continue

        # ── Indicator flags ──
        rsi          = float(sig.get('rsi', 50) or 50)
        ema20        = float(sig.get('ema20', 0) or 0)
        ema50        = float(sig.get('ema50', 0) or 0)
        macd         = float(sig.get('macd', 0) or 0)
        macd_sig_val = float(sig.get('macd_signal', 0) or 0)
        vol_spike    = float(sig.get('volume_spike', 1) or 1)
        news_score   = float(sig.get('news_score', 0) or 0)
        prob         = float(sig.get('probability_up', 50) or 50)

        rsi_bull     = 50 <= rsi <= 70
        ema_cross    = ema20 > ema50
        macd_bull    = macd > macd_sig_val
        vol_high     = vol_spike >= 1.5
        news_pos     = news_score > 0.1

        # ── Simulate forward outcome ──
        outcome = 'UNKNOWN'
        candles = market_map.get(sym, [])
        if candles and sig_ts:
            forward = [c for c in candles
                       if isinstance(c.get('open_time'), datetime)
                       and c['open_time'] > sig_ts
                       and c['open_time'] <= sig_ts + timedelta(hours=SIGNAL_WINDOW_HOURS)]
            if forward:
                entry = forward[0].get('close', 0)
                if entry > 0:
                    tp_price = entry * (1 + TP)
                    sl_price = entry * (1 - SL)
                    for c in forward:
                        if c.get('low', 0) <= sl_price:
                            outcome = 'LOSS'
                            break
                        if c.get('high', 0) >= tp_price:
                            outcome = 'WIN'
                            break

        outcomes.append({
            'symbol':       sym,
            'timestamp':    sig_ts,
            'score':        score,
            'prob':         prob,
            'rsi':          rsi,
            'rsi_bull':     rsi_bull,
            'ema_cross':    ema_cross,
            'macd_bull':    macd_bull,
            'vol_high':     vol_high,
            'news_pos':     news_pos,
            'outcome':      outcome,
        })

    return outcomes


# ══════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════

def _analyse_indicators(outcomes: List[Dict]) -> Dict:
    """
    Compute win rate for each indicator flag.
    Returns dict keyed by indicator name.
    """
    indicators = {
        'rsi':    {'flag': 'rsi_bull',  'label': 'RSI Bullish (50–70)'},
        'ema':    {'flag': 'ema_cross', 'label': 'EMA20 > EMA50 Cross'},
        'macd':   {'flag': 'macd_bull', 'label': 'MACD Bullish Cross'},
        'volume': {'flag': 'vol_high',  'label': 'Volume Spike ≥1.5×'},
        'news':   {'flag': 'news_pos',  'label': 'News Positive'},
    }

    # Filter only resolved outcomes
    resolved = [o for o in outcomes if o['outcome'] in ('WIN', 'LOSS')]
    if not resolved:
        logger.warning("[Learning] No resolved outcomes to analyse")
        return {}

    stats = {}
    for key, meta in indicators.items():
        flag = meta['flag']
        # Trades where this indicator fired
        active = [o for o in resolved if o.get(flag)]
        inactive = [o for o in resolved if not o.get(flag)]

        if len(active) < 3:
            # Not enough data — mark as neutral
            stats[key] = {
                'label':         meta['label'],
                'win_rate':      0.5,
                'trades':        len(active),
                'wins':          0,
                'losses':        0,
                'inactive_wr':   0.5,
                'contribution':  0.0,
                'confidence':    'LOW',
            }
            continue

        wins   = sum(1 for o in active if o['outcome'] == 'WIN')
        losses = len(active) - wins
        wr     = wins / len(active)

        inact_wins = sum(1 for o in inactive if o['outcome'] == 'WIN')
        inact_wr   = inact_wins / len(inactive) if inactive else 0.5

        # Contribution = win rate when active vs baseline
        baseline    = sum(1 for o in resolved if o['outcome'] == 'WIN') / len(resolved)
        contribution = wr - baseline   # positive = helps | negative = hurts

        confidence = 'HIGH' if len(active) >= 30 else 'MEDIUM' if len(active) >= 15 else 'LOW'

        stats[key] = {
            'label':        meta['label'],
            'win_rate':     round(wr, 4),
            'trades':       len(active),
            'wins':         wins,
            'losses':       losses,
            'inactive_wr':  round(inact_wr, 4),
            'contribution': round(contribution, 4),
            'confidence':   confidence,
        }

    # Probability: correlation — categorise prob buckets
    prob_buckets = [(35, 50), (50, 60), (60, 70), (70, 100)]
    prob_analysis = {}
    for lo, hi in prob_buckets:
        bucket = [o for o in resolved if lo <= o['prob'] < hi]
        if bucket:
            wr = sum(1 for o in bucket if o['outcome'] == 'WIN') / len(bucket)
            prob_analysis[f'prob_{lo}_{hi}'] = round(wr, 3)
    if prob_analysis:
        stats['probability'] = {
            'label':        'Probability Engine',
            'win_rate':     round(sum(prob_analysis.values()) / len(prob_analysis), 4),
            'trades':       len(resolved),
            'wins':         sum(1 for o in resolved if o['outcome'] == 'WIN'),
            'losses':       sum(1 for o in resolved if o['outcome'] == 'LOSS'),
            'bucket_wr':    prob_analysis,
            'contribution': 0.0,
            'confidence':   'HIGH' if len(resolved) >= 50 else 'MEDIUM',
        }

    return stats


def _analyse_thresholds(outcomes: List[Dict]) -> Dict:
    """Find optimal score and probability thresholds."""
    resolved = [o for o in outcomes if o['outcome'] in ('WIN', 'LOSS')]
    if not resolved:
        return {}

    best_score_threshold = 30
    best_score_wr        = 0.0
    for threshold in range(30, 80, 5):
        subset = [o for o in resolved if o['score'] >= threshold]
        if len(subset) < 5:
            continue
        wr = sum(1 for o in subset if o['outcome'] == 'WIN') / len(subset)
        if wr > best_score_wr:
            best_score_wr        = wr
            best_score_threshold = threshold

    best_prob_threshold = 30
    best_prob_wr        = 0.0
    for threshold in range(30, 80, 5):
        subset = [o for o in resolved if o['prob'] >= threshold]
        if len(subset) < 5:
            continue
        wr = sum(1 for o in subset if o['outcome'] == 'WIN') / len(subset)
        if wr > best_prob_wr:
            best_prob_wr        = wr
            best_prob_threshold = threshold

    return {
        'optimal_score_threshold': best_score_threshold,
        'score_threshold_wr':      round(best_score_wr, 4),
        'optimal_prob_threshold':  best_prob_threshold,
        'prob_threshold_wr':       round(best_prob_wr, 4),
    }


# ══════════════════════════════════════════════════════════════
# WEIGHT ADAPTATION
# ══════════════════════════════════════════════════════════════

def _compute_new_weights(current: Dict, stats: Dict) -> Dict:
    """
    Adjust weights based on indicator win rates.

    Rules:
      - win_rate > 0.60 → increase weight by up to 10%
      - win_rate < 0.40 → decrease weight by up to 10%
      - 0.40–0.60     → no change
      - Max ±10% change per cycle (safety cap)
      - Re-normalise to sum = 1.0 after adjustment
    """
    new = {k: v for k, v in current.items()}

    for key, stat in stats.items():
        if key not in new:
            continue
        if stat.get('trades', 0) < MIN_TRADES_REQUIRED:
            continue

        wr         = stat.get('win_rate', 0.5)
        confidence = stat.get('confidence', 'LOW')
        cur        = new[key]

        # Scale adjustment by confidence
        scale = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.2}.get(confidence, 0.2)

        if wr > 0.60:
            # Increase — proportional to excess win rate
            delta = min(MAX_WEIGHT_CHANGE_PCT, (wr - 0.60) * scale)
            new[key] = cur + delta
        elif wr < 0.40:
            # Decrease
            delta = min(MAX_WEIGHT_CHANGE_PCT, (0.40 - wr) * scale)
            new[key] = max(0.02, cur - delta)   # never zero
        # else: no change

    # Normalise so weights sum to 1.0
    total = sum(new.values())
    if total > 0:
        new = {k: round(v / total, 6) for k, v in new.items()}

    return new


def _describe_improvements(old: Dict, new: Dict, stats: Dict) -> List[str]:
    """Human-readable list of weight changes."""
    msgs = []
    for key in old:
        if key not in new:
            continue
        delta = new[key] - old[key]
        if abs(delta) < 0.001:
            continue
        direction = "↑ increased" if delta > 0 else "↓ decreased"
        wr = stats.get(key, {}).get('win_rate')
        wr_str = f" (win rate: {wr*100:.1f}%)" if wr else ""
        msgs.append(f"{key}: {old[key]:.3f} → {new[key]:.3f} {direction}{wr_str}")
    if not msgs:
        msgs.append("No weight changes needed — all indicators within normal range")
    return msgs


# ══════════════════════════════════════════════════════════════
# MONGODB PERSISTENCE
# ══════════════════════════════════════════════════════════════

def _load_current_weights(db) -> Dict:
    """Load latest model weights from DB, or return defaults."""
    try:
        doc = db[COLL_MODEL_WEIGHTS].find_one(
            {'type': 'current'},
            sort=[('cycle_ts', -1)],
            projection={'_id': 0, 'weights': 1}
        )
        if doc and doc.get('weights'):
            w = doc['weights']
            # Ensure all keys exist (forward compat)
            for k, v in DEFAULT_WEIGHTS.items():
                if k not in w:
                    w[k] = v
            return w
    except Exception as e:
        logger.warning(f"[Learning] Weight load error: {e}")
    return {k: v for k, v in DEFAULT_WEIGHTS.items()}


def _save_indicator_stats(db, stats: Dict, cycle_ts: datetime):
    """Upsert per-indicator stats into indicator_stats collection."""
    col = db[COLL_INDICATOR_STATS]
    try:
        for key, stat in stats.items():
            doc = {
                'indicator':     key,
                'cycle_ts':      cycle_ts,
                'updated_at':    datetime.utcnow(),
                **stat,
            }
            col.update_one(
                {'indicator': key},
                {'$set': doc},
                upsert=True
            )
        # Ensure index
        col.create_index([('indicator', 1)], unique=True, background=True)
        col.create_index([('cycle_ts', -1)], background=True)
        logger.info(f"[Learning] Saved {len(stats)} indicator stats")
    except Exception as e:
        logger.error(f"[Learning] Indicator stats save error: {e}")


def _save_model_weights(db, new_weights: Dict, backup: Dict,
                        stats: Dict, thresholds: Dict, cycle_ts: datetime):
    """
    Save new weights to model_weights collection.
    Always store a rollback backup document before overwriting current.
    """
    col = db[COLL_MODEL_WEIGHTS]
    try:
        # 1. Save rollback backup (never overwritten — timestamped)
        col.insert_one({
            'type':          'backup',
            'weights':       backup,
            'cycle_ts':      cycle_ts,
            'created_at':    datetime.utcnow(),
        })

        # 2. Upsert current weights (single canonical document)
        col.update_one(
            {'type': 'current'},
            {'$set': {
                'type':          'current',
                'weights':       new_weights,
                'thresholds':    thresholds,
                'indicator_wr':  {k: round(v.get('win_rate', 0.5), 4)
                                  for k, v in stats.items()},
                'cycle_ts':      cycle_ts,
                'updated_at':    datetime.utcnow(),
            }},
            upsert=True
        )

        # Ensure indexes
        col.create_index([('type', 1)], background=True)
        col.create_index([('cycle_ts', -1)], background=True)

        logger.info(f"[Learning] Saved model weights: {new_weights}")
    except Exception as e:
        logger.error(f"[Learning] Weight save error: {e}")


def _empty_result() -> Dict:
    return {
        'weights':         {k: v for k, v in DEFAULT_WEIGHTS.items()},
        'indicator_stats': {},
        'thresholds':      {},
        'improvements':    ['No learning data available'],
        'trade_count':     0,
        'cycle_ts':        datetime.utcnow(),
    }


# ══════════════════════════════════════════════════════════════
# FIX 2: XGBOOST LEARNING ENGINE
# ══════════════════════════════════════════════════════════════

MIN_XGB_SAMPLES = 50   # need this many resolved outcomes to use XGBoost

def _compute_new_weights_xgb(outcomes: List[Dict], current: Dict) -> Dict:
    """
    Fix 2: XGBoost-based weight computation.
    Trains a binary classifier (WIN=1, LOSS=0) on indicator feature combinations.
    Uses feature_importances_ as the new indicator weights.
    Falls back to statistical _compute_new_weights() if < MIN_XGB_SAMPLES.

    Features: rsi_bull, ema_cross, macd_bull, vol_high, news_pos, prob (normalised)
    """
    resolved = [o for o in outcomes if o.get('outcome') in ('WIN', 'LOSS')]
    if len(resolved) < MIN_XGB_SAMPLES:
        logger.info(f"[XGBoost] Only {len(resolved)} samples — using statistical fallback")
        stats = _analyse_indicators(outcomes)
        return _compute_new_weights(current, stats)

    try:
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        FEATURES = ['rsi_bull', 'ema_cross', 'macd_bull', 'vol_high', 'news_pos']

        X = np.array([
            [int(o.get(f, False)) for f in FEATURES] + [float(o.get('prob', 50)) / 100.0]
            for o in resolved
        ], dtype=float)
        y = np.array([1 if o['outcome'] == 'WIN' else 0 for o in resolved])

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )

        # Cross-validate to check if model is meaningful
        if len(resolved) >= 100:
            cv_scores = cross_val_score(xgb, X, y, cv=5, scoring='roc_auc')
            auc = cv_scores.mean()
            logger.info(f"[XGBoost] CV AUC={auc:.3f} on {len(resolved)} samples")
        else:
            auc = 0.5  # not enough for CV

        xgb.fit(X, y)
        importances = xgb.feature_importances_   # shape: [n_features]

        # Map importances to weight keys
        feat_names = FEATURES + ['probability']
        weight_map = {
            'rsi_bull':  'rsi',
            'ema_cross': 'ema',
            'macd_bull': 'macd',
            'vol_high':  'volume',
            'news_pos':  'news',
            'probability': 'probability',
        }

        # Build raw weights from feature importance
        raw = {}
        for i, feat in enumerate(feat_names):
            key = weight_map.get(feat, feat)
            if key in current:
                raw[key] = max(0.02, float(importances[i]))

        # Blend with current weights (70% XGBoost, 30% current) for stability
        # This prevents drastic swings in a single cycle
        blended = {}
        for key in current:
            xgb_w = raw.get(key, current[key])
            blended[key] = 0.70 * xgb_w + 0.30 * current[key]

        # Apply ±10% change cap (same safety as statistical method)
        capped = {}
        for key in current:
            cur = current[key]
            new_w = blended[key]
            delta = new_w - cur
            if abs(delta) > MAX_WEIGHT_CHANGE_PCT:
                new_w = cur + (MAX_WEIGHT_CHANGE_PCT * (1 if delta > 0 else -1))
            capped[key] = max(0.02, new_w)

        # Normalise to sum = 1.0
        total = sum(capped.values())
        if total > 0:
            capped = {k: round(v / total, 6) for k, v in capped.items()}

        logger.info(f"[XGBoost] New weights: {capped} (AUC={auc:.3f})")
        return capped

    except ImportError:
        logger.warning("[XGBoost] xgboost not installed — using statistical fallback")
        stats = _analyse_indicators(outcomes)
        return _compute_new_weights(current, stats)
    except Exception as e:
        logger.error(f"[XGBoost] Training error: {e} — using statistical fallback")
        stats = _analyse_indicators(outcomes)
        return _compute_new_weights(current, stats)


# ══════════════════════════════════════════════════════════════
# FIX 6: PER-COIN LEARNING
# ══════════════════════════════════════════════════════════════

def run_per_coin_learning(db, outcomes_all: List[Dict],
                          top_n: int = 10) -> Dict:
    """
    Fix 6: Train separate XGBoost models for the top_n most-traded coins.
    Stores per-coin weights in model_weights with type='per_coin'.
    Returns dict: {symbol: weights_dict}
    """
    if not outcomes_all:
        return {}

    # Count symbols
    from collections import Counter
    sym_counts = Counter(o['symbol'] for o in outcomes_all if o.get('symbol'))
    top_symbols = [s for s, _ in sym_counts.most_common(top_n)]

    per_coin_weights = {}
    cycle_ts = datetime.utcnow()
    global_weights = _load_current_weights(db)

    for sym in top_symbols:
        coin_outcomes = [o for o in outcomes_all if o.get('symbol') == sym]
        resolved = [o for o in coin_outcomes if o.get('outcome') in ('WIN', 'LOSS')]

        if len(resolved) < 20:
            logger.debug(f"[PerCoin] {sym}: only {len(resolved)} samples — skipping")
            continue

        weights = _compute_new_weights_xgb(coin_outcomes, global_weights)
        per_coin_weights[sym] = weights

        # Save to DB
        try:
            db[COLL_MODEL_WEIGHTS].update_one(
                {'type': 'per_coin', 'symbol': sym},
                {'$set': {
                    'type':       'per_coin',
                    'symbol':     sym,
                    'weights':    weights,
                    'cycle_ts':   cycle_ts,
                    'updated_at': datetime.utcnow(),
                    'sample_count': len(resolved),
                }},
                upsert=True
            )
            logger.info(f"[PerCoin] {sym}: weights saved ({len(resolved)} samples)")
        except Exception as e:
            logger.error(f"[PerCoin] DB save error {sym}: {e}")

    logger.info(f"[PerCoin] Trained {len(per_coin_weights)}/{len(top_symbols)} coins")
    return per_coin_weights


def get_current_weights(db=None, symbol: str = None) -> Dict:
    """
    Enhanced: check per-coin weights first (if symbol provided), fall back to global.
    Used by ranking_engine to get the best weights for a specific coin.
    """
    own_conn = (db is None)
    if own_conn:
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
        db = client[settings.DATABASE_NAME]
    try:
        # Try per-coin first
        if symbol:
            doc = db[COLL_MODEL_WEIGHTS].find_one(
                {'type': 'per_coin', 'symbol': symbol},
                {'_id': 0, 'weights': 1}
            )
            if doc and doc.get('weights'):
                logger.debug(f"[Weights] Using per-coin weights for {symbol}")
                return doc['weights']
        # Fall back to global
        return _load_current_weights(db)
    finally:
        if own_conn:
            client.close()



if __name__ == '__main__':
    import json

    print()
    print("=" * 65)
    print("  SELF-LEARNING AI — STANDALONE TEST RUN")
    print("=" * 65)
    print()

    result = run_learning_cycle(lookback_days=90)

    print(f"\nLearning cycle complete.")
    print(f"  Trade outcomes:  {result['trade_count']}")
    print(f"  Cycle time:      {result['cycle_ts'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("INDICATOR WIN RATES:")
    print(f"  {'Indicator':<14} {'Win Rate':>9} {'Trades':>7} {'W':>5} {'L':>5} "
          f"{'Contribution':>14} {'Conf':<8}")
    print(f"  {'-'*65}")
    for key, stat in result['indicator_stats'].items():
        wr    = stat.get('win_rate', 0) * 100
        tr    = stat.get('trades', 0)
        w     = stat.get('wins', 0)
        l     = stat.get('losses', 0)
        contr = stat.get('contribution', 0)
        conf  = stat.get('confidence', '')
        lbl   = stat.get('label', key)
        print(f"  {lbl:<30} {wr:>8.1f}% {tr:>7} {w:>5} {l:>5} "
              f"  {contr:>+.4f}     {conf:<8}")

    print()
    print("OPTIMAL THRESHOLDS:")
    t = result.get('thresholds', {})
    print(f"  Score threshold:  {t.get('optimal_score_threshold', 'N/A')} "
          f"(win rate: {t.get('score_threshold_wr', 0)*100:.1f}%)")
    print(f"  Prob threshold:   {t.get('optimal_prob_threshold', 'N/A')} "
          f"(win rate: {t.get('prob_threshold_wr', 0)*100:.1f}%)")

    print()
    print("NEW MODEL WEIGHTS:")
    for k, v in result['weights'].items():
        print(f"  {k:<14}: {v:.4f}")

    print()
    print("CHANGES:")
    for msg in result['improvements']:
        print(f"  • {msg}")

    print()

    # Verify DB
    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[settings.DATABASE_NAME]
    n_stats   = db[COLL_INDICATOR_STATS].count_documents({})
    n_weights = db[COLL_MODEL_WEIGHTS].count_documents({})
    print(f"MongoDB: indicator_stats={n_stats} docs | model_weights={n_weights} docs")
    client.close()

    print()
    print("=" * 65)
    print("  LEARNING CYCLE COMPLETE")
    print("=" * 65)
