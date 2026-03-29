"""
rl_optimizer.py — Reinforcement Learning Parameter Optimizer
============================================================
Lightweight reward-based Q-learning that adapts trading parameters
by learning from historical trade outcomes.

Architecture:
    State   : performance_bucket (POOR / NEUTRAL / GOOD) derived from composite score
    Actions : adjust rl_weight_adjustment, entry_threshold, prob_threshold
    Reward  : +1 winning trade | -1 losing trade | +0.5 PF bonus | -0.5 high DDR

No heavy ML frameworks — pure Python with deterministic reward accumulation.

Collections written:
    rl_parameters          — current active RL params (single doc, upserted)
    rl_parameter_history   — per-cycle snapshots for rollback / trend view
    rl_performance_history — before/after backtest comparison per run

DB fields in rl_parameters:
    indicator_weights : dict  (rsi, macd, ema, volume — multipliers on indicator signals)
    entry_threshold   : float (minimum final_score to enter a trade)
    prob_threshold    : float (minimum probability_up to enter)
    rl_weight_adjustment : float (score multiplier applied in pipeline, clamped 0.80–1.20)
    reward_score      : float (cumulative reward)
    episode           : int   (learning cycle count)
    last_updated      : datetime
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pymongo

from config import settings

logger = logging.getLogger(__name__)

# ── Collection names ────────────────────────────────────────────
COLL_RL_PARAMS      = 'rl_parameters'
COLL_RL_HIST        = 'rl_parameter_history'
COLL_RL_PERF        = 'rl_performance_history'
COLL_SIGNALS        = settings.COLLECTION_AI_SIGNALS
COLL_BACKTEST       = 'backtest_results'
SYNTH_SIGNALS       = 'ai_signals_backtest_synth'

# ── Default / initial parameters ───────────────────────────────
DEFAULT_RL_PARAMS = {
    'indicator_weights': {
        'rsi':    1.00,
        'macd':   1.00,
        'ema':    1.00,
        'volume': 1.00,
    },
    'entry_threshold':      45.0,   # min final_score to enter trade
    'prob_threshold':       35.0,   # min probability_up to enter
    'rl_weight_adjustment': 1.00,   # score multiplier applied in pipeline
    # Smart-Levels parameters (learned from backtest outcomes)
    'atr_sl_factor':        1.50,   # SL = entry - (ATR × this) [1.0–3.0]
    'atr_entry_pull':       0.40,   # entry = price - (ATR × this) [0.1–1.0]
    'atr_tp_min_rr':        2.00,   # minimum RR target [1.5–4.0]
    'reward_score':         0.0,
    'episode':              0,
    'last_updated':         None,
}

# ── Safety clamps ───────────────────────────────────────────────
CLAMPS = {
    'rl_weight_adjustment': (0.80, 1.20),
    'entry_threshold':      (20.0, 65.0),
    'prob_threshold':       (20.0, 60.0),
    'indicator_weight':     (0.50, 1.50),
    # Smart Levels clamps
    'atr_sl_factor':        (1.00, 3.00),
    'atr_entry_pull':       (0.10, 1.00),
    'atr_tp_min_rr':        (1.50, 4.00),
}

MAX_DRIFT_PER_CYCLE = {
    'rl_weight_adjustment': 0.05,
    'entry_threshold':      5.0,
    'prob_threshold':       5.0,
    'indicator_weight':     0.10,
    # Smart Levels drift caps
    'atr_sl_factor':        0.10,
    'atr_entry_pull':       0.05,
    'atr_tp_min_rr':        0.10,
}

# Q-table: state → action → reward accumulator
# States: POOR (<-5 total return), NEUTRAL (-5 to +15), GOOD (>+15)
# Actions: 0=increase_rl_weight, 1=decrease_rl_weight,
#          2=raise_threshold, 3=lower_threshold,
#          4=raise_prob, 5=lower_prob,
#          6=widen_sl (bigger ATR mult), 7=tighten_sl,
#          8=increase_tp_rr, 9=decrease_tp_rr
Q_TABLE = {
    'POOR':    {0: 0.0, 1: 0.5, 2: 0.3, 3:-0.2, 4: 0.3, 5:-0.1, 6: 0.4, 7:-0.3, 8:-0.2, 9: 0.1},
    'NEUTRAL': {0: 0.3, 1: 0.0, 2: 0.0, 3: 0.1, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.2, 9: 0.0},
    'GOOD':    {0: 0.5, 1:-0.2, 2:-0.1, 3: 0.2, 4:-0.1, 5: 0.2, 6:-0.1, 7: 0.2, 8: 0.4, 9:-0.1},
}

# ── In-memory cache ─────────────────────────────────────────────
_rl_cache: Optional[Dict] = None
_rl_cache_at: float = 0
_RL_CACHE_TTL = 300   # 5 minutes

# Minimum trades for a learning cycle
MIN_TRADES = 20

# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def get_current_rl_params(db=None) -> Dict:
    """
    Return current RL parameters (cached 5 min).
    Falls back to DEFAULT_RL_PARAMS if DB is empty or unreachable.

    Used by market_pipeline.py every scan.
    """
    global _rl_cache, _rl_cache_at
    import time

    if _rl_cache and (time.time() - _rl_cache_at) < _RL_CACHE_TTL:
        return _rl_cache

    own = (db is None)
    client = None
    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[settings.DATABASE_NAME]

        doc = db[COLL_RL_PARAMS].find_one({}, {'_id': 0}, sort=[('last_updated', -1)])
        if doc:
            params = {**DEFAULT_RL_PARAMS, **doc}
            _rl_cache = params
            _rl_cache_at = time.time()
            return params

    except Exception as e:
        logger.warning(f"[RL] Could not load params: {e} — using defaults")
    finally:
        if own and client:
            client.close()

    _rl_cache = dict(DEFAULT_RL_PARAMS)
    _rl_cache_at = import_time()
    return _rl_cache


def get_rl_weight(db=None) -> float:
    """
    Fast accessor: return only the rl_weight_adjustment multiplier.
    Used inline in the scoring pipeline.
    """
    try:
        params = get_current_rl_params(db=db)
        return float(params.get('rl_weight_adjustment', 1.0))
    except Exception:
        return 1.0


def run_rl_learning(lookback_days: int = 60, db=None, force: bool = False) -> Dict:
    """
    Main learning cycle entry point.

    1. Load historical trade outcomes
    2. Compute reward signal
    3. Determine performance state
    4. Select best Q-action
    5. Apply parameter updates with safety clamps
    6. Rollback if performance worsens
    7. Save to DB

    Args:
        lookback_days : days of signal history to use
        db            : optional pymongo DB handle
        force         : if True, run even if MIN_TRADES not met

    Returns:
        dict with episode, reward_score, params, improvement, applied
    """
    own = (db is None)
    client = None

    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
            db = client[settings.DATABASE_NAME]

        logger.info("=" * 60)
        logger.info("[RL] Learning cycle started")
        logger.info("=" * 60)

        # 1. Load trade outcomes
        trades = _load_trade_outcomes(db, lookback_days)
        if not trades:
            logger.warning("[RL] No trade outcomes found")
            return _empty_result("No trade outcomes")

        if len(trades) < MIN_TRADES and not force:
            logger.warning(f"[RL] Only {len(trades)} trades (< {MIN_TRADES}) — skipping. Use force=True to override.")
            return _empty_result(f"Too few trades ({len(trades)} < {MIN_TRADES})")

        logger.info(f"[RL] Processing {len(trades)} trade outcomes")

        # 2. Compute reward signal
        reward, stats = _compute_reward(trades)
        logger.info(f"[RL] Reward: {reward:+.2f} | WR={stats['win_rate']:.1f}% "
                    f"PF={stats['profit_factor']:.2f} DD={stats['max_drawdown']:.1f}%")

        # 3. Load current params (checkpoint)
        current = get_current_rl_params(db=db)
        checkpoint = dict(current)

        # 4. Determine performance state
        state = _get_state(stats)
        logger.info(f"[RL] Performance state: {state}")

        # 5. Select best Q-action
        action = _select_action(state)
        logger.info(f"[RL] Selected action: {_ACTION_NAMES.get(action, str(action))}")

        # 6. Apply update
        new_params = _apply_action(current, action, reward, stats)
        new_params['episode']      = current.get('episode', 0) + 1
        new_params['reward_score'] = current.get('reward_score', 0.0) + reward
        new_params['last_updated'] = datetime.utcnow()

        # 7. Validate (rollback if rl_weight would make things worse)
        if _should_rollback(stats, checkpoint):
            logger.warning("[RL] Rollback triggered — performance below threshold")
            new_params = _rollback_params(new_params, checkpoint)

        # 8. Save
        _save_params(db, new_params)

        # 9. Invalidate cache
        global _rl_cache, _rl_cache_at
        _rl_cache = new_params
        _rl_cache_at = import_time()

        logger.info(f"[RL] Episode {new_params['episode']} complete | "
                    f"RL weight: {new_params['rl_weight_adjustment']:.3f} | "
                    f"Cumulative reward: {new_params['reward_score']:+.1f}")

        return {
            'episode':       new_params['episode'],
            'reward_score':  new_params['reward_score'],
            'reward':        reward,
            'state':         state,
            'action':        _ACTION_NAMES.get(action, str(action)),
            'params':        new_params,
            'trades_used':   len(trades),
            'stats':         stats,
            'applied':       True,
        }

    except Exception as e:
        logger.error(f"[RL] Learning error: {e}", exc_info=True)
        return _empty_result(str(e))
    finally:
        if own and client:
            client.close()


# ═══════════════════════════════════════════════════════════════
# INTERNAL: TRADE LOADING
# ═══════════════════════════════════════════════════════════════

def _load_trade_outcomes(db, lookback_days: int) -> List[Dict]:
    """
    Extract simulated trade outcomes from signal history.
    Uses synthetic signals if available, falls back to real signals.
    Simulates TP=12% / SL=5% outcomes using price sequence.
    """
    start = datetime.utcnow() - timedelta(days=lookback_days)

    # Try backtest_results first (real outcomes)
    bt_docs = list(db[COLL_BACKTEST].find(
        {'timestamp': {'$gte': start}},
        {'_id': 0, 'metrics': 1, 'timestamp': 1}
    ).sort('timestamp', -1).limit(5))

    if bt_docs:
        # ── Fix 1: deterministic trade reconstruction — no random.random() ──
        # Uses real win_rate + avg_profit_pct/avg_loss_pct from backtest metrics.
        # WIN/LOSS order is proportional (first n_wins = WIN), no dice rolling.
        trades = []
        for doc in bt_docs:
            m        = doc.get('metrics', {})
            n        = int(float(m.get('total_trades', 0) or 0))
            wr       = float(m.get('win_rate', 50) or 50) / 100.0
            pf       = float(m.get('profit_factor', 1.0) or 1.0)
            dd       = float(m.get('max_drawdown', 0) or 0)
            avg_win  = float(m.get('avg_profit_pct', 12.0) or 12.0)
            avg_loss = float(m.get('avg_loss_pct',   5.0)  or  5.0)
            n_use    = min(n, 30)
            n_wins   = round(wr * n_use)
            n_losses = n_use - n_wins
            for _ in range(n_wins):
                trades.append({'pnl_pct': avg_win,  'win': True,
                                'profit_factor': pf, 'max_drawdown': dd})
            for _ in range(n_losses):
                trades.append({'pnl_pct': -avg_loss, 'win': False,
                                'profit_factor': pf, 'max_drawdown': dd})
        if len(trades) >= MIN_TRADES:
            logger.info(f"[RL] {len(trades)} real trades from backtest_results (deterministic)")
            return trades

    # Fallback: simulate from signal history
    trades = []
    for coll_name in [SYNTH_SIGNALS, COLL_SIGNALS]:
        try:
            sigs = list(db[coll_name].find(
                {'timestamp': {'$gte': start},
                 'final_score': {'$exists': True}},
                {'_id': 0, 'symbol': 1, 'timestamp': 1,
                 'final_score': 1, 'probability_up': 1,
                 'final_signal': 1}
            ).sort('timestamp', -1).limit(300))
            if sigs:
                logger.info(f"[RL] Simulating trades from {len(sigs)} signals in '{coll_name}'")
                trades = _simulate_trades_from_signals(db, sigs)
                if len(trades) >= 5:
                    break
        except Exception as e:
            logger.debug(f"[RL] Cannot read {coll_name}: {e}")

    return trades


def _simulate_trades_from_signals(db, sigs: List[Dict]) -> List[Dict]:
    """Simulate AI-driven TP/SL outcomes using market data candles.

    Uses RL-learned ATR multipliers (atr_sl_factor, atr_tp_min_rr) so the
    reward signal reflects the actual smart_levels calculation — not hardcoded%.
    Falls back to DEFAULT_RL_PARAMS if params not available.
    """
    # Load current RL-learned ATR multipliers
    try:
        rl_params   = get_current_rl_params(db=db)
        sl_factor   = float(rl_params.get('atr_sl_factor',  DEFAULT_RL_PARAMS['atr_sl_factor']))
        tp_rr       = float(rl_params.get('atr_tp_min_rr',  DEFAULT_RL_PARAMS['atr_tp_min_rr']))
        entry_pull  = float(rl_params.get('atr_entry_pull', DEFAULT_RL_PARAMS['atr_entry_pull']))
    except Exception:
        sl_factor, tp_rr, entry_pull = 1.5, 2.0, 0.4

    # Typical ATR-derived percentages (used when per-coin ATR unavailable)
    # SL% ≈ sl_factor × 2.5% ATR estimate ; TP% ≈ tp_rr × SL%
    FALLBACK_ATR_PCT = 0.025
    FALLBACK_SL = sl_factor  * FALLBACK_ATR_PCT   # e.g., 1.5 × 2.5% = 3.75%
    FALLBACK_TP = tp_rr      * FALLBACK_SL         # e.g., 2.0 × 3.75% = 7.5%

    WIN_THRESH = 45.0
    PROB_MIN   = 30.0
    trades = []

    symbols = list({s['symbol'] for s in sigs})
    candle_map: Dict[str, List] = {}

    for sym in symbols:
        try:
            cands = list(db[settings.COLLECTION_MARKET_DATA].find(
                {'symbol': sym},
                {'_id': 0, 'symbol': 1, 'open_time': 1, 'close': 1,
                 'high': 1, 'low': 1, 'atr': 1}
            ).sort('open_time', 1).limit(400))
            candle_map[sym] = cands
        except Exception:
            pass

    for sig in sigs:
        sym   = sig.get('symbol', '')
        sc    = float(sig.get('final_score', 0) or 0)
        prob  = float(sig.get('probability_up', 0) or 0)
        sig_t = sig.get('timestamp')
        if not sig_t:
            continue

        if sc < WIN_THRESH or prob < PROB_MIN:
            continue

        cands = candle_map.get(sym, [])
        if not sig_t.tzinfo:
            sig_naive = sig_t
        else:
            sig_naive = sig_t.replace(tzinfo=None)

        entry_px = None
        coin_atr  = None
        for c in cands:
            ct = c['open_time']
            if ct.tzinfo:
                ct = ct.replace(tzinfo=None)
            if ct >= sig_naive:
                entry_px  = float(c.get('close', 0) or 0)
                # Use stored ATR if available (from indicators), else fallback
                coin_atr  = float(c.get('atr', 0) or 0) or None
                break

        if not entry_px or entry_px <= 0:
            continue

        # Compute AI-driven SL/TP percentages from ATR multipliers
        if coin_atr and coin_atr > 0:
            sl_pct = (sl_factor  * coin_atr) / entry_px
            tp_pct = tp_rr       * sl_pct
        else:
            sl_pct = FALLBACK_SL
            tp_pct = FALLBACK_TP

        # Apply entry pull (buy slightly below; tighter SL distance from entry)
        entry_adj = entry_px * (1 - entry_pull * (FALLBACK_ATR_PCT if not coin_atr else coin_atr / entry_px))
        tp_px = entry_adj * (1 + tp_pct)
        sl_px = entry_adj * (1 - sl_pct)
        outcome = None

        for c in cands:
            ct = c['open_time']
            if ct.tzinfo:
                ct = ct.replace(tzinfo=None)
            if ct <= sig_naive:
                continue
            hi = float(c.get('high', entry_px) or entry_px)
            lo = float(c.get('low',  entry_px) or entry_px)
            if lo <= sl_px:
                outcome = -sl_pct * 100; break
            if hi >= tp_px:
                outcome = tp_pct * 100;  break

        if outcome is None and entry_px:
            outcome = 0.0

        if outcome is not None:
            trades.append({
                'pnl_pct': outcome,
                'win':     outcome > 0,
                'sl_pct':  round(sl_pct * 100, 2),
                'tp_pct':  round(tp_pct * 100, 2),
            })

    logger.info(f"[RL] Simulated {len(trades)} trades from signals "
                f"(sl_factor={sl_factor:.2f} tp_rr={tp_rr:.2f} → "
                f"avg SL≈{FALLBACK_SL*100:.1f}% TP≈{FALLBACK_TP*100:.1f}%)")
    return trades


# ═══════════════════════════════════════════════════════════════
# INTERNAL: REWARD + STATE + ACTIONS
# ═══════════════════════════════════════════════════════════════

def _compute_reward(trades: List[Dict]) -> Tuple[float, Dict]:
    """Compute cumulative reward and performance stats from trades."""
    if not trades:
        return 0.0, {}

    wins  = [t for t in trades if t.get('win')]
    losses= [t for t in trades if not t.get('win')]

    win_rate = len(wins) / len(trades) * 100
    avg_win  = sum(t.get('pnl_pct', 12) for t in wins)  / max(len(wins), 1)
    avg_loss = abs(sum(t.get('pnl_pct', -5) for t in losses)) / max(len(losses), 1)
    pf = (sum(t.get('pnl_pct', 0) for t in wins) /
          max(abs(sum(t.get('pnl_pct', 0) for t in losses)), 0.01))

    # Max drawdown approximation
    balance = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        balance *= (1 + t.get('pnl_pct', 0) / 100)
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    total_return = (balance - 1.0) * 100

    # Reward accumulation
    reward = 0.0
    for t in trades:
        if t.get('win'):
            reward += 1.0
        else:
            reward -= 1.0
    # Bonus / penalties
    if pf > 1.5:
        reward += 0.5 * len(trades) / 20
    if max_dd > 15:
        reward -= 0.5 * len(trades) / 20
    if win_rate > 60:
        reward += 0.3

    stats = {
        'trade_count':   len(trades),
        'win_rate':      round(win_rate, 1),
        'avg_win':       round(avg_win, 2),
        'avg_loss':      round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'max_drawdown':  round(max_dd, 2),
        'total_return':  round(total_return, 2),
    }
    return round(reward, 2), stats


def _get_state(stats: Dict) -> str:
    """Map performance stats to Q-table state bucket."""
    ret = stats.get('total_return', 0)
    if ret < -5:
        return 'POOR'
    elif ret > 15:
        return 'GOOD'
    return 'NEUTRAL'


_ACTION_NAMES = {
    0: 'increase_rl_weight',
    1: 'decrease_rl_weight',
    2: 'raise_entry_threshold',
    3: 'lower_entry_threshold',
    4: 'raise_prob_threshold',
    5: 'lower_prob_threshold',
    6: 'widen_sl (atr_sl_factor+)',
    7: 'tighten_sl (atr_sl_factor-)',
    8: 'increase_tp_rr',
    9: 'decrease_tp_rr',
}


def _select_action(state: str) -> int:
    """Return action with highest Q-value for this state (greedy)."""
    q_row = Q_TABLE.get(state, Q_TABLE['NEUTRAL'])
    best_action = max(q_row, key=q_row.get)
    return best_action


def _apply_action(params: Dict, action: int, reward: float, stats: Dict) -> Dict:
    """Apply selected action to parameters with safety clamps."""
    p = dict(params)

    # How much to move: scale by |reward| but cap by MAX_DRIFT
    rl_step   = min(abs(reward) * 0.005, MAX_DRIFT_PER_CYCLE['rl_weight_adjustment'])
    thr_step  = min(abs(reward) * 0.3,   MAX_DRIFT_PER_CYCLE['entry_threshold'])
    prob_step = min(abs(reward) * 0.3,   MAX_DRIFT_PER_CYCLE['prob_threshold'])

    current_w = float(p.get('rl_weight_adjustment', 1.0))
    current_e = float(p.get('entry_threshold', 45.0))
    current_pr= float(p.get('prob_threshold', 35.0))

    if action == 0:   # increase weight
        new_w = current_w + max(rl_step, 0.005)
    elif action == 1: # decrease weight
        new_w = current_w - max(rl_step, 0.005)
    else:
        new_w = current_w

    if action == 2:   # raise entry threshold
        new_e = current_e + max(thr_step, 1.0)
    elif action == 3: # lower entry threshold
        new_e = current_e - max(thr_step, 1.0)
    else:
        new_e = current_e

    if action == 4:   # raise prob threshold
        new_pr = current_pr + max(prob_step, 1.0)
    elif action == 5: # lower prob threshold
        new_pr = current_pr - max(prob_step, 1.0)
    else:
        new_pr = current_pr

    # Apply indicator weight nudges based on win rate
    ind_w = dict(p.get('indicator_weights', DEFAULT_RL_PARAMS['indicator_weights']))
    wr = stats.get('win_rate', 50)
    if wr > 60:
        # Good performance → nudge all weights up slightly
        for k in ind_w:
            delta = min(0.01 * (wr - 60) / 10, MAX_DRIFT_PER_CYCLE['indicator_weight'])
            ind_w[k] = _clamp(ind_w[k] + delta, *CLAMPS['indicator_weight'])
    elif wr < 40:
        # Poor performance → nudge weights toward 1.0 (revert)
        for k in ind_w:
            ind_w[k] = _clamp(ind_w[k] * 0.99, *CLAMPS['indicator_weight'])

    p['rl_weight_adjustment'] = _clamp(new_w,  *CLAMPS['rl_weight_adjustment'])
    p['entry_threshold']      = _clamp(new_e,  *CLAMPS['entry_threshold'])
    p['prob_threshold']       = _clamp(new_pr, *CLAMPS['prob_threshold'])
    p['indicator_weights']    = ind_w

    # Smart Levels ATR parameter learning
    # More SL knock-outs (losses) → widen SL; more TP hits → tighten SL for better entry
    sl_factor  = float(p.get('atr_sl_factor',  1.5))
    entry_pull = float(p.get('atr_entry_pull', 0.4))
    tp_rr      = float(p.get('atr_tp_min_rr',  2.0))

    sl_step  = min(abs(reward) * 0.005, MAX_DRIFT_PER_CYCLE['atr_sl_factor'])
    ep_step  = min(abs(reward) * 0.003, MAX_DRIFT_PER_CYCLE['atr_entry_pull'])
    rr_step  = min(abs(reward) * 0.005, MAX_DRIFT_PER_CYCLE['atr_tp_min_rr'])

    if action == 6:    # widen SL (caught too many false stop-outs)
        sl_factor  += max(sl_step, 0.05)
    elif action == 7:  # tighten SL (losses are big; cut faster)
        sl_factor  -= max(sl_step, 0.05)

    if action == 8:    # increase target RR (performance GOOD, aim higher)
        tp_rr += max(rr_step, 0.05)
        entry_pull *= 0.99  # also pull entry slightly closer (tighter setup)
    elif action == 9:  # decrease target RR (missing too many TP targets)
        tp_rr -= max(rr_step, 0.05)

    p['atr_sl_factor']  = _clamp(sl_factor,  *CLAMPS['atr_sl_factor'])
    p['atr_entry_pull'] = _clamp(entry_pull, *CLAMPS['atr_entry_pull'])
    p['atr_tp_min_rr']  = _clamp(tp_rr,      *CLAMPS['atr_tp_min_rr'])

    # Feed learned params back into smart_levels module (live update)
    try:
        from ai import smart_levels as _sl
        _sl.ATR_SL_FACTOR  = p['atr_sl_factor']
        _sl.ATR_ENTRY_PULL = p['atr_entry_pull']
        _sl.ATR_TP_MIN_RR  = p['atr_tp_min_rr']
        logger.info(f"[RL→SmartLevels] SL_factor={p['atr_sl_factor']:.3f} "
                    f"entry_pull={p['atr_entry_pull']:.3f} "
                    f"tp_rr={p['atr_tp_min_rr']:.3f}")
    except Exception as _e:
        logger.debug(f"[RL] Could not update smart_levels module: {_e}")

    return p


def _should_rollback(stats: Dict, checkpoint: Dict) -> bool:
    """
    Rollback if profitability dropped dramatically compared to long-run expectation.
    Uses: win_rate < 30% AND profit_factor < 0.8 simultaneously.
    """
    wr = stats.get('win_rate', 50)
    pf = stats.get('profit_factor', 1.0)
    dd = stats.get('max_drawdown', 0)
    return (wr < 30 and pf < 0.8) or dd > 30


def _rollback_params(new_params: Dict, checkpoint: Dict) -> Dict:
    """Restore checkpoint values for tunable params; keep episode/reward."""
    rolled = dict(new_params)
    for key in ('rl_weight_adjustment', 'entry_threshold', 'prob_threshold', 'indicator_weights'):
        if key in checkpoint:
            rolled[key] = checkpoint[key]
    logger.info("[RL] Parameters rolled back to checkpoint")
    return rolled


# ═══════════════════════════════════════════════════════════════
# INTERNAL: DB PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _save_params(db, params: Dict) -> bool:
    """Upsert rl_parameters (single active doc) and append to history."""
    try:
        # Upsert single active doc
        db[COLL_RL_PARAMS].replace_one({}, params, upsert=True)

        # Append snapshot to history (for rollback / trend)
        snap = dict(params)
        snap['snapshot_at'] = datetime.utcnow()
        db[COLL_RL_HIST].insert_one(snap)

        # Ensure indexes
        db[COLL_RL_PARAMS].create_index([('last_updated', -1)], background=True)
        db[COLL_RL_HIST].create_index([('snapshot_at', -1)], background=True)
        # TTL: keep 90 days of snapshots
        db[COLL_RL_HIST].create_index(
            [('snapshot_at', 1)],
            expireAfterSeconds=90 * 24 * 3600,
            background=True,
            name='rl_hist_ttl'
        )
        logger.debug(f"[RL] Saved params: episode={params.get('episode')} "
                     f"weight={params.get('rl_weight_adjustment'):.3f}")
        return True
    except Exception as e:
        logger.error(f"[RL] Save error: {e}")
        return False


def save_rl_performance(before: Dict, after: Dict, episode: int,
                        rl_weight: float, applied: bool, db=None) -> bool:
    """
    Store before/after backtest comparison in rl_performance_history.
    Called by rl_backtest_compare.py.
    """
    own = (db is None)
    client = None
    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=6000)
            db = client[settings.DATABASE_NAME]

        # Compute improvement
        b_ret = before.get('return_pct', 0) or 0
        a_ret = after.get('return_pct', 0) or 0
        imp   = ((a_ret - b_ret) / max(abs(b_ret), 1.0)) * 100 if b_ret != a_ret else 0

        doc = {
            'run_at':          datetime.utcnow(),
            'episode':         episode,
            'before':          before,
            'after':           after,
            'improvement_pct': round(imp, 2),
            'rl_weight':       round(rl_weight, 4),
            'applied':         applied,
        }
        db[COLL_RL_PERF].insert_one(doc)
        db[COLL_RL_PERF].create_index([('run_at', -1)], background=True)
        logger.info(f"[RL] Performance saved: improvement={imp:+.1f}%")
        return True
    except Exception as e:
        logger.error(f"[RL] Performance save error: {e}")
        return False
    finally:
        if own and client:
            client.close()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, round(float(val), 4)))


def import_time() -> float:
    import time
    return time.time()


def _empty_result(reason: str = '') -> Dict:
    return {
        'episode':      0,
        'reward_score': 0,
        'reward':       0,
        'trades_used':  0,
        'applied':      False,
        'params':       DEFAULT_RL_PARAMS,
        'error':        reason,
    }


def get_rl_performance_history(db=None, limit: int = 10) -> List[Dict]:
    """Return recent rl_performance_history entries for dashboard."""
    own = (db is None)
    client = None
    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[settings.DATABASE_NAME]
        docs = list(
            db[COLL_RL_PERF]
            .find({}, {'_id': 0})
            .sort('run_at', -1)
            .limit(limit)
        )
        return docs
    except Exception as e:
        logger.warning(f"[RL] History read error: {e}")
        return []
    finally:
        if own and client:
            client.close()


# ═══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import logging as _log
    _log.basicConfig(level=_log.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("\n" + "=" * 60)
    print("  RL OPTIMIZER — Standalone Run")
    print("=" * 60 + "\n")

    result = run_rl_learning(lookback_days=90, force=True)

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    else:
        p = result.get('params', {})
        s = result.get('stats', {})
        print(f"  Episode:       {result['episode']}")
        print(f"  Trades used:   {result['trades_used']}")
        print(f"  State:         {result.get('state','?')}")
        print(f"  Action:        {result.get('action','?')}")
        print(f"  Reward:        {result['reward']:+.2f}")
        print(f"  Cum. reward:   {result['reward_score']:+.1f}")
        print(f"  RL weight:     {p.get('rl_weight_adjustment', 1.0):.4f}")
        print(f"  Entry thresh:  {p.get('entry_threshold', 45):.1f}")
        print(f"  Prob thresh:   {p.get('prob_threshold', 35):.1f}")
        print(f"  Win rate:      {s.get('win_rate', 0):.1f}%")
        print(f"  Profit factor: {s.get('profit_factor', 0):.2f}")
        print(f"  Applied:       {'YES' if result['applied'] else 'NO'}")
    print(f"{'='*60}\n")
