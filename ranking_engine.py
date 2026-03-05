"""
Ranking Engine — Best Coin Ranking AI System
============================================
Identifies and ranks the TOP crypto trading opportunities using a weighted
composite score from: technical indicators, probability engine, news sentiment,
volume strength, and trend strength.

Formula:
  rank_score = (technical_score * 0.35) +
               (probability_up    * 0.30) +
               (news_sentiment    * 0.15) +
               (volume_strength   * 0.10) +
               (trend_strength    * 0.10)

Entry Filters:
  - signal IN ['BUY', 'STRONG_BUY', 'HOLD']
  - technical_score >= 40
  - probability_up  >= 35
  - volume > volume_ma
  - price > ema20

Outputs:
  - Top 3, Top 5, Full list
  - Saves to MongoDB `ranked_opportunities` collection
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import pymongo

from config import settings
from ai.smart_levels import compute_smart_levels

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
COLLECTION = "ranked_opportunities"

# Weights must sum to 1.0
WEIGHT_TECHNICAL  = 0.35
WEIGHT_PROB_UP    = 0.30
WEIGHT_NEWS       = 0.15
WEIGHT_VOLUME     = 0.10
WEIGHT_TREND      = 0.10

# Entry filters
FILTER_SIGNALS    = {'BUY', 'STRONG_BUY', 'HOLD', 'SELL'}  # Include SELL — in BEAR markets all coins are SELL
FILTER_MIN_TECH   = 40
FILTER_MIN_PROB   = 35


# ══════════════════════════════════════════════════════════════
# NORMALIZER HELPERS
# ══════════════════════════════════════════════════════════════

def _norm(value: float, lo: float, hi: float) -> float:
    """Normalise value to 0–100 given a known range."""
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100))


def _compute_volume_strength(coin: Dict) -> float:
    """
    Volume strength 0–100.
    Uses volume_spike field (already calculated by market_pipeline).
    If volume_spike >= 2 → very high; 1 → average.
    """
    spike = float(coin.get('volume_spike', 1.0) or 1.0)
    # Map: 0×→0, 1×→50, 2×→75, 3+×→100
    return min(100.0, (spike / 3.0) * 100.0)


def _compute_trend_strength(coin: Dict) -> float:
    """
    Trend strength 0–100.
    price > ema20 > ema50  → strong uptrend (100)
    price > ema20 only     → moderate (65)
    price < ema20          → downtrend (0)
    Also accounts for breakout_score.
    """
    price  = float(coin.get('price', 0) or 0)
    ema20  = float(coin.get('ema20', price) or price)
    ema50  = float(coin.get('ema50', ema20) or ema20)
    bs     = float(coin.get('breakout_score', 0) or 0)  # 0–100

    if price <= 0 or ema20 <= 0:
        return 50.0

    if price > ema20 and ema20 > ema50:
        base = 80.0
    elif price > ema20:
        base = 60.0
    elif price > ema50:
        base = 40.0
    else:
        base = 15.0

    # Blend in breakout score (20% weight)
    return min(100.0, base * 0.8 + bs * 0.2)


def _normalize_news_score(coin: Dict) -> float:
    """
    News sentiment → 0–100.
    news_score is in [-1, 1]; map to [0, 100].
    """
    ns = float(coin.get('news_score', 0) or 0)
    return (ns + 1.0) / 2.0 * 100.0


# NOTE: _compute_entry_levels() replaced by ai.smart_levels.compute_smart_levels()
# Kept as fallback for direct calls from legacy code.
def _compute_entry_levels(price: float, tp_pct: float = 0.12,
                           sl_pct: float = 0.05) -> Dict:
    """Legacy fallback. Use compute_smart_levels() for AI-driven calculation."""
    entry  = round(price, 8)
    tp     = round(price * (1 + tp_pct), 8)
    sl     = round(price * (1 - sl_pct), 8)
    rr     = round(tp_pct / sl_pct, 2)
    return {'entry_price': entry, 'take_profit': tp, 'stop_loss': sl, 'risk_reward_ratio': rr}


# ══════════════════════════════════════════════════════════════
# MAIN RANKING FUNCTION
# ══════════════════════════════════════════════════════════════

def rank_coins(scan_results: List[Dict]) -> List[Dict]:
    """
    Takes the full market scan output and returns ranked list.

    Parameters
    ----------
    scan_results : List[Dict]
        Output from market_pipeline.scan_market()

    Returns
    -------
    List[Dict]
        All qualifying coins sorted by rank_score descending.
        Each dict has: symbol, rank_score, signal, technical_score,
        probability_up, news_sentiment, news_sentiment_normalised,
        volume_strength, trend_strength, entry_price, stop_loss,
        take_profit, risk_reward_ratio, created_at.
    """
    ranked = []

    for coin in scan_results:
        # ── 1. Entry filters ──
        signal = coin.get('final_signal', coin.get('signal', ''))
        if signal not in FILTER_SIGNALS:
            continue

        tech  = float(coin.get('technical_score', coin.get('profit_score', 0)) or 0)
        prob  = float(coin.get('probability_up', 0) or 0)
        price = float(coin.get('price', 0) or 0)
        ema20 = float(coin.get('ema20', price) or price)
        vol   = float(coin.get('volume', 0) or 0)

        if tech < FILTER_MIN_TECH:
            continue
        if prob < FILTER_MIN_PROB:
            continue
        if price <= 0:
            continue
        # price > ema20 trend filter (with 1% slack)
        if ema20 > 0 and price < ema20 * 0.99:
            continue

        # ── 2. Compute component scores (0–100) ──
        tech_norm  = min(100.0, tech)
        prob_norm  = min(100.0, prob)
        news_norm  = _normalize_news_score(coin)
        vol_str    = _compute_volume_strength(coin)
        trend_str  = _compute_trend_strength(coin)

        # ── 3. Composite rank score ──
        rank_score = (
            tech_norm  * WEIGHT_TECHNICAL +
            prob_norm  * WEIGHT_PROB_UP   +
            news_norm  * WEIGHT_NEWS      +
            vol_str    * WEIGHT_VOLUME    +
            trend_str  * WEIGHT_TREND
        )
        rank_score = round(rank_score, 2)

        # ── 4. AI-Driven Entry / SL / TP (ai.smart_levels) ────────────────────
        # Uses ATR, RSI, EMA20/50, probability_up, whale_signal,
        # news_score, market_regime, nearest_support/resistance.
        # Zero hardcoded percentages.
        levels = compute_smart_levels(coin)

        # ── 5. Composite rank score (add trade quality as tiebreaker) ───
        quality = levels.get('trade_quality_score', 0)
        # Blend trade quality into rank: 5% weight on quality (0–100)
        rank_score_final = round(rank_score * 0.95 + quality * 0.05, 2)

        entry = {
            'symbol':                  coin.get('symbol', ''),
            'rank_score':              rank_score_final,
            'raw_rank_score':          rank_score,         # pre-quality blend
            'signal':                  signal,
            'final_signal':            signal,
            'technical_score':         round(tech, 1),
            'probability_up':          round(prob, 1),
            'probability_down':        round(float(coin.get('probability_down', 100 - prob) or 0), 1),
            'news_sentiment':          coin.get('news_sentiment', 'NEUTRAL'),
            'news_score':              round(float(coin.get('news_score', 0) or 0), 3),
            'news_sentiment_norm':     round(news_norm, 1),
            'volume_strength':         round(vol_str, 1),
            'trend_strength':          round(trend_str, 1),
            'final_score':             float(coin.get('final_score', 0) or 0),
            # Risk model fields
            'risk_adjusted_score':     float(coin.get('risk_adjusted_score', 0) or 0),
            'risk_score':              float(coin.get('risk_score', 0) or 0),
            'risk_level':              coin.get('risk_level', ''),
            # Indicators — stored so dashboard can feed them into smart_levels
            'price':                   price,
            'rsi':                     float(coin.get('rsi', 0) or 0),
            'rsi_4h':                  float(coin.get('rsi_4h', 50) or 50),
            'ema20':                   float(coin.get('ema20', 0) or 0) or None,
            'ema50':                   float(coin.get('ema50', 0) or 0) or None,
            'atr':                     float(coin.get('atr', 0) or 0) or levels.get('levels_atr') or None,
            'atr_pct':                 levels.get('levels_atr_pct', 0),
            'volume_spike':            float(coin.get('volume_spike', 1) or 1),
            'volatility':              float(coin.get('volatility', 0) or 0),
            'nearest_support':         coin.get('nearest_support'),
            'nearest_resistance':      coin.get('nearest_resistance'),
            'mtf_confirmed':           bool(coin.get('mtf_confirmed', False)),
            # Context
            'whale_signal':            coin.get('whale_signal', 'NONE'),
            'whale_score':             float(coin.get('whale_score', 50) or 50),
            'whale_buy_pressure':      float(coin.get('whale_buy_pressure', 0) or 0),
            'whale_sell_pressure':     float(coin.get('whale_sell_pressure', 0) or 0),
            'market_regime':           coin.get('market_regime', 'UNKNOWN'),
            # AI-Driven Entry Levels (computed at scan time — re-computed live in dashboard)
            'entry_price':             levels['entry_price'],
            'stop_loss':               levels['stop_loss'],
            'take_profit':             levels['take_profit'],
            'risk_reward_ratio':       levels['risk_reward_ratio'],
            'trade_quality_score':     levels.get('trade_quality_score', 0),
            # AI Level Transparency (what drove the calculation)
            'entry_logic':             levels.get('entry_logic', ''),
            'sl_logic':                levels.get('sl_logic', ''),
            'tp_logic':                levels.get('tp_logic', ''),
            'created_at':              datetime.utcnow(),
        }
        ranked.append(entry)

    # Sort descending by rank_score
    ranked.sort(key=lambda x: x['rank_score'], reverse=True)

    # Assign rank position
    for i, r in enumerate(ranked, 1):
        r['rank'] = i

    logger.info(f"[Ranking] {len(scan_results)} coins scanned → {len(ranked)} qualified")
    if ranked:
        top = ranked[0]
        logger.info(f"[Ranking] Top coin: {top['symbol']} score={top['rank_score']}")
    return ranked


# ══════════════════════════════════════════════════════════════
# MONGODB PERSISTENCE
# ══════════════════════════════════════════════════════════════

def save_rankings(ranked: List[Dict], db, all_scanned_symbols: List[str] = None) -> int:
    """
    Save ranked opportunities to MongoDB.
    Two-layer stale cleanup:
      1. Delete records for symbols that were scanned but filtered out this cycle.
      2. Delete ANY record older than 20 min that wasn't refreshed (TTL purge).
         This catches coins that silently dropped out (SELL signal, errors, etc.)
    """
    if db is None:
        return 0
    try:
        from datetime import timedelta
        col = db[COLLECTION]

        # Ensure indexes
        col.create_index([('rank_score', pymongo.DESCENDING)])
        col.create_index([('symbol', pymongo.ASCENDING)], unique=True)
        col.create_index([('created_at', pymongo.DESCENDING)])

        current_ranked_symbols = {r['symbol'] for r in ranked}

        # ── Layer 1: delete symbols scanned-but-filtered ──────────────────────
        if all_scanned_symbols:
            stale_symbols = [s for s in all_scanned_symbols if s not in current_ranked_symbols]
            if stale_symbols:
                r1 = col.delete_many({'symbol': {'$in': stale_symbols}})
                if r1.deleted_count:
                    logger.info(f"[Ranking] Removed {r1.deleted_count} filtered-out symbols")

        # ── Layer 2: TTL purge — remove records not refreshed in this cycle ──
        # Any record whose created_at is more than 20 min old AND is NOT in
        # the current ranked set is definitely stale (price has changed).
        cutoff = datetime.utcnow() - timedelta(minutes=20)
        r2 = col.delete_many({
            'created_at': {'$lt': cutoff},
            'symbol': {'$nin': list(current_ranked_symbols)}
        })
        if r2.deleted_count:
            logger.info(f"[Ranking] TTL purge: removed {r2.deleted_count} stale records "
                        f"(older than 20 min, not in current top)")

        if not ranked:
            return 0

        # ── Upsert qualifying coins ────────────────────────────────────────────
        batch_ts = datetime.utcnow()
        inserted = 0
        for doc in ranked:
            doc['batch_ts'] = batch_ts
            col.update_one(
                {'symbol': doc['symbol']},
                {'$set': doc},
                upsert=True
            )
            inserted += 1

        logger.info(f"[Ranking] Saved {inserted} ranked coins to '{COLLECTION}'")
        return inserted
    except Exception as e:
        logger.error(f"[Ranking] Save error: {e}")
        return 0


# ══════════════════════════════════════════════════════════════
# QUERY HELPERS
# ══════════════════════════════════════════════════════════════

def get_top_opportunities(db, n: int = 10) -> List[Dict]:
    """Load top-N ranked coins from MongoDB, sorted by rank_score."""
    try:
        col = db[COLLECTION]
        docs = list(col.find({}, {'_id': 0}).sort('rank_score', pymongo.DESCENDING).limit(n))
        return docs
    except Exception as e:
        logger.error(f"[Ranking] Query error: {e}")
        return []


def get_ranking_summary(db) -> Dict:
    """Return summary stats about current rankings."""
    try:
        col = db[COLLECTION]
        total = col.count_documents({})
        buy_count  = col.count_documents({'signal': 'BUY'})
        hold_count = col.count_documents({'signal': 'HOLD'})
        best = col.find_one({}, {'symbol': 1, 'rank_score': 1, 'signal': 1, '_id': 0},
                             sort=[('rank_score', pymongo.DESCENDING)])
        last = col.find_one({}, {'created_at': 1, '_id': 0},
                             sort=[('created_at', pymongo.DESCENDING)])
        return {
            'total_ranked': total,
            'buy_count': buy_count,
            'hold_count': hold_count,
            'top_coin': best,
            'last_updated': last.get('created_at') if last else None,
        }
    except Exception as e:
        logger.error(f"[Ranking] Summary error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════

def run_ranking_from_db(db) -> List[Dict]:
    """
    Load latest AI signals from DB and re-rank them.
    Used when ranking needs to run independently of live scan.
    """
    try:
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
            {'$replaceRoot': {'newRoot': '$doc'}},
        ]
        signals = list(db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
        logger.info(f"[Ranking] Loaded {len(signals)} latest signals from DB")
        ranked = rank_coins(signals)
        saved  = save_rankings(ranked, db)
        return ranked
    except Exception as e:
        logger.error(f"[Ranking] run_ranking_from_db error: {e}")
        return []


if __name__ == '__main__':
    import pymongo as _pymongo
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[settings.DATABASE_NAME]

    print("=" * 60)
    print("  RANKING ENGINE — STANDALONE RUN")
    print("=" * 60)

    ranked = run_ranking_from_db(db)

    print(f"\nTotal ranked coins: {len(ranked)}")
    if not ranked:
        print("No coins passed the entry filters (BUY/HOLD + tech>=40 + prob>=35)")
        print("Current signal distribution may be mostly SELL — normal in bear market.")
    else:
        print(f"\n{'─'*65}")
        print(f"  {'#':>3} {'Symbol':<14} {'Score':>6} {'Signal':<10} {'Tech':>5} {'Prob':>5} {'News':<10} {'RR':>4}")
        print(f"{'─'*65}")
        for r in ranked[:15]:
            print(f"  {r['rank']:>3} {r['symbol']:<14} {r['rank_score']:>5.1f} "
                  f"{r['signal']:<10} {r['technical_score']:>4.0f} {r['probability_up']:>4.0f}% "
                  f"{r['news_sentiment']:<10} {r['risk_reward_ratio']:>3.1f}x")

        print(f"\n{'─'*65}")
        print("TOP 3:")
        for r in ranked[:3]:
            print(f"  #{r['rank']} {r['symbol']}")
            print(f"     Rank Score:    {r['rank_score']}")
            print(f"     Signal:        {r['signal']}")
            print(f"     Tech Score:    {r['technical_score']}")
            print(f"     Prob Up:       {r['probability_up']}%")
            print(f"     Entry:         ${r['entry_price']:.6f}")
            print(f"     Stop Loss:     ${r['stop_loss']:.6f}")
            print(f"     Take Profit:   ${r['take_profit']:.6f}")
            print(f"     R:R:           {r['risk_reward_ratio']}x")

    summary = get_ranking_summary(db)
    print(f"\nDB Summary: {summary}")
    client.close()
