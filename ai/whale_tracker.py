"""
ai/whale_tracker.py — Whale Intelligence Engine v2
================================================
Detects large capital movements using multiple Binance data sources:

    1. aggTrades   — large trade detection (>= threshold USDT)
    2. Order book  — bid/ask imbalance (buy vs sell wall)
    3. 24h ticker  — exchange flow bias from volume/price divergence
    4. Klines      — volume spike scoring (existing logic, enhanced)
    5. Depth snap  — top-of-book pressure ratio

Produces 5 normalized metrics → composite whale_score (0-100)
and normalised pressure scores (-1 to +1).

Writes to MongoDB collection: whale_data

Public API:
    detect_whale_activity(klines, symbol=None, save_to_db=False, db=None)
    run_whale_scan(symbols, db=None)
    get_whale_summary(symbol, db, hours=1)
"""

import logging
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)

# ── Binance endpoints ────────────────────────────────────────────────────────
BINANCE_API = "https://api.binance.com"

# ── Thresholds & weights ─────────────────────────────────────────────────────
LARGE_TRADE_USDT      = 50_000      # fallback threshold (per-coin Z-score used if history available)
AGG_TRADES_LIMIT      = 500         # number of aggTrades to fetch
DEPTH_LEVELS          = 20          # order book levels to check
WHALE_WEIGHT          = 10          # % contribution to final_score (default)
ZSCORE_LOOKBACK_DAYS  = 7           # days of history for per-coin baseline
ZSCORE_SIGMA          = 2.0         # standard deviations above mean = whale trade

# Component weights (must sum to 1.0)
W_AGG       = 0.35
W_ORDERBOOK = 0.25
W_TICKER    = 0.15
W_KLINES    = 0.15
W_DEPTH     = 0.10

# ── Retry helpers ─────────────────────────────────────────────────────────────
def _get(url: str, params: dict = None, retries: int = 3, timeout: int = 8) -> Optional[dict]:
    """HTTP GET with retry + exponential back-off. Returns None on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                logger.warning(f"[Whale] Rate-limited — wait 10s")
                _time.sleep(10)
            else:
                logger.debug(f"[Whale] HTTP error ({attempt}): {e}")
                _time.sleep(1.5 * attempt)
        except Exception as e:
            logger.debug(f"[Whale] Request error ({attempt}): {e}")
            _time.sleep(1.5 * attempt)
    return None


# ── Per-coin adaptive threshold (Z-score based) ───────────────────────────────
_coin_threshold_cache: dict = {}
_THRESHOLD_CACHE_TTL = 3600  # refresh every hour

def _get_per_coin_threshold(symbol: str, db=None) -> float:
    """
    Adaptive whale trade threshold per coin using 7-day history.
    Small-cap coins (low activity) get lower threshold; BTC/ETH get higher.
    Falls back to LARGE_TRADE_USDT if no history in DB.
    """
    import time as _t
    from datetime import datetime as _dt, timedelta as _td
    now = _t.time()
    cached = _coin_threshold_cache.get(symbol)
    if cached and (now - cached[1]) < _THRESHOLD_CACHE_TTL:
        return cached[0]
    try:
        if db is None:
            return LARGE_TRADE_USDT
        since = _dt.utcnow() - _td(days=ZSCORE_LOOKBACK_DAYS)
        docs = list(db['whale_data'].find(
            {'symbol': symbol, 'timestamp': {'$gte': since}},
            {'_id': 0, 'whale_score': 1}
        ).limit(300))
        if len(docs) < 10:
            return LARGE_TRADE_USDT
        scores = [float(d.get('whale_score', 50)) for d in docs]
        import statistics as _stats
        mean_s = _stats.mean(scores)
        # Scale threshold by coin's typical activity level
        # mean_score 50 (neutral) -> 1x default; 80 (active) -> 1.33x; 30 (quiet) -> 0.5x
        threshold = LARGE_TRADE_USDT * max(0.1, mean_s / 50.0)
        threshold = max(5_000, min(500_000, threshold))  # clamp $5k–$500k
        _coin_threshold_cache[symbol] = (threshold, now)
        logger.debug(f"[Whale/ZScore] {symbol}: adaptive=${threshold:,.0f} (mean={mean_s:.1f})")
        return threshold
    except Exception as e:
        logger.debug(f"[Whale/ZScore] fallback {symbol}: {e}")
        return LARGE_TRADE_USDT


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AGG TRADES — Detect large buy/sell trades
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_agg_trades(symbol: str) -> Dict:
    """
    Fetch last N aggTrades.
    Returns buy_volume, sell_volume, large_trade_count, large_trade_ratio,
    whale_buy_pressure, whale_sell_pressure.
    """
    empty = {
        'buy_volume': 0, 'sell_volume': 0,
        'large_trade_count': 0, 'large_trade_ratio': 0.0,
        'whale_buy_pressure': 0.0, 'whale_sell_pressure': 0.0,
        'agg_score': 50.0,
    }
    try:
        data = _get(f"{BINANCE_API}/api/v3/aggTrades",
                    {'symbol': symbol, 'limit': AGG_TRADES_LIMIT})
        if not data:
            return empty

        total_buy_vol  = 0.0
        total_sell_vol = 0.0
        large_buy_vol  = 0.0
        large_sell_vol = 0.0
        large_count    = 0

        # Fix 3: use per-coin adaptive threshold instead of fixed $50k
        _adaptive_thresh = _get_per_coin_threshold(symbol)

        for t in data:
            qty   = float(t.get('q', 0))
            price = float(t.get('p', 0))
            notional = qty * price
            is_sell = bool(t.get('m', False))  # m=True: maker=buyer, so aggressor sold

            if is_sell:
                total_sell_vol += notional
                if notional >= _adaptive_thresh:
                    large_sell_vol += notional
                    large_count += 1
            else:
                total_buy_vol += notional
                if notional >= _adaptive_thresh:
                    large_buy_vol += notional
                    large_count += 1

        total_vol = total_buy_vol + total_sell_vol
        large_vol = large_buy_vol + large_sell_vol

        large_trade_ratio  = large_vol / total_vol if total_vol > 0 else 0.0
        whale_buy_pressure = large_buy_vol  / max(large_vol, 1)
        whale_sell_pressure= large_sell_vol / max(large_vol, 1)

        # Convert to score (0-100): high buy pressure → high bullish score
        # Formula: 50 + (buy_frac - 0.5) * 100 → clamped 0-100
        agg_score = 50.0 + (whale_buy_pressure - 0.5) * 100.0
        agg_score = max(0.0, min(100.0, agg_score))

        # If large_trade_ratio very low → unknown, stay neutral
        if large_trade_ratio < 0.02:
            agg_score = 50.0

        return {
            'buy_volume':        round(total_buy_vol, 0),
            'sell_volume':       round(total_sell_vol, 0),
            'large_trade_count': large_count,
            'large_trade_ratio': round(large_trade_ratio, 4),
            'whale_buy_pressure': round(whale_buy_pressure, 4),
            'whale_sell_pressure': round(whale_sell_pressure, 4),
            'agg_score':         round(agg_score, 2),
        }

    except Exception as e:
        logger.debug(f"[Whale/aggTrades] {symbol}: {e}")
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ORDER BOOK — Bid/Ask imbalance
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_order_book(symbol: str) -> Dict:
    """
    Fetch top DEPTH_LEVELS bid/ask, compute volume imbalance.
    Imbalance > 0 → buy wall (bullish). < 0 → sell wall (bearish).
    """
    empty = {'bid_vol': 0, 'ask_vol': 0, 'imbalance': 0.0, 'ob_score': 50.0}
    try:
        data = _get(f"{BINANCE_API}/api/v3/depth",
                    {'symbol': symbol, 'limit': DEPTH_LEVELS})
        if not data:
            return empty

        bid_vol = sum(float(b[0]) * float(b[1]) for b in data.get('bids', []))
        ask_vol = sum(float(a[0]) * float(a[1]) for a in data.get('asks', []))
        total   = bid_vol + ask_vol

        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
        # imbalance in [-1, +1]

        # Convert to 0-100 score (50 = neutral)
        ob_score = 50.0 + imbalance * 50.0
        ob_score = max(0.0, min(100.0, ob_score))

        return {
            'bid_vol':    round(bid_vol, 0),
            'ask_vol':    round(ask_vol, 0),
            'imbalance':  round(imbalance, 4),
            'ob_score':   round(ob_score, 2),
        }
    except Exception as e:
        logger.debug(f"[Whale/orderBook] {symbol}: {e}")
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 24H TICKER — Exchange flow bias
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_ticker_flow(symbol: str) -> Dict:
    """
    Use 24h ticker to estimate exchange flow bias.
    If price > weightedAvgPrice and volume high → net buying.
    """
    empty = {'price_vs_avg': 0.0, 'volume_ratio': 0.0,
             'exchange_flow_bias': 0.0, 'ticker_score': 50.0}
    try:
        data = _get(f"{BINANCE_API}/api/v3/ticker/24hr",
                    {'symbol': symbol})
        if not data or not isinstance(data, dict):
            return empty

        close_price  = float(data.get('lastPrice', 0) or 0)
        avg_price    = float(data.get('weightedAvgPrice', close_price) or close_price)
        vol          = float(data.get('volume', 0) or 0)
        count        = int(data.get('count', 0) or 0)
        price_pct    = float(data.get('priceChangePercent', 0) or 0)

        price_vs_avg = (close_price - avg_price) / avg_price if avg_price > 0 else 0
        # positive → price above avg (buying above VWAP = bullish)

        # Flow bias: combine price direction + volume signal
        flow_bias = price_vs_avg * 2.0 + (price_pct / 100.0) * 0.5
        flow_bias = max(-1.0, min(1.0, flow_bias))

        ticker_score = 50.0 + flow_bias * 50.0
        ticker_score = max(0.0, min(100.0, ticker_score))

        return {
            'price_vs_avg':       round(price_vs_avg, 4),
            'price_change_pct':   round(price_pct, 2),
            'exchange_flow_bias': round(flow_bias, 4),
            'ticker_score':       round(ticker_score, 2),
        }
    except Exception as e:
        logger.debug(f"[Whale/ticker] {symbol}: {e}")
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KLINES — Enhanced volume spike scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _score_klines(klines: List[Dict]) -> Dict:
    """
    Enhanced version of existing klines whale detection.
    Returns vol_spike_score (0-100) and buy_pressure.
    """
    empty = {'vol_ratio': 1.0, 'buy_pressure': 0.5, 'klines_score': 50.0,
             'volume_spike_score': 1.0}
    if not klines or len(klines) < 20:
        return empty
    try:
        recent = klines[-20:]
        volumes = [float(k.get('volume', 0)) for k in recent]
        avg_vol = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        latest_vol = volumes[-1]
        vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0

        # Up/down volume (last 10 candles)
        up_vol   = sum(float(k.get('volume', 0))
                       for k in recent[-10:]
                       if float(k.get('close', 0)) >= float(k.get('open', 0)))
        down_vol = sum(float(k.get('volume', 0))
                       for k in recent[-10:]
                       if float(k.get('close', 0)) < float(k.get('open', 0)))
        total_v  = up_vol + down_vol
        buy_pressure = up_vol / total_v if total_v > 0 else 0.5

        # Volume spike score (0-100)
        if vol_ratio >= 5.0:
            spike_score = 100.0
        elif vol_ratio >= 3.0:
            spike_score = 80.0
        elif vol_ratio >= 2.0:
            spike_score = 60.0
        elif vol_ratio >= 1.5:
            spike_score = 40.0
        else:
            spike_score = 20.0

        # Combine: bias toward buy pressure
        klines_score = spike_score * buy_pressure + spike_score * (1 - buy_pressure) * 0.3
        klines_score = max(0.0, min(100.0, klines_score))

        return {
            'vol_ratio':         round(vol_ratio, 2),
            'buy_pressure':      round(buy_pressure, 4),
            'klines_score':      round(klines_score, 2),
            'volume_spike_score': round(vol_ratio, 2),
        }
    except Exception as e:
        logger.debug(f"[Whale/klines] {e}")
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEPTH SNAPSHOT — Top-of-book pressure
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_depth_pressure(symbol: str) -> Dict:
    """
    Fetch shallow order book (top 5) to detect immediate pressure.
    Useful for detecting spoofing or genuine wall.
    """
    empty = {'top_bid_wall': 0.0, 'top_ask_wall': 0.0, 'depth_score': 50.0}
    try:
        data = _get(f"{BINANCE_API}/api/v3/depth",
                    {'symbol': symbol, 'limit': 5})
        if not data:
            return empty

        top_bid = sum(float(b[0]) * float(b[1]) for b in data.get('bids', [])[:3])
        top_ask = sum(float(a[0]) * float(a[1]) for a in data.get('asks', [])[:3])
        total   = top_bid + top_ask

        depth_score = 50.0 + ((top_bid - top_ask) / total) * 50.0 if total > 0 else 50.0
        depth_score = max(0.0, min(100.0, depth_score))

        return {
            'top_bid_wall': round(top_bid, 0),
            'top_ask_wall': round(top_ask, 0),
            'depth_score':  round(depth_score, 2),
        }
    except Exception as e:
        logger.debug(f"[Whale/depth] {symbol}: {e}")
        return empty


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_composite(agg: Dict, ob: Dict, ticker: Dict,
                       kl: Dict, depth: Dict) -> Dict:
    """
    Weighted average of 5 component scores (0-100 each) → composite_score.
    Normalized to [-1, +1] as whale_score_norm for AI formula.
    """
    composite = (
        agg.get('agg_score', 50)     * W_AGG
        + ob.get('ob_score', 50)     * W_ORDERBOOK
        + ticker.get('ticker_score', 50) * W_TICKER
        + kl.get('klines_score', 50) * W_KLINES
        + depth.get('depth_score', 50) * W_DEPTH
    )
    composite = round(max(0.0, min(100.0, composite)), 2)

    # Normalize to [-1, +1] for scoring formula
    whale_score_norm = round((composite - 50.0) / 50.0, 4)

    # Human-readable signal
    bp = agg.get('whale_buy_pressure', 0.5)
    if composite >= 70:
        signal = 'ACCUMULATION'
    elif composite <= 30:
        signal = 'DISTRIBUTION'
    elif composite >= 55 and bp >= 0.55:
        signal = 'ACCUMULATION'
    elif composite <= 45 and bp <= 0.45:
        signal = 'DISTRIBUTION'
    else:
        signal = 'NONE'

    return {
        'whale_score':      composite,
        'whale_score_norm': whale_score_norm,   # -1 to +1
        'whale_signal':     signal,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC: detect_whale_activity (backward-compatible + enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_whale_activity(klines: List[Dict],
                          symbol: str = None,
                          save_to_db: bool = False,
                          db=None) -> Dict:
    """
    Full whale detection (backward-compatible with old signature).

    Always runs klines analysis (no API call needed).
    If symbol provided: also runs aggTrades + order book + ticker + depth.
    If save_to_db=True: saves to whale_data collection.

    Returns:
        whale_score (0-100), whale_signal, whale_buy_pressure,
        whale_sell_pressure, large_trade_ratio, volume_spike_score,
        exchange_flow_bias, whale_score_norm (-1 to +1)
    """
    try:
        kl = _score_klines(klines)

        if symbol:
            agg    = _fetch_agg_trades(symbol)
            ob     = _fetch_order_book(symbol)
            ticker = _fetch_ticker_flow(symbol)
            depth  = _fetch_depth_pressure(symbol)
            _time.sleep(0.05)   # tiny pause to avoid API spam
        else:
            agg    = {'agg_score': 50}
            ob     = {'ob_score': 50}
            ticker = {'ticker_score': 50}
            depth  = {'depth_score': 50}

        composite = _compute_composite(agg, ob, ticker, kl, depth)

        result = {
            # Core output (backward-compat)
            'whale_score':          composite['whale_score'],
            'whale_signal':         composite['whale_signal'],
            'vol_ratio':            kl.get('vol_ratio', 1.0),
            'buy_pressure':         kl.get('buy_pressure', 0.5),

            # New detailed metrics
            'whale_buy_pressure':   agg.get('whale_buy_pressure', 0.5),
            'whale_sell_pressure':  agg.get('whale_sell_pressure', 0.5),
            'large_trade_ratio':    agg.get('large_trade_ratio', 0.0),
            'large_trade_count':    agg.get('large_trade_count', 0),
            'volume_spike_score':   kl.get('volume_spike_score', 1.0),
            'exchange_flow_bias':   ticker.get('exchange_flow_bias', 0.0),
            'order_book_imbalance': ob.get('imbalance', 0.0),

            # Normalized score for AI formula
            'whale_score_norm':     composite['whale_score_norm'],

            # Component scores (debug)
            'components': {
                'agg_score':    agg.get('agg_score', 50),
                'ob_score':     ob.get('ob_score', 50),
                'ticker_score': ticker.get('ticker_score', 50),
                'klines_score': kl.get('klines_score', 50),
                'depth_score':  depth.get('depth_score', 50),
            },
        }

        if save_to_db and db is not None and symbol:
            _save_whale_data(db, symbol, result)

        return result

    except Exception as e:
        logger.error(f"[Whale] detect_whale_activity error: {e}")
        return {
            'whale_score': 0, 'whale_signal': 'NONE',
            'vol_ratio': 1.0, 'buy_pressure': 0.5,
            'whale_buy_pressure': 0.5, 'whale_sell_pressure': 0.5,
            'large_trade_ratio': 0.0, 'large_trade_count': 0,
            'volume_spike_score': 1.0, 'exchange_flow_bias': 0.0,
            'order_book_imbalance': 0.0, 'whale_score_norm': 0.0,
            'components': {},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC: run_whale_scan (multi-symbol)
# ═══════════════════════════════════════════════════════════════════════════════

def run_whale_scan(symbols: List[str], db=None) -> List[Dict]:
    """
    Run whale detection for a list of symbols and save to whale_data.
    Suitable for scheduled calls (5-10 min cadence).

    Returns list of whale result dicts with 'symbol' key added.
    """
    results = []
    for sym in symbols:
        try:
            from services.binance_scanner import get_klines
            klines = get_klines(sym, '15m', 50)
            if not klines:
                klines = []

            w = detect_whale_activity(klines, symbol=sym,
                                      save_to_db=(db is not None),
                                      db=db)
            w['symbol'] = sym
            results.append(w)
            _time.sleep(0.15)   # 150ms between symbols
        except Exception as e:
            logger.error(f"[Whale] {sym}: {e}")

    logger.info(f"[Whale] Scanned {len(results)} symbols")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DB helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Index setup (run once at startup, not on every save) ───────────────────
_whale_indexes_created = False

def _ensure_whale_indexes(db) -> None:
    """Create whale_data indexes once. Safe to call multiple times (idempotent)."""
    global _whale_indexes_created
    if _whale_indexes_created:
        return
    try:
        db['whale_data'].create_index(
            [('symbol', 1), ('timestamp', -1)],
            background=True, name='whale_sym_ts'
        )
        # TTL index: auto-delete docs older than 14 days
        # NOTE: MongoDB only allows one TTL index per collection.
        # If 'whale_data_ttl' already exists with different params, drop and recreate.
        existing = db['whale_data'].index_information()
        if 'whale_data_ttl' not in existing:
            db['whale_data'].create_index(
                [('timestamp', 1)],
                expireAfterSeconds=14 * 24 * 3600,
                background=True, name='whale_data_ttl'
            )
        _whale_indexes_created = True
    except Exception as e:
        logger.debug(f"[Whale] Index setup: {e}")


def _save_whale_data(db, symbol: str, whale_result: Dict) -> bool:
    """Save whale metrics to whale_data collection."""
    try:
        # Ensure indexes on first call (not every save)
        _ensure_whale_indexes(db)

        doc = {
            'symbol':         symbol,
            'timestamp':      datetime.utcnow(),
            'whale_score':    whale_result.get('whale_score', 0),
            'whale_score_norm': whale_result.get('whale_score_norm', 0.0),
            'whale_signal':   whale_result.get('whale_signal', 'NONE'),
            'source':         'binance_multi',
            'metrics': {
                'whale_buy_pressure':   whale_result.get('whale_buy_pressure', 0.5),
                'whale_sell_pressure':  whale_result.get('whale_sell_pressure', 0.5),
                'large_trade_ratio':    whale_result.get('large_trade_ratio', 0.0),
                'large_trade_count':    whale_result.get('large_trade_count', 0),
                'volume_spike_score':   whale_result.get('volume_spike_score', 1.0),
                'exchange_flow_bias':   whale_result.get('exchange_flow_bias', 0.0),
                'order_book_imbalance': whale_result.get('order_book_imbalance', 0.0),
                'buy_pressure':         whale_result.get('buy_pressure', 0.5),
                'vol_ratio':            whale_result.get('vol_ratio', 1.0),
                'components':           whale_result.get('components', {}),
            },
        }
        db['whale_data'].insert_one(doc)
        return True
    except Exception as e:
        logger.error(f"[Whale] DB save error: {e}")
        return False


def get_whale_summary(symbol: str, db, hours: int = 6) -> Dict:
    """
    Load recent whale data for a symbol from DB for dashboard.
    Returns averaged metrics over last `hours` hours.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        docs = list(
            db['whale_data']
            .find({'symbol': symbol, 'timestamp': {'$gte': since}},
                  {'_id': 0})
            .sort('timestamp', -1)
            .limit(72)
        )
        if not docs:
            return {}

        scores = [d.get('whale_score', 50) for d in docs]
        signals= [d.get('whale_signal', 'NONE') for d in docs]
        avg_score = round(sum(scores) / len(scores), 1)

        # Most common signal
        from collections import Counter
        trend = Counter(signals).most_common(1)[0][0]

        latest = docs[0]
        m = latest.get('metrics', {})
        return {
            'symbol':            symbol,
            'avg_whale_score':   avg_score,
            'latest_score':      latest.get('whale_score', 0),
            'whale_signal':      latest.get('whale_signal', 'NONE'),
            'trend':             trend,
            'whale_buy_pressure':  m.get('whale_buy_pressure', 0.5),
            'whale_sell_pressure': m.get('whale_sell_pressure', 0.5),
            'large_trade_ratio':   m.get('large_trade_ratio', 0.0),
            'exchange_flow_bias':  m.get('exchange_flow_bias', 0.0),
            'order_book_imbalance': m.get('order_book_imbalance', 0.0),
            'data_points':       len(docs),
        }
    except Exception as e:
        logger.error(f"[Whale] Summary error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import logging as _log
    _log.basicConfig(level=_log.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from services.binance_scanner import get_klines

    print("\n" + "=" * 60)
    print("  WHALE INTELLIGENCE ENGINE v2 — Test Run")
    print("=" * 60 + "\n")

    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    for sym in test_symbols:
        klines = get_klines(sym, '15m', 50)
        if not klines:
            print(f"  {sym}: No klines")
            continue
        r = detect_whale_activity(klines, symbol=sym)
        print(f"  {sym:10}: Score={r['whale_score']:5.1f} | "
              f"Signal={r['whale_signal']:12} | "
              f"BuyPress={r['whale_buy_pressure']:.2f} | "
              f"LargeTrades={r['large_trade_count']:3} | "
              f"FlowBias={r['exchange_flow_bias']:+.3f} | "
              f"OB={r['order_book_imbalance']:+.3f}")
    print()
