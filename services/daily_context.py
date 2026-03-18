"""
Daily Context v1 — 90-Day Trend Analysis for Live Signal Enhancement
=====================================================================
Reads the 90-day daily candles already stored in the `market_data` MongoDB
collection (populated by collect_historical_data() at scheduler startup) and
computes:

  - daily_trend         : UPTREND | DOWNTREND | SIDEWAYS
  - trend_strength_d    : 0-100, how strong the daily trend is
  - daily_ema20         : EMA20 on daily candles (short-term daily baseline)
  - daily_ema50         : EMA50 on daily candles (long-term daily baseline)
  - distance_from_ema20d: price % above/below daily EMA20
  - daily_support_zone  : nearest major support on the daily chart
  - daily_resistance_zone: nearest major resistance on the daily chart
  - trend_alignment_mult: multiplier for final_score in scan_market()
      1.15 → 15m BUY signal aligns with daily uptrend (extra confidence)
      0.85 → 15m BUY signal fights daily downtrend (reduce confidence)
      1.00 → sideways or no daily data

Key design: NO new Binance API calls. Uses only data already in MongoDB.
Fast: typically < 100ms for 90 coins (pure DB read + numpy).

Usage:
    from services.daily_context import get_daily_context
    ctx = get_daily_context(['BTCUSDT', 'ETHUSDT'])
    mult = ctx['BTCUSDT']['trend_alignment_mult']  # apply to final_score
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# How many daily candles to use (should match HISTORICAL_DAYS in settings)
DAILY_LOOKBACK = 90

# Trend classification thresholds (price vs EMA20d)
UPTREND_THRESHOLD   = 0.01    # price > EMA20d * 1.01 AND EMA20d > EMA50d
DOWNTREND_THRESHOLD = -0.01   # price < EMA20d * 0.99 AND EMA20d < EMA50d

# Alignment multiplier values
MULT_ALIGN_UPTREND   = 1.15   # 15m signal aligns with daily uptrend: boost by 15%
MULT_ALIGN_DOWNTREND = 0.85   # 15m signal fights daily downtrend: reduce by 15%
MULT_ALIGN_SIDEWAYS  = 1.00   # No clear daily trend: neutral


def get_daily_context(symbols: List[str], db=None) -> Dict[str, Dict]:
    """
    Load daily candles from market_data for the given symbols and compute
    trend context. Returns a dict keyed by symbol.

    Parameters
    ----------
    symbols : list of str
        E.g. ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    db : pymongo.database.Database, optional
        Pass an existing DB connection to avoid opening a new one.
        If None, opens its own short-lived connection.

    Returns
    -------
    dict
        {
            'BTCUSDT': {
                'daily_trend': 'UPTREND',
                'trend_strength_d': 75.0,
                'daily_ema20': 84000.0,
                'daily_ema50': 80000.0,
                'distance_from_ema20d': 2.4,     # % above EMA20d
                'daily_support_zone': 82000.0,
                'daily_resistance_zone': 90000.0,
                'trend_alignment_mult': 1.15,
            },
            ...
        }
        Missing symbols get a neutral entry (mult=1.0, trend=SIDEWAYS).
    """
    _own_client = None
    try:
        if db is None:
            import pymongo
            _own_client = pymongo.MongoClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=8000
            )
            db = _own_client[settings.DATABASE_NAME]

        col = db[settings.COLLECTION_MARKET_DATA]
        cutoff = datetime.utcnow() - timedelta(days=DAILY_LOOKBACK + 5)  # small buffer

        context: Dict[str, Dict] = {}

        for sym in symbols:
            try:
                # Fetch the last 90 daily candles for this symbol (fast index scan)
                candles = list(
                    col.find(
                        {'symbol': sym, 'open_time': {'$gte': cutoff}},
                        {'close': 1, 'high': 1, 'low': 1, 'open_time': 1, '_id': 0}
                    ).sort('open_time', 1).limit(DAILY_LOOKBACK)
                )

                if len(candles) < 20:
                    # Not enough data — return neutral
                    context[sym] = _neutral_context()
                    continue

                closes = np.array([float(c['close']) for c in candles])
                highs  = np.array([float(c.get('high', c['close'])) for c in candles])
                lows   = np.array([float(c.get('low',  c['close'])) for c in candles])
                price  = float(closes[-1])

                # ── EMA20d and EMA50d ──────────────────────────────────────
                ema20d = _ema(closes, 20)
                ema50d = _ema(closes, 50) if len(closes) >= 50 else ema20d

                # ── Trend classification ───────────────────────────────────
                # UPTREND:   price > EMA20d AND EMA20d > EMA50d
                # DOWNTREND: price < EMA20d AND EMA20d < EMA50d
                # SIDEWAYS:  mixed signals
                pct_vs_ema20 = (price / ema20d - 1.0) if ema20d > 0 else 0.0
                ema_aligned_up   = ema20d > ema50d
                ema_aligned_down = ema20d < ema50d

                if pct_vs_ema20 >= UPTREND_THRESHOLD and ema_aligned_up:
                    daily_trend = 'UPTREND'
                    mult        = MULT_ALIGN_UPTREND
                elif pct_vs_ema20 <= DOWNTREND_THRESHOLD and ema_aligned_down:
                    daily_trend = 'DOWNTREND'
                    mult        = MULT_ALIGN_DOWNTREND
                else:
                    daily_trend = 'SIDEWAYS'
                    mult        = MULT_ALIGN_SIDEWAYS

                # ── Trend strength 0-100 ───────────────────────────────────
                # How far price is from EMA50d relative to ATR (daily)
                atr_d = _atr(highs, lows, closes, 14)
                if atr_d > 0:
                    distance_atr = abs(price - ema50d) / atr_d
                    trend_strength = min(100.0, distance_atr * 25.0)
                else:
                    trend_strength = 50.0

                # ── Daily Support / Resistance (simplified pivot) ──────────
                # Use the last 20 daily candles — major levels only
                sup, res = _daily_sr(lows[-20:], highs[-20:], price)

                context[sym] = {
                    'daily_trend':          daily_trend,
                    'trend_strength_d':     round(trend_strength, 1),
                    'daily_ema20':          round(ema20d, 6),
                    'daily_ema50':          round(ema50d, 6),
                    'distance_from_ema20d': round(pct_vs_ema20 * 100, 2),  # %
                    'daily_support_zone':   round(sup, 6) if sup else None,
                    'daily_resistance_zone':round(res, 6) if res else None,
                    'trend_alignment_mult': mult,
                }

            except Exception as e:
                logger.debug(f"[DailyCtx] {sym}: {e}")
                context[sym] = _neutral_context()

        logger.info(
            f"[DailyCtx] {len(context)} symbols analysed — "
            f"UP: {sum(1 for v in context.values() if v['daily_trend']=='UPTREND')} "
            f"DOWN: {sum(1 for v in context.values() if v['daily_trend']=='DOWNTREND')} "
            f"SIDE: {sum(1 for v in context.values() if v['daily_trend']=='SIDEWAYS')}"
        )
        return context

    except Exception as e:
        logger.error(f"[DailyCtx] Failed: {e}")
        return {sym: _neutral_context() for sym in symbols}
    finally:
        if _own_client:
            try:
                _own_client.close()
            except Exception:
                pass


# ── Private helpers ────────────────────────────────────────────────────────────

def _neutral_context() -> Dict:
    """Return a neutral (no-op) context when data is unavailable."""
    return {
        'daily_trend':          'SIDEWAYS',
        'trend_strength_d':     50.0,
        'daily_ema20':          None,
        'daily_ema50':          None,
        'distance_from_ema20d': 0.0,
        'daily_support_zone':   None,
        'daily_resistance_zone':None,
        'trend_alignment_mult': 1.00,  # Neutral: no boost, no penalty
    }


def _ema(values: np.ndarray, period: int) -> float:
    """Compute EMA of the last `period` bars using numpy."""
    if len(values) < period:
        period = len(values)
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = float(v) * k + ema * (1 - k)
    return ema


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    """Compute ATR (Average True Range) over last `period` bars."""
    try:
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1])
            )
            tr_list.append(tr)
        if not tr_list:
            return 0.0
        return float(np.mean(tr_list[-period:]))
    except Exception:
        return 0.0


def _daily_sr(lows: np.ndarray, highs: np.ndarray, price: float):
    """
    Simplified daily support/resistance: lowest of lows below price,
    highest of highs above price in the last 20 daily candles.
    """
    try:
        sup = max((l for l in lows if l < price), default=None)
        res = min((h for h in highs if h > price), default=None)
        return sup, res
    except Exception:
        return None, None


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    ctx = get_daily_context(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'])
    print("\n=== Daily Context Results ===")
    for sym, data in ctx.items():
        print(
            f"{sym:12s} trend={data['daily_trend']:9s} "
            f"mult={data['trend_alignment_mult']:.2f} "
            f"ema20d={data['daily_ema20']} "
            f"dist={data['distance_from_ema20d']:+.1f}% "
            f"sup={data['daily_support_zone']} "
            f"res={data['daily_resistance_zone']}"
        )
