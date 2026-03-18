"""
Hourly Analysis v1 — 1H Candle Context for Position Trade Detection
====================================================================
Fetches 200 × 1h candles (~8 days of hourly data) from Binance and computes
medium-term indicators for POSITION trade classification (1–3 day holds).

Why 1h candles?
  - 15m candles  = ~50h of history → good for SWING trades (4–12h)
  - 1h candles   = ~8 days of history → good for POSITION trades (1–3 days)
  - Daily candles = 90 days of history → good for TREND direction

What it computes (per coin):
  - rsi_1h           : RSI on 1h candles (medium-term momentum)
  - ema20_1h         : EMA20 on 1h (medium-term trend baseline)
  - ema50_1h         : EMA50 on 1h
  - macd_hist_1h     : MACD histogram on 1h (positive = bullish momentum)
  - hourly_trend     : UPTREND | DOWNTREND | SIDEWAYS
  - hourly_momentum  : BULLISH | BEARISH | NEUTRAL
  - position_score   : 0–100 composite readiness for a 1–3 day position

position_score formula:
  = (rsi_zone_1h * 0.30) + (trend_alignment_1h * 0.40) + (macd_momentum_1h * 0.30)
  where each component is 0–100

Usage:
    from services.hourly_analysis import get_hourly_analysis
    result = get_hourly_analysis('BTCUSDT')
    trade_type_eligible = result['position_score'] > 60
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
POSITION_SCORE_MIN   = 60    # Minimum position_score to classify as POSITION trade
TREND_SCORE_MIN      = 50    # Minimum trend alignment for POSITION eligibility

RSI_1H_IDEAL_LOW  = 45
RSI_1H_IDEAL_HIGH = 65
RSI_1H_OVERBOUGHT = 70


def get_hourly_analysis(symbol: str, klines_1h=None) -> Dict:
    """
    Compute 1h-based indicators for a single coin.

    Parameters
    ----------
    symbol : str
        E.g. 'BTCUSDT'
    klines_1h : list of dict, optional
        Pre-fetched 1h klines (pass to avoid a second Binance API call if
        you already have them). If None, fetches 200 bars from Binance.

    Returns
    -------
    dict with keys: rsi_1h, ema20_1h, ema50_1h, macd_hist_1h,
                    hourly_trend, hourly_momentum, position_score
    On error, returns a neutral dict with position_score=0.
    """
    try:
        if klines_1h is None:
            from services.binance_scanner import get_klines
            klines_1h = get_klines(symbol, '1h', 200)

        if not klines_1h or len(klines_1h) < 30:
            return _neutral_hourly(symbol)

        closes = np.array([float(k['close']) for k in klines_1h])
        highs  = np.array([float(k['high'])  for k in klines_1h])
        lows   = np.array([float(k['low'])   for k in klines_1h])
        price  = float(closes[-1])

        # ── RSI (14) on 1h ────────────────────────────────────────────────
        rsi_1h = _rsi(closes, 14)

        # ── EMA20 / EMA50 on 1h ───────────────────────────────────────────
        ema20_1h = _ema(closes, 20)
        ema50_1h = _ema(closes, 50) if len(closes) >= 50 else ema20_1h

        # ── MACD histogram on 1h (12, 26, 9) ─────────────────────────────
        macd_hist_1h = _macd_histogram(closes)

        # ── Hourly trend ──────────────────────────────────────────────────
        # UPTREND: price > EMA20_1h > EMA50_1h
        # DOWNTREND: price < EMA20_1h < EMA50_1h
        if price > ema20_1h and ema20_1h > ema50_1h:
            hourly_trend = 'UPTREND'
        elif price < ema20_1h and ema20_1h < ema50_1h:
            hourly_trend = 'DOWNTREND'
        else:
            hourly_trend = 'SIDEWAYS'

        # ── Hourly momentum (MACD direction) ─────────────────────────────
        if macd_hist_1h > 0:
            hourly_momentum = 'BULLISH'
        elif macd_hist_1h < 0:
            hourly_momentum = 'BEARISH'
        else:
            hourly_momentum = 'NEUTRAL'

        # ── Position Score (0–100) ────────────────────────────────────────
        # Component 1: RSI zone on 1h (prefers 45–65 reset zone)
        rsi_zone_score = _rsi_zone_score(rsi_1h)

        # Component 2: Trend alignment on 1h
        if hourly_trend == 'UPTREND':
            trend_align_score = 90.0
        elif hourly_trend == 'SIDEWAYS':
            trend_align_score = 50.0
        else:
            trend_align_score = 15.0  # Downtrend = bad for position entry

        # Component 3: MACD momentum (positive histogram = building momentum)
        if macd_hist_1h > 0:
            # Normalize: stronger positive = higher score
            macd_score = min(100.0, 50.0 + abs(macd_hist_1h) / (price / 100.0) * 200)
        elif macd_hist_1h < 0:
            macd_score = max(0.0, 50.0 - abs(macd_hist_1h) / (price / 100.0) * 200)
        else:
            macd_score = 50.0

        position_score = (
            rsi_zone_score    * 0.30 +
            trend_align_score * 0.40 +
            macd_score        * 0.30
        )
        position_score = round(min(100.0, max(0.0, position_score)), 1)

        return {
            'symbol':          symbol,
            'rsi_1h':          round(rsi_1h, 1),
            'ema20_1h':        round(ema20_1h, 6),
            'ema50_1h':        round(ema50_1h, 6),
            'macd_hist_1h':    round(macd_hist_1h, 6),
            'hourly_trend':    hourly_trend,
            'hourly_momentum': hourly_momentum,
            'position_score':  position_score,
        }

    except Exception as e:
        logger.debug(f"[HourlyAnalysis] {symbol}: {e}")
        return _neutral_hourly(symbol)


def get_hourly_analysis_batch(symbols: list, max_workers: int = 5) -> Dict[str, Dict]:
    """
    Fetch 1h analysis for multiple symbols in parallel using ThreadPoolExecutor.
    Limits concurrency to avoid hitting Binance rate limits.

    Parameters
    ----------
    symbols : list
        List of symbol strings, e.g. ['BTCUSDT', 'ETHUSDT']
    max_workers : int
        Thread pool size. Default 5 is safe for Binance public API limits.

    Returns
    -------
    dict : {symbol: analysis_dict}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_hourly_analysis, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:
                logger.debug(f"[HourlyBatch] {sym}: {e}")
                results[sym] = _neutral_hourly(sym)

    return results


# ── Private helpers ────────────────────────────────────────────────────────────

def _neutral_hourly(symbol: str) -> Dict:
    """Return a neutral result when hourly data is unavailable."""
    return {
        'symbol':          symbol,
        'rsi_1h':          50.0,
        'ema20_1h':        None,
        'ema50_1h':        None,
        'macd_hist_1h':    0.0,
        'hourly_trend':    'SIDEWAYS',
        'hourly_momentum': 'NEUTRAL',
        'position_score':  0.0,   # 0 means: don't classify as POSITION
    }


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI over the last `period` bars."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(values: np.ndarray, period: int) -> float:
    """Exponential Moving Average of the entire series."""
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = float(v) * k + ema * (1 - k)
    return ema


def _macd_histogram(closes: np.ndarray,
                    fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """Compute latest MACD histogram value = MACD line - Signal line."""
    if len(closes) < slow + signal:
        return 0.0
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    # Approximate signal line (EMA of MACD over last `signal` bars)
    # We compute MACD for each of the last `signal+5` bars for EMA input
    macd_vals = []
    for i in range(-(signal + 5), 0):
        ef = _ema(closes[:i] if i < 0 else closes, fast)
        es = _ema(closes[:i] if i < 0 else closes, slow)
        macd_vals.append(ef - es)
    signal_line = _ema(np.array(macd_vals), signal)
    return macd_line - signal_line


def _rsi_zone_score(rsi: float) -> float:
    """
    Score the RSI value for position trade suitability (same logic as ranking engine).
    Best zone 45-65: score 100. Overbought >70: score drops steeply.
    """
    if RSI_1H_IDEAL_LOW <= rsi <= RSI_1H_IDEAL_HIGH:
        return 100.0
    elif RSI_1H_IDEAL_HIGH < rsi <= RSI_1H_OVERBOUGHT:
        return 100.0 - ((rsi - RSI_1H_IDEAL_HIGH) / (RSI_1H_OVERBOUGHT - RSI_1H_IDEAL_HIGH)) * 35.0
    elif 35 < rsi < RSI_1H_IDEAL_LOW:
        return 60.0
    elif rsi > RSI_1H_OVERBOUGHT:
        return 20.0
    else:
        return 25.0


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        r = get_hourly_analysis(sym)
        print(
            f"{sym:12s} trend={r['hourly_trend']:9s} "
            f"mom={r['hourly_momentum']:7s} "
            f"rsi1h={r['rsi_1h']:.1f} "
            f"pos_score={r['position_score']:.1f}"
        )
