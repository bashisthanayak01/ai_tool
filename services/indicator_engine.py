"""
Technical Indicator Engine v2 — RSI, EMA, MACD, Volume, Volatility,
                                Breakout, ATR (exported), Support/Resistance
Upgrades:
  - ATR value now exported (for ATR-based TP/SL)
  - Support/Resistance levels detected (pivot points method)
  - near_support / near_resistance flags added
  - sr_quality score added (how clean the S/R zone is)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_indicators(klines: List[Dict]) -> Optional[Dict]:
    """
    Calculate all technical indicators from klines.
    Returns dict with: price, rsi, ema20, ema50, macd, macd_signal,
    macd_histogram, volume, volume_spike, volatility, breakout_score,
    atr (new), atr_pct (new), support_levels, resistance_levels,
    near_support, near_resistance, sr_quality
    """
    try:
        if not klines or len(klines) < 50:
            return None

        df = pd.DataFrame(klines)
        close  = df['close'].astype(float)
        high   = df['high'].astype(float)
        low    = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # ── RSI (14) ──────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        # ── EMA 20 & 50 ──────────────────────────────────────────
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # ── MACD (12, 26, 9) ─────────────────────────────────────
        ema12          = close.ewm(span=12, adjust=False).mean()
        ema26          = close.ewm(span=26, adjust=False).mean()
        macd_line      = ema12 - ema26
        macd_signal    = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        # ── Volume spike ─────────────────────────────────────────
        vol_avg   = volume.rolling(20).mean()
        vol_spike = (volume.iloc[-1] / vol_avg.iloc[-1]
                     if vol_avg.iloc[-1] > 0 else 1.0)

        # ── ATR (14) — True Range ─────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        price   = float(close.iloc[-1])
        atr_pct = (atr_val / price * 100) if price > 0 else 0
        volatility = atr_pct   # Keep backward compat

        # ── Breakout (20-period) ─────────────────────────────────
        high_20 = high.rolling(20).max()
        low_20  = low.rolling(20).min()
        p_range = high_20.iloc[-1] - low_20.iloc[-1]
        if p_range > 0:
            breakout_score = ((price - low_20.iloc[-1]) / p_range) * 100
        else:
            breakout_score = 50.0

        # ── Support & Resistance (Pivot Points) ──────────────────
        sr = _detect_support_resistance(high, low, close, price, atr_val)

        return {
            'symbol':          klines[0].get('symbol', ''),
            'price':           round(price, 6),
            'rsi':             round(float(rsi.iloc[-1]), 2),
            'ema20':           round(float(ema20.iloc[-1]), 6),
            'ema50':           round(float(ema50.iloc[-1]), 6),
            'macd':            round(float(macd_line.iloc[-1]), 6),
            'macd_signal':     round(float(macd_signal.iloc[-1]), 6),
            'macd_histogram':  round(float(macd_histogram.iloc[-1]), 6),
            'volume':          round(float(volume.iloc[-1]), 2),
            'volume_spike':    round(float(vol_spike), 2),
            'volatility':      round(float(volatility), 2),
            'breakout_score':  round(float(breakout_score), 2),
            # New: ATR (for dynamic TP/SL)
            'atr':             round(atr_val, 6),
            'atr_pct':         round(atr_pct, 3),
            # New: Support/Resistance
            'support_levels':  sr['support_levels'],
            'resistance_levels': sr['resistance_levels'],
            'near_support':    sr['near_support'],
            'near_resistance': sr['near_resistance'],
            'sr_quality':      sr['sr_quality'],
            'nearest_support': sr['nearest_support'],
            'nearest_resistance': sr['nearest_resistance'],
        }

    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None


def _detect_support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    price: float,
    atr: float,
    lookback: int = 50,
    tolerance_atr: float = 0.5,
) -> Dict:
    """
    Detect support and resistance levels using pivot highs/lows.
    A pivot high = candle whose high is higher than N candles on each side.
    A pivot low  = candle whose low is lower than N candles on each side.

    Returns:
        support_levels, resistance_levels, near_support, near_resistance,
        sr_quality (0-100), nearest_support, nearest_resistance
    """
    try:
        n = 3   # pivot window
        tolerance = atr * tolerance_atr   # price within 0.5 ATR of level

        highs  = list(high.iloc[-lookback:])
        lows   = list(low.iloc[-lookback:])
        closes = list(close.iloc[-lookback:])

        pivots_high = []
        pivots_low  = []

        for i in range(n, len(highs) - n):
            # Pivot high
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                pivots_high.append(highs[i])

            # Pivot low
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                pivots_low.append(lows[i])

        # Cluster levels (merge levels within 1 ATR of each other)
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(set(levels))
            clustered = []
            group = [levels[0]]
            for v in levels[1:]:
                if v - group[-1] < atr * 1.5:
                    group.append(v)
                else:
                    clustered.append(round(sum(group) / len(group), 6))
                    group = [v]
            clustered.append(round(sum(group) / len(group), 6))
            return clustered

        support_levels    = [s for s in cluster(pivots_low)  if s < price]
        resistance_levels = [r for r in cluster(pivots_high) if r > price]

        # Nearest levels
        nearest_support    = max(support_levels)    if support_levels    else None
        nearest_resistance = min(resistance_levels) if resistance_levels else None

        # Near support/resistance flags
        near_support    = (nearest_support is not None and
                           abs(price - nearest_support) <= tolerance)
        near_resistance = (nearest_resistance is not None and
                           abs(price - nearest_resistance) <= tolerance)

        # S/R Quality: how many levels exist (more = more reliable)
        total_sr = len(support_levels) + len(resistance_levels)
        sr_quality = min(100, total_sr * 15)

        return {
            'support_levels':     [round(s, 6) for s in support_levels[-3:]],
            'resistance_levels':  [round(r, 6) for r in resistance_levels[:3]],
            'near_support':       near_support,
            'near_resistance':    near_resistance,
            'sr_quality':         sr_quality,
            'nearest_support':    round(nearest_support, 6) if nearest_support else None,
            'nearest_resistance': round(nearest_resistance, 6) if nearest_resistance else None,
        }

    except Exception as e:
        logger.debug(f"S/R detection error: {e}")
        return {
            'support_levels': [], 'resistance_levels': [],
            'near_support': False, 'near_resistance': False,
            'sr_quality': 0, 'nearest_support': None, 'nearest_resistance': None,
        }


if __name__ == '__main__':
    from services.binance_scanner import get_klines
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        klines = get_klines(sym, '15m', 200)
        if klines:
            r = analyze_indicators(klines)
            if r:
                print(f"\n{sym}:")
                print(f"  Price={r['price']}  RSI={r['rsi']}  ATR={r['atr_pct']:.2f}%")
                print(f"  Support:    {r['support_levels']}")
                print(f"  Resistance: {r['resistance_levels']}")
                print(f"  NearSupport={r['near_support']}  NearResist={r['near_resistance']}  SRQ={r['sr_quality']}")
