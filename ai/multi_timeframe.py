"""
ai/multi_timeframe.py — Multi-Timeframe Confirmation Engine
============================================================
Upgrade #1 (High Impact): Requires 15m AND 4h charts to agree
before generating a BUY signal.

Logic:
  - Fast timeframe (15m): current momentum + entry timing
  - Slow timeframe (4h): trend direction + regime confirmation
  - Only BUY when BOTH align → dramatically reduces false signals

Score output:
  mtf_score         (0-100): combined alignment score
  mtf_confirmed     (bool):  True = both timeframes bullish
  mtf_bias          (str):   BULLISH / BEARISH / MIXED / NEUTRAL
  timeframe_4h      (dict):  4h indicators summary
  alignment_reason  (str):   human-readable explanation
"""

import logging
import time as _time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"


# ─────────────────────────────────────────────────────────────────
# FETCH 4H KLINES
# ─────────────────────────────────────────────────────────────────

def get_4h_klines(symbol: str, limit: int = 100) -> Optional[List[Dict]]:
    """Fetch 4-hour klines from Binance (no DB — fresh each call)."""
    try:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={'symbol': symbol, 'interval': '4h', 'limit': limit},
            timeout=8
        )
        resp.raise_for_status()
        raw = resp.json()
        return [_parse_kline(k) for k in raw]
    except Exception as e:
        logger.debug(f"[MTF] 4h klines error for {symbol}: {e}")
        return None


def _parse_kline(k: list) -> Dict:
    return {
        'open':   float(k[1]),
        'high':   float(k[2]),
        'low':    float(k[3]),
        'close':  float(k[4]),
        'volume': float(k[5]),
    }


# ─────────────────────────────────────────────────────────────────
# 4H QUICK ANALYSIS
# ─────────────────────────────────────────────────────────────────

def _analyze_4h(klines_4h: List[Dict]) -> Dict:
    """
    Calculate key 4h indicators.
    Returns trend, rsi_4h, ema_aligned, momentum, bias.
    """
    if not klines_4h or len(klines_4h) < 26:
        return {'bias': 'NEUTRAL', 'trend_score': 50, 'rsi_4h': 50}

    closes = [k['close'] for k in klines_4h]
    highs  = [k['high']  for k in klines_4h]
    lows   = [k['low']   for k in klines_4h]
    vols   = [k['volume'] for k in klines_4h]

    # EMA 20 + 50 on 4h
    def ema(data, span):
        k_factor = 2 / (span + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(v * k_factor + result[-1] * (1 - k_factor))
        return result

    ema20_4h = ema(closes, 20)
    ema50_4h = ema(closes, 50)
    price_4h = closes[-1]

    # RSI 14 on 4h
    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        d = closes[-i] - closes[-i-1]
        if d > 0:
            gains.append(d)
        else:
            losses.append(abs(d))
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    rsi_4h = 100 - (100 / (1 + avg_gain / avg_loss))

    # Trend: price vs EMAs
    ema_bullish = (price_4h > ema20_4h[-1] > ema50_4h[-1])
    ema_bearish = (price_4h < ema20_4h[-1] < ema50_4h[-1])

    # Momentum: is current 4h candle bullish?
    last_close = closes[-1]
    last_open  = klines_4h[-1]['open']
    candle_bull = last_close > last_open

    # Volume trend: last 3 candles above average?
    avg_vol = sum(vols[:-3]) / max(len(vols) - 3, 1)
    vol_increasing = sum(vols[-3:]) / 3 > avg_vol * 1.1

    # Compute 4h score (0-100)
    score = 50
    if ema_bullish:
        score += 25
    elif ema_bearish:
        score -= 25
    if rsi_4h > 50 and rsi_4h < 70:
        score += 10
    elif rsi_4h >= 70:
        score -= 5  # overbought on 4h — caution
    elif rsi_4h < 40:
        score -= 10
    if candle_bull:
        score += 10
    if vol_increasing:
        score += 5
    score = max(0, min(100, score))

    # Bias
    if score >= 65:
        bias = 'BULLISH'
    elif score <= 35:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    return {
        'bias':         bias,
        'trend_score':  round(score, 1),
        'rsi_4h':       round(rsi_4h, 1),
        'ema_bullish':  ema_bullish,
        'ema_bearish':  ema_bearish,
        'candle_bull':  candle_bull,
        'price_4h':     round(price_4h, 6),
        'ema20_4h':     round(ema20_4h[-1], 6),
        'ema50_4h':     round(ema50_4h[-1], 6),
        'vol_increasing': vol_increasing,
    }


# ─────────────────────────────────────────────────────────────────
# ALIGNMENT SCORING
# ─────────────────────────────────────────────────────────────────

def _compute_alignment(ind_15m: Dict, tf4h: Dict) -> Dict:
    """
    Combine 15m and 4h signals into a single alignment score.

    Rules:
      - Both bullish   → CONFIRMED BUY (high score)
      - 4h bullish, 15m neutral → POSSIBLE (medium score)
      - 4h bearish     → BLOCKED regardless of 15m
      - Disagreement   → MIXED (lower score, no entry)
    """
    rsi_15m      = float(ind_15m.get('rsi', 50) or 50)
    macd         = float(ind_15m.get('macd', 0) or 0)
    macd_sig     = float(ind_15m.get('macd_signal', 0) or 0)
    price        = float(ind_15m.get('price', 0) or 0)
    ema20_15m    = float(ind_15m.get('ema20', price) or price)
    ema50_15m    = float(ind_15m.get('ema50', price) or price)
    breakout_15m = float(ind_15m.get('breakout_score', 50) or 50)

    bias_4h      = tf4h.get('bias', 'NEUTRAL')
    score_4h     = tf4h.get('trend_score', 50)
    rsi_4h       = tf4h.get('rsi_4h', 50)
    ema_bull_4h  = tf4h.get('ema_bullish', False)

    # 15m trend
    trend_15m_bull = (price > ema20_15m > ema50_15m)
    macd_bull_15m  = (macd > macd_sig)
    rsi_15m_ok     = (40 <= rsi_15m <= 70)  # not overbought/oversold

    # Alignment matrix
    if bias_4h == 'BEARISH':
        # 4h is bearish → block all entries regardless
        bias    = 'BEARISH'
        score   = max(0.0, score_4h * 0.3)
        confirmed = False
        reason  = f"4h BEARISH (score={score_4h:.0f}) — entry blocked"

    elif bias_4h == 'BULLISH' and trend_15m_bull and macd_bull_15m:
        # Perfect alignment: 4h+15m trend + MACD
        bias      = 'BULLISH'
        score     = min(100.0, (score_4h + (rsi_15m / 100 * 30) + 20))
        confirmed = True
        reason    = "4h BULLISH + 15m trend aligned + MACD positive"

    elif bias_4h == 'BULLISH' and (trend_15m_bull or macd_bull_15m):
        # Partial 15m alignment
        bias    = 'BULLISH'
        score   = min(100.0, (score_4h * 0.7 + 25))
        confirmed = rsi_15m_ok and rsi_4h < 72
        reason  = "4h BULLISH + partial 15m alignment"

    elif bias_4h == 'NEUTRAL' and trend_15m_bull and macd_bull_15m:
        # 4h neutral, 15m strongly bullish → proceed cautiously
        bias    = 'MIXED'
        score   = min(100.0, 45 + breakout_15m * 0.2)
        confirmed = False
        reason  = "4h NEUTRAL — 15m bullish but no higher-timeframe confirmation"

    else:
        bias    = 'MIXED'
        score   = 40.0
        confirmed = False
        reason  = f"No clear alignment: 4h={bias_4h} 15m_trend={trend_15m_bull}"

    return {
        'mtf_score':       round(score, 1),
        'mtf_confirmed':   confirmed,
        'mtf_bias':        bias,
        'alignment_reason':reason,
        'rsi_15m':         round(rsi_15m, 1),
        'rsi_4h':          round(rsi_4h, 1),
        'trend_15m_bull':  trend_15m_bull,
        'macd_bull_15m':   macd_bull_15m,
        'timeframe_4h':    tf4h,
    }


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def get_mtf_confirmation(symbol: str, ind_15m: Dict,
                         cache: Dict = None) -> Dict:
    """
    Main entry point. Fetches 4h data and computes alignment.

    Args:
        symbol:  e.g. 'BTCUSDT'
        ind_15m: indicators dict from analyze_indicators() (15m)
        cache:   optional dict to cache 4h data between calls
                 { symbol: (tf4h_dict, fetch_time) }

    Returns:
        mtf_score, mtf_confirmed, mtf_bias, alignment_reason,
        timeframe_4h, rsi_15m, rsi_4h
    """
    neutral = {
        'mtf_score': 50.0, 'mtf_confirmed': False,
        'mtf_bias': 'NEUTRAL', 'alignment_reason': 'No 4h data',
        'rsi_15m': 50, 'rsi_4h': 50, 'timeframe_4h': {},
        'trend_15m_bull': False, 'macd_bull_15m': False,
    }
    try:
        import time as t
        now = t.time()

        # Check cache (valid for 4h = 14400s, but use 1h for freshness)
        if cache is not None and symbol in cache:
            tf4h, fetched_at = cache[symbol]
            if now - fetched_at < 3600:  # 1h cache
                return _compute_alignment(ind_15m, tf4h)

        # Fetch 4h klines
        klines_4h = get_4h_klines(symbol, limit=60)
        if not klines_4h:
            return neutral

        tf4h = _analyze_4h(klines_4h)

        # Store in cache
        if cache is not None:
            cache[symbol] = (tf4h, now)

        result = _compute_alignment(ind_15m, tf4h)
        logger.debug(
            f"[MTF] {symbol}: {result['mtf_bias']} "
            f"score={result['mtf_score']:.0f} "
            f"confirmed={result['mtf_confirmed']}"
        )
        return result

    except Exception as e:
        logger.error(f"[MTF] {symbol}: {e}")
        return neutral


# ─────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import logging as _log
    _log.basicConfig(level=_log.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from services.binance_scanner import get_klines
    from services.indicator_engine import analyze_indicators

    cache = {}
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']

    print("\n" + "=" * 65)
    print("  MULTI-TIMEFRAME CONFIRMATION ENGINE TEST")
    print("=" * 65)
    for sym in symbols:
        klines = get_klines(sym, '15m', 100)
        if not klines:
            continue
        ind = analyze_indicators(klines)
        if not ind:
            continue
        r = get_mtf_confirmation(sym, ind, cache=cache)
        conf = "✅" if r['mtf_confirmed'] else "❌"
        print(
            f"  {sym:10}: {conf} {r['mtf_bias']:8} "
            f"score={r['mtf_score']:5.1f} "
            f"RSI15m={r['rsi_15m']:.0f} "
            f"RSI4h={r['rsi_4h']:.0f} "
            f"| {r['alignment_reason']}"
        )
    print()
