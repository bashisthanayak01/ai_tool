"""
Whale Movement Detection — detects accumulation/distribution via volume & price anomalies
"""

import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_whale_activity(klines: List[Dict]) -> Dict:
    """
    Detect whale movements using volume spikes, large candle moves,
    and abnormal order flow (volume proxy).

    Args:
        klines: List of kline dicts (must have close, high, low, volume)

    Returns:
        {whale_score: 0-100, whale_signal: ACCUMULATION|DISTRIBUTION|NONE}
    """
    try:
        if not klines or len(klines) < 20:
            return {'whale_score': 0, 'whale_signal': 'NONE'}

        # Get recent data
        recent = klines[-20:]
        volumes = [k['volume'] for k in recent]
        closes = [k['close'] for k in recent]

        avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
        latest_vol = volumes[-1]

        # ── Factor 1: Volume spike ratio (0-30) ──
        vol_ratio = latest_vol / avg_volume if avg_volume > 0 else 1.0
        if vol_ratio >= 5.0:
            vol_score = 30
        elif vol_ratio >= 3.0:
            vol_score = 25
        elif vol_ratio >= 2.0:
            vol_score = 18
        elif vol_ratio >= 1.5:
            vol_score = 10
        else:
            vol_score = 3

        # ── Factor 2: Large candle body (0-25) ──
        latest = klines[-1]
        body = abs(latest['close'] - latest['open'])
        candle_range = latest['high'] - latest['low']
        body_ratio = body / candle_range if candle_range > 0 else 0

        # Compare body size to avg candle range
        avg_range = sum(k['high'] - k['low'] for k in recent[:-1]) / len(recent[:-1])
        body_vs_avg = body / avg_range if avg_range > 0 else 0

        if body_vs_avg >= 3.0:
            body_score = 25
        elif body_vs_avg >= 2.0:
            body_score = 18
        elif body_vs_avg >= 1.5:
            body_score = 12
        else:
            body_score = 4

        # ── Factor 3: Sudden liquidity increase (consecutive volume surge) (0-25) ──
        recent_vols = volumes[-5:]
        vol_increase_count = sum(1 for i in range(1, len(recent_vols))
                                  if recent_vols[i] > recent_vols[i-1] * 1.3)
        if vol_increase_count >= 3:
            liquidity_score = 25
        elif vol_increase_count >= 2:
            liquidity_score = 15
        elif vol_increase_count >= 1:
            liquidity_score = 8
        else:
            liquidity_score = 2

        # ── Factor 4: Abnormal order flow proxy (0-20) ──
        # Detect if volume is concentrated in up-moves or down-moves
        up_volume = sum(k['volume'] for k in recent[-10:] if k['close'] >= k['open'])
        down_volume = sum(k['volume'] for k in recent[-10:] if k['close'] < k['open'])
        total_vol = up_volume + down_volume

        if total_vol > 0:
            buy_pressure = up_volume / total_vol
        else:
            buy_pressure = 0.5

        if buy_pressure >= 0.75 or buy_pressure <= 0.25:
            flow_score = 20
        elif buy_pressure >= 0.65 or buy_pressure <= 0.35:
            flow_score = 13
        else:
            flow_score = 4

        # ── Total whale score ──
        whale_score = min(100, vol_score + body_score + liquidity_score + flow_score)

        # ── Determine signal ──
        price_change = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0

        if whale_score >= 40:
            if price_change > 0 and buy_pressure >= 0.55:
                whale_signal = 'ACCUMULATION'
            elif price_change < 0 and buy_pressure < 0.45:
                whale_signal = 'DISTRIBUTION'
            else:
                whale_signal = 'ACCUMULATION' if buy_pressure >= 0.5 else 'DISTRIBUTION'
        else:
            whale_signal = 'NONE'

        return {
            'whale_score': whale_score,
            'whale_signal': whale_signal,
            'vol_ratio': round(vol_ratio, 2),
            'buy_pressure': round(buy_pressure, 2),
        }

    except Exception as e:
        logger.error(f"Whale detection error: {e}")
        return {'whale_score': 0, 'whale_signal': 'NONE'}


if __name__ == "__main__":
    from services.binance_scanner import get_klines
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        klines = get_klines(sym, '15m', 200)
        if klines:
            r = detect_whale_activity(klines)
            print(f"{sym}: Score={r['whale_score']}, Signal={r['whale_signal']}, "
                  f"VolRatio={r.get('vol_ratio')}, BuyPressure={r.get('buy_pressure')}")
