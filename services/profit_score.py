"""
Profit Score Calculator — outputs profit_score, confidence, risk_level, signal
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_profit_score(indicators: Dict) -> Dict:
    """
    Calculate profit score (0-100), confidence (0-100), risk_level, and signal.

    Factors:
        Trend Strength (EMA20 vs EMA50)   — 25 pts
        RSI Position                      — 20 pts
        Volume Spike                      — 15 pts
        Momentum (MACD)                   — 15 pts
        Volatility                        — 10 pts
        Price Breakout                    — 15 pts
    """
    try:
        trend = _trend_score(indicators)
        rsi = _rsi_score(indicators)
        volume = _volume_score(indicators)
        momentum = _momentum_score(indicators)
        volatility = _volatility_score(indicators)
        breakout = _breakout_score(indicators)

        profit_score = round(min(100, max(0, trend + rsi + volume + momentum + volatility + breakout)), 2)

        # Confidence: how many factors agree
        factor_scores = [trend / 25, rsi / 20, volume / 15, momentum / 15, volatility / 10, breakout / 15]
        strong_factors = sum(1 for f in factor_scores if f >= 0.6)
        confidence = round(min(100, (strong_factors / len(factor_scores)) * 100), 2)

        # Risk level
        vol = indicators.get('volatility', 0)
        if vol >= 5 or profit_score < 30:
            risk_level = "High"
        elif vol >= 2 or profit_score < 50:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Signal — BUY requires 78+ (was 75 — too easy to trigger)
        if profit_score >= 78:
            signal = "BUY"
        elif profit_score >= 45:
            signal = "HOLD"
        else:
            signal = "SELL"

        return {
            'profit_score': profit_score,
            'confidence': confidence,
            'risk_level': risk_level,
            'signal': signal,
            'breakdown': {
                'trend': trend,
                'rsi': rsi,
                'volume': volume,
                'momentum': momentum,
                'volatility': volatility,
                'breakout': breakout
            }
        }

    except Exception as e:
        logger.error(f"Profit score error: {e}")
        return {
            'profit_score': 50.0,
            'confidence': 0.0,
            'risk_level': 'Medium',
            'signal': 'HOLD',
            'breakdown': {}
        }


def _trend_score(ind: Dict) -> float:
    """EMA alignment: 0-25"""
    try:
        price = ind['price']
        ema20 = ind['ema20']
        ema50 = ind['ema50']
        if price > ema20 > ema50:
            return 25
        elif ema20 > ema50:
            return 18
        elif price > ema20:
            return 12
        elif price < ema20 < ema50:
            return 3
        else:
            return 7
    except:
        return 0


def _rsi_score(ind: Dict) -> float:
    """RSI momentum: 0-20. Best entry zone: 45-60 (not overbought, trending up)."""
    rsi = ind.get('rsi', 50)
    if 45 <= rsi <= 60:
        return 20    # Sweet spot: trending bullish, not overbought
    elif 60 < rsi <= 70:
        return 15    # Still ok but getting extended
    elif 30 <= rsi < 45:
        return 12    # Oversold bounce potential
    elif 70 < rsi <= 80:
        return 5     # FIX: overbought = high reversal risk, was incorrectly 13
    elif rsi > 80:
        return 1     # Extremely overbought — avoid
    elif rsi < 30:
        return 8     # Deeply oversold — risky but potential bounce
    else:
        return 5


def _volume_score(ind: Dict) -> float:
    """Volume spike: 0-15"""
    vs = ind.get('volume_spike', 1.0)
    if vs >= 3.0:
        return 15
    elif vs >= 2.0:
        return 13
    elif vs >= 1.5:
        return 10
    elif vs >= 1.0:
        return 6
    else:
        return 3


def _momentum_score(ind: Dict) -> float:
    """MACD momentum: 0-15"""
    macd = ind.get('macd', 0)
    sig = ind.get('macd_signal', 0)
    hist = ind.get('macd_histogram', 0)
    if macd > sig and hist > 0:
        return 15
    elif macd > sig:
        return 11
    elif macd > 0:
        return 7
    elif hist > 0:
        return 4
    else:
        return 1


def _volatility_score(ind: Dict) -> float:
    """Volatility opportunity: 0-10"""
    v = ind.get('volatility', 0)
    if v >= 5:
        return 10
    elif v >= 3:
        return 8
    elif v >= 2:
        return 6
    elif v >= 1:
        return 4
    else:
        return 2


def _breakout_score(ind: Dict) -> float:
    """Price breakout position: 0-15"""
    bs = ind.get('breakout_score', 50)
    if bs >= 90:
        return 15
    elif bs >= 75:
        return 12
    elif bs >= 60:
        return 9
    elif bs >= 40:
        return 6
    else:
        return 3


if __name__ == "__main__":
    test = {
        'price': 51234, 'rsi': 62, 'ema20': 50987, 'ema50': 49654,
        'macd': 123, 'macd_signal': 98, 'macd_histogram': 25,
        'volume': 12345, 'volume_spike': 2.3, 'volatility': 3.2,
        'breakout_score': 78
    }
    r = calculate_profit_score(test)
    print(f"Score: {r['profit_score']}/100  Confidence: {r['confidence']}%")
    print(f"Risk: {r['risk_level']}  Signal: {r['signal']}")
    print(f"Breakdown: {r['breakdown']}")
