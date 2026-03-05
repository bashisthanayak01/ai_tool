"""
Probability Engine — estimates probability of price going UP in 4-12 hours
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_probability(indicators: Dict, news_data: Dict,
                          whale_data: Dict, regime_data: Dict,
                          horizon_hours: int = 8) -> Dict:
    """
    Estimate probability that price goes UP in the next horizon.

    Formula:
        probability =
            trend_score * 0.25 +
            momentum_score * 0.20 +
            volume_score * 0.05 +
            news_score * 0.25 +
            whale_score * 0.15 +
            regime_alignment * 0.10

    Returns:
        {probability_up: 0-100, confidence: 0-100, horizon_hours: int}
    """
    try:
        trend = _trend_factor(indicators)
        momentum = _momentum_factor(indicators)
        volume = _volume_factor(indicators)
        news = _news_factor(news_data)
        whale = _whale_factor(whale_data)
        regime = _regime_factor(regime_data, indicators)

        probability = (
            trend * 0.25 +
            momentum * 0.20 +
            volume * 0.05 +
            news * 0.25 +
            whale * 0.15 +
            regime * 0.10
        )

        probability = round(min(100, max(0, probability)), 2)

        # Confidence: how many factors agree (above 50)
        factors = [trend, momentum, volume, news, whale, regime]
        agreeing = sum(1 for f in factors if f > 55)
        disagreeing = sum(1 for f in factors if f < 40)

        if agreeing >= 4:
            confidence = min(95, 60 + agreeing * 5)
        elif disagreeing >= 3:
            confidence = max(15, 40 - disagreeing * 5)
        else:
            confidence = 50

        return {
            'probability_up': probability,
            'probability_confidence': confidence,
            'horizon_hours': horizon_hours,
        }

    except Exception as e:
        logger.error(f"Probability error: {e}")
        return {'probability_up': 50.0, 'probability_confidence': 0, 'horizon_hours': horizon_hours}


def _trend_factor(ind: Dict) -> float:
    """EMA alignment → 0-100"""
    try:
        price = ind.get('price', 0)
        ema20 = ind.get('ema20', price)
        ema50 = ind.get('ema50', price)
        if price > ema20 > ema50:
            return 85
        elif ema20 > ema50:
            return 65
        elif price > ema50:
            return 55
        elif price < ema20 < ema50:
            return 15
        else:
            return 35
    except:
        return 50


def _momentum_factor(ind: Dict) -> float:
    """MACD + RSI → 0-100"""
    rsi = ind.get('rsi', 50)
    macd_hist = ind.get('macd_histogram', 0)

    score = 50
    if rsi > 60:
        score += 15
    elif rsi < 40:
        score -= 15

    if macd_hist > 0:
        score += 20
    elif macd_hist < 0:
        score -= 20

    return min(100, max(0, score))


def _volume_factor(ind: Dict) -> float:
    """Volume spike → 0-100"""
    vs = ind.get('volume_spike', 1.0)
    if vs >= 3.0:
        return 90
    elif vs >= 2.0:
        return 75
    elif vs >= 1.5:
        return 60
    elif vs >= 1.0:
        return 45
    else:
        return 25


def _news_factor(news: Dict) -> float:
    """News sentiment → 0-100"""
    score = news.get('news_score', 0)  # -1 to +1
    # Map -1..+1 → 0..100
    return round(max(0, min(100, (score + 1) * 50)), 2)


def _whale_factor(whale: Dict) -> float:
    """Whale activity → 0-100"""
    ws = whale.get('whale_score', 0)
    signal = whale.get('whale_signal', 'NONE')
    if signal == 'ACCUMULATION':
        return min(100, ws + 20)
    elif signal == 'DISTRIBUTION':
        return max(0, 50 - ws)
    else:
        return 50


def _regime_factor(regime: Dict, ind: Dict) -> float:
    """Market regime alignment → 0-100"""
    r = regime.get('regime', 'SIDEWAYS')
    price = ind.get('price', 0)
    ema20 = ind.get('ema20', price)

    bullish_coin = price > ema20

    if r == 'BULL' and bullish_coin:
        return 90
    elif r == 'BULL':
        return 60
    elif r == 'BEAR' and not bullish_coin:
        return 15
    elif r == 'BEAR':
        return 35
    else:
        return 50


if __name__ == "__main__":
    r = calculate_probability(
        {'price': 51000, 'rsi': 62, 'ema20': 50900, 'ema50': 49600,
         'macd_histogram': 25, 'volume_spike': 2.3},
        {'news_score': 0.5},
        {'whale_score': 45, 'whale_signal': 'ACCUMULATION'},
        {'regime': 'BULL'},
    )
    print(f"Probability Up: {r['probability_up']}%, Confidence: {r['probability_confidence']}%")
