"""
Market Regime Detection — Bull / Bear / Sideways using BTC as market proxy
Cached for 5 minutes. Regime history persisted to regime_history collection.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pymongo
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache
_regime_cache: Optional[Dict] = None
_regime_cached_at: float = 0
_CACHE_TTL = 300  # 5 minutes


def detect_market_regime(btc_klines: List[Dict] = None) -> Dict:
    """
    Detect global market regime using BTC data.
    Cached for 5 minutes.

    Indicators:
        - EMA50 vs EMA200
        - Market volatility
        - Momentum (price vs EMA50)
        - Trend consistency

    Returns:
        {regime: BULL|BEAR|SIDEWAYS, confidence: 0-100, btc_trend: str}
    """
    global _regime_cache, _regime_cached_at

    # Check cache
    if _regime_cache and (time.time() - _regime_cached_at) < _CACHE_TTL:
        return _regime_cache

    try:
        # Fetch BTC data if not provided
        if btc_klines is None:
            from services.binance_scanner import get_klines
            btc_klines = get_klines('BTCUSDT', '1d', 250)

        if not btc_klines or len(btc_klines) < 200:
            # Try with what we have
            if not btc_klines or len(btc_klines) < 50:
                return {'regime': 'SIDEWAYS', 'confidence': 0, 'btc_trend': 'UNKNOWN'}

        closes = [k['close'] for k in btc_klines]

        # ── EMA 50 vs EMA 200 ──
        ema50 = _ema(closes, 50)
        ema200 = _ema(closes, min(200, len(closes) - 1))

        ema50_val = ema50[-1] if ema50 else closes[-1]
        ema200_val = ema200[-1] if ema200 else closes[-1]
        price = closes[-1]

        ema_cross = (ema50_val - ema200_val) / ema200_val * 100 if ema200_val > 0 else 0

        # ── Volatility (20-day) ──
        recent = closes[-20:]
        avg = sum(recent) / len(recent)
        variance = sum((c - avg) ** 2 for c in recent) / len(recent)
        volatility = (variance ** 0.5) / avg * 100

        # ── Momentum: price vs EMA50 ──
        momentum = (price - ema50_val) / ema50_val * 100 if ema50_val > 0 else 0

        # ── Trend consistency: how many of last 10 days closed above EMA50 ──
        ema50_recent = ema50[-10:] if len(ema50) >= 10 else ema50
        close_recent = closes[-10:]
        above_count = sum(1 for i in range(len(ema50_recent))
                          if close_recent[i] > ema50_recent[i])
        consistency = above_count / len(ema50_recent) if ema50_recent else 0.5

        # ── Score ──
        bull_score = 0
        bear_score = 0

        # EMA cross
        if ema_cross > 2:
            bull_score += 30
        elif ema_cross > 0:
            bull_score += 15
        elif ema_cross < -2:
            bear_score += 30
        else:
            bear_score += 15

        # Momentum
        if momentum > 3:
            bull_score += 25
        elif momentum > 0:
            bull_score += 10
        elif momentum < -3:
            bear_score += 25
        else:
            bear_score += 10

        # Consistency
        if consistency >= 0.7:
            bull_score += 25
        elif consistency >= 0.5:
            bull_score += 10
        elif consistency <= 0.3:
            bear_score += 25
        else:
            bear_score += 10

        # Volatility (high vol = uncertainty)
        vol_penalty = min(20, volatility * 3)

        # Determine regime
        net = bull_score - bear_score
        if net >= 25:
            regime = 'BULL'
            confidence = min(95, 50 + net)
        elif net <= -25:
            regime = 'BEAR'
            confidence = min(95, 50 + abs(net))
        else:
            regime = 'SIDEWAYS'
            confidence = max(20, 60 - abs(net))

        # Adjust for volatility
        confidence = max(10, int(confidence - vol_penalty / 2))

        # Trend description
        if regime == 'BULL':
            btc_trend = f"BTC above EMA50 ({ema_cross:+.1f}%), momentum {momentum:+.1f}%"
        elif regime == 'BEAR':
            btc_trend = f"BTC below EMA50 ({ema_cross:+.1f}%), momentum {momentum:+.1f}%"
        else:
            btc_trend = f"BTC consolidating, volatility {volatility:.1f}%"

        result = {
            'regime': regime,
            'confidence': confidence,
            'btc_trend': btc_trend,
            'ema_cross': round(ema_cross, 2),
            'momentum': round(momentum, 2),
            'volatility': round(volatility, 2),
        }

        # Cache
        _regime_cache = result
        _regime_cached_at = time.time()

        logger.info(f"Market Regime: {regime} (confidence {confidence}%)")
        return result

    except Exception as e:
        logger.error(f"Market regime error: {e}")
        return {'regime': 'SIDEWAYS', 'confidence': 0, 'btc_trend': 'ERROR'}


def _ema(data: list, period: int) -> list:
    """Calculate EMA for a list of floats"""
    if len(data) < period:
        return data
    k = 2 / (period + 1)
    ema_vals = [sum(data[:period]) / period]
    for price in data[period:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


# ══════════════════════════════════════════════════════════════
# REGIME HISTORY PERSISTENCE
# ══════════════════════════════════════════════════════════════

COLL_REGIME_HISTORY = 'regime_history'


def save_regime_history(regime_data: Dict, btc_price: float = None, db=None) -> bool:
    """
    Persist current regime snapshot to regime_history collection.
    Called by scheduler every hour.

    Args:
        regime_data: dict from detect_market_regime()
        btc_price:   current BTC close price (optional)
        db:          optional pymongo database handle

    Returns:
        True if saved, False on error
    """
    own = (db is None)
    client = None
    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=6000)
            db = client[settings.DATABASE_NAME]

        doc = {
            'regime':      regime_data.get('regime', 'SIDEWAYS'),
            'confidence':  regime_data.get('confidence', 0),
            'btc_trend':   regime_data.get('btc_trend', ''),
            'ema_cross':   regime_data.get('ema_cross', 0),
            'momentum':    regime_data.get('momentum', 0),
            'volatility':  regime_data.get('volatility', 0),
            'btc_price':   btc_price,
            'detected_at': datetime.utcnow(),
        }

        col = db[COLL_REGIME_HISTORY]
        col.insert_one(doc)

        # Ensure indexes
        col.create_index([('detected_at', -1)], background=True)
        col.create_index([('regime', 1), ('detected_at', -1)], background=True)

        # TTL: delete records older than 30 days automatically
        col.create_index(
            [('detected_at', 1)],
            expireAfterSeconds=30 * 24 * 3600,
            background=True,
            name='regime_ttl'
        )

        logger.debug(f"[RegimeHistory] Saved: {doc['regime']} conf={doc['confidence']}%")
        return True

    except Exception as e:
        logger.error(f"[RegimeHistory] Save error: {e}")
        return False
    finally:
        if own and client:
            client.close()


def get_regime_history(days: int = 7, db=None) -> List[Dict]:
    """
    Return regime history for the last N days.
    Used by dashboard chart.

    Returns list of dicts: {regime, confidence, detected_at}
    """
    own = (db is None)
    client = None
    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=6000)
            db = client[settings.DATABASE_NAME]

        since = datetime.utcnow() - timedelta(days=days)
        docs = list(
            db[COLL_REGIME_HISTORY]
            .find(
                {'detected_at': {'$gte': since}},
                {'_id': 0, 'regime': 1, 'confidence': 1, 'btc_price': 1,
                 'detected_at': 1, 'ema_cross': 1, 'momentum': 1}
            )
            .sort('detected_at', 1)
        )
        return docs

    except Exception as e:
        logger.warning(f"[RegimeHistory] Read error: {e}")
        return []
    finally:
        if own and client:
            client.close()


def get_regime_summary(days: int = 7, db=None) -> Dict:
    """
    Return aggregated regime summary for dashboard display.
    Shows distribution of BULL/BEAR/SIDEWAYS over the period.
    """
    history = get_regime_history(days=days, db=db)
    if not history:
        return {'total': 0, 'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0,
                'dominant': 'UNKNOWN', 'current': None}

    counts = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0}
    for h in history:
        r = h.get('regime', 'SIDEWAYS')
        counts[r] = counts.get(r, 0) + 1

    dominant = max(counts, key=counts.get)
    return {
        'total':    len(history),
        'BULL':     counts.get('BULL', 0),
        'BEAR':     counts.get('BEAR', 0),
        'SIDEWAYS': counts.get('SIDEWAYS', 0),
        'dominant': dominant,
        'current':  history[-1] if history else None,
    }


if __name__ == "__main__":
    r = detect_market_regime()
    print(f"Regime: {r['regime']}, Confidence: {r['confidence']}%")
    print(f"Trend: {r['btc_trend']}")
    saved = save_regime_history(r)
    print(f"Saved to DB: {saved}")
