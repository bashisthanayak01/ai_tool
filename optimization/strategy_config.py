"""
strategy_config.py — Active Strategy Configuration Manager
===========================================================
Provides a single source of truth for trading parameters:
  - get_active_strategy_config()  → used by scanner + backtester
  - set_active_strategy_config()  → used by auto_optimizer
  - get_regime_adjusted_config()  → returns regime-modified thresholds

Collections written:
    strategy_configs   — stored configs (one active at a time)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pymongo
from config import settings

logger = logging.getLogger(__name__)

# ── Default fallback config ────────────────────────────────────
DEFAULT_CONFIG = {
    'take_profit':     0.08,   # 8% TP — 2:1 R:R ratio
    'stop_loss':       0.04,   # 4% SL — tighter than TP
    'min_score':       50,     # Only strong signals (was 30)
    'min_probability': 40,     # Minimum confidence (was 30)
    'allow_hold_entry': False, # BUY signals only — HOLD = no trade
    'signal_window_hours': 48,
    'max_open_positions': 5,
    'fee_rate': 0.00075,
}

COLL_STRATEGY_CONFIGS = 'strategy_configs'

# ── Regime multipliers ─────────────────────────────────────────
REGIME_SCORE_MULTIPLIER = {
    'BULL':     1.10,   # Boost score slightly in strong uptrend
    'BEAR':     0.80,   # Penalise score — require stronger confirmation
    'SIDEWAYS': 0.95,   # Mild penalty — reduce noise trades
    'NEUTRAL':  1.00,
}

REGIME_MIN_SCORE_ADJUSTMENT = {
    'BULL':     -3,   # Lower bar in bull (more opportunities)
    'BEAR':     +8,   # Raise bar in bear (survive drawdown)
    'SIDEWAYS': +3,   # Slightly tighter
    'NEUTRAL':   0,
}

REGIME_MIN_PROB_ADJUSTMENT = {
    'BULL':     -3,
    'BEAR':     +8,
    'SIDEWAYS': +3,
    'NEUTRAL':   0,
}

# ── Confidence gate ─────────────────────────────────────────────
REGIME_CONFIDENCE_THRESHOLD = 55   # % below this → use NEUTRAL adjustments


def _get_db(client_ref=None):
    """Open a fresh MongoDB connection."""
    c = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=6000)
    return c, c[settings.DATABASE_NAME]


def get_active_strategy_config(db=None) -> Dict:
    """
    Return the currently active strategy config from DB.
    Falls back to DEFAULT_CONFIG if nothing is stored.

    Safe to call from scanner / backtester at any time.
    """
    own = (db is None)
    client = None
    try:
        if own:
            client, db = _get_db()

        doc = db[COLL_STRATEGY_CONFIGS].find_one(
            {'active': True},
            sort=[('created_at', -1)],
            projection={'_id': 0}
        )
        if doc and doc.get('params'):
            cfg = {**DEFAULT_CONFIG, **doc['params']}
            logger.debug(f"[StrategyConfig] Loaded active config id={doc.get('config_id')}")
            return cfg

    except Exception as e:
        logger.warning(f"[StrategyConfig] DB read error: {e} — using defaults")
    finally:
        if own and client:
            client.close()

    return dict(DEFAULT_CONFIG)


def get_regime_adjusted_config(regime_data: Dict = None, db=None) -> Dict:
    """
    Return base active config with regime-specific threshold adjustments.

    Args:
        regime_data: dict with keys 'regime' (str) and 'confidence' (int/float)
        db: optional pymongo db handle

    Returns:
        Config dict with adjusted min_score, min_probability, score_multiplier
    """
    base = get_active_strategy_config(db=db)

    regime = 'NEUTRAL'
    confidence = 0

    if regime_data:
        regime = regime_data.get('regime', 'NEUTRAL')
        confidence = float(regime_data.get('confidence', 0))

    # Below confidence gate → treat as NEUTRAL (no adjustment)
    if confidence < REGIME_CONFIDENCE_THRESHOLD:
        regime = 'NEUTRAL'
        logger.debug(f"[StrategyConfig] Regime confidence {confidence}% < {REGIME_CONFIDENCE_THRESHOLD}% → NEUTRAL")

    # Apply adjustments
    adj_score = REGIME_MIN_SCORE_ADJUSTMENT.get(regime, 0)
    adj_prob  = REGIME_MIN_PROB_ADJUSTMENT.get(regime, 0)
    multiplier = REGIME_SCORE_MULTIPLIER.get(regime, 1.0)

    cfg = dict(base)
    cfg['min_score']       = max(10, base['min_score'] + adj_score)
    cfg['min_probability'] = max(10, base['min_probability'] + adj_prob)
    cfg['score_multiplier'] = multiplier
    cfg['applied_regime']  = regime
    cfg['regime_confidence'] = confidence

    return cfg


def set_active_strategy_config(params: Dict, performance: Dict,
                                config_id: str = None,
                                regime: str = 'ALL',
                                db=None) -> bool:
    """
    Save a new config as the active one.
    Deactivates all previous active configs before inserting.

    Safety: will NOT switch if last switch was < 24 hours ago.

    Args:
        params:      dict with TP, SL, min_score, min_probability
        performance: dict with return_pct, win_rate, profit_factor, max_drawdown
        config_id:   optional string ID (auto-generated if None)
        regime:      regime context when this config was optimized
        db:          optional pymongo db handle

    Returns:
        True if saved successfully, False otherwise
    """
    own = (db is None)
    client = None
    try:
        if own:
            client, db = _get_db()

        col = db[COLL_STRATEGY_CONFIGS]

        # Safety: check last switch time
        last_active = col.find_one({'active': True}, sort=[('created_at', -1)])
        if last_active:
            last_at = last_active.get('created_at', datetime.min)
            if (datetime.utcnow() - last_at) < timedelta(hours=24):
                logger.warning(
                    f"[StrategyConfig] Config switch blocked: last switch was "
                    f"{(datetime.utcnow()-last_at).total_seconds()/3600:.1f}h ago (< 24h)"
                )
                return False

        # Deactivate all old configs
        col.update_many({'active': True}, {'$set': {'active': False}})

        # Composite score
        comp = (
            performance.get('return_pct', 0) * 0.4 +
            performance.get('win_rate', 0)   * 0.3 +
            performance.get('profit_factor', 0) * 0.2 -
            performance.get('max_drawdown', 0)  * 0.1
        )

        doc = {
            'config_id':   config_id or f"cfg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'params':      params,
            'performance': {**performance, 'composite_score': round(comp, 4)},
            'regime':      regime,
            'created_at':  datetime.utcnow(),
            'active':      True,
        }

        col.insert_one(doc)

        # Ensure indexes
        col.create_index([('active', 1)], background=True)
        col.create_index([('created_at', -1)], background=True)
        col.create_index([('config_id', 1)], unique=True, background=True)

        logger.info(
            f"[StrategyConfig] New active config saved: {doc['config_id']} "
            f"| TP={params.get('take_profit',0)*100:.0f}% "
            f"| SL={params.get('stop_loss',0)*100:.0f}% "
            f"| Score≥{params.get('min_score',0)} "
            f"| Composite={comp:.4f}"
        )
        return True

    except Exception as e:
        logger.error(f"[StrategyConfig] Save error: {e}")
        return False
    finally:
        if own and client:
            client.close()


def list_strategy_configs(db=None, limit: int = 10) -> list:
    """Return recent strategy configs for dashboard display."""
    own = (db is None)
    client = None
    try:
        if own:
            client, db = _get_db()
        docs = list(
            db[COLL_STRATEGY_CONFIGS].find({}, {'_id': 0})
            .sort('created_at', -1)
            .limit(limit)
        )
        return docs
    except Exception as e:
        logger.warning(f"[StrategyConfig] List error: {e}")
        return []
    finally:
        if own and client:
            client.close()
