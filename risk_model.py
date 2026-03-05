"""
Risk Model — Risk-Adjusted Professional Scoring
================================================
Takes the raw scan result (indicators + final AI score) and computes a
risk-adjusted score that penalises dangerous setups and rewards
high-quality, liquid, low-volatility opportunities.

FORMULA (all terms clamped before summing):
  risk_adjusted_score =
      base_technical_score      (0-100, from profit_score / final_score)
    + probability_weight        (0-15)
    + news_weight               (0-10)
    + liquidity_bonus           (0-10)
    + rr_bonus                  (-5 to +10) [penalty if RR < 1.5, bonus if > 2.0]
    - volatility_penalty        (0-25)
    - drawdown_penalty          (0-15)

Final: normalise to [0, 100].

New fields written to each coin dict:
  risk_adjusted_score   float 0-100
  risk_score            float 0-100   (inverted: high score = low risk)
  risk_level            str  LOW | MEDIUM | HIGH
  volatility_penalty    float (deducted points)
  drawdown_penalty      float (deducted points)
  liquidity_bonus       float (added points)
  rr_bonus              float (added or subtracted)
  probability_weight    float (added points)
  news_weight           float (added points)
  atr_pct               float ATR as % of price
  recent_drawdown       float max drawdown % over recent window
  liquidity_score       float 0-100
  risk_reward_ratio     float TP/SL ratio
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning Constants ──────────────────────────────────────────
# TP / SL assumptions (same as backtester optimal params)
_TP_PCT = 0.12
_SL_PCT = 0.05
_BASE_RR = _TP_PCT / _SL_PCT          # 2.4×

# Volatility thresholds (ATR % of price)
_VOL_LOW    = 1.5   # ATR% < 1.5 → minimal penalty
_VOL_MED    = 3.0   # ATR% 1.5-3 → moderate penalty
_VOL_HIGH   = 5.0   # ATR% 3-5   → heavy penalty
_VOL_SEVERE = 8.0   # ATR% > 8   → max penalty

# Drawdown thresholds (%)
_DD_MILD   = 5.0
_DD_MOD    = 10.0
_DD_HEAVY  = 20.0

# Volume-to-MA thresholds for liquidity
_LIQ_POOR  = 0.5
_LIQ_OK    = 1.0
_LIQ_GOOD  = 2.0
_LIQ_GREAT = 3.0


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════

def apply_risk_model(coin: Dict, klines: List[Dict] = None) -> Dict:
    """
    Apply the risk model to a single coin result dict.

    Parameters
    ----------
    coin    : result dict from scan_market() / market_pipeline.py
    klines  : raw kline list (optional) — used for recent drawdown calculation.
              If None, drawdown uses only the market_data fields already in `coin`.

    Returns
    -------
    coin dict with risk fields added in-place (and returned).
    """
    try:
        # ── 1. Base technical score ──
        base = float(coin.get('technical_score',
                              coin.get('profit_score',
                              coin.get('final_score', 50))) or 50)

        # ── 2. Probability weight  (0–15) ──
        prob = float(coin.get('probability_up', 50) or 50)
        probability_weight = _prob_weight(prob)

        # ── 3. News confidence weight  (0–10) ──
        news_score  = float(coin.get('news_score', 0) or 0)     # -1..+1
        news_weight = _news_weight(news_score)

        # ── 4. Liquidity bonus  (0–10) ──
        vol_spike = float(coin.get('volume_spike', 1.0) or 1.0)
        liquidity_score  = _compute_liquidity_score(vol_spike)
        liquidity_bonus  = _liquidity_bonus(vol_spike)

        # ── 5. Volatility penalty  (0–25) ──
        # Use ATR% already computed by indicator_engine
        atr_pct = float(coin.get('volatility', 0) or 0)   # already ATR% of price
        volatility_penalty = _volatility_penalty(atr_pct)

        # ── 6. Drawdown penalty  (0–15) ──
        recent_drawdown = _estimate_recent_drawdown(coin, klines)
        drawdown_penalty = _drawdown_penalty(recent_drawdown)

        # ── 7. R:R bonus/penalty  (-5..+10) ──
        # We use a fixed TP/SL model (same as backtester optimal params)
        rr_val  = round(_TP_PCT / _SL_PCT, 2)          # 2.4
        rr_bonus = _rr_bonus(rr_val)

        # ── 8. Composite risk-adjusted score ──
        raw = (
            base
            + probability_weight
            + news_weight
            + liquidity_bonus
            + rr_bonus
            - volatility_penalty
            - drawdown_penalty
        )
        risk_adjusted_score = round(max(0.0, min(100.0, raw)), 2)

        # ── 9. Risk score (inverted risk view: high = safe) ──
        # Penalise by how much was deducted vs max possible penalty
        penalty_total = volatility_penalty + drawdown_penalty
        max_penalty   = 25 + 15  # max totals
        risk_score    = round(max(0.0, min(100.0, 100 - (penalty_total / max_penalty) * 100)), 2)

        # ── 10. Risk level badge ──
        if risk_score >= 67:
            risk_level = "LOW"
        elif risk_score >= 34:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Annotate coin dict
        coin.update({
            'risk_adjusted_score':  risk_adjusted_score,
            'risk_score':           risk_score,
            'risk_level':           risk_level,
            'volatility_penalty':   round(volatility_penalty, 2),
            'drawdown_penalty':     round(drawdown_penalty, 2),
            'liquidity_bonus':      round(liquidity_bonus, 2),
            'liquidity_score':      round(liquidity_score, 2),
            'rr_bonus':             round(rr_bonus, 2),
            'probability_weight':   round(probability_weight, 2),
            'news_weight_risk':     round(news_weight, 2),
            'atr_pct':              round(atr_pct, 3),
            'recent_drawdown':      round(recent_drawdown, 2),
            'risk_reward_ratio':    rr_val,
        })

        logger.debug(
            f"[Risk] {coin.get('symbol','?')} -> RA={risk_adjusted_score} "
            f"RS={risk_score} ({risk_level}) "
            f"| VolPen={volatility_penalty:.1f} DDPen={drawdown_penalty:.1f} "
            f"LiqBonus={liquidity_bonus:.1f} RRBonus={rr_bonus:.1f}"
        )

    except Exception as e:
        logger.error(f"[Risk] apply_risk_model error for {coin.get('symbol','?')}: {e}")
        # Safe defaults on error
        coin.setdefault('risk_adjusted_score',  float(coin.get('final_score', 50) or 50))
        coin.setdefault('risk_score',            50.0)
        coin.setdefault('risk_level',            'MEDIUM')
        coin.setdefault('volatility_penalty',    0.0)
        coin.setdefault('drawdown_penalty',      0.0)
        coin.setdefault('liquidity_bonus',       0.0)
        coin.setdefault('liquidity_score',       50.0)
        coin.setdefault('rr_bonus',              0.0)
        coin.setdefault('probability_weight',    0.0)
        coin.setdefault('news_weight_risk',      0.0)
        coin.setdefault('atr_pct',               0.0)
        coin.setdefault('recent_drawdown',       0.0)
        coin.setdefault('risk_reward_ratio',     _BASE_RR)

    return coin


def apply_risk_model_batch(coins: List[Dict], klines_map: Dict = None) -> List[Dict]:
    """
    Apply risk model to a full scan result list.

    Parameters
    ----------
    coins      : list of coin dicts from scan_market()
    klines_map : optional dict {symbol: klines_list}

    Returns
    -------
    Same list with risk fields added to each coin.
    """
    klines_map = klines_map or {}
    for coin in coins:
        sym    = coin.get('symbol', '')
        klines = klines_map.get(sym)
        apply_risk_model(coin, klines)
    return coins


# ══════════════════════════════════════════════════════════════
# COMPONENT CALCULATORS
# ══════════════════════════════════════════════════════════════

def _prob_weight(prob: float) -> float:
    """Probability up → added score points (0–15)."""
    if prob >= 70:   return 15.0
    if prob >= 60:   return 12.0
    if prob >= 50:   return 9.0
    if prob >= 40:   return 6.0
    if prob >= 35:   return 3.0
    return 0.0


def _news_weight(news_score: float) -> float:
    """News score (-1..+1) → added/deducted points (-10..+10)."""
    # Map [-1,+1] → [-10,+10] linear
    return round(news_score * 10, 2)


def _compute_liquidity_score(vol_spike: float) -> float:
    """Vol spike ratio → liquidity score 0-100."""
    if vol_spike >= _LIQ_GREAT: return 100.0
    if vol_spike >= _LIQ_GOOD:  return 75.0
    if vol_spike >= _LIQ_OK:    return 50.0
    if vol_spike >= _LIQ_POOR:  return 25.0
    return 10.0


def _liquidity_bonus(vol_spike: float) -> float:
    """Vol spike → liquidity bonus points (0–10)."""
    if vol_spike >= _LIQ_GREAT: return 10.0
    if vol_spike >= _LIQ_GOOD:  return 7.0
    if vol_spike >= _LIQ_OK:    return 4.0
    if vol_spike >= _LIQ_POOR:  return 1.0
    return -3.0   # penalty for very thin volume


def _volatility_penalty(atr_pct: float) -> float:
    """ATR% → volatility penalty points (0–25).

    Higher volatility = wider price swings = harder to exit at SL/TP → penalty.
    """
    if atr_pct >= _VOL_SEVERE: return 25.0
    if atr_pct >= _VOL_HIGH:   return 18.0
    if atr_pct >= _VOL_MED:    return 10.0
    if atr_pct >= _VOL_LOW:    return 4.0
    return 0.0


def _drawdown_penalty(dd_pct: float) -> float:
    """Recent max drawdown % → penalty points (0–15)."""
    if dd_pct >= _DD_HEAVY: return 15.0
    if dd_pct >= _DD_MOD:   return 9.0
    if dd_pct >= _DD_MILD:  return 4.0
    return 0.0


def _rr_bonus(rr: float) -> float:
    """Risk:Reward ratio → bonus/penalty (-5..+10)."""
    if rr >= 3.0:   return 10.0
    if rr >= 2.5:   return 7.0
    if rr >= 2.0:   return 5.0
    if rr >= 1.5:   return 2.0
    if rr >= 1.0:   return 0.0
    return -5.0     # below 1:1 — penalise heavily


def _estimate_recent_drawdown(coin: Dict, klines: List[Dict] = None) -> float:
    """
    Estimate recent max drawdown %.
    Uses last 20 klines if provided, otherwise falls back to
    (high_20 - low_20) / high_20 approximation from breakout_score / volatility.
    """
    try:
        if klines and len(klines) >= 5:
            closes = [float(k.get('close', 0)) for k in klines[-20:] if k.get('close')]
            if len(closes) >= 2:
                peak = max(closes)
                trough_after_peak = min(closes[closes.index(peak):]) if peak in closes else min(closes)
                if peak > 0:
                    return round((peak - trough_after_peak) / peak * 100, 2)

        # Fallback: approximate from volatility and price range
        # breakout_score tells us where price is in recent 20-bar range
        bs  = float(coin.get('breakout_score', 50) or 50)   # 0=bottom, 100=top of range
        vol = float(coin.get('volatility', 2) or 2)         # ATR%

        # Rough estimate: if atr% = 3%, over 20 bars the range ~= 3*sqrt(20) ≈ 13%
        # And if breakout_score = 30 (near bottom), drawdown ≈ (100-30)% of that range
        range_pct   = vol * (20 ** 0.5)
        dd_fraction = (100 - bs) / 100
        return round(range_pct * dd_fraction, 2)

    except Exception:
        return 5.0   # safe fallback


# ══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import pymongo
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from config import settings

    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[settings.DATABASE_NAME]

    # Load latest signals
    pipeline = [
        {'$sort': {'timestamp': -1}},
        {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
        {'$replaceRoot': {'newRoot': '$doc'}},
        {'$limit': 90},
    ]
    signals = list(db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
    print(f"\n=== RISK MODEL — STANDALONE TEST ===")
    print(f"Loaded {len(signals)} signals from DB\n")

    results = apply_risk_model_batch(signals)

    # Print comparison table
    print(f"{'Symbol':<14} {'Old Score':>9} {'RA Score':>9} {'Risk Sc':>8} "
          f"{'Level':<8} {'VolPen':>7} {'DDPen':>6} {'LiqBon':>7} {'RR':>5}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get('risk_adjusted_score', 0), reverse=True)[:20]:
        sym   = r.get('symbol', '')
        old   = r.get('final_score', r.get('technical_score', 0))
        ra    = r.get('risk_adjusted_score', 0)
        rs    = r.get('risk_score', 0)
        lvl   = r.get('risk_level', '')
        vp    = r.get('volatility_penalty', 0)
        dp    = r.get('drawdown_penalty', 0)
        lb    = r.get('liquidity_bonus', 0)
        rr    = r.get('risk_reward_ratio', 0)
        delta = ra - old
        print(f"{sym:<14} {old:>9.1f} {ra:>9.1f} {rs:>8.1f} "
              f"{lvl:<8} {vp:>7.1f} {dp:>6.1f} {lb:>7.1f} {rr:>5.1f}")

    # Save updated risk fields to DB
    print(f"\nSaving risk fields to MongoDB ai_signals...")
    updated = 0
    col = db[settings.COLLECTION_AI_SIGNALS]
    for r in results:
        sym = r.get('symbol')
        if not sym:
            continue
        update_fields = {
            'risk_adjusted_score': r.get('risk_adjusted_score'),
            'risk_score':          r.get('risk_score'),
            'risk_level':          r.get('risk_level'),
            'volatility_penalty':  r.get('volatility_penalty'),
            'drawdown_penalty':    r.get('drawdown_penalty'),
            'liquidity_bonus':     r.get('liquidity_bonus'),
            'liquidity_score':     r.get('liquidity_score'),
            'rr_bonus':            r.get('rr_bonus'),
            'atr_pct':             r.get('atr_pct'),
            'recent_drawdown':     r.get('recent_drawdown'),
            'risk_reward_ratio':   r.get('risk_reward_ratio'),
        }
        res = col.update_many({'symbol': sym}, {'$set': update_fields})
        updated += res.modified_count

    print(f"Updated {updated} signal documents with risk fields")
    client.close()
    print("\n=== COMPLETE ===")
