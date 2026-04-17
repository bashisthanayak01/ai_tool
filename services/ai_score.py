"""
Final AI Score — Tech(70%) + News(30%) + Whale(10% additive)
Updated formula per system upgrade v3 (Whale Intelligence).
"""

import logging
from typing import Dict
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_final_score(profit_data: Dict, news_data: Dict,
                          indicators: Dict, whale_data: Dict = None,
                          probability_data: Dict = None,
                          regime_data: Dict = None) -> Dict:
    """
    Final AI score formula (v3 — Whale Intelligence):

        base_score  = technical_score * 0.70 + news_impact * 0.30
        whale_adj   = base_score + (whale_score_norm * WHALE_WEIGHT)
        final_score = whale_adj * regime_multiplier

    whale_score_norm is in [-1, +1] from the new whale engine.
    WHALE_WEIGHT is configured in settings (default 10).
    Regime multiplier only applied when regime confidence >= 55%.
    Clamped to 0-100.
    Thresholds: BUY >= 75 | HOLD >= 45 | SELL < 45
    """
    try:
        technical = profit_data.get('profit_score', 50)           # 0-100
        news_score = news_data.get('news_score', 0.0)             # -1 to +1
        whale_raw  = (whale_data or {})
        prob_up    = float((probability_data or {}).get('probability_up', 50) or 50)
        prob_down  = round(100.0 - prob_up, 2)

        # Map news_score (-1..+1) → news_impact (0..100)
        news_impact = (news_score + 1) * 50

        # Base formula: Tech 70% + News 30%
        base_score = technical * 0.70 + news_impact * 0.30

        # ── Whale contribution (additive) ──────────────────────
        # whale_score_norm is in [-1, +1] from upgraded tracker
        # If old whale (only whale_score 0-100), convert to norm
        whale_norm = whale_raw.get('whale_score_norm', None)
        if whale_norm is None:
            # Backward compat: whale_score 0-100 → norm
            old_ws = float(whale_raw.get('whale_score', 50) or 50)
            whale_norm = (old_ws - 50.0) / 50.0
        whale_norm  = max(-1.0, min(1.0, float(whale_norm)))
        whale_weight = getattr(settings, 'WHALE_WEIGHT', 10)
        whale_contrib = whale_norm * whale_weight
        score_before_whale = round(min(100, max(0, base_score)), 2)
        whale_adj_score    = base_score + whale_contrib

        # ── Regime multiplier ───────────────────────────────
        REGIME_MULT = {'BULL': 1.10, 'BEAR': 0.80, 'SIDEWAYS': 0.95}
        regime_mult  = 1.0
        applied_regime = 'NEUTRAL'
        if regime_data:
            r_name = regime_data.get('regime', 'SIDEWAYS')
            r_conf = float(regime_data.get('confidence', 0))
            if r_conf >= 55:
                regime_mult    = REGIME_MULT.get(r_name, 1.0)
                applied_regime = r_name

        final_score = round(min(100, max(0, whale_adj_score * regime_mult)), 2)

        # Signal thresholds — calibrated for BEAR/SIDEWAYS regime reality
        # BEAR regime applies ×0.80 which reduces scores significantly.
        # Example: tech=72 base=71 → after BEAR×0.80 = 57 → must be BUY
        # BUY  >= 55  (was 75, then 65 — both too high for BEAR regime)
        # HOLD >= 35  (covers mid-range coins)
        # SELL  < 35
        if final_score >= 55:
            signal = "BUY"
        elif final_score >= 35:
            signal = "HOLD"
        else:
            signal = "SELL"

        return {
            'final_score':    final_score,
            'final_signal':   signal,
            'confidence':     profit_data.get('confidence', 0),
            'risk_level':     profit_data.get('risk_level', 'Medium'),
            # Breakdown for transparency
            'technical_score': round(technical, 2),
            'news_weight':     round(news_impact * 0.30, 2),
            'news_impact':     round(news_impact, 2),
            # Probability complement
            'probability_down': prob_down,
            # Whale contribution
            'whale_contrib':      round(whale_contrib, 2),
            'whale_score_norm':   round(whale_norm, 4),
            'score_before_whale': score_before_whale,
            # Regime info
            'score_before_regime': round(min(100, max(0, whale_adj_score)), 2),
            'regime_multiplier':   round(regime_mult, 3),
            'applied_regime':      applied_regime,
            # Legacy compat
            'prob_contrib': round(prob_up * 0.15, 2),
        }

    except Exception as e:
        logger.error(f"Final score error: {e}")
        return {
            'final_score': 50.0, 'final_signal': 'HOLD',
            'confidence': 0, 'risk_level': 'Medium',
            'technical_score': 50, 'news_weight': 0,
            'news_impact': 50, 'whale_contrib': 0, 'prob_contrib': 0,
            'probability_down': 50.0,
        }


if __name__ == "__main__":
    r = calculate_final_score(
        {'profit_score': 72, 'confidence': 83, 'risk_level': 'Low'},
        {'news_score': 0.6},
        {},
        {'whale_score': 55},
        {'probability_up': 68},
    )
    print(f"Final: {r['final_score']}/100 ({r['final_signal']})")
    print(f"Tech={r['technical_score']}*0.7 = {r['technical_score']*0.7:.1f}")
    print(f"News impact={r['news_impact']}*0.3 = {r['news_weight']:.1f}")
