"""
Ranking Engine v2 — Best Coin Ranking AI System
================================================
Identifies and ranks the TOP crypto trading opportunities.

Philosophy (v2 upgrade):
  The old engine rewarded coins already in a strong uptrend, which meant it
  often highlighted coins that had ALREADY pumped 10–20%. These are bad entry
  points — high risk of reversal, poor risk/reward for a new position.

  v2 now actively seeks PRE-BREAKOUT setups:
    - Coins that are FLAT or slightly recovering (not already pumped)
    - RSI in the 45–62 reset zone (momentum recovering, not overbought)
    - Volume coiling then surging (institutional accumulation signal)
    - Price close to key support (asymmetric risk/reward)
    - Price NOT overextended above EMA20 (no chasing pumps)

Composite Score Formula (v2):
  rank_score =
    (technical_score      * WEIGHT_TECHNICAL)    +  # 0.25 — AI signal quality
    (probability_up       * WEIGHT_PROB_UP)      +  # 0.25 — probability engine
    (news_sentiment       * WEIGHT_NEWS)         +  # 0.10 — news sentiment
    (volume_strength      * WEIGHT_VOLUME)       +  # 0.08 — volume vs average
    (trend_strength       * WEIGHT_TREND)        +  # 0.07 — price vs EMA
    (rsi_zone_score       * WEIGHT_RSI_ZONE)     +  # 0.10 — RSI setup quality
    (volume_coiling_score * WEIGHT_VOL_COILING)  +  # 0.08 — dry-up→surge pattern
    (support_proximity    * WEIGHT_SUPPORT_PROX) +  # 0.07 — near support level
    (entry_timing_score   * WEIGHT_ENTRY_TIMING)    # 0.00 — blended penalty factor

  An OVEREXTENSION PENALTY and 24H GAIN PENALTY are applied as multipliers
  after the composite score, not as additive components. This creates a smooth
  reduction rather than a cliff-edge cutoff.

Hard Filters (exclude entirely):
  - signal NOT in ['BUY', 'STRONG_BUY', 'HOLD', 'SELL']
  - technical_score < 40
  - probability_up < 35
  - price <= 0
  - 24h gain > HARD_EXCLUDE_24H_GAIN_PCT (already pumped way too much)

Outputs:
  - Top 3, Top 5, Full list
  - Saves to MongoDB `ranked_opportunities` collection
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import pymongo

from config import settings
from ai.smart_levels import compute_smart_levels

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
COLLECTION = "ranked_opportunities"

# ── Composite Score Weights (must sum to 1.0) ──────────────────
# These weights determine what the engine values most.
# v2: Old weights heavily favoured already-trending coins (tech+trend = 45%).
#     New weights balance momentum with SETUP QUALITY (RSI zone, coiling, support).
WEIGHT_TECHNICAL    = 0.25   # AI technical signal quality
WEIGHT_PROB_UP      = 0.25   # Probability engine confidence
WEIGHT_NEWS         = 0.10   # News sentiment contribution
WEIGHT_VOLUME       = 0.08   # Current volume vs 20-bar average
WEIGHT_TREND        = 0.07   # Price position vs EMA20/50
WEIGHT_RSI_ZONE     = 0.10   # RSI setup quality (prefers 45–62 reset zone)
WEIGHT_VOL_COILING  = 0.08   # Volume dry-up→surge pattern detection
WEIGHT_SUPPORT_PROX = 0.07   # Proximity to key support level
# NOTE: WEIGHT_ENTRY_TIMING is a multiplier, not additive — applied after composite

# ── Hard Filters (disqualify entirely) ────────────────────────
FILTER_SIGNALS              = {'BUY', 'STRONG_BUY', 'HOLD', 'SELL'}
FILTER_MIN_TECH             = 40    # Minimum technical score
FILTER_MIN_PROB             = 35    # Minimum probability up %
HARD_EXCLUDE_24H_GAIN_PCT   = 12.0  # Skip coins already up >12% in 24h — too late to enter

# ── Soft Penalty Thresholds ───────────────────────────────────
# These reduce rank_score smoothly rather than hard-excluding.
SOFT_PENALTY_24H_GAIN_START = 5.0   # Start penalising gains above this %
SOFT_PENALTY_24H_GAIN_MAX   = 12.0  # At this gain, penalty multiplier reaches ~0.50
OVEREXTENSION_THRESHOLD_PCT = 6.0   # Price >6% above EMA20 = overextended (start penalty)
OVEREXTENSION_MAX_PCT       = 15.0  # Price >15% above EMA20 = max penalty (multiplier ~0.35)

# ── RSI Zone Definitions ──────────────────────────────────────
# Best setup zone = RSI recovering from oversold, not yet overbought.
RSI_IDEAL_LOW   = 45    # Lower bound of ideal setup zone
RSI_IDEAL_HIGH  = 62    # Upper bound of ideal setup zone (above = already moving)
RSI_OVERBOUGHT  = 70    # Above this = late, likely to pull back soon


# ══════════════════════════════════════════════════════════════
# SECTION 1: NORMALIZER HELPERS
# ══════════════════════════════════════════════════════════════

def _norm(value: float, lo: float, hi: float) -> float:
    """Normalise value to 0–100 given a known range [lo, hi]."""
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100))


# ══════════════════════════════════════════════════════════════
# SECTION 2: ORIGINAL SCORING COMPONENTS (v1 — kept for compatibility)
# ══════════════════════════════════════════════════════════════

def _compute_volume_strength(coin: Dict) -> float:
    """
    Volume strength 0–100.
    Uses volume_spike field (ratio of current volume to 20-bar avg).
    spike >= 2× = strong (score 67+), 1× = average (score 33), <0.5 = weak.
    """
    spike = float(coin.get('volume_spike', 1.0) or 1.0)
    # Map: 0×→0, 1×→33, 2×→67, 3+×→100
    return min(100.0, (spike / 3.0) * 100.0)


def _compute_trend_strength(coin: Dict) -> float:
    """
    Trend alignment score 0–100 based on price vs EMA20/EMA50.
    NOTE (v2): This is intentionally kept in the formula but with reduced weight
    (0.07 vs old 0.10) because strong trend = already moved = worse entry.
    The overextension penalty below provides the counterbalance.

    price > ema20 > ema50  → aligned uptrend   → 80 base
    price > ema20 only     → moderate uptrend  → 60 base
    price > ema50 only     → weak support      → 40 base
    price < ema50          → downtrend         → 15 base
    Blended with breakout_score (20% weight) for additional signal.
    """
    price  = float(coin.get('price', 0) or 0)
    ema20  = float(coin.get('ema20', price) or price)
    ema50  = float(coin.get('ema50', ema20) or ema20)
    bs     = float(coin.get('breakout_score', 0) or 0)  # 0–100 from indicator engine

    if price <= 0 or ema20 <= 0:
        return 50.0

    if price > ema20 and ema20 > ema50:
        base = 80.0
    elif price > ema20:
        base = 60.0
    elif price > ema50:
        base = 40.0
    else:
        base = 15.0

    return min(100.0, base * 0.8 + bs * 0.2)


def _normalize_news_score(coin: Dict) -> float:
    """
    News sentiment → 0–100.
    news_score from news_collector is in [-1, +1]; maps to [0, 100].
    Neutral = 50, fully bullish = 100, fully bearish = 0.
    """
    ns = float(coin.get('news_score', 0) or 0)
    return (ns + 1.0) / 2.0 * 100.0


# ══════════════════════════════════════════════════════════════
# SECTION 3: NEW v2 SCORING COMPONENTS — SETUP QUALITY
# These functions identify PRE-BREAKOUT setups, not already-pumped coins.
# ══════════════════════════════════════════════════════════════

def _compute_rsi_zone_score(coin: Dict) -> float:
    """
    RSI Setup Quality Score — 0 to 100.

    Philosophy:
      The best entry point RSI is NOT the highest — it's the RESET zone.
      When RSI pulls back to 45–62 after a strong move, it signals:
        - The coin had strength (was above 60+)
        - It pulled back (healthy reset, not a collapse)
        - It's ready to continue the move (momentum recovering)

      RSI > 70 (overbought) = bad entry, likely to cool off
      RSI < 35 (oversold)   = could bounce, but also could keep falling
      RSI 45–62 (reset zone) = ideal setup zone

    Scoring:
      45–62  →  100  (ideal: momentum reset, ready to resume)
      62–70  →  70   (decent but getting extended)
      35–45  →  60   (recovering from dip, acceptable)
      70–80  →  35   (overbought, avoid or take profits)
      > 80   →  10   (extremely overbought, very high reversal risk)
      < 35   →  30   (oversold, high risk, may keep falling)
    """
    rsi = float(coin.get('rsi', 50) or 50)

    if RSI_IDEAL_LOW <= rsi <= RSI_IDEAL_HIGH:   # 45–62: optimal setup zone
        return 100.0
    elif RSI_IDEAL_HIGH < rsi <= RSI_OVERBOUGHT:  # 62–70: decent but getting late
        # Linear decay: 100 at 62, 70 at 70
        return 100.0 - ((rsi - RSI_IDEAL_HIGH) / (RSI_OVERBOUGHT - RSI_IDEAL_HIGH)) * 30.0
    elif 35 < rsi < RSI_IDEAL_LOW:               # 35–45: recovering from dip
        return 60.0
    elif RSI_OVERBOUGHT < rsi <= 80:              # 70–80: overbought
        return 35.0
    elif rsi > 80:                                # >80: extremely overbought
        return 10.0
    else:                                         # <35: oversold danger zone
        return 30.0


def _compute_volume_coiling_score(coin: Dict) -> float:
    """
    Volume Coiling Detection Score — 0 to 100.

    Philosophy:
      Institutional accumulation typically looks like this:
        1. Volume DRIES UP over several candles (sellers exhausted, smart money
           quietly buying at low prices without moving price)
        2. Then volume SURGES upward as price breaks out

      This pattern (coiling → surge) is one of the most reliable pre-breakout
      signals in crypto. The opposite (already high volume for many candles) =
      the move is already underway and latecomers are buying the top.

    How it's computed:
      - volume_spike = current_volume / 20-bar_average (from indicator engine)
      - If spike just happened (1.5–3×) after being low = SETUP forming → high score
      - If spike is massive (>3×) = may already be mid-breakout → moderate score
      - If volume was consistently low (spike <0.8×) = no interest → low score

    Data available:
      volume_spike: ratio of latest candle volume to 20-bar moving average
      (computed by indicator_engine.py, already in the coin dict)
    """
    spike = float(coin.get('volume_spike', 1.0) or 1.0)

    # Ideal coiling surge: volume picks up meaningfully but isn't a blowoff top
    if 1.5 <= spike <= 3.0:    # Healthy volume pickup — setup forming
        return 90.0 + (spike - 1.5) * 6.7   # Peaks at 100 around 2× spike
    elif 3.0 < spike <= 5.0:   # Strong breakout volume — already moving, decent
        return 75.0
    elif spike > 5.0:           # Extreme spike — blowoff top risk, late entry
        return 40.0
    elif 1.0 <= spike < 1.5:   # Normal volume — neutral, no coiling signal
        return 50.0
    else:                       # <1× — volume drying up (consolidation phase)
        # Low volume = coin coiling. Score it higher than "no interest"
        # because this could be the pre-surge setup
        return 55.0 + (1.0 - spike) * 30.0  # Higher score for tighter coiling


def _compute_support_proximity_score(coin: Dict) -> float:
    """
    Support Proximity Score — 0 to 100.

    Philosophy:
      Buying near a key support level gives the BEST risk/reward:
        - Stop loss is close (just below support) → small risk
        - Reward is the next resistance → large potential gain
        - If support holds → profit; if breaks → small loss

      Buying far from support (mid-range or near resistance) = bad R:R:
        - Stop is far away (below support) → large risk
        - Already close to resistance → limited upside

    Scoring:
      near_support = True (within 0.5 ATR of support)  → 100
      near_resistance = True                            → 20 (opposite of ideal)
      Neither (mid-range)                               → 50 (neutral)
      sr_quality > 60 + near_support                   → extra bonus

    Data:
      near_support, near_resistance, sr_quality — from indicator_engine.py
      (computed via pivot point detection across last 50 candles)
    """
    near_sup  = bool(coin.get('near_support', False))
    near_res  = bool(coin.get('near_resistance', False))
    sr_qual   = float(coin.get('sr_quality', 0) or 0)   # 0–100, how many S/R levels exist

    if near_sup and not near_res:
        # Ideal: at support, away from resistance — perfect asymmetric entry
        bonus = min(10.0, sr_qual / 10.0)  # Up to +10 bonus for clean S/R quality
        return min(100.0, 85.0 + bonus)
    elif near_res and not near_sup:
        # Bad: price near resistance = limited upside, risky
        return 20.0
    elif near_sup and near_res:
        # Compressed range — tight coil, could go either way
        return 55.0
    else:
        # No clear S/R proximity — neutral
        return 50.0


def _compute_24h_gain_penalty_multiplier(coin: Dict) -> float:
    """
    24h Gain Penalty Multiplier — 0.35 to 1.0 (applied to final rank_score).

    Philosophy:
      A coin that already pumped 10%+ in the last 24 hours is a LATE entry:
        - You are buying after everyone else already bought
        - The risk of a pullback/reversal is high
        - The remaining upside is reduced
        - The best traders were in BEFORE the 10% move, not after

      This penalty multiplier reduces the rank_score for already-pumped coins
      so they don't appear at the top of the opportunity list.

    Penalty curve:
      24h gain < 5%   → no penalty (multiplier = 1.0)  — fresh
      24h gain = 8%   → multiplier ≈ 0.75              — moderate penalty
      24h gain = 12%  → multiplier ≈ 0.50              — heavy penalty
      24h gain > 12%  → hard-excluded by HARD_EXCLUDE_24H_GAIN_PCT filter

    Data source:
      price_change_24h_pct — computed from klines by scan_market().
      Falls back to 0 if not available (old signals before this field existed).
    """
    gain = float(coin.get('price_change_24h_pct', 0) or 0)

    if gain <= SOFT_PENALTY_24H_GAIN_START:    # 0–5%: no penalty
        return 1.0

    # Linear penalty from 5% gain (multiplier=1.0) to 12% gain (multiplier=0.35)
    range_pct = SOFT_PENALTY_24H_GAIN_MAX - SOFT_PENALTY_24H_GAIN_START  # = 7
    how_far   = min(gain - SOFT_PENALTY_24H_GAIN_START, range_pct) / range_pct
    return round(1.0 - (how_far * 0.65), 3)  # 1.0 → 0.35


def _compute_overextension_penalty_multiplier(coin: Dict) -> float:
    """
    Overextension Penalty Multiplier — 0.35 to 1.0 (applied to final rank_score).

    Philosophy:
      When price is far above its EMA20, it is OVEREXTENDED:
        - EMA20 is the short-term 'fair value' baseline
        - Price 2–3% above EMA20 = normal healthy uptrend
        - Price 8–15% above EMA20 = parabolic, due for a mean-reversion pullback
        - Buying into overextension = buying at the peak of a short-term move

      This multiplier penalises rank_score when price is stretched too far
      above EMA20, discouraging late entries into extended moves.

    Penalty curve:
      price/EMA20 - 1 < 6%   → no penalty (multiplier = 1.0)
      price/EMA20 - 1 = 10%  → multiplier ≈ 0.72
      price/EMA20 - 1 = 15%  → multiplier ≈ 0.35

    Note: Only applied on the UPSIDE. If price is below EMA20, this
    function returns 1.0 (the downside filter is handled in rank_coins()).
    """
    price = float(coin.get('price', 0) or 0)
    ema20 = float(coin.get('ema20', price) or price)

    if price <= 0 or ema20 <= 0:
        return 1.0   # Can't compute — no penalty

    pct_above = ((price / ema20) - 1.0) * 100.0

    if pct_above <= OVEREXTENSION_THRESHOLD_PCT:  # < 6%: normal, no penalty
        return 1.0

    # Linear decay from 6% (multiplier=1.0) to 15%+ (multiplier=0.35)
    range_pct = OVEREXTENSION_MAX_PCT - OVEREXTENSION_THRESHOLD_PCT  # = 9
    how_far   = min(pct_above - OVEREXTENSION_THRESHOLD_PCT, range_pct) / range_pct
    return round(1.0 - (how_far * 0.65), 3)  # 1.0 → 0.35


# NOTE: _compute_entry_levels() replaced by ai.smart_levels.compute_smart_levels()
# Kept as fallback for direct calls from legacy code.
def _compute_entry_levels(price: float, tp_pct: float = 0.12,
                           sl_pct: float = 0.05) -> Dict:
    """Legacy fallback. Use compute_smart_levels() for AI-driven calculation."""
    entry  = round(price, 8)
    tp     = round(price * (1 + tp_pct), 8)
    sl     = round(price * (1 - sl_pct), 8)
    rr     = round(tp_pct / sl_pct, 2)
    return {'entry_price': entry, 'take_profit': tp, 'stop_loss': sl, 'risk_reward_ratio': rr}


# ══════════════════════════════════════════════════════════════
# MAIN RANKING FUNCTION
# ══════════════════════════════════════════════════════════════

def rank_coins(scan_results: List[Dict]) -> List[Dict]:
    """
    Takes the full market scan output and returns ranked list of FRESH setups.

    v2 Changes vs v1:
      - Added 4 new scoring components: RSI zone, volume coiling,
        support proximity (weighted additively in composite score)
      - Added 2 penalty multipliers applied AFTER composite:
        overextension penalty (price too far above EMA20)
        24h gain penalty (coin already pumped, bad entry timing)
      - Added hard filter: skip coins up >12% in 24h
      - Rebalanced weights (tech+trend reduced from 45%→32%, setup quality = 25%)

    Parameters
    ----------
    scan_results : List[Dict]
        Output from market_pipeline.scan_market(). Each dict must have at
        minimum: symbol, final_signal, technical_score, probability_up,
        price, ema20, rsi, volume_spike, near_support, near_resistance,
        sr_quality. Optional: price_change_24h_pct (added by scan_market v2).

    Returns
    -------
    List[Dict]
        All qualifying coins sorted by rank_score descending.
        Entry dict includes all scoring component scores for transparency.
    """
    ranked = []

    for coin in scan_results:
        # ── 1. Basic entry filters ─────────────────────────────────────────
        signal = coin.get('final_signal', coin.get('signal', ''))
        if signal not in FILTER_SIGNALS:
            continue

        tech  = float(coin.get('technical_score', coin.get('profit_score', 0)) or 0)
        prob  = float(coin.get('probability_up', 0) or 0)
        price = float(coin.get('price', 0) or 0)
        ema20 = float(coin.get('ema20', price) or price)

        if tech < FILTER_MIN_TECH:
            continue
        if prob < FILTER_MIN_PROB:
            continue
        if price <= 0:
            continue
        # Allow slight slack (1%) below EMA20 — coin may be in healthy pullback
        if ema20 > 0 and price < ema20 * 0.99:
            continue

        # ── 2. Hard filter: Skip coins already way up in 24h ──────────────
        # A coin up >12% in 24h is a late entry — high reversal risk,
        # poor R:R for a new position. Don't show it as an opportunity.
        gain_24h = float(coin.get('price_change_24h_pct', 0) or 0)
        if gain_24h > HARD_EXCLUDE_24H_GAIN_PCT:
            logger.debug(
                f"[Ranking] {coin.get('symbol')} excluded: 24h gain {gain_24h:.1f}% "
                f"> hard limit {HARD_EXCLUDE_24H_GAIN_PCT}%"
            )
            continue

        # ── 3. Compute all component scores (each 0–100) ───────────────────
        tech_norm         = min(100.0, tech)
        prob_norm         = min(100.0, prob)
        news_norm         = _normalize_news_score(coin)
        vol_str           = _compute_volume_strength(coin)
        trend_str         = _compute_trend_strength(coin)
        # v2 new components:
        rsi_zone          = _compute_rsi_zone_score(coin)
        vol_coiling       = _compute_volume_coiling_score(coin)
        support_prox      = _compute_support_proximity_score(coin)

        # ── 4. Composite rank score (weighted sum of all components) ────────
        # Weights are defined at the top of this file with full explanations.
        rank_score = (
            tech_norm    * WEIGHT_TECHNICAL    +   # AI signal quality
            prob_norm    * WEIGHT_PROB_UP      +   # probability engine
            news_norm    * WEIGHT_NEWS         +   # news sentiment
            vol_str      * WEIGHT_VOLUME       +   # volume vs average
            trend_str    * WEIGHT_TREND        +   # price vs EMA alignment
            rsi_zone     * WEIGHT_RSI_ZONE     +   # RSI reset zone quality (NEW)
            vol_coiling  * WEIGHT_VOL_COILING  +   # dry-up→surge detection (NEW)
            support_prox * WEIGHT_SUPPORT_PROX     # support proximity (NEW)
        )
        rank_score = round(rank_score, 2)

        # ── 5. Apply penalty multipliers ──────────────────────────────────
        # These reduce rank_score smoothly for overextended or already-pumped
        # coins. Applied AFTER composite so they're transparent in the logs.
        overext_mult  = _compute_overextension_penalty_multiplier(coin)  # 0.35–1.0
        gain_mult     = _compute_24h_gain_penalty_multiplier(coin)       # 0.35–1.0
        penalty_mult  = overext_mult * gain_mult  # Combined penalty
        rank_score_penalised = round(rank_score * penalty_mult, 2)

        if penalty_mult < 0.95:   # Log when penalty is significant
            logger.debug(
                f"[Ranking] {coin.get('symbol')} penalty: "
                f"overext={overext_mult:.2f} gain24h={gain_mult:.2f} → "
                f"{rank_score:.1f} → {rank_score_penalised:.1f}"
            )

        # ── 6. AI-Driven Entry / SL / TP (ai.smart_levels) ───────────────
        # Uses ATR, RSI, EMA20/50, probability_up, whale_signal,
        # news_score, market_regime, nearest_support/resistance.
        levels = compute_smart_levels(coin)

        # ── 7. Blend trade quality as tie-breaker (5% weight) ─────────────
        # trade_quality_score is from smart_levels — reflects R:R and setup quality.
        quality = levels.get('trade_quality_score', 0)
        rank_score_final = round(rank_score_penalised * 0.95 + quality * 0.05, 2)

        entry = {
            'symbol':                  coin.get('symbol', ''),
            'rank_score':              rank_score_final,

            # ── Score transparency (so analysts understand each component) ──
            'raw_rank_score':          rank_score,             # pre-penalty composite
            'rank_score_penalised':    rank_score_penalised,   # after overext+24h penalty
            'overext_penalty':         round(overext_mult, 3), # 1.0 = no penalty
            'gain24h_penalty':         round(gain_mult, 3),    # 1.0 = no penalty
            'price_change_24h_pct':    round(gain_24h, 2),     # 24h % gain/loss
            'rsi_zone_score':          round(rsi_zone, 1),     # setup zone quality
            'vol_coiling_score':       round(vol_coiling, 1),  # dry-up→surge
            'support_proximity_score': round(support_prox, 1), # near support?

            # ── Core fields ───────────────────────────────────────────────
            'signal':                  signal,
            'final_signal':            signal,
            'technical_score':         round(tech, 1),
            'probability_up':          round(prob, 1),
            'probability_down':        round(float(coin.get('probability_down', 100 - prob) or 0), 1),
            'news_sentiment':          coin.get('news_sentiment', 'NEUTRAL'),
            'news_score':              round(float(coin.get('news_score', 0) or 0), 3),
            'news_sentiment_norm':     round(news_norm, 1),
            'volume_strength':         round(vol_str, 1),
            'trend_strength':          round(trend_str, 1),
            'final_score':             float(coin.get('final_score', 0) or 0),

            # ── Risk model fields ─────────────────────────────────────────
            'risk_adjusted_score':     float(coin.get('risk_adjusted_score', 0) or 0),
            'risk_score':              float(coin.get('risk_score', 0) or 0),
            'risk_level':              coin.get('risk_level', ''),

            # ── Technical indicators (for dashboard smart_levels recompute) ─
            'price':                   price,
            'rsi':                     float(coin.get('rsi', 0) or 0),
            'rsi_4h':                  float(coin.get('rsi_4h', 50) or 50),
            'ema20':                   float(coin.get('ema20', 0) or 0) or None,
            'ema50':                   float(coin.get('ema50', 0) or 0) or None,
            'atr':                     float(coin.get('atr', 0) or 0) or levels.get('levels_atr') or None,
            'atr_pct':                 levels.get('levels_atr_pct', 0),
            'volume_spike':            float(coin.get('volume_spike', 1) or 1),
            'volatility':              float(coin.get('volatility', 0) or 0),
            'nearest_support':         coin.get('nearest_support'),
            'nearest_resistance':      coin.get('nearest_resistance'),
            'near_support':            bool(coin.get('near_support', False)),
            'near_resistance':         bool(coin.get('near_resistance', False)),
            'sr_quality':              float(coin.get('sr_quality', 0) or 0),
            'mtf_confirmed':           bool(coin.get('mtf_confirmed', False)),

            # ── Market context ─────────────────────────────────────────────
            'whale_signal':            coin.get('whale_signal', 'NONE'),
            'whale_score':             float(coin.get('whale_score', 50) or 50),
            'whale_buy_pressure':      float(coin.get('whale_buy_pressure', 0) or 0),
            'whale_sell_pressure':     float(coin.get('whale_sell_pressure', 0) or 0),
            'market_regime':           coin.get('market_regime', 'UNKNOWN'),

            # ── v3: Multi-timeframe classification ────────────────────────────
            # trade_type: SWING (15m) | POSITION (1h+15m) | TREND (daily+1h+15m)
            # Assigned by market_pipeline based on daily_trend + hourly_trend + position_score.
            'trade_type':              coin.get('trade_type', 'SWING'),

            # ── v3: 90-day daily trend context (from market_data DB collection) ─
            # trend_alignment_mult: multiplier already applied to final_score
            # (1.15 for UPTREND, 0.85 for DOWNTREND, 1.00 for SIDEWAYS/missing)
            'daily_trend':             coin.get('daily_trend', 'SIDEWAYS'),
            'trend_alignment_mult':    float(coin.get('trend_alignment_mult', 1.0) or 1.0),
            'daily_ema20':             coin.get('daily_ema20'),
            'daily_ema50':             coin.get('daily_ema50'),
            'daily_support_zone':      coin.get('daily_support_zone'),
            'daily_resistance_zone':   coin.get('daily_resistance_zone'),
            'distance_from_ema20d':    float(coin.get('distance_from_ema20d', 0.0) or 0.0),

            # ── v3: 1-hour candle analysis (200×1h = ~8 days) ────────────────
            # position_score: 0–100 composite score for POSITION trade readiness
            # Key thresholds: ≥70 → potential TREND, ≥60 → potential POSITION
            'hourly_trend':            coin.get('hourly_trend', 'SIDEWAYS'),
            'hourly_momentum':         coin.get('hourly_momentum', 'NEUTRAL'),
            'position_score':          float(coin.get('position_score', 0.0) or 0.0),
            'rsi_1h':                  float(coin.get('rsi_1h', 50.0) or 50.0),

            # ── AI Entry Levels (smart_levels — recomputed live in dashboard too) ─
            'entry_price':             levels['entry_price'],
            'stop_loss':               levels['stop_loss'],
            'take_profit':             levels['take_profit'],
            'risk_reward_ratio':       levels['risk_reward_ratio'],
            'trade_quality_score':     levels.get('trade_quality_score', 0),
            'entry_logic':             levels.get('entry_logic', ''),
            'sl_logic':                levels.get('sl_logic', ''),
            'tp_logic':                levels.get('tp_logic', ''),
            'created_at':              datetime.utcnow(),
        }
        ranked.append(entry)

    # Sort descending by final rank_score
    ranked.sort(key=lambda x: x['rank_score'], reverse=True)

    # Assign rank position and top_rank (top 3 get 1/2/3, rest get None)
    for i, r in enumerate(ranked, 1):
        r['rank'] = i
        r['top_rank'] = i if i <= 3 else None

    logger.info(
        f"[Ranking] {len(scan_results)} coins scanned → {len(ranked)} qualified "
        f"(excluded {len(scan_results) - len(ranked)} by filters)"
    )
    if ranked:
        top = ranked[0]
        logger.info(
            f"[Ranking] #1: {top['symbol']} score={top['rank_score']} "
            f"rsi={top.get('rsi',0):.0f} gain24h={top.get('price_change_24h_pct',0):.1f}% "
            f"near_sup={top.get('near_support',False)}"
        )
    return ranked


# ══════════════════════════════════════════════════════════════
# MONGODB PERSISTENCE
# ══════════════════════════════════════════════════════════════

def save_rankings(ranked: List[Dict], db, all_scanned_symbols: List[str] = None) -> int:
    """
    Save ranked opportunities to MongoDB.
    Two-layer stale cleanup:
      1. Delete records for symbols that were scanned but filtered out this cycle.
      2. Delete ANY record older than 20 min that wasn't refreshed (TTL purge).
         This catches coins that silently dropped out (SELL signal, errors, etc.)
    """
    if db is None:
        return 0
    try:
        from datetime import timedelta
        col = db[COLLECTION]

        # Ensure indexes
        col.create_index([('rank_score', pymongo.DESCENDING)])
        col.create_index([('symbol', pymongo.ASCENDING)], unique=True)
        col.create_index([('created_at', pymongo.DESCENDING)])

        current_ranked_symbols = {r['symbol'] for r in ranked}

        # ── Layer 1: delete symbols scanned-but-filtered ──────────────────────
        if all_scanned_symbols:
            stale_symbols = [s for s in all_scanned_symbols if s not in current_ranked_symbols]
            if stale_symbols:
                r1 = col.delete_many({'symbol': {'$in': stale_symbols}})
                if r1.deleted_count:
                    logger.info(f"[Ranking] Removed {r1.deleted_count} filtered-out symbols")

        # ── Layer 2: TTL purge — remove records not refreshed in this cycle ──
        # Any record whose created_at is more than 20 min old AND is NOT in
        # the current ranked set is definitely stale (price has changed).
        cutoff = datetime.utcnow() - timedelta(minutes=20)
        r2 = col.delete_many({
            'created_at': {'$lt': cutoff},
            'symbol': {'$nin': list(current_ranked_symbols)}
        })
        if r2.deleted_count:
            logger.info(f"[Ranking] TTL purge: removed {r2.deleted_count} stale records "
                        f"(older than 20 min, not in current top)")

        if not ranked:
            return 0

        # ── Upsert qualifying coins ────────────────────────────────────────────
        batch_ts = datetime.utcnow()
        inserted = 0
        for doc in ranked:
            doc['batch_ts'] = batch_ts
            col.update_one(
                {'symbol': doc['symbol']},
                {'$set': doc},
                upsert=True
            )
            inserted += 1

        logger.info(f"[Ranking] Saved {inserted} ranked coins to '{COLLECTION}'")
        return inserted
    except Exception as e:
        logger.error(f"[Ranking] Save error: {e}")
        return 0


# ══════════════════════════════════════════════════════════════
# QUERY HELPERS
# ══════════════════════════════════════════════════════════════

def get_top_opportunities(db, n: int = 10) -> List[Dict]:
    """Load top-N ranked coins from MongoDB, sorted by rank_score."""
    try:
        col = db[COLLECTION]
        docs = list(col.find({}, {'_id': 0}).sort('rank_score', pymongo.DESCENDING).limit(n))
        return docs
    except Exception as e:
        logger.error(f"[Ranking] Query error: {e}")
        return []


def get_ranking_summary(db) -> Dict:
    """Return summary stats about current rankings."""
    try:
        col = db[COLLECTION]
        total = col.count_documents({})
        buy_count  = col.count_documents({'signal': 'BUY'})
        hold_count = col.count_documents({'signal': 'HOLD'})
        best = col.find_one({}, {'symbol': 1, 'rank_score': 1, 'signal': 1, '_id': 0},
                             sort=[('rank_score', pymongo.DESCENDING)])
        last = col.find_one({}, {'created_at': 1, '_id': 0},
                             sort=[('created_at', pymongo.DESCENDING)])
        return {
            'total_ranked': total,
            'buy_count': buy_count,
            'hold_count': hold_count,
            'top_coin': best,
            'last_updated': last.get('created_at') if last else None,
        }
    except Exception as e:
        logger.error(f"[Ranking] Summary error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════

def run_ranking_from_db(db) -> List[Dict]:
    """
    Load latest AI signals from DB and re-rank them.
    Used when ranking needs to run independently of live scan.
    """
    try:
        # idx_signal_lookup index handles sort without RAM limit
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
            {'$replaceRoot': {'newRoot': '$doc'}},
        ]
        signals = list(db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
        logger.info(f"[Ranking] Loaded {len(signals)} latest signals from DB")
        ranked = rank_coins(signals)
        saved  = save_rankings(ranked, db)
        return ranked
    except Exception as e:
        logger.error(f"[Ranking] run_ranking_from_db error: {e}")
        return []


if __name__ == '__main__':
    import pymongo as _pymongo
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[settings.DATABASE_NAME]

    print("=" * 60)
    print("  RANKING ENGINE — STANDALONE RUN")
    print("=" * 60)

    ranked = run_ranking_from_db(db)

    print(f"\nTotal ranked coins: {len(ranked)}")
    if not ranked:
        print("No coins passed the entry filters (BUY/HOLD + tech>=40 + prob>=35)")
        print("Current signal distribution may be mostly SELL — normal in bear market.")
    else:
        print(f"\n{'─'*65}")
        print(f"  {'#':>3} {'Symbol':<14} {'Score':>6} {'Signal':<10} {'Tech':>5} {'Prob':>5} {'News':<10} {'RR':>4}")
        print(f"{'─'*65}")
        for r in ranked[:15]:
            print(f"  {r['rank']:>3} {r['symbol']:<14} {r['rank_score']:>5.1f} "
                  f"{r['signal']:<10} {r['technical_score']:>4.0f} {r['probability_up']:>4.0f}% "
                  f"{r['news_sentiment']:<10} {r['risk_reward_ratio']:>3.1f}x")

        print(f"\n{'─'*65}")
        print("TOP 3:")
        for r in ranked[:3]:
            print(f"  #{r['rank']} {r['symbol']}")
            print(f"     Rank Score:    {r['rank_score']}")
            print(f"     Signal:        {r['signal']}")
            print(f"     Tech Score:    {r['technical_score']}")
            print(f"     Prob Up:       {r['probability_up']}%")
            print(f"     Entry:         ${r['entry_price']:.6f}")
            print(f"     Stop Loss:     ${r['stop_loss']:.6f}")
            print(f"     Take Profit:   ${r['take_profit']:.6f}")
            print(f"     R:R:           {r['risk_reward_ratio']}x")

    summary = get_ranking_summary(db)
    print(f"\nDB Summary: {summary}")
    client.close()
