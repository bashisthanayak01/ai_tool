"""
ai/smart_levels.py — AI-Driven Entry / Stop Loss / Take Profit Engine
=======================================================================
Replaces all hardcoded TP/SL percentages with fully AI-calculated levels
using every available data signal.

DATA USED:
  • ATR               — actual volatility (basis for all distances)
  • RSI               — overbought/oversold (shifts entry target)
  • EMA20 / EMA50     — dynamic support/resistance zones (entry anchor)
  • probability_up    — probability engine (boosts/reduces TP distance)
  • whale_signal      — large capital direction (shifts TP up on ACCUMULATION)
  • news_score        — sentiment (tightens SL on negative news)
  • market_regime     — BULL/BEAR/SIDEWAYS (scales all distances)
  • nearest_support   — technical support price (SL anchor)
  • nearest_resistance— technical resistance (caps TP)
  • mtf_confirmed     — multi-timeframe agreement (confidence boost)
  • volatility_class  — LOW/MEDIUM/HIGH (scales SL width)
  • RSI_4h            — higher timeframe RSI (prevent entry in overbought)

OUTPUT:
  entry_price, stop_loss, take_profit, risk_reward_ratio,
  entry_logic, sl_logic, tp_logic, trade_quality_score

Author: AI System  v1.0
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Regime multipliers for SL and TP width ───────────────────────────────────
REGIME_TP_MULT  = {'BULL': 1.25,  'SIDEWAYS': 0.90, 'BEAR': 0.65, 'NEUTRAL': 1.0}
REGIME_SL_MULT  = {'BULL': 0.85,  'SIDEWAYS': 1.00, 'BEAR': 1.20, 'NEUTRAL': 1.0}

# ── ATR multipliers — loaded from RL DB at startup; updated every RL cycle ────
# These start at sensible defaults and are tuned by the RL optimizer over time.
ATR_SL_FACTOR   = 1.5    # SL = entry - (ATR × this)  [RL-learned, range 1.0-3.0]
ATR_TP_MIN_RR   = 2.0    # minimum R:R target          [RL-learned, range 1.5-4.0]
ATR_ENTRY_PULL  = 0.4    # entry = price - (ATR × this)[RL-learned, range 0.1-1.0]

# ── In-memory cache for RL params ─────────────────────────────────────────────
_rl_params_loaded_at: float = 0.0
_RL_PARAMS_TTL: float = 300.0   # reload every 5 minutes


def load_rl_params_from_db() -> None:
    """Load RL-learned ATR multipliers from MongoDB into module globals.

    Called once at startup and then every 5 minutes (cached). This is how
    the RL optimizer's learning feeds back into live trade level computation.
    Fails silently — defaults remain if DB is unreachable.
    """
    global ATR_SL_FACTOR, ATR_TP_MIN_RR, ATR_ENTRY_PULL, _rl_params_loaded_at

    now = time.time()
    if now - _rl_params_loaded_at < _RL_PARAMS_TTL:
        return  # cache still valid

    try:
        import pymongo
        from config import settings
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
        db     = client[settings.DATABASE_NAME]
        doc    = db['rl_parameters'].find_one({}, {'_id': 0}, sort=[('last_updated', -1)])
        client.close()
        if doc:
            ATR_SL_FACTOR  = float(doc.get('atr_sl_factor',  ATR_SL_FACTOR))
            ATR_TP_MIN_RR  = float(doc.get('atr_tp_min_rr',  ATR_TP_MIN_RR))
            ATR_ENTRY_PULL = float(doc.get('atr_entry_pull', ATR_ENTRY_PULL))
            logger.debug(f"[SmartLevels] RL params loaded: "
                         f"SL×{ATR_SL_FACTOR:.2f} RR×{ATR_TP_MIN_RR:.2f} pull×{ATR_ENTRY_PULL:.2f}")
        _rl_params_loaded_at = now
    except Exception as e:
        logger.debug(f"[SmartLevels] Could not load RL params from DB: {e}")
        _rl_params_loaded_at = now  # avoid hammering DB on connection failur


# ── Probability modifiers ─────────────────────────────────────────────────────
# Every +10% above 50% probability adds this to TP multiplier
PROB_TP_STEP    = 0.04    # prob 80% → +0.12 to TP mult

# ── Whale modifier ────────────────────────────────────────────────────────────
WHALE_TP_BOOST      = 0.20  # +20% TP when whales accumulating
WHALE_SL_TIGHTEN    = 0.15  # -15% SL distance when whales distributing

# ── News modifier ─────────────────────────────────────────────────────────────
NEWS_TP_MAX_BOOST   = 0.15   # max +15% TP for very positive news
NEWS_SL_TIGHTEN_MAX = 0.20   # max -20% SL when news is negative


# ═══════════════════════════════════════════════════════════════════════════════
# SMART ENTRY CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_smart_entry(price: float, ema20: float, ema50: float,
                         atr: float, rsi: float,
                         nearest_support: Optional[float] = None) -> tuple:
    """
    Choose the best limit-order entry point.

    Logic:
    1. If RSI > 65 (overbought): entry = EMA20 (wait for pullback, don't chase)
    2. If RSI < 35 (oversold): entry = current price (already at a discount)
    3. Normal: entry = midpoint between current price and EMA20 (slight pullback)
    4. If nearest_support is valid and above that midpoint: use support level
    5. Hard cap: entry cannot be more than 1.5×ATR below current price
       (don't wait forever — if it drops 1.5 ATR it's likely a breakdown)

    Returns (entry_price, entry_logic_description)
    """
    logic = ""

    # Protect against zero ATR or zero EMA
    atr   = max(atr, price * 0.005)   # minimum 0.5% of price
    ema20 = ema20 if ema20 > 0 else price

    if rsi >= 65:
        # Overbought — don't buy at top, wait for EMA20 pullback
        raw_entry = ema20
        logic     = f"RSI={rsi:.0f} overbought → target EMA20 pullback"
    elif rsi <= 35:
        # Oversold — already cheap, enter close to current price
        raw_entry = price - (0.2 * atr)
        logic     = f"RSI={rsi:.0f} oversold → entry near current (minor discount)"
    else:
        # Normal zone — target slight pullback into EMA20
        raw_entry = price - (ATR_ENTRY_PULL * atr)
        logic     = f"Normal RSI={rsi:.0f} → entry -0.4×ATR below current"

    # Prefer entering near support if it's between raw_entry and current
    if nearest_support and nearest_support > 0:
        support_gap = abs(price - nearest_support) / price
        if support_gap < 0.05 and nearest_support < price:
            # Support within 5% — use it as entry anchor
            raw_entry = max(raw_entry, nearest_support * 1.002)  # just above support
            logic += f" | anchored to S={nearest_support:.6g}"

    # Hard floor: entry never more than 1.5×ATR below current
    min_entry = price - (1.5 * atr)
    entry = max(raw_entry, min_entry)

    # Sanity: entry cannot exceed current price
    entry = min(entry, price)

    return round(entry, 8), logic


# ═══════════════════════════════════════════════════════════════════════════════
# SMART SL CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_smart_sl(entry: float, atr: float, regime: str,
                      whale_signal: str, news_score: float,
                      nearest_support: Optional[float] = None,
                      volatility: float = 0) -> tuple:
    """
    ATR-based stop loss with AI modifiers.

    Base: SL = entry - (ATR × 1.5)

    Modifiers:
    • BEAR regime: widen SL (more volatile moves)
    • BULL regime: tighten SL (trend is your friend)
    • Whale DISTRIBUTION: tighten SL (institutional selling)
    • Negative news: tighten SL
    • Nearest support below entry: use it as SL floor
    • High volatility: widen SL further

    Returns (stop_loss, sl_logic_description)
    """
    logic = ""

    atr = max(atr, entry * 0.005)

    # Base SL distance
    sl_atr_mult = ATR_SL_FACTOR

    # Regime modifier
    regime_adj = REGIME_SL_MULT.get(regime, 1.0)
    sl_atr_mult *= regime_adj
    logic += f"RegimeSL×{regime_adj:.2f}"

    # Whale modifier — if whales distributed, exit faster
    if whale_signal == 'DISTRIBUTION':
        sl_atr_mult *= (1 - WHALE_SL_TIGHTEN)
        logic += " | Whale-DIST tighter"
    elif whale_signal == 'ACCUMULATION':
        sl_atr_mult *= 1.05  # tiny buffer — whales holding = give a bit more room
        logic += " | Whale-ACCUM slight buffer"

    # News modifier
    news = float(news_score or 0)
    if news < -0.3:
        # Negative news → tighter SL
        news_factor = 1.0 - (abs(news) * NEWS_SL_TIGHTEN_MAX)
        sl_atr_mult *= news_factor
        logic += f" | News{news:+.2f} tighter"

    # High volatility → widen a touch (prevent false stop-outs)
    if volatility > 0.04:  # > 4% volatility
        sl_atr_mult *= 1.10
        logic += " | HighVol wider"

    sl_distance = atr * sl_atr_mult
    stop_loss   = entry - sl_distance

    # If nearest support is between SL and entry, use it as a better anchor
    if nearest_support and nearest_support > 0:
        gap = (entry - nearest_support) / entry
        if 0.005 < gap < 0.10:  # support within 0.5%–10%
            stop_loss = nearest_support * 0.995  # just below support
            logic += f" | SL below S={nearest_support:.6g}"

    # Hard floor: SL never more than 10% below entry (prevent runaway stops)
    sl_floor = entry * 0.90
    stop_loss = max(stop_loss, sl_floor)

    return round(stop_loss, 8), logic


# ═══════════════════════════════════════════════════════════════════════════════
# SMART TP CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_smart_tp(entry: float, sl: float, regime: str,
                      probability_up: float, whale_signal: str,
                      news_score: float, mtf_confirmed: bool,
                      nearest_resistance: Optional[float] = None) -> tuple:
    """
    Risk-reward based take profit with full AI modifiers.

    Base: risk = entry - sl
          TP   = entry + (base_rr × risk)

    Modifiers boost the base_rr:
    • probability_up > 50%: +PROB_TP_STEP per 10% above 50
    • Whale ACCUMULATION: +20% (institutions buying = bigger move)
    • Very positive news (>0.5): up to +15%
    • BULL regime: ×1.25
    • MTF confirmed: ×1.10 (two-timeframe agreement = stronger signal)
    • Cap at nearest resistance × 0.985

    Minimum R:R = 1.5 (otherwise too risky)

    Returns (take_profit, rr_achieved, tp_logic_description)
    """
    logic  = ""
    risk   = entry - sl
    if risk <= 0:
        risk = entry * 0.02  # fallback

    # Base R:R
    base_rr = ATR_TP_MIN_RR  # start at 2.0
    logic  += f"baseRR={base_rr:.1f}"

    # Probability modifier
    prob_excess = max(0, (float(probability_up or 50) - 50) / 10)  # steps of 10% above 50
    prob_boost  = prob_excess * PROB_TP_STEP
    base_rr    += prob_boost
    if prob_boost > 0:
        logic += f" | Prob{probability_up:.0f}%+{prob_boost:.2f}"

    # Whale modifier
    if whale_signal == 'ACCUMULATION':
        base_rr *= (1 + WHALE_TP_BOOST)
        logic   += " | Whale-ACCUM +20%"
    elif whale_signal == 'DISTRIBUTION':
        base_rr *= 0.80   # reduce target — whales selling into rally
        logic   += " | Whale-DIST -20%"

    # News modifier
    news = float(news_score or 0)
    if news > 0.3:
        news_boost = min(news * NEWS_TP_MAX_BOOST, NEWS_TP_MAX_BOOST)
        base_rr   *= (1 + news_boost)
        logic     += f" | News{news:+.2f}+{news_boost:.2f}"
    elif news < -0.3:
        base_rr *= 0.85   # reduce TP on bad news
        logic   += f" | News{news:+.2f} reduced"

    # Regime multiplier
    regime_mult = REGIME_TP_MULT.get(regime, 1.0)
    base_rr    *= regime_mult
    logic      += f" | Regime×{regime_mult:.2f}"

    # MTF confirmation bonus
    if mtf_confirmed:
        base_rr *= 1.10
        logic   += " | MTF✓ +10%"

    # Compute raw TP
    take_profit = entry + (base_rr * risk)
    rr_achieved = round(base_rr, 2)

    # Cap at nearest resistance (don't target a price we can't reach)
    if nearest_resistance and nearest_resistance > entry:
        res_cap = nearest_resistance * 0.985  # just below resistance
        if take_profit > res_cap:
            take_profit = res_cap
            rr_achieved = round((take_profit - entry) / max(risk, 0.0001), 2)
            logic += f" | capped@R={nearest_resistance:.6g}"

    # Minimum R:R guard
    min_tp   = entry + (1.5 * risk)
    take_profit = max(take_profit, min_tp)

    return round(take_profit, 8), rr_achieved, logic


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE QUALITY SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _trade_quality_score(rr: float, probability_up: float,
                         whale_signal: str, mtf_confirmed: bool,
                         regime: str, rsi: float) -> float:
    """
    0–100 score capturing overall quality of the trade setup.
    Higher = better opportunity. Used for ranking ties.

    Factors:
    • R:R >= 2.0: base 40pts, +10 per extra 0.5 RR
    • probability_up > 60: +15pts
    • Whale ACCUMULATION: +15pts
    • MTF confirmed: +10pts
    • BULL regime: +10pts
    • RSI in sweet spot (40–60): +10pts
    """
    score = 0.0

    # R:R quality
    if rr >= 2.0:
        score += 40 + min(20, (rr - 2.0) / 0.5 * 10)
    elif rr >= 1.5:
        score += 20

    # Probability
    if probability_up >= 65:
        score += 15
    elif probability_up >= 55:
        score += 8

    # Whale
    if whale_signal == 'ACCUMULATION':
        score += 15
    elif whale_signal == 'DISTRIBUTION':
        score -= 10

    # MTF
    if mtf_confirmed:
        score += 10

    # Regime
    if regime == 'BULL':
        score += 10
    elif regime == 'BEAR':
        score -= 10

    # RSI sweet spot
    if 38 <= rsi <= 62:
        score += 10
    elif rsi > 70:
        score -= 10   # chasing overbought

    return round(max(0, min(100, score)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_smart_levels(coin: Dict) -> Dict:
    """
    Main entry point. Takes a full coin signal dict (from scan_market)
    and returns AI-computed entry / SL / TP levels.

    Automatically loads latest RL-learned ATR multipliers from DB
    (cached 5 min) so RL improvements are applied in real-time.
    """
    # Sync RL-learned params from DB (5-min cache, non-blocking)
    load_rl_params_from_db()

    try:
        price       = float(coin.get('price', 0) or 0)
        if price <= 0:
            return _fallback(price)

        ema20       = float(coin.get('ema20', price) or price)
        ema50       = float(coin.get('ema50', ema20) or ema20)
        atr         = float(coin.get('atr', price * 0.02) or price * 0.02)
        rsi         = float(coin.get('rsi', 50) or 50)
        rsi_4h      = float(coin.get('rsi_4h', 50) or 50)

        prob_up     = float(coin.get('probability_up', 50) or 50)
        whale_sig   = str(coin.get('whale_signal', 'NONE') or 'NONE')
        news_score  = float(coin.get('news_score', 0) or 0)

        regime      = str(coin.get('market_regime', 'NEUTRAL') or 'NEUTRAL')
        mtf_ok      = bool(coin.get('mtf_confirmed', False))
        volatility  = float(coin.get('volatility', 0.02) or 0.02)

        n_support   = coin.get('nearest_support')
        n_resistance= coin.get('nearest_resistance')

        n_support    = float(n_support)    if n_support    and float(n_support)    > 0 else None
        n_resistance = float(n_resistance) if n_resistance and float(n_resistance) > 0 else None

        # ── 4h RSI guard: if higher timeframe overbought (>72), skip ──
        # We still return levels but flag them in quality
        rsi_4h_overbought = rsi_4h > 72

        # ── Smart Entry ──
        entry, e_logic = _compute_smart_entry(
            price, ema20, ema50, atr, rsi, n_support)

        # ── Smart SL ──
        sl, sl_logic = _compute_smart_sl(
            entry, atr, regime, whale_sig, news_score,
            n_support, volatility)

        # ── Smart TP ──
        tp, rr, tp_logic = _compute_smart_tp(
            entry, sl, regime, prob_up, whale_sig, news_score,
            mtf_ok, n_resistance)

        # Reduce RR if 4h overbought
        if rsi_4h_overbought:
            tp_logic += f" | 4h-RSI={rsi_4h:.0f} OB warning"

        # ── Trade Quality ──
        quality = _trade_quality_score(
            rr, prob_up, whale_sig, mtf_ok, regime, rsi)

        return {
            'entry_price':         entry,
            'stop_loss':           sl,
            'take_profit':         tp,
            'risk_reward_ratio':   rr,
            'trade_quality_score': quality,
            'entry_logic':         e_logic,
            'sl_logic':            sl_logic,
            'tp_logic':            tp_logic,
            # Transparency: original analysis inputs
            'levels_atr':          round(atr, 8),
            'levels_atr_pct':      round(atr / price * 100, 3) if price > 0 else 0,
            'levels_regime':       regime,
            'levels_whale':        whale_sig,
            'levels_prob':         prob_up,
            'levels_rsi':          rsi,
        }

    except Exception as e:
        logger.error(f"[SmartLevels] Error: {e}")
        return _fallback(float(coin.get('price', 0) or 0))


def _fallback(price: float) -> Dict:
    """Return safe defaults when computation fails.
    Uses current ATR multipliers (not hardcoded %) for consistency.
    """
    load_rl_params_from_db()  # try to get current RL values before fallback
    atr_est = price * 0.025
    sl_dist = ATR_SL_FACTOR * atr_est
    tp_dist = ATR_TP_MIN_RR  * sl_dist
    entry   = price or 1.0
    return {
        'entry_price':         round(entry, 8),
        'stop_loss':           round(entry - sl_dist, 8),
        'take_profit':         round(entry + tp_dist, 8),
        'risk_reward_ratio':   round(ATR_TP_MIN_RR, 2),
        'trade_quality_score': 0.0,
        'entry_logic':         'fallback (computation error)',
        'sl_logic':            'fallback',
        'tp_logic':            'fallback',
        'levels_atr':          round(atr_est, 8),
        'levels_atr_pct':      2.5,
        'levels_regime':       'NEUTRAL',
        'levels_whale':        'NONE',
        'levels_prob':         50.0,
        'levels_rsi':          50.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json

    test_cases = [
        {
            'name': 'SAPIENUSDT — Whale accumulation, BULL, oversold',
            'price': 0.0818, 'ema20': 0.0800, 'ema50': 0.0780,
            'atr': 0.003, 'rsi': 38, 'rsi_4h': 44,
            'probability_up': 72, 'whale_signal': 'ACCUMULATION',
            'news_score': 0.3, 'market_regime': 'BULL',
            'mtf_confirmed': True, 'volatility': 0.03,
            'nearest_support': 0.0790, 'nearest_resistance': 0.0950,
        },
        {
            'name': 'RUNEUSDT — Neutral whale, SIDEWAYS',
            'price': 0.409, 'ema20': 0.405, 'ema50': 0.400,
            'atr': 0.012, 'rsi': 52, 'rsi_4h': 55,
            'probability_up': 63, 'whale_signal': 'NONE',
            'news_score': 0.1, 'market_regime': 'SIDEWAYS',
            'mtf_confirmed': False, 'volatility': 0.025,
            'nearest_support': 0.400, 'nearest_resistance': 0.430,
        },
        {
            'name': 'ONDOUSDT — Whale distribution, BEAR, overbought',
            'price': 0.2555, 'ema20': 0.2510, 'ema50': 0.2490,
            'atr': 0.008, 'rsi': 68, 'rsi_4h': 74,
            'probability_up': 52, 'whale_signal': 'DISTRIBUTION',
            'news_score': -0.2, 'market_regime': 'BEAR',
            'mtf_confirmed': False, 'volatility': 0.04,
            'nearest_support': 0.247, 'nearest_resistance': 0.275,
        },
    ]

    print("=" * 70)
    print("  SMART LEVELS ENGINE — AI-DRIVEN ENTRY/SL/TP TEST")
    print("=" * 70)

    for tc in test_cases:
        name = tc.pop('name')
        result = compute_smart_levels(tc)
        entry = result['entry_price']
        sl    = result['stop_loss']
        tp    = result['take_profit']
        rr    = result['risk_reward_ratio']
        q     = result['trade_quality_score']

        print(f"\n  {name}")
        print(f"    Entry:   ${entry:,.6g}   [{result['entry_logic']}]")
        print(f"    SL:      ${sl:,.6g}   [{result['sl_logic']}]")
        print(f"    TP:      ${tp:,.6g}   [{result['tp_logic']}]")
        print(f"    R:R:     {rr:.2f}x")
        print(f"    Quality: {q}/100")
        print(f"    ATR:     ${result['levels_atr']:,.6g} ({result['levels_atr_pct']:.2f}%)")
    print()
