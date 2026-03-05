"""
ai/portfolio_manager.py — Adaptive Position Sizing + Correlation Filter
========================================================================
Upgrade #2 + #3 (High Impact):

  2. Kelly Criterion position sizing:
     - Estimates optimal bet size based on historical win rate + RR ratio
     - Prevents over-betting during uncertain markets
     - Floor/ceiling: 1% to 10% of balance per trade

  3. Correlation filter:
     - Tracks which coins are currently in open positions
     - Blocks new entries on coins that move together (BTC+ETH+BNB family)
     - Prevents portfolio from moving as one block in a market crash

Public API:
    get_position_size(balance, score, confidence, win_rate, avg_win, avg_loss)
    is_correlated_entry_blocked(symbol, open_positions)
    get_portfolio_stats(open_positions)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# CORRELATION GROUPS
# ─────────────────────────────────────────────────────────────────
# Coins within the same group tend to move together.
# We limit to MAX 2 open positions from the same group at once.

CORRELATION_GROUPS = {
    'BTC_FAMILY': [
        'BTCUSDT', 'WBTCUSDT', 'BTCDOMUSDT',
    ],
    'ETH_FAMILY': [
        'ETHUSDT', 'STETHUSDT', 'WETHUSDT', 'ETHWUSDT',
    ],
    'LAYER1': [
        'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT',
        'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'SEIUSDT',
    ],
    'LARGE_CAP': [
        'BNBUSDT', 'XRPUSDT', 'TRXUSDT',
    ],
    'DEFI': [
        'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'CRVUSDT',
        'COMPUSDT', 'SNXUSDT', 'SUSHIUSDT', 'YFIUSDT',
    ],
    'AI_TOKENS': [
        'FETUSDT', 'AGIXUSDT', 'RNDRUSDT', 'WLDUSDT',
        'TAOUSDT', 'GRTUSDT',
    ],
    'LAYER2': [
        'MATICUSDT', 'ARBUSDT', 'OPUSDT', 'ZKUSDT',
        'SCROLLUSDT', 'IMXUSDT', 'STRKUSDT',
    ],
    'MEME': [
        'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT',
        'BONKUSDT', 'WIFUSDT',
    ],
}

# Max open positions per correlation group
MAX_PER_GROUP = 2

# Default sizing fallback
DEFAULT_RISK_PCT = 0.05  # 5%
MIN_RISK_PCT     = 0.01  # 1% minimum
MAX_RISK_PCT     = 0.10  # 10% maximum


# ─────────────────────────────────────────────────────────────────
# KELLY CRITERION POSITION SIZING
# ─────────────────────────────────────────────────────────────────

def get_position_size(
    balance: float,
    score: float = 65.0,
    confidence: float = 50.0,
    win_rate: float = 0.50,
    avg_win_pct: float = 6.0,
    avg_loss_pct: float = 3.0,
    half_kelly: bool = True,
) -> Dict:
    """
    Kelly Criterion: f* = (bp - q) / b
    where:
        b = avg_win / avg_loss (reward-to-risk ratio)
        p = win probability
        q = 1 - p

    Then scale by signal confidence to be more conservative.
    Uses Half-Kelly (f*/2) for safety — standard practice.

    Returns:
        position_size_usdt, risk_pct, kelly_fraction, sizing_reason
    """
    try:
        p = max(0.1, min(0.95, win_rate))   # probability of win
        q = 1.0 - p
        b = avg_win_pct / max(avg_loss_pct, 0.1)   # reward/risk

        # Kelly fraction
        kelly_f = (b * p - q) / b

        # Half-Kelly for safety
        if half_kelly:
            kelly_f = kelly_f / 2.0

        # Scale by signal quality (score + confidence)
        quality_factor = min(1.0, (score / 100.0) * 0.6 +
                                   (confidence / 100.0) * 0.4)

        adjusted = kelly_f * quality_factor

        # Apply floor/ceiling
        risk_pct = max(MIN_RISK_PCT, min(MAX_RISK_PCT, adjusted))

        # If Kelly is negative (edge is negative) → use minimum
        if kelly_f <= 0:
            risk_pct = MIN_RISK_PCT
            reason = f"Kelly negative ({kelly_f:.3f}) — WR={win_rate:.0%} insufficient edge → min bet {MIN_RISK_PCT*100:.0f}%"
        elif adjusted < MIN_RISK_PCT:
            risk_pct = MIN_RISK_PCT
            reason = f"Kelly too small ({adjusted:.3f}) → floored at {MIN_RISK_PCT*100:.0f}%"
        elif adjusted > MAX_RISK_PCT:
            risk_pct = MAX_RISK_PCT
            reason = f"Kelly capped at max {MAX_RISK_PCT*100:.0f}% (raw={adjusted:.3f})"
        else:
            reason = (
                f"Kelly={kelly_f:.3f} × quality={quality_factor:.2f} "
                f"→ {risk_pct*100:.1f}% (b={b:.2f}, p={p:.0%})"
            )

        position_size = balance * risk_pct

        return {
            'position_size_usdt': round(position_size, 2),
            'risk_pct':           round(risk_pct, 4),
            'kelly_fraction':     round(kelly_f, 4),
            'quality_factor':     round(quality_factor, 3),
            'sizing_reason':      reason,
            'win_rate_used':      round(p, 3),
            'rr_ratio_used':      round(b, 2),
        }

    except Exception as e:
        logger.debug(f"[Kelly] Sizing error: {e}")
        return {
            'position_size_usdt': balance * DEFAULT_RISK_PCT,
            'risk_pct':           DEFAULT_RISK_PCT,
            'kelly_fraction':     0.0,
            'quality_factor':     0.5,
            'sizing_reason':      f"Error fallback {DEFAULT_RISK_PCT*100:.0f}%",
            'win_rate_used':      win_rate,
            'rr_ratio_used':      avg_win_pct / max(avg_loss_pct, 0.1),
        }


# ─────────────────────────────────────────────────────────────────
# CORRELATION FILTER
# ─────────────────────────────────────────────────────────────────

def _get_group(symbol: str) -> Optional[str]:
    """Return the correlation group for a symbol, or None."""
    for group, members in CORRELATION_GROUPS.items():
        if symbol in members:
            return group
    return None


def is_correlated_entry_blocked(
    symbol: str,
    open_positions: Dict,
    max_per_group: int = MAX_PER_GROUP,
) -> Dict:
    """
    Check if too many correlated coins are already open.

    Args:
        symbol:         Coin to check (e.g. 'SOLUSDT')
        open_positions: Dict of currently open positions {symbol: data}
        max_per_group:  Max allowed positions from same group (default 2)

    Returns:
        blocked (bool), group (str), group_count (int), reason (str)
    """
    group = _get_group(symbol)
    if group is None:
        # Not in any known group — allow freely
        return {
            'blocked':     False,
            'group':       None,
            'group_count': 0,
            'reason':      'Symbol not in correlation group — allowed',
        }

    # Count how many in same group are already open
    group_members = CORRELATION_GROUPS[group]
    open_in_group = [s for s in open_positions if s in group_members]
    count = len(open_in_group)

    if count >= max_per_group:
        return {
            'blocked':     True,
            'group':       group,
            'group_count': count,
            'reason': (
                f"Correlation block: {count}/{max_per_group} {group} positions open "
                f"({', '.join(open_in_group)})"
            ),
        }

    return {
        'blocked':     False,
        'group':       group,
        'group_count': count,
        'reason':      f"{group}: {count}/{max_per_group} positions open — allowed",
    }


# ─────────────────────────────────────────────────────────────────
# PORTFOLIO STATS (for dashboard)
# ─────────────────────────────────────────────────────────────────

def get_portfolio_stats(open_positions: Dict, balance: float = 0.0) -> Dict:
    """
    Summary stats for current portfolio composition.
    Useful for dashboard display.
    """
    group_counts = {}
    for sym in open_positions:
        grp = _get_group(sym)
        label = grp or 'OTHER'
        group_counts[label] = group_counts.get(label, 0) + 1

    total_positions = len(open_positions)
    total_risk = sum(
        p.get('position_size', 0) for p in open_positions.values()
    )
    risk_pct = (total_risk / balance * 100) if balance > 0 else 0

    return {
        'total_positions':  total_positions,
        'total_risk_usdt':  round(total_risk, 2),
        'total_risk_pct':   round(risk_pct, 1),
        'group_breakdown':  group_counts,
        'diversified':      len(group_counts) >= 2,
    }


# ─────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  PORTFOLIO MANAGER TEST")
    print("=" * 55)

    # Kelly sizing test
    test_cases = [
        (1000, 75, 70, 0.60, 6.0, 3.0, "High confidence, good WR"),
        (1000, 65, 50, 0.45, 6.0, 3.0, "Medium confidence, borderline WR"),
        (1000, 82, 85, 0.70, 8.0, 3.0, "Very high confidence, excellent WR"),
        (1000, 50, 40, 0.30, 5.0, 4.0, "Low confidence, bad WR → min bet"),
    ]

    print("\n  Kelly Criterion Sizing:")
    for balance, score, conf, wr, aw, al, label in test_cases:
        r = get_position_size(balance, score, conf, wr, aw, al)
        print(
            f"    {label[:35]:35}: "
            f"size=${r['position_size_usdt']:.2f} "
            f"({r['risk_pct']*100:.1f}%) "
            f"kelly={r['kelly_fraction']:.3f}"
        )

    # Correlation filter test
    print("\n  Correlation Filter:")
    mock_open = {
        'SOLUSDT': {'position_size': 50},
        'AVAXUSDT': {'position_size': 50},
    }
    tests = ['NEARUSDT', 'BTCUSDT', 'ETHUSDT', 'APTUSDT', 'PEPEUSDT']
    for sym in tests:
        r = is_correlated_entry_blocked(sym, mock_open)
        status = "🚫 BLOCKED" if r['blocked'] else "✅ ALLOWED"
        print(f"    {sym:12}: {status} — {r['reason']}")
    print()
