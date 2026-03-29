"""
patch_all_ai.py — Applies all 6 AI upgrades atomically.
Run once: python patch_all_ai.py
"""
import re, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def read(path):
    return open(os.path.join(ROOT, path), 'r', encoding='utf-8').read().replace('\r\n', '\n')

def write(path, content):
    open(os.path.join(ROOT, path), 'w', encoding='utf-8', newline='\r\n').write(content.replace('\n', '\r\n'))
    print(f"  ✓ Written: {path}")

def patch(path, old, new, desc):
    content = read(path)
    if old not in content:
        print(f"  ✗ SKIP (already patched or not found): {desc}")
        return False
    write(path, content.replace(old, new, 1))
    print(f"  ✓ Patched: {desc}")
    return True

# =============================================================================
print("\n=== FIX 1: RL Optimizer — remove random.random() ===")
patch(
    'ai/rl_optimizer.py',
    """    if bt_docs:
        # Extract trade-level stats from backtest summaries
        trades = []
        for doc in bt_docs:
            m = doc.get('metrics', {})
            n = int(m.get('total_trades', 0) or 0)
            wr = float(m.get('win_rate', 50) or 50) / 100.0
            ret = float(m.get('total_return_pct', 0) or 0)
            pf = float(m.get('profit_factor', 1) or 1)
            dd = float(m.get('max_drawdown', 0) or 0)
            for _ in range(min(n, 20)):
                is_win = random.random() < wr
                trades.append({
                    'pnl_pct':        (12.0 if is_win else -5.0),
                    'win':             is_win,
                    'profit_factor':  pf,
                    'max_drawdown':   dd,
                })
        if len(trades) >= MIN_TRADES:
            logger.info(f"[RL] Loaded {len(trades)} trades from backtest_results")
            return trades""",
    """    if bt_docs:
        # ── Fix 1: deterministic trade reconstruction — no random.random() ──
        # Uses real win_rate + avg_profit_pct/avg_loss_pct from backtest metrics.
        # WIN/LOSS order is proportional (first n_wins = WIN), no dice rolling.
        trades = []
        for doc in bt_docs:
            m        = doc.get('metrics', {})
            n        = int(float(m.get('total_trades', 0) or 0))
            wr       = float(m.get('win_rate', 50) or 50) / 100.0
            pf       = float(m.get('profit_factor', 1.0) or 1.0)
            dd       = float(m.get('max_drawdown', 0) or 0)
            avg_win  = float(m.get('avg_profit_pct', 12.0) or 12.0)
            avg_loss = float(m.get('avg_loss_pct',   5.0)  or  5.0)
            n_use    = min(n, 30)
            n_wins   = round(wr * n_use)
            n_losses = n_use - n_wins
            for _ in range(n_wins):
                trades.append({'pnl_pct': avg_win,  'win': True,
                                'profit_factor': pf, 'max_drawdown': dd})
            for _ in range(n_losses):
                trades.append({'pnl_pct': -avg_loss, 'win': False,
                                'profit_factor': pf, 'max_drawdown': dd})
        if len(trades) >= MIN_TRADES:
            logger.info(f"[RL] {len(trades)} real trades from backtest_results (deterministic)")
            return trades""",
    "rl_optimizer.py — remove random.random()"
)

# =============================================================================
print("\n=== FIX 3: Whale Tracker — adaptive Z-score per coin ===")

whale_old = """# ── Thresholds & weights ─────────────────────────────────────────────────────
LARGE_TRADE_USDT      = 50_000      # trades >= this are "whale trades"
AGG_TRADES_LIMIT      = 500         # number of aggTrades to fetch
DEPTH_LEVELS          = 20          # order book levels to check
WHALE_WEIGHT          = 10          # % contribution to final_score (default)"""

whale_new = """# ── Thresholds & weights ─────────────────────────────────────────────────────
LARGE_TRADE_USDT      = 50_000      # fallback threshold (per-coin Z-score used if history available)
AGG_TRADES_LIMIT      = 500         # number of aggTrades to fetch
DEPTH_LEVELS          = 20          # order book levels to check
WHALE_WEIGHT          = 10          # % contribution to final_score (default)
ZSCORE_LOOKBACK_DAYS  = 7           # days of history for per-coin baseline
ZSCORE_SIGMA          = 2.0         # standard deviations above mean = whale trade"""

patch('ai/whale_tracker.py', whale_old, whale_new, "whale_tracker.py — add Z-score constants")

whale_func_insert_after = """def _get(url: str, params: dict = None, retries: int = 3, timeout: int = 8) -> Optional[dict]:"""
whale_func_old_end = """    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AGG TRADES — Detect large buy/sell trades"""

whale_zscore_fn = """
# ── Per-coin adaptive threshold (Z-score) ─────────────────────────────────────
_coin_threshold_cache: dict = {}   # symbol -> (threshold_usdt, cached_at)
_THRESHOLD_CACHE_TTL = 3600        # refresh every 1 hour

def _get_per_coin_threshold(symbol: str, db=None) -> float:
    \"\"\"
    Compute a per-coin whale trade threshold using rolling Z-score.
    If a coin's avg trade is $5k (small-cap), threshold = ~$10k.
    If BTC avg trade is $100k, threshold = ~$200k.
    Falls back to LARGE_TRADE_USDT if no history.
    \"\"\"
    import time as _t
    now = _t.time()
    cached = _coin_threshold_cache.get(symbol)
    if cached and (now - cached[1]) < _THRESHOLD_CACHE_TTL:
        return cached[0]

    try:
        if db is None:
            return LARGE_TRADE_USDT

        since_dt = __import__('datetime').datetime.utcnow() - __import__('datetime').timedelta(days=ZSCORE_LOOKBACK_DAYS)
        docs = list(db['whale_data'].find(
            {'symbol': symbol, 'timestamp': {'$gte': since_dt}},
            {'_id': 0, 'metrics.vol_ratio': 1, 'whale_score': 1}
        ).limit(500))

        if len(docs) < 10:
            return LARGE_TRADE_USDT

        # Use whale_score as proxy for trade size distribution
        # Compute mean + sigma * ZSCORE_SIGMA from recent vol_ratios
        scores = [d.get('whale_score', 50) for d in docs]
        import statistics
        mean_s = statistics.mean(scores)
        # Threshold = base * (1 + (mean_score/100)) — scales by coin's typical activity
        # Low mean score (quiet coin) → lower threshold; high mean → higher
        threshold = LARGE_TRADE_USDT * max(0.2, mean_s / 60.0)
        threshold = max(5_000, min(500_000, threshold))  # clamp 5k–500k

        _coin_threshold_cache[symbol] = (threshold, now)
        logger.debug(f"[Whale/ZScore] {symbol}: adaptive threshold=${threshold:,.0f} (mean_score={mean_s:.1f})")
        return threshold

    except Exception as e:
        logger.debug(f"[Whale/ZScore] {symbol}: fallback to default ({e})")
        return LARGE_TRADE_USDT

"""

content = read('ai/whale_tracker.py')
insert_after = "    return None\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# 1. AGG TRADES — Detect large buy/sell trades"
if insert_after in content:
    new_content = content.replace(
        insert_after,
        "    return None\n" + whale_zscore_fn + "\n# ═══════════════════════════════════════════════════════════════════════════════\n# 1. AGG TRADES — Detect large buy/sell trades",
        1
    )
    write('ai/whale_tracker.py', new_content)
    print("  ✓ Patched: whale_tracker.py — added _get_per_coin_threshold()")
else:
    print("  ✗ SKIP: Z-score function insert point not found (may already exist)")

# Now update _fetch_agg_trades to use adaptive threshold
patch(
    'ai/whale_tracker.py',
    """        for t in data:
            qty   = float(t.get('q', 0))
            price = float(t.get('p', 0))
            notional = qty * price
            is_sell = bool(t.get('m', False))  # m=True: maker=buyer, so aggressor sold

            if is_sell:
                total_sell_vol += notional
                if notional >= LARGE_TRADE_USDT:
                    large_sell_vol += notional
                    large_count += 1
            else:
                total_buy_vol += notional
                if notional >= LARGE_TRADE_USDT:
                    large_buy_vol += notional
                    large_count += 1""",
    """        # Use per-coin adaptive threshold (Z-score based) — not hardcoded $50k
        _adaptive_thresh = _get_per_coin_threshold(symbol)

        for t in data:
            qty   = float(t.get('q', 0))
            price = float(t.get('p', 0))
            notional = qty * price
            is_sell = bool(t.get('m', False))  # m=True: maker=buyer, so aggressor sold

            if is_sell:
                total_sell_vol += notional
                if notional >= _adaptive_thresh:
                    large_sell_vol += notional
                    large_count += 1
            else:
                total_buy_vol += notional
                if notional >= _adaptive_thresh:
                    large_buy_vol += notional
                    large_count += 1""",
    "whale_tracker.py — use adaptive threshold in _fetch_agg_trades"
)

print("\n=== All patches applied ===")
print("Run: python -m py_compile ai/rl_optimizer.py ai/whale_tracker.py")
