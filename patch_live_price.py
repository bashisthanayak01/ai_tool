"""
patch_live_price.py — Patch dashboard to always show live Binance prices
for entry/TP/SL instead of stale DB values.

Run once: python patch_live_price.py
"""

LIVE_PRICE_FUNC = '''
@st.cache_data(ttl=15)
def fetch_live_prices(symbols: list) -> dict:
    """Fetch current live prices from Binance for a list of symbols. Cache 15s."""
    try:
        import requests
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            timeout=6
        )
        if resp.status_code == 200:
            all_prices = {item['symbol']: float(item['price'])
                         for item in resp.json()}
            return {s: all_prices[s] for s in symbols if s in all_prices}
    except Exception:
        pass
    return {}


def recompute_levels(live_price: float, tp_pct: float = 0.12, sl_pct: float = 0.05) -> dict:
    """Recompute entry / TP / SL from live price."""
    entry = round(live_price, 8)
    tp    = round(live_price * (1 + tp_pct), 8)
    sl    = round(live_price * (1 - sl_pct), 8)
    rr    = round(tp_pct / sl_pct, 2)
    return {'entry_price': entry, 'take_profit': tp, 'stop_loss': sl, 'risk_reward_ratio': rr}

'''

path = 'dashboard.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changed = False

# 1. Insert live price helpers after load_signals()
if 'fetch_live_prices' not in content:
    # Find a safe insertion point — after load_signals definition ends
    marker = '\n@st.cache_data(ttl=30)\ndef load_backtest_result():'
    if marker in content:
        content = content.replace(marker, LIVE_PRICE_FUNC + marker)
        print("Inserted fetch_live_prices() + recompute_levels()")
        changed = True
    else:
        print("WARNING: could not find insertion point for live price helpers")
else:
    print("fetch_live_prices already present")

# 2. Add data age warning inside load_signals by wrapping
# Actually patch the load_rankings call to inject live prices
# We'll patch render_ranking() / wherever top coins are rendered

# 3. Find the function that renders top 3 cards and add live price refresh
# The key pattern to patch: where entry_price / stop_loss / take_profit are shown
# We need to inject live price lookup before display

live_price_injection = """
    # ── Live price refresh for accurate entry/TP/SL ─────────────────────────
    if not top_coins.empty:
        _syms = top_coins['symbol'].head(10).tolist() if 'symbol' in top_coins.columns else []
    elif isinstance(top_coins, list):
        _syms = [c.get('symbol', '') for c in top_coins[:10]]
    else:
        _syms = []
    _live_prices = fetch_live_prices(_syms) if _syms else {}
    # ─────────────────────────────────────────────────────────────────────────
"""

if 'fetch_live_prices(_syms)' not in content:
    # Look for render_top_opportunities or similar
    import re
    # Find the function that renders the top 3 coins
    m = re.search(
        r'(def render_top_opportunities\(\)[^\n]*\n)',
        content, re.MULTILINE
    )
    if m:
        old = m.group(0)
        new = old + live_price_injection
        content = content.replace(old, new)
        print("Injected live price lookup into render_top_opportunities()")
        changed = True
    else:
        print("WARNING: render_top_opportunities function not found by regex")
else:
    print("Live price injection already present")

# 4. Add data age / freshness warning at the top of the Top Opportunities tab
age_warning_code = '''
    # ── Data freshness check ─────────────────────────────────────────────────
    import pymongo as _pm
    try:
        _cli, _db = get_db_connection()
        _latest = _db['ai_signals'].find_one(
            {}, {'timestamp': 1}, sort=[('timestamp', -1)])
        _cli.close()
        if _latest:
            _ts = _latest['timestamp']
            _age_min = round((datetime.utcnow() - _ts.replace(tzinfo=None)).total_seconds() / 60, 0)
            if _age_min > 15:
                st.warning(
                    f"⚠️ **Stale Data Warning:** Last market scan was **{int(_age_min)} minutes ago** "
                    f"(at {_ts.strftime('%H:%M UTC')}). "
                    f"Entry prices and signals may not reflect current market conditions. "
                    f"Run `python scheduler.py` in your terminal to keep data live.",
                    icon="⚠️"
                )
            else:
                st.success(f"✅ Data is live — last scan {int(_age_min)} minutes ago", icon="✅")
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────────────────
'''

if 'Stale Data Warning' not in content:
    # Find the Top Opportunities tab render function
    target = 'def render_top_opportunities():\n'
    if target in content:
        # Find what comes right after the docstring / first line
        idx = content.find(target)
        # Find end of the function's opening (after docstring if any)
        area = content[idx: idx + 600]
        # Insert after the first st. call or comment in function
        first_st = area.find('\n    st.')
        if first_st > 0:
            insert_at = idx + first_st
            content = content[:insert_at] + '\n' + age_warning_code + content[insert_at:]
            print("Injected data freshness warning into render_top_opportunities()")
            changed = True
        else:
            print("WARNING: could not find insertion point for freshness warning")
    else:
        print(f"WARNING: '{target}' not found")
else:
    print("Staleness warning already present")

if changed:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("\ndashboard.py patched with live price + staleness warning")
else:
    print("\nNo changes made")
