"""
services/paper_trading.py — Automatic Paper Trading Tracker
============================================================
Auto-logs every conviction pick that appears on the board.
Tracks Entry → TP / SL hit automatically every scan.
No manual clicking required.

Collections:
    paper_trades  — open + closed simulated trades

Public API:
    log_new_conviction_picks(picks_doc, db)  — called after conviction job
    check_open_trades(db)                    — called after every scan
    get_paper_trade_summary(db)              — for dashboard display
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pymongo
from config import settings

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
COLL = 'paper_trades'

# ── Index setup ────────────────────────────────────────────────────────────────
_indexes_done = False

def _ensure_indexes(db) -> None:
    global _indexes_done
    if _indexes_done:
        return
    try:
        db[COLL].create_index([('symbol', 1), ('opened_at', -1)], background=True)
        db[COLL].create_index([('status', 1)], background=True)
        db[COLL].create_index(
            [('opened_at', 1)],
            expireAfterSeconds=90 * 24 * 3600,
            background=True, name='paper_trades_ttl'
        )
        _indexes_done = True
    except Exception as e:
        logger.debug(f"[PaperTrade] Index: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LOG NEW PICKS — called after conviction job runs
# ══════════════════════════════════════════════════════════════════════════════

def log_new_conviction_picks(picks_doc: Dict, db) -> int:
    """
    Auto-log new conviction picks as open paper trades.
    Only logs picks that don't already have an OPEN trade.
    Returns count of new trades opened.
    """
    if not picks_doc:
        return 0

    _ensure_indexes(db)
    opened = 0
    now_ist = datetime.now(IST)

    for trade_type in ['swing', 'position', 'trend']:
        for pick in picks_doc.get(trade_type, []):
            sym    = pick.get('symbol')
            entry  = pick.get('entry_price')
            tp     = pick.get('take_profit')
            sl     = pick.get('stop_loss')
            score  = pick.get('avg_rank_score', 0)
            rr     = pick.get('risk_reward_ratio')
            streak = pick.get('streak', 0)

            if not all([sym, entry, tp, sl]):
                continue
            if float(entry) <= 0:
                continue

            # Safety: recalculate R:R from entry/tp/sl if stored value is bad
            _ep = float(entry); _tp = float(tp); _sl = float(sl)
            if (rr is None or float(rr) <= 0) and _ep > _sl and _tp > _ep:
                rr = round((_tp - _ep) / (_ep - _sl), 2)

            # Skip if already have an OPEN trade for this symbol
            existing = db[COLL].find_one({'symbol': sym, 'status': 'OPEN'})
            if existing:
                continue

            # Calculate expected profit %
            tp_pct  = (float(tp) / float(entry) - 1) * 100
            sl_pct  = (1 - float(sl) / float(entry)) * 100

            doc = {
                'symbol':      sym,
                'trade_type':  trade_type.upper(),
                'status':      'OPEN',
                'entry_price': float(entry),
                'take_profit': float(tp),
                'stop_loss':   float(sl),
                'expected_tp_pct': round(tp_pct, 2),
                'expected_sl_pct': round(sl_pct, 2),
                'risk_reward': float(rr) if rr else None,
                'ai_score':    float(score),
                'streak':      int(streak),
                'opened_at':   datetime.utcnow(),
                'opened_ist':  now_ist.strftime('%Y-%m-%d %H:%M IST'),
                'closed_at':   None,
                'outcome':     None,
                'actual_pnl_pct': None,
                'exit_price':  None,
            }
            try:
                db[COLL].insert_one(doc)
                opened += 1
                logger.info(f"[PaperTrade] Opened: {sym} ({trade_type.upper()}) "
                            f"Entry={entry:.4g} TP={tp:.4g} SL={sl:.4g} "
                            f"(+{tp_pct:.1f}% / -{sl_pct:.1f}%)")
            except Exception as e:
                logger.error(f"[PaperTrade] Insert error {sym}: {e}")

    logger.info(f"[PaperTrade] {opened} new paper trades opened")
    return opened


# ══════════════════════════════════════════════════════════════════════════════
# CHECK OPEN TRADES — called after every scan
# ══════════════════════════════════════════════════════════════════════════════

def check_open_trades(db) -> int:
    """
    Check all OPEN paper trades against current live prices.
    Closes any trade that hit TP or SL.
    Returns number of trades closed.
    """
    open_trades = list(db[COLL].find({'status': 'OPEN'}, {'_id': 1,
        'symbol': 1, 'entry_price': 1, 'take_profit': 1, 'stop_loss': 1,
        'trade_type': 1, 'opened_at': 1}))

    if not open_trades:
        return 0

    # Fetch current prices from latest ai_signals
    symbols = [t['symbol'] for t in open_trades]
    latest_prices = {}
    try:
        for sym in symbols:
            doc = db[settings.COLLECTION_AI_SIGNALS].find_one(
                {'symbol': sym},
                {'_id': 0, 'current_price': 1, 'close': 1},
                sort=[('timestamp', -1)]
            )
            if doc:
                price = doc.get('current_price') or doc.get('close')
                if price:
                    latest_prices[sym] = float(price)
    except Exception as e:
        logger.warning(f"[PaperTrade] Price fetch error: {e}")
        return 0

    closed = 0
    now_utc = datetime.utcnow()
    now_ist = datetime.now(IST)

    for trade in open_trades:
        sym    = trade['symbol']
        price  = latest_prices.get(sym)
        if not price:
            continue

        entry = float(trade['entry_price'])
        tp    = float(trade['take_profit'])
        sl    = float(trade['stop_loss'])
        age_h = (now_utc - trade['opened_at']).total_seconds() / 3600

        outcome     = None
        exit_price  = None
        pnl_pct     = None

        if price >= tp:
            outcome    = 'WIN'
            exit_price = tp
            pnl_pct    = (tp / entry - 1) * 100
        elif price <= sl:
            outcome    = 'LOSS'
            exit_price = sl
            pnl_pct    = (sl / entry - 1) * 100
        elif age_h > 72:
            # Auto-close after 72h at current price (position expired)
            outcome    = 'EXPIRED'
            exit_price = price
            pnl_pct    = (price / entry - 1) * 100

        if outcome:
            try:
                db[COLL].update_one(
                    {'_id': trade['_id']},
                    {'$set': {
                        'status':       'CLOSED',
                        'outcome':      outcome,
                        'exit_price':   exit_price,
                        'actual_pnl_pct': round(pnl_pct, 2),
                        'closed_at':    now_utc,
                        'closed_ist':   now_ist.strftime('%Y-%m-%d %H:%M IST'),
                        'hold_hours':   round(age_h, 1),
                    }}
                )
                closed += 1
                emoji = '✅' if outcome == 'WIN' else '❌'
                logger.info(f"[PaperTrade] {emoji} Closed: {sym} → {outcome} "
                            f"{pnl_pct:+.1f}% (held {age_h:.0f}h)")
            except Exception as e:
                logger.error(f"[PaperTrade] Close error {sym}: {e}")

    if closed:
        logger.info(f"[PaperTrade] {closed} trades closed this check")
    return closed


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY — for dashboard
# ══════════════════════════════════════════════════════════════════════════════

def get_paper_trade_summary(db, days: int = 30) -> Dict:
    """Return paper trading P&L summary for dashboard display."""
    try:
        since = datetime.utcnow() - timedelta(days=days)
        all_trades = list(db[COLL].find(
            {'opened_at': {'$gte': since}},
            {'_id': 0, 'symbol': 1, 'trade_type': 1, 'status': 1,
             'outcome': 1, 'actual_pnl_pct': 1, 'expected_tp_pct': 1,
             'expected_sl_pct': 1, 'risk_reward': 1, 'ai_score': 1,
             'streak': 1, 'opened_ist': 1, 'closed_ist': 1, 'hold_hours': 1,
             'entry_price': 1, 'take_profit': 1, 'stop_loss': 1}
        ).sort('opened_at', -1))

        open_trades   = [t for t in all_trades if t['status'] == 'OPEN']
        closed_trades = [t for t in all_trades if t['status'] == 'CLOSED']
        wins   = [t for t in closed_trades if t.get('outcome') == 'WIN']
        losses = [t for t in closed_trades if t.get('outcome') == 'LOSS']

        win_rate = len(wins) / max(len(closed_trades), 1) * 100
        avg_win  = sum(t.get('actual_pnl_pct', 0) for t in wins)  / max(len(wins), 1)
        avg_loss = sum(t.get('actual_pnl_pct', 0) for t in losses) / max(len(losses), 1)
        total_pnl = sum(t.get('actual_pnl_pct', 0) for t in closed_trades)

        return {
            'open_trades':    open_trades,
            'closed_trades':  closed_trades[:20],  # last 20
            'total_trades':   len(all_trades),
            'open_count':     len(open_trades),
            'closed_count':   len(closed_trades),
            'win_count':      len(wins),
            'loss_count':     len(losses),
            'win_rate_pct':   round(win_rate, 1),
            'avg_win_pct':    round(avg_win, 2),
            'avg_loss_pct':   round(avg_loss, 2),
            'total_pnl_pct':  round(total_pnl, 2),
            'lookback_days':  days,
        }
    except Exception as e:
        logger.error(f"[PaperTrade] Summary error: {e}")
        return {}
