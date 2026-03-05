"""
auto_optimizer.py — Automated Strategy Parameter Optimization
=============================================================
Runs a grid search over trading parameters, evaluates each config
using historical signals + candle data, and saves the best config
to the `strategy_configs` MongoDB collection.

Composite score formula:
    score = return_pct * 0.4 + win_rate * 0.3 + profit_factor * 0.2 - drawdown * 0.1

Grid:
    take_profit:     [7%, 8%, 9%, 10%, 12%]     → 5 values  (all > max SL)
    stop_loss:       [3%, 4%, 5%, 6%]           → 4 values
    min_score:       [45, 50, 55, 60]           → 4 values  (tighter — no weak signals)
    min_probability: [35, 40, 45, 50]           → 4 values
    Total: 5×4×4×4 = 320 configurations
    Note: configs where SL >= TP are automatically skipped (bad R:R)

Used by:
    scheduler.py → run_optimization_job() (weekly)
    run manually: python -m optimization.auto_optimizer
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pymongo

from config import settings
from optimization.strategy_config import (
    set_active_strategy_config,
    get_active_strategy_config,
    DEFAULT_CONFIG,
    COLL_STRATEGY_CONFIGS,
)

logger = logging.getLogger(__name__)

# ── Grid definition ───────────────────────────────────────────
PARAM_GRID = {
    'take_profit':     [0.07, 0.08, 0.09, 0.10, 0.12],  # All > max SL=6%
    'stop_loss':       [0.03, 0.04, 0.05, 0.06],
    'min_score':       [45, 50, 55, 60],                 # Raised lower bound from 35
    'min_probability': [35, 40, 45, 50],                 # Raised lower bound from 30
}

# Fixed simulation constants
FEE_RATE     = 0.00075    # 0.075% taker fee (Binance)
RISK_PER_TRADE = 0.05     # 5% per position
MAX_POSITIONS  = 5
SIGNAL_WINDOW  = 48 * 3600   # 48h — match backtester default

# Safety threshold: minimum composite improvement over current config
MIN_IMPROVEMENT_PCT = 0.0   # accept any improvement (or first run)

# ── Collection constants ───────────────────────────────────────
COLL_SIGNALS    = settings.COLLECTION_AI_SIGNALS
COLL_MARKET     = settings.COLLECTION_MARKET_DATA
SYNTH_COLL      = 'ai_signals_backtest_synth'


def run_optimization(lookback_days: int = 90, db=None,
                     force_apply: bool = False) -> Dict:
    """
    Main entry point: run grid search and optionally apply best config.

    Args:
        lookback_days: how far back to look for signals + candles
        db:            optional pymongo db (opens own if None)
        force_apply:   if True, bypass 24h safety gate

    Returns:
        dict with keys: best_config, tested_count, improvement_pct,
                        best_perf, all_results
    """
    own = (db is None)
    client = None

    try:
        if own:
            client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
            db = client[settings.DATABASE_NAME]

        logger.info("=" * 65)
        logger.info("[Optimizer] AUTO STRATEGY OPTIMIZATION STARTED")
        logger.info(f"[Optimizer] Lookback: {lookback_days} days | Force: {force_apply}")
        logger.info("=" * 65)

        # Load all data once
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info("[Optimizer] Loading candles + signals...")
        candles, signal_map = _load_data(db, start_date, end_date)

        if not candles:
            logger.warning("[Optimizer] No candle data — aborting")
            return _empty_result("No candle data found")

        total_signals = sum(len(v) for v in signal_map.values())
        logger.info(f"[Optimizer] Loaded {len(candles)} candles, {total_signals} signals")

        if total_signals < 20:
            logger.warning("[Optimizer] Not enough signals for reliable optimization")
            return _empty_result("Too few signals")

        # Build grid
        configs = _build_grid()
        logger.info(f"[Optimizer] Testing {len(configs)} configurations...")

        # Run grid
        results = []
        for i, cfg in enumerate(configs):
            perf = _simulate(cfg, candles, signal_map)
            comp = _composite_score(perf)
            results.append({'cfg': cfg, 'perf': perf, 'composite': comp})
            if (i + 1) % 80 == 0:
                logger.info(f"[Optimizer] Progress: {i+1}/{len(configs)} configs tested")

        # Sort by composite score
        results.sort(key=lambda x: x['composite'], reverse=True)

        # Filter to configs with at least some trades
        valid = [r for r in results if r['perf'].get('trades', 0) >= 5]
        if not valid:
            logger.warning("[Optimizer] No configs produced ≥5 trades")
            return _empty_result("No valid configs")

        best = valid[0]
        best_cfg  = best['cfg']
        best_perf = best['perf']
        best_comp = best['composite']

        logger.info(f"[Optimizer] Best config: {_fmt_cfg(best_cfg)}")
        logger.info(f"[Optimizer] Performance: trades={best_perf['trades']} "
                    f"return={best_perf['return_pct']:+.2f}% "
                    f"winrate={best_perf['win_rate']:.1f}% "
                    f"PF={best_perf['profit_factor']:.2f} "
                    f"DD={best_perf['max_drawdown']:.2f}%")
        logger.info(f"[Optimizer] Composite score: {best_comp:.4f}")

        # Compare with current config
        current_cfg  = get_active_strategy_config(db=db)
        current_perf = _simulate(current_cfg, candles, signal_map)
        current_comp = _composite_score(current_perf)
        improvement  = best_comp - current_comp

        logger.info(f"[Optimizer] Current config composite: {current_comp:.4f} | "
                    f"Improvement: {improvement:+.4f}")

        # Apply best config if improved (or no active config exists)
        applied = False
        current_active = db[COLL_STRATEGY_CONFIGS].find_one({'active': True})
        should_apply = (
            force_apply or
            current_active is None or
            improvement > MIN_IMPROVEMENT_PCT
        )

        if should_apply:
            params = {
                'take_profit':     best_cfg['take_profit'],
                'stop_loss':       best_cfg['stop_loss'],
                'min_score':       best_cfg['min_score'],
                'min_probability': best_cfg['min_probability'],
                'allow_hold_entry': False,           # BUY-only entries always
                'signal_window_hours': 48,
                'max_open_positions': 5,
                'fee_rate': FEE_RATE,
            }
            applied = set_active_strategy_config(
                params=params,
                performance=best_perf,
                config_id=f"opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                db=db,
            )
            if applied:
                logger.info("[Optimizer] ✅ New config applied to strategy_configs")
            else:
                logger.info("[Optimizer] ⚠️ Config not applied (safety gate or no improvement)")
        else:
            logger.info("[Optimizer] No improvement — keeping current config")

        # Log optimization audit record always
        _save_optimization_audit(db, best_cfg, best_perf, best_comp,
                                  len(configs), applied)

        return {
            'best_config':    best_cfg,
            'best_perf':      best_perf,
            'composite':      round(best_comp, 4),
            'tested_count':   len(configs),
            'valid_count':    len(valid),
            'current_composite': round(current_comp, 4),
            'improvement_pct':   round(improvement, 4),
            'applied':        applied,
            'all_results':    valid[:10],   # top 10
        }

    except Exception as e:
        logger.error(f"[Optimizer] Error: {e}", exc_info=True)
        return _empty_result(str(e))
    finally:
        if own and client:
            client.close()


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def _load_data(db, start_date: datetime, end_date: datetime) -> Tuple[List, Dict]:
    """Load candles and signals from DB into memory."""
    # Load candles
    all_candles = []
    symbols = db[COLL_MARKET].distinct('symbol')
    for sym in symbols:
        cands = list(db[COLL_MARKET].find(
            {'symbol': sym, 'open_time': {'$gte': start_date, '$lte': end_date}},
            {'_id': 0, 'symbol': 1, 'open_time': 1, 'close': 1, 'high': 1, 'low': 1}
        ).sort('open_time', 1))
        for c in cands:
            c['_sym'] = sym
        all_candles.extend(cands)
    all_candles.sort(key=lambda x: x.get('open_time', datetime.min))

    # Load signals (prefer synthetic, fall back to real)
    signal_map: Dict[str, List] = {}
    for coll_name in [SYNTH_COLL, COLL_SIGNALS]:
        try:
            sigs = list(db[coll_name].find(
                {'timestamp': {'$gte': start_date, '$lte': end_date}},
                {'_id': 0, 'symbol': 1, 'timestamp': 1,
                 'final_signal': 1, 'final_score': 1, 'probability_up': 1}
            ).sort('timestamp', 1))
            if sigs:
                for s in sigs:
                    sym = s.get('symbol', '')
                    if sym not in signal_map:
                        signal_map[sym] = []
                    signal_map[sym].append(s)
                logger.info(f"[Optimizer] Loaded {len(sigs)} signals from '{coll_name}'")
                break
        except Exception:
            pass

    return all_candles, signal_map


# ══════════════════════════════════════════════════════════════
# GRID BUILDING
# ══════════════════════════════════════════════════════════════

def _build_grid() -> List[Dict]:
    """Produce all parameter combinations."""
    grid = []
    for tp in PARAM_GRID['take_profit']:
        for sl in PARAM_GRID['stop_loss']:
            for ms in PARAM_GRID['min_score']:
                for mp in PARAM_GRID['min_probability']:
                    grid.append({
                        'take_profit':     tp,
                        'stop_loss':       sl,
                        'min_score':       ms,
                        'min_probability': mp,
                    })
    return grid


# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════

def _naive(dt: datetime) -> datetime:
    """Strip timezone info for arithmetic."""
    if dt is None:
        return datetime.min
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _simulate(cfg: Dict, all_candles: List, signal_map: Dict) -> Dict:
    """
    Run a in-memory simulation of a given config against all_candles.
    Returns performance metrics.
    """
    initial = 1000.0
    balance = initial
    open_pos: Dict = {}
    trades: List[Dict] = []
    equity_pts = [initial]
    peak = initial

    tp  = cfg.get('take_profit',     DEFAULT_CONFIG['take_profit'])
    sl  = cfg.get('stop_loss',       DEFAULT_CONFIG['stop_loss'])
    ms  = cfg.get('min_score',       DEFAULT_CONFIG['min_score'])
    mp  = cfg.get('min_probability', DEFAULT_CONFIG['min_probability'])
    fee = FEE_RATE

    for candle in all_candles:
        sym   = candle['_sym']
        price = float(candle.get('close', 0) or 0)
        high  = float(candle.get('high', price) or price)
        low   = float(candle.get('low', price) or price)
        ctime = _naive(candle.get('open_time', datetime.min))

        if price <= 0:
            continue

        sigs = signal_map.get(sym, [])

        # ── Manage open position ──
        if sym in open_pos:
            pos   = open_pos[sym]
            entry = pos['entry_price']
            tp_px = entry * (1 + tp)
            sl_px = entry * (1 - sl)
            exit_px = None
            reason  = None

            if low <= sl_px:
                exit_px = sl_px; reason = 'SL'
            elif high >= tp_px:
                exit_px = tp_px; reason = 'TP'
            else:
                for s in sigs:
                    td = (ctime - _naive(s.get('timestamp'))).total_seconds()
                    if 0 <= td <= SIGNAL_WINDOW and s.get('final_signal') == 'SELL':
                        exit_px = price; reason = 'SIG'; break

            if exit_px:
                size   = pos['size']
                pnl_pct = (exit_px - entry) / entry * 100
                profit  = size * (exit_px - entry) / entry - (size * 2 * fee)
                balance += size + profit
                trades.append({'pnl_pct': pnl_pct, 'profit': profit, 'reason': reason})
                del open_pos[sym]

        # ── Check entry ──
        elif len(open_pos) < MAX_POSITIONS:
            for s in sigs:
                td  = (ctime - _naive(s.get('timestamp'))).total_seconds()
                if 0 <= td <= SIGNAL_WINDOW:
                    sig  = s.get('final_signal', '')
                    sc   = float(s.get('final_score', 0) or 0)
                    prob = float(s.get('probability_up', 0) or 0)
                    if sig == 'BUY' and sc >= ms and prob >= mp:
                        size = balance * RISK_PER_TRADE
                        if size > 1.0 and balance >= size:
                            balance -= size
                            open_pos[sym] = {'entry_price': price, 'size': size}
                        break

        # Track equity
        eq = balance + sum(p['size'] for p in open_pos.values())
        equity_pts.append(eq)
        if eq > peak:
            peak = eq

    # Close remaining open positions at last known price
    for sym, pos in list(open_pos.items()):
        last_px = pos['entry_price']
        for c in reversed(all_candles):
            if c['_sym'] == sym and float(c.get('close', 0) or 0) > 0:
                last_px = float(c['close']); break
        pnl = (last_px - pos['entry_price']) / pos['entry_price'] * 100
        profit = pos['size'] * (last_px - pos['entry_price']) / pos['entry_price']
        balance += pos['size'] + profit
        trades.append({'pnl_pct': pnl, 'profit': profit, 'reason': 'END'})

    if not trades:
        return {'trades': 0, 'return_pct': 0, 'win_rate': 0,
                'profit_factor': 0, 'max_drawdown': 0,
                'tp_hits': 0, 'sl_hits': 0, 'expectancy': 0}

    pnls    = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers  = [p for p in pnls if p <= 0]
    gp = sum(t['profit'] for t in trades if t['profit'] > 0)
    gl = abs(sum(t['profit'] for t in trades if t['profit'] < 0))

    wr  = len(winners) / len(trades)
    avg_w = sum(winners) / len(winners) if winners else 0
    avg_l = abs(sum(losers) / len(losers)) if losers else 0
    exp   = avg_w * wr - avg_l * (1 - wr)

    max_dd = 0
    pk = equity_pts[0]
    for e in equity_pts:
        if e > pk: pk = e
        dd = (pk - e) / pk * 100 if pk > 0 else 0
        if dd > max_dd: max_dd = dd

    return {
        'trades':        len(trades),
        'return_pct':    round((balance - initial) / initial * 100, 2),
        'win_rate':      round(wr * 100, 1),
        'profit_factor': round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0),
        'max_drawdown':  round(max_dd, 2),
        'tp_hits':       sum(1 for t in trades if t['reason'] == 'TP'),
        'sl_hits':       sum(1 for t in trades if t['reason'] == 'SL'),
        'expectancy':    round(exp, 2),
    }


# ══════════════════════════════════════════════════════════════
# SCORING + HELPERS
# ══════════════════════════════════════════════════════════════

def _composite_score(perf: Dict) -> float:
    """
    Composite score formula:
        score = return * 0.4 + win_rate * 0.3 + profit_factor * 0.2 - drawdown * 0.1
    """
    if perf.get('trades', 0) == 0:
        return -9999.0
    return (
        perf.get('return_pct',    0) * 0.4 +
        perf.get('win_rate',      0) * 0.3 +
        perf.get('profit_factor', 0) * 0.2 -
        perf.get('max_drawdown',  0) * 0.1
    )


def _fmt_cfg(cfg: Dict) -> str:
    return (
        f"TP={cfg['take_profit']*100:.0f}% "
        f"SL={cfg['stop_loss']*100:.0f}% "
        f"Score≥{cfg['min_score']} "
        f"Prob≥{cfg['min_probability']}"
    )


def _empty_result(reason: str = '') -> Dict:
    return {
        'best_config': None,
        'best_perf': {},
        'composite': 0,
        'tested_count': 0,
        'valid_count': 0,
        'current_composite': 0,
        'improvement_pct': 0,
        'applied': False,
        'all_results': [],
        'error': reason,
    }


def _save_optimization_audit(db, best_cfg, best_perf, best_comp,
                              tested_count, applied):
    """Write an audit record to optimization_log collection."""
    try:
        db['optimization_log'].insert_one({
            'run_at':        datetime.utcnow(),
            'tested_configs': tested_count,
            'best_cfg':      best_cfg,
            'best_perf':     best_perf,
            'composite':     round(best_comp, 4),
            'applied':       applied,
        })
        db['optimization_log'].create_index(
            [('run_at', -1)], background=True
        )
    except Exception as e:
        logger.warning(f"[Optimizer] Audit save error: {e}")


# ══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print()
    print("=" * 65)
    print("  AUTO STRATEGY OPTIMIZER — Standalone Run")
    print("=" * 65)
    print()

    result = run_optimization(lookback_days=90, force_apply=True)

    print(f"\n{'='*65}")
    print("  OPTIMIZATION RESULTS")
    print(f"{'='*65}")
    print(f"  Configs tested:    {result['tested_count']}")
    print(f"  Configs with ≥5 trades: {result['valid_count']}")
    print(f"  Composite (best):  {result['composite']:.4f}")
    print(f"  Composite (old):   {result['current_composite']:.4f}")
    print(f"  Improvement:       {result['improvement_pct']:+.4f}")
    print(f"  Applied to DB:     {'YES ✅' if result['applied'] else 'NO (safety gate)'}")

    if result.get('best_config'):
        bc = result['best_config']
        bp = result['best_perf']
        print(f"\n  BEST CONFIGURATION")
        print(f"  Take Profit:  {bc['take_profit']*100:.0f}%")
        print(f"  Stop Loss:    {bc['stop_loss']*100:.0f}%")
        print(f"  Min Score:    {bc['min_score']}")
        print(f"  Min Prob:     {bc['min_probability']}%")
        print(f"\n  PERFORMANCE")
        print(f"  Trades:       {bp.get('trades',0)}")
        print(f"  Return:       {bp.get('return_pct',0):+.2f}%")
        print(f"  Win Rate:     {bp.get('win_rate',0):.1f}%")
        print(f"  Profit Factor:{bp.get('profit_factor',0):.2f}")
        print(f"  Max Drawdown: {bp.get('max_drawdown',0):.2f}%")
        print(f"  Expectancy:   {bp.get('expectancy',0):+.2f}%/trade")

    if result.get('error'):
        print(f"\n  ERROR: {result['error']}")

    print(f"\n{'='*65}")
    print("  Done.")
