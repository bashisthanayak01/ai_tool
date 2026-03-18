"""
Signal Persistence v1 — Consistency Tracking Across Scans
==========================================================
Solves the core problem: "Top 3 changes every scan — how do I know
which coins are GENUINELY good?" 

HOW IT WORKS
------------
Every scan (~every 15 min), the scheduler calls `save_scan_snapshot()`
with the top-20 ranked coins. This writes a small record per coin to the
`signal_persistence` MongoDB collection.

Then once per hour, `get_conviction_picks()` queries the last 2 hours of
records (= last 8 scans roughly) and returns only coins that appeared in
the top-10 in ≥5 of those 8 scans. These are the HIGH CONVICTION PICKS.

HIGH CONVICTION CRITERIA (per trade type)
-----------------------------------------
SWING    : top-10 in ≥5/8 scans + trade_type=SWING + RSI 45-70 + gain24h<8%
POSITION : top-10 in ≥5/8 scans + trade_type=POSITION + position_score>60
TREND    : top-10 in ≥5/8 scans + trade_type=TREND + daily_trend=UPTREND

DATA STORED PER SCAN (each record in signal_persistence)
---------------------------------------------------------
symbol, scan_time, rank_position, rank_score, trade_type,
price, rsi, price_change_24h_pct, near_support, position_score,
daily_trend, hourly_trend

AUTO-CLEANUP
------------
TTL index on scan_time: records auto-expire after 7 days.
No manual cleanup needed.

Usage
-----
    from services.signal_persistence import save_scan_snapshot, get_conviction_picks

    # After each scan (in scheduler.run_scan):
    save_scan_snapshot(ranked_top20, db=_rank_db)

    # Hourly (in scheduler.run_conviction_update):
    picks = get_conviction_picks(db=_rank_db)
    # picks = {'SWING': [...], 'POSITION': [...], 'TREND': [...]}
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── Collection name ────────────────────────────────────────────────────────────
COLLECTION = 'signal_persistence'
CONVICTION_COLLECTION = 'conviction_picks'

# ── Conviction thresholds ──────────────────────────────────────────────────────
LOOKBACK_HOURS       = 2      # How many hours of scan history to analyse
MIN_APPEARANCES      = 5      # Coin must appear in top-10 in ≥5 of last 8 scans
TOP_N_PER_SCAN       = 10     # Only track coins in top-10 (not top-20)
MAX_SAVES_PER_SCAN   = 20     # Save top-20 per scan (for flexibility)

# Per-type conviction filters
SWING_MAX_GAIN_24H   = 8.0    # % — coin mustn't already be up more than this
SWING_RSI_MIN        = 35.0
SWING_RSI_MAX        = 72.0
POSITION_SCORE_MIN   = 60.0   # Minimum position_score for POSITION picks
TREND_ALIGN_REQUIRED = 'UPTREND'  # daily_trend must be UPTREND for TREND picks


def save_scan_snapshot(ranked_coins: List[Dict], db=None) -> int:
    """
    Save top-20 ranked coins from the current scan to signal_persistence.
    Called by scheduler.run_scan() after every scan completes.

    Parameters
    ----------
    ranked_coins : list
        Output of rank_coins() — already sorted by rank_score descending.
        Only the first MAX_SAVES_PER_SCAN entries are saved.
    db : pymongo.database.Database, optional
        Existing DB connection. If None, opens its own.

    Returns
    -------
    int : number of records saved
    """
    _own_client = None
    try:
        if db is None:
            import pymongo
            _own_client = pymongo.MongoClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=8000
            )
            db = _own_client[settings.DATABASE_NAME]

        col = db[COLLECTION]
        scan_time = datetime.utcnow()
        records = []

        for i, coin in enumerate(ranked_coins[:MAX_SAVES_PER_SCAN], 1):
            records.append({
                'symbol':              coin.get('symbol', ''),
                'scan_time':           scan_time,
                'rank_position':       i,
                'rank_score':          float(coin.get('rank_score', 0) or 0),
                'trade_type':          coin.get('trade_type', 'SWING'),
                'price':               float(coin.get('price', 0) or 0),
                'rsi':                 float(coin.get('rsi', 50) or 50),
                'price_change_24h_pct':float(coin.get('price_change_24h_pct', 0) or 0),
                'near_support':        bool(coin.get('near_support', False)),
                'position_score':      float(coin.get('position_score', 0) or 0),
                'daily_trend':         coin.get('daily_trend', 'SIDEWAYS'),
                'hourly_trend':        coin.get('hourly_trend', 'SIDEWAYS'),
                'probability_up':      float(coin.get('probability_up', 50) or 50),
                'technical_score':     float(coin.get('technical_score', 0) or 0),
                # Store entry levels for conviction board display
                'entry_price':         coin.get('entry_price'),
                'stop_loss':           coin.get('stop_loss'),
                'take_profit':         coin.get('take_profit'),
                'risk_reward_ratio':   coin.get('risk_reward_ratio'),
            })

        if records:
            col.insert_many(records, ordered=False)
            logger.info(f"[Persistence] Saved {len(records)} records for scan at {scan_time}")
            return len(records)

        return 0

    except Exception as e:
        logger.error(f"[Persistence] save_scan_snapshot error: {e}")
        return 0
    finally:
        if _own_client:
            try:
                _own_client.close()
            except Exception:
                pass


def get_conviction_picks(db=None, lookback_hours: int = LOOKBACK_HOURS) -> Dict[str, List[Dict]]:
    """
    Analyse the last `lookback_hours` of scan snapshots and return coins
    that appeared consistently in the top-10 across multiple scans.

    This is the core function — its output drives the High Conviction Board
    on the dashboard.

    Parameters
    ----------
    db : pymongo.database.Database, optional
    lookback_hours : int
        How many hours of history to look back (default 2h = ~8 scans).

    Returns
    -------
    dict:
        {
            'SWING':    [list of conviction pick dicts, best first],
            'POSITION': [...],
            'TREND':    [...],
            'metadata': {
                'scans_analysed': 8,
                'lookback_hours': 2,
                'generated_at': datetime
            }
        }
    Each pick dict includes:
        symbol, avg_rank_score, appearances, streak,
        avg_rank_position, trade_type, + latest entry levels
    """
    _own_client = None
    try:
        if db is None:
            import pymongo
            _own_client = pymongo.MongoClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=8000
            )
            db = _own_client[settings.DATABASE_NAME]

        col = db[COLLECTION]
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Fetch all records in the lookback window
        records = list(col.find(
            {'scan_time': {'$gte': cutoff}},
            {'_id': 0}
        ).sort('scan_time', 1))

        if not records:
            logger.info("[Persistence] No scan records found in lookback window")
            return _empty_picks()

        # Count distinct scan_times (= number of scans analysed)
        scan_times = sorted(set(r['scan_time'] for r in records))
        n_scans    = len(scan_times)

        # Group records by symbol
        by_symbol: Dict[str, List[Dict]] = {}
        for r in records:
            sym = r.get('symbol', '')
            if not sym:
                continue
            by_symbol.setdefault(sym, []).append(r)

        # Count appearances in top-10 per scan
        picks = {'SWING': [], 'POSITION': [], 'TREND': []}

        for sym, sym_records in by_symbol.items():
            # Only count records where the coin was in the top-10
            top10_records = [r for r in sym_records if r.get('rank_position', 99) <= TOP_N_PER_SCAN]
            appearances   = len(top10_records)

            if appearances < MIN_APPEARANCES:
                continue  # Not consistent enough

            # Compute streak (consecutive recent scans in top-10)
            streak = _compute_streak(sym, scan_times, top10_records)

            # Average score and position across appearances
            avg_score = round(sum(r['rank_score'] for r in top10_records) / appearances, 2)
            avg_pos   = round(sum(r['rank_position'] for r in top10_records) / appearances, 1)

            # Use the most recent record for current values
            latest = sorted(top10_records, key=lambda r: r['scan_time'])[-1]
            trade_type   = latest.get('trade_type', 'SWING')
            daily_trend  = latest.get('daily_trend', 'SIDEWAYS')
            pos_score    = float(latest.get('position_score', 0) or 0)
            gain_24h     = float(latest.get('price_change_24h_pct', 0) or 0)
            rsi          = float(latest.get('rsi', 50) or 50)

            pick = {
                'symbol':           sym,
                'avg_rank_score':   avg_score,
                'appearances':      appearances,
                'streak':           streak,
                'n_scans':          n_scans,
                'consistency_pct':  round(appearances / n_scans * 100, 0) if n_scans > 0 else 0,
                'avg_rank_position':avg_pos,
                'trade_type':       trade_type,
                'daily_trend':      daily_trend,
                'hourly_trend':     latest.get('hourly_trend', 'SIDEWAYS'),
                'position_score':   pos_score,
                'price':            latest.get('price'),
                'rsi':              rsi,
                'price_change_24h_pct': gain_24h,
                'near_support':     latest.get('near_support', False),
                'probability_up':   latest.get('probability_up', 50),
                'technical_score':  latest.get('technical_score', 0),
                # Entry levels from most recent scan
                'entry_price':      latest.get('entry_price'),
                'stop_loss':        latest.get('stop_loss'),
                'take_profit':      latest.get('take_profit'),
                'risk_reward_ratio':latest.get('risk_reward_ratio'),
            }

            # ── Assign to EXACTLY ONE trade type bucket (highest priority wins) ─
            # Priority: TREND (strictest, longest hold) > POSITION > SWING (default)
            # This ensures each coin appears in only ONE column on the dashboard.

            if (daily_trend == TREND_ALIGN_REQUIRED and
                    pos_score >= POSITION_SCORE_MIN and
                    avg_pos <= 8):
                # TREND: daily uptrend + 1h momentum + consistently high ranked
                picks['TREND'].append(pick)

            elif (pos_score >= POSITION_SCORE_MIN and
                      latest.get('hourly_trend') == 'UPTREND'):
                # POSITION: strong 1h uptrend + good position score (but no daily UPTREND)
                picks['POSITION'].append(pick)

            elif (SWING_RSI_MIN <= rsi <= SWING_RSI_MAX and
                      gain_24h <= SWING_MAX_GAIN_24H):
                # SWING: RSI in good range, not already pumped — default short-term pick
                picks['SWING'].append(pick)

        # Sort each bucket by avg_rank_score descending, keep top 5
        for tt in picks:
            picks[tt].sort(key=lambda x: x['avg_rank_score'], reverse=True)
            picks[tt] = picks[tt][:5]

        picks['metadata'] = {
            'scans_analysed': n_scans,
            'lookback_hours': lookback_hours,
            'generated_at':   datetime.utcnow(),
            'total_symbols_tracked': len(by_symbol),
        }

        logger.info(
            f"[Persistence] Conviction picks: "
            f"SWING={len(picks['SWING'])} "
            f"POSITION={len(picks['POSITION'])} "
            f"TREND={len(picks['TREND'])} "
            f"from {n_scans} scans in last {lookback_hours}h"
        )
        return picks

    except Exception as e:
        logger.error(f"[Persistence] get_conviction_picks error: {e}")
        return _empty_picks()
    finally:
        if _own_client:
            try:
                _own_client.close()
            except Exception:
                pass


def save_conviction_picks(picks: Dict, db=None) -> bool:
    """
    Save conviction picks to the `conviction_picks` collection so the
    dashboard can read them without re-computing.
    Called hourly by scheduler.run_conviction_update().
    """
    _own_client = None
    try:
        if db is None:
            import pymongo
            _own_client = pymongo.MongoClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=8000
            )
            db = _own_client[settings.DATABASE_NAME]

        col = db[CONVICTION_COLLECTION]
        doc = {
            'generated_at': datetime.utcnow(),
            'swing':     picks.get('SWING', []),
            'position':  picks.get('POSITION', []),
            'trend':     picks.get('TREND', []),
            'metadata':  picks.get('metadata', {}),
        }
        col.replace_one({}, doc, upsert=True)
        logger.info("[Persistence] Conviction picks saved to DB")
        return True

    except Exception as e:
        logger.error(f"[Persistence] save_conviction_picks error: {e}")
        return False
    finally:
        if _own_client:
            try:
                _own_client.close()
            except Exception:
                pass


def get_saved_conviction_picks(db=None) -> Optional[Dict]:
    """
    Read the most recently saved conviction picks from the DB.
    Used by the dashboard (fast read, no computation).

    Returns None if no picks have been saved yet (first hour of running).
    """
    _own_client = None
    try:
        if db is None:
            import pymongo
            _own_client = pymongo.MongoClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=8000
            )
            db = _own_client[settings.DATABASE_NAME]

        doc = db[CONVICTION_COLLECTION].find_one({}, {'_id': 0})
        return doc

    except Exception as e:
        logger.debug(f"[Persistence] get_saved_conviction_picks: {e}")
        return None
    finally:
        if _own_client:
            try:
                _own_client.close()
            except Exception:
                pass


def setup_persistence_indexes(db) -> bool:
    """
    Create required indexes on signal_persistence and conviction_picks.
    Called once at scheduler startup.

    Indexes:
    - signal_persistence: TTL on scan_time (auto-expire 7 days)
    - signal_persistence: compound (symbol, scan_time) for fast lookups
    - conviction_picks: no special index needed (single-doc collection)
    """
    try:
        from pymongo import ASCENDING, DESCENDING

        col = db[COLLECTION]

        # TTL index: auto-delete records older than 7 days
        try:
            col.create_index(
                [('scan_time', ASCENDING)],
                expireAfterSeconds=7 * 86400,
                name='idx_persistence_ttl'
            )
        except Exception:
            pass  # Already exists

        # Compound index for fast lookups during get_conviction_picks()
        try:
            col.create_index(
                [('symbol', ASCENDING), ('scan_time', DESCENDING)],
                name='idx_persistence_symbol_time'
            )
        except Exception:
            pass

        logger.info("[Persistence] Indexes ready for signal_persistence collection")
        return True

    except Exception as e:
        logger.warning(f"[Persistence] Index setup warning: {e}")
        return False


# ── Private helpers ────────────────────────────────────────────────────────────

def _compute_streak(symbol: str, all_scan_times: list, top10_records: List[Dict]) -> int:
    """
    Compute how many CONSECUTIVE recent scans the coin appeared in top-10.
    E.g. if the last 4 scans all have the coin, streak=4.
    """
    try:
        top10_times = set(r['scan_time'] for r in top10_records)
        streak = 0
        for st in reversed(all_scan_times):
            if st in top10_times:
                streak += 1
            else:
                break
        return streak
    except Exception:
        return 0


def _empty_picks() -> Dict:
    """Return empty conviction picks structure."""
    return {
        'SWING': [], 'POSITION': [], 'TREND': [],
        'metadata': {
            'scans_analysed': 0,
            'lookback_hours': LOOKBACK_HOURS,
            'generated_at': datetime.utcnow(),
        }
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    picks = get_conviction_picks()
    for tt in ['SWING', 'POSITION', 'TREND']:
        print(f"\n=== {tt} Conviction Picks ===")
        for p in picks.get(tt, []):
            print(
                f"  {p['symbol']:12s} score={p['avg_rank_score']:.1f} "
                f"streak={p['streak']} appear={p['appearances']}/{p['n_scans']} "
                f"rsi={p['rsi']:.0f} gain24h={p['price_change_24h_pct']:+.1f}%"
            )
    meta = picks.get('metadata', {})
    print(f"\nAnalysed {meta.get('scans_analysed',0)} scans in last {meta.get('lookback_hours',2)}h")
