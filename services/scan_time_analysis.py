"""
services/scan_time_analysis.py — Best Trading Hours (IST) Analysis
===================================================================
MIN_DAYS_REQUIRED = 2  # At least 2 days of signal history to draw conclusions by IST hour,
measures win rate per hour (BUY signals that hit +5% within 24h).

Public API:
    get_best_scan_hours(db, lookback_days=30) -> Dict
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# Win = price went up by at least WIN_PCT within FORWARD_HOURS
WIN_PCT     = 0.04   # 4% gain = win
FORWARD_HRS = 24     # look forward 24h after signal


def get_best_scan_hours(db, lookback_days: int = 30) -> Dict:
    """
    Analyse historical BUY signals to find which IST hours produce most wins.

    Returns:
        hourly_win_rates: dict {hour_IST: {'win_rate': float, 'count': int}}
        best_hours: list of top 5 IST hours (int)
        worst_hours: list of bottom 3 IST hours (int)
        chart_data: list of {hour, win_rate, count} sorted by hour for bar chart
    """
    from config import settings

    try:
        since = datetime.utcnow() - timedelta(days=lookback_days)

        # Load BUY signals
        signals = list(db[settings.COLLECTION_AI_SIGNALS].find(
            {'timestamp': {'$gte': since},
             'final_signal': {'$in': ['BUY', 'STRONG BUY']}},
            {'_id': 0, 'symbol': 1, 'timestamp': 1,
             'final_score': 1, 'current_price': 1, 'close': 1}
        ).sort('timestamp', 1).limit(5000))

        if len(signals) < 20:
            logger.info("[ScanTime] Insufficient signals for analysis")
            return _empty_result()

        # Pre-load market data for forward price check
        symbols = list({s['symbol'] for s in signals})
        candle_map = {}
        try:
            cursor = db[settings.COLLECTION_MARKET_DATA].find(
                {'symbol': {'$in': symbols},
                 'open_time': {'$gte': since}},
                {'_id': 0, 'symbol': 1, 'open_time': 1, 'high': 1, 'close': 1}
            ).sort('open_time', 1)
            for c in cursor:
                sym = c['symbol']
                if sym not in candle_map:
                    candle_map[sym] = []
                candle_map[sym].append(c)
        except Exception as e:
            logger.warning(f"[ScanTime] Candle load: {e}")

        # Bucket by IST hour
        hour_data: Dict[int, Dict] = {h: {'wins': 0, 'total': 0} for h in range(24)}

        for sig in signals:
            ts  = sig.get('timestamp')
            sym = sig.get('symbol', '')
            ep  = sig.get('current_price') or sig.get('close')

            if not ts or not ep:
                continue

            # Convert to IST hour
            if ts.tzinfo is None:
                ts_utc = ts.replace(tzinfo=UTC)
            else:
                ts_utc = ts
            ts_ist = ts_utc.astimezone(IST)
            hour   = ts_ist.hour

            # Simulate forward: did price rise WIN_PCT within FORWARD_HRS?
            outcome = 'UNKNOWN'
            candles = candle_map.get(sym, [])
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            fwd = [c for c in candles
                   if isinstance(c.get('open_time'), datetime)
                   and c['open_time'].replace(tzinfo=None) > ts_naive
                   and c['open_time'].replace(tzinfo=None) <= ts_naive + timedelta(hours=FORWARD_HRS)]

            if fwd:
                target = float(ep) * (1 + WIN_PCT)
                for c in fwd:
                    if float(c.get('high', 0) or 0) >= target:
                        outcome = 'WIN'
                        break
                if outcome != 'WIN':
                    outcome = 'LOSS'

            if outcome in ('WIN', 'LOSS'):
                hour_data[hour]['total'] += 1
                if outcome == 'WIN':
                    hour_data[hour]['wins'] += 1

        # Compute win rates
        MIN_SIGNALS = 3
        hourly = {}
        chart  = []
        for h in range(24):
            d = hour_data[h]
            if d['total'] >= MIN_SIGNALS:
                wr = d['wins'] / d['total'] * 100
            else:
                wr = None   # not enough data
            hourly[h] = {'win_rate': round(wr, 1) if wr is not None else None,
                         'count': d['total']}
            chart.append({
                'hour_ist':  h,
                'label_ist': f"{h:02d}:00",
                'win_rate':  round(wr, 1) if wr is not None else 0,
                'count':     d['total'],
                'has_data':  wr is not None,
            })

        # Best / worst hours (only where we have data)
        valid = [(h, hourly[h]['win_rate']) for h in range(24)
                 if hourly[h]['win_rate'] is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        best_hours  = [h for h, _ in valid[:5]]
        worst_hours = [h for h, _ in valid[-3:]]

        logger.info(f"[ScanTime] Analysis done. Best IST hours: {best_hours}")
        return {
            'hourly_win_rates': hourly,
            'chart_data':       sorted(chart, key=lambda x: x['hour_ist']),
            'best_hours_ist':   best_hours,
            'worst_hours_ist':  worst_hours,
            'signals_analysed': len([s for s in signals]),
            'lookback_days':    lookback_days,
            'analysed_at':      datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[ScanTime] Error: {e}")
        return _empty_result()


def _empty_result() -> Dict:
    return {
        'hourly_win_rates': {},
        'chart_data': [],
        'best_hours_ist': [],
        'worst_hours_ist': [],
        'signals_analysed': 0,
    }
