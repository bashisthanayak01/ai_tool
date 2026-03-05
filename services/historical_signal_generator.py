"""
services/historical_signal_generator.py
=========================================
Generates historical AI signals from stored market_data candles.

For each symbol, loads all candles sorted by open_time.
Uses a sliding window of 50+ candles to compute indicators, AI score,
probability, and risk model — then UPSERTs into ai_signals.

SAFETY RULES:
  • Never overwrites signals that already exist (UPSERT by symbol + timestamp)
  • Only generates signals for timestamps NOT already in ai_signals
  • Skips candles with insufficient history (< 50 prior candles)
  • Handles NaN/None safely throughout
  • Batch upserts for performance
"""

import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import pymongo

sys.path.insert(0, '.')

from config import settings
from services.indicator_engine import analyze_indicators
from services.profit_score import calculate_profit_score
from services.ai_score import calculate_final_score
from ai.probability_engine import calculate_probability
from risk_model import apply_risk_model

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_WINDOW    = 60          # minimum candles needed to compute indicators
BATCH_SIZE    = 200         # upsert batch size
DEFAULT_NEWS  = {'news_score': 0.0, 'sentiment': 'NEUTRAL',
                 'confidence': 0, 'headline_count': 0,
                 'top_headline': '', 'news_available': False}
DEFAULT_WHALE = {'whale_score': 50, 'whale_signal': 'NONE'}
DEFAULT_REGIME = {'regime': 'NEUTRAL', 'confidence': 50}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0) -> float:
    """Return float, replacing NaN/None with default."""
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _candles_to_klines(candles: List[Dict]) -> List[Dict]:
    """
    Convert MongoDB market_data docs to the klines format expected
    by analyze_indicators (requires keys: open, high, low, close, volume, symbol).
    """
    klines = []
    for c in candles:
        klines.append({
            'symbol': c.get('symbol', ''),
            'open':   _safe_float(c.get('open'),   0.0),
            'high':   _safe_float(c.get('high'),   0.0),
            'low':    _safe_float(c.get('low'),    0.0),
            'close':  _safe_float(c.get('close'),  0.0),
            'volume': _safe_float(c.get('volume'), 0.0),
        })
    return klines


def _get_existing_signal_timestamps(db, symbol: str) -> set:
    """Return set of existing signal timestamps for this symbol (date-only precision)."""
    col = db[settings.COLLECTION_AI_SIGNALS]
    docs = col.find({'symbol': symbol}, {'timestamp': 1, '_id': 0})
    dates = set()
    for d in docs:
        ts = d.get('timestamp')
        if ts:
            dates.add(ts.date())
    return dates


def _build_signal_doc(symbol: str, timestamp: datetime, indicators: Dict,
                      profit: Dict, final: Dict, prob: Dict,
                      news: Dict = None, regime: Dict = None) -> Dict:
    """Assemble a complete ai_signal document from component scores."""
    news  = news  or DEFAULT_NEWS
    regime = regime or DEFAULT_REGIME

    result = {
        # Identity
        'symbol':       symbol,
        'timestamp':    timestamp,
        'timeframe':    '1d',
        'source':       'historical_generator',
        'created_at':   datetime.utcnow(),

        # Price & indicators
        'price':          _safe_float(indicators.get('price')),
        'rsi':            _safe_float(indicators.get('rsi')),
        'ema20':          _safe_float(indicators.get('ema20')),
        'ema50':          _safe_float(indicators.get('ema50')),
        'macd':           _safe_float(indicators.get('macd')),
        'macd_signal':    _safe_float(indicators.get('macd_signal')),
        'macd_histogram': _safe_float(indicators.get('macd_histogram')),
        'volume':         _safe_float(indicators.get('volume')),
        'volume_spike':   _safe_float(indicators.get('volume_spike'), 1.0),
        'volatility':     _safe_float(indicators.get('volatility')),
        'breakout_score': _safe_float(indicators.get('breakout_score')),

        # Profit / signal
        'profit_score':  _safe_float(profit.get('profit_score')),
        'confidence':    _safe_float(profit.get('confidence')),
        'risk_level':    profit.get('risk_level', 'Medium'),
        'signal':        profit.get('signal', 'HOLD'),

        # Probability
        'probability_up':         _safe_float(prob.get('probability_up'), 50.0),
        'probability_down':       round(100.0 - _safe_float(prob.get('probability_up'), 50.0), 2),
        'probability_confidence': _safe_float(prob.get('probability_confidence'), 0.0),

        # Final AI score
        'final_score':     _safe_float(final.get('final_score')),
        'final_signal':    final.get('final_signal', 'HOLD'),
        'technical_score': _safe_float(final.get('technical_score')),
        'news_impact':     _safe_float(final.get('news_impact')),
        'news_weight':     _safe_float(final.get('news_weight')),
        'whale_contrib':   _safe_float(final.get('whale_contrib')),
        'prob_contrib':    _safe_float(final.get('prob_contrib')),

        # News (neutral for historical since we don't have past news)
        'news_score':    _safe_float(news.get('news_score')),
        'news_sentiment': news.get('sentiment', 'NEUTRAL'),
        'news_confidence': _safe_float(news.get('confidence')),
        'headline_count':  int(news.get('headline_count', 0)),
        'news_available':  bool(news.get('news_available', False)),

        # Whale (neutral for historical)
        'whale_score':  _safe_float(DEFAULT_WHALE['whale_score']),
        'whale_signal': DEFAULT_WHALE['whale_signal'],

        # Market regime
        'market_regime':     regime.get('regime', 'NEUTRAL'),
        'regime_confidence': _safe_float(regime.get('confidence'), 50.0),

        # Indicators snapshot for backtester
        'indicators': {
            'rsi':          _safe_float(indicators.get('rsi')),
            'ema20':        _safe_float(indicators.get('ema20')),
            'ema50':        _safe_float(indicators.get('ema50')),
            'volume_ratio': _safe_float(indicators.get('volume_spike'), 1.0),
        },
    }

    # Apply risk model in-place
    apply_risk_model(result)

    return result


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_historical_signals(
    db,
    days: int = 180,
    symbols: List[str] = None,
    overwrite_existing: bool = False,
) -> Dict:
    """
    Generate historical AI signals from market_data candles.

    Parameters
    ----------
    db                 : pymongo database handle
    days               : how many days back to generate signals for
    symbols            : list of symbols to process (None = all in market_data)
    overwrite_existing : if False (default), skips timestamps already in ai_signals

    Returns
    -------
    dict with: total_inserted, total_skipped, symbols_processed, errors
    """
    col_md  = db[settings.COLLECTION_MARKET_DATA]
    col_sig = db[settings.COLLECTION_AI_SIGNALS]

    # Ensure indexes exist
    try:
        col_sig.create_index(
            [('symbol', pymongo.ASCENDING), ('timestamp', pymongo.DESCENDING)],
            background=True, name='idx_sym_ts'
        )
    except Exception:
        pass

    # Cutoff date
    cutoff_dt = datetime.utcnow() - timedelta(days=days)

    # Get symbols to process
    if symbols is None:
        symbols = col_md.distinct('symbol')
    symbols = sorted(symbols)

    logger.info(f"[HistGen] Starting historical signal generation")
    logger.info(f"[HistGen] Symbols: {len(symbols)} | Lookback: {days}d | "
                f"Cutoff: {cutoff_dt.date()}")

    total_inserted  = 0
    total_skipped   = 0
    errors          = []

    for sym_idx, symbol in enumerate(symbols, 1):
        try:
            # Load ALL candles for this symbol (sorted ascending)
            all_candles = list(
                col_md.find({'symbol': symbol})
                       .sort('open_time', pymongo.ASCENDING)
            )

            if len(all_candles) < MIN_WINDOW:
                logger.warning(f"[HistGen] {symbol}: only {len(all_candles)} candles — skip")
                continue

            # Get existing signal dates for this symbol
            if not overwrite_existing:
                existing_dates = _get_existing_signal_timestamps(db, symbol)
            else:
                existing_dates = set()

            # Find the start index where we have MIN_WINDOW history
            ops = []
            inserted_sym  = 0
            skipped_sym   = 0

            for idx in range(MIN_WINDOW - 1, len(all_candles)):
                candle = all_candles[idx]
                ts = candle.get('open_time')

                # Skip if candle is outside our date range
                if ts and ts < cutoff_dt:
                    continue

                # Skip if signal already exists for this date
                if not overwrite_existing and ts and ts.date() in existing_dates:
                    skipped_sym += 1
                    continue

                # Use candles up to and including current index as sliding window
                window = all_candles[max(0, idx - MIN_WINDOW + 1): idx + 1]
                klines = _candles_to_klines(window)

                # Compute indicators
                indicators = analyze_indicators(klines)
                if indicators is None:
                    continue

                # Override symbol (analyze_indicators uses klines[0]['symbol'])
                indicators['symbol'] = symbol

                # Compute profit score
                profit = calculate_profit_score(indicators)

                # Probability (use neutral news/whale/regime for historical)
                prob = calculate_probability(
                    indicators, DEFAULT_NEWS, DEFAULT_WHALE, DEFAULT_REGIME
                )

                # Final AI score
                final = calculate_final_score(
                    profit, DEFAULT_NEWS, indicators, DEFAULT_WHALE, prob
                )

                # Build signal doc
                signal_ts = ts if ts else datetime.utcnow()
                doc = _build_signal_doc(
                    symbol=symbol,
                    timestamp=signal_ts,
                    indicators=indicators,
                    profit=profit,
                    final=final,
                    prob=prob,
                )

                # Prepare upsert operation
                ops.append(pymongo.UpdateOne(
                    {'symbol': symbol, 'timestamp': signal_ts},
                    {'$setOnInsert': doc},  # Only insert, never overwrite
                    upsert=True
                ))
                inserted_sym += 1

                # Flush batch
                if len(ops) >= BATCH_SIZE:
                    try:
                        col_sig.bulk_write(ops, ordered=False)
                    except Exception as be:
                        logger.warning(f"[HistGen] Batch write warning: {be}")
                    ops = []

            # Flush remaining
            if ops:
                try:
                    col_sig.bulk_write(ops, ordered=False)
                except Exception as be:
                    logger.warning(f"[HistGen] Final batch write warning: {be}")

            total_inserted += inserted_sym
            total_skipped  += skipped_sym

            if sym_idx % 10 == 0 or sym_idx == len(symbols):
                logger.info(
                    f"[HistGen] [{sym_idx}/{len(symbols)}] {symbol}: "
                    f"+{inserted_sym} inserted, {skipped_sym} skipped | "
                    f"Running total: {total_inserted}"
                )

        except Exception as e:
            logger.error(f"[HistGen] Error processing {symbol}: {e}")
            errors.append({'symbol': symbol, 'error': str(e)})

    result = {
        'total_inserted':  total_inserted,
        'total_skipped':   total_skipped,
        'symbols_processed': len(symbols),
        'errors':          errors,
    }
    logger.info(f"[HistGen] DONE → {total_inserted} inserted | {total_skipped} skipped | "
                f"{len(errors)} errors across {len(symbols)} symbols")
    return result


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 65)
    print("  HISTORICAL SIGNAL GENERATOR — STANDALONE RUN")
    print("=" * 65)

    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[settings.DATABASE_NAME]

    before = db[settings.COLLECTION_AI_SIGNALS].count_documents({})
    print(f"\n  Signals before: {before}")

    t0 = time.time()
    result = generate_historical_signals(db, days=180)
    elapsed = time.time() - t0

    after = db[settings.COLLECTION_AI_SIGNALS].count_documents({})

    print(f"\n  Signals after:      {after}")
    print(f"  Net inserted:       {result['total_inserted']}")
    print(f"  Skipped (existing): {result['total_skipped']}")
    print(f"  Symbols processed:  {result['symbols_processed']}")
    print(f"  Errors:             {len(result['errors'])}")
    print(f"  Time elapsed:       {elapsed:.1f}s")

    if result['errors']:
        print("\n  ERRORS:")
        for e in result['errors'][:5]:
            print(f"    {e}")

    client.close()
