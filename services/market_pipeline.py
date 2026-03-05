"""
Market Pipeline v2 — all engines + hourly signal history storage + DB-backed news cache
"""

import logging
import time as _time
from datetime import datetime, timedelta
from typing import List, Dict

from services.binance_scanner import get_top_symbols, get_klines, get_klines_since
from services.indicator_engine import analyze_indicators
from services.profit_score import calculate_profit_score
from services.ai_score import calculate_final_score
from ai.whale_tracker import detect_whale_activity          # upgraded v2
from ai.multi_timeframe import get_mtf_confirmation         # NEW: 4h alignment
from news.news_collector import get_news_sentiment
from ai.market_regime import detect_market_regime
from ai.probability_engine import calculate_probability
from ai.ranking_engine import rank_coins
from risk_model import apply_risk_model
from optimization.strategy_config import get_regime_adjusted_config
from ai.rl_optimizer import get_current_rl_params
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def should_store_signals() -> bool:
    """
    Save signals on every scan (every 5 min) so the dashboard always shows
    the latest live price. Uses upsert — no duplicates, no DB bloat.
    """
    return True


def collect_historical_data(symbols: List[str]) -> Dict:
    """
    Incremental historical data collection.
    First run: HISTORICAL_DAYS of 1d candles. Later: only new candles.
    """
    from database.mongo_client import mongo_client

    stats = {'inserted': 0, 'updated': 0, 'skipped': 0, 'processed': 0}

    if not mongo_client.connect():
        return stats

    total = mongo_client.get_market_data_count()
    if total == 0:
        logger.info(f"INITIAL LOAD: Fetching {settings.HISTORICAL_DAYS} days...")
    else:
        logger.info(f"INCREMENTAL: {total} candles in DB, new only...")

    for i, sym in enumerate(symbols, 1):
        try:
            last = mongo_client.get_last_candle_time(sym)
            if last is None:
                start = datetime.utcnow() - timedelta(days=settings.HISTORICAL_DAYS)
                logger.info(f"[{i}/{len(symbols)}] {sym}: Full {settings.HISTORICAL_DAYS}d")
            else:
                start = last
                logger.info(f"[{i}/{len(symbols)}] {sym}: From {last.strftime('%Y-%m-%d')}")

            klines = get_klines_since(sym, start, interval='1d')
            if klines:
                r = mongo_client.upsert_market_data(klines)
                stats['inserted'] += r['inserted']
                stats['updated'] += r['updated']
                stats['skipped'] += r['skipped']
                stats['processed'] += 1
            _time.sleep(0.1)
        except Exception as e:
            logger.error(f"  Error {sym}: {e}")

    logger.info(f"Historical: +{stats['inserted']} new, ~{stats['updated']} updated")
    mongo_client.close()
    return stats


def scan_market(limit: int = 90) -> List[Dict]:
    """
    Full market scan with all engines integrated:
    1. Detect market regime (BTC)
    2. For each symbol:
       a. Fetch 15m klines
       b. Technical indicators
       c. Profit score + confidence
       d. News sentiment (3-tier cache: memory → DB → live)
       e. Whale detection
       f. Probability engine
       g. Final AI score: technical(70%) + news(30%)
       h. Risk-Adjusted Score (NEW): volatility_penalty, drawdown_penalty,
          liquidity_bonus, rr_bonus, probability_weight, news_weight
    3. Rank all coins → Top 3
    """
    # Use a DEDICATED local connection — never the shared mongo_client singleton.
    # The shared singleton is used concurrently by whale_tracker and news_collector
    # (APScheduler jobs). Closing it here would cause 'MongoClient after close'.
    import pymongo as _pymongo
    _scan_client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
    db = _scan_client[settings.DATABASE_NAME]

    logger.info(f"{'='*60}")
    logger.info(f"FULL MARKET SCAN v2 — {limit} symbols")
    logger.info(f"{'='*60}")

    # Step 1: Market regime (cached, fast)
    regime = detect_market_regime()
    logger.info(f"Market Regime: {regime['regime']} ({regime['confidence']}%)")

    # Load regime-adjusted strategy config from DB (min_score, min_prob, score multiplier)
    strategy_cfg = get_regime_adjusted_config(regime_data=regime)
    logger.info(
        f"[Config] TP={strategy_cfg.get('take_profit',0)*100:.0f}% "
        f"SL={strategy_cfg.get('stop_loss',0)*100:.0f}% "
        f"MinScore={strategy_cfg.get('min_score',30)} "
        f"Regime={strategy_cfg.get('applied_regime','?')} "
        f"Mult={strategy_cfg.get('score_multiplier',1.0):.2f}"
    )

    # Load RL params once per scan (cached 5 min — very fast)
    rl_params = get_current_rl_params(db=db)
    rl_weight  = float(rl_params.get('rl_weight_adjustment', 1.0))
    logger.info(f"[RL] Weight adjustment: {rl_weight:.4f} | "
                f"episode={rl_params.get('episode', 0)} | "
                f"entry_thr={rl_params.get('entry_threshold', 45):.1f}")

    # Step 2: Get symbols
    symbols = get_top_symbols(limit)
    if not symbols:
        logger.error("No symbols fetched")
        _scan_client.close()
        return []

    results = []
    failed = 0
    scan_time = datetime.utcnow()
    mtf_cache: dict = {}   # shared 4h cache for multi-timeframe (1h TTL per symbol)

    for i, sym in enumerate(symbols, 1):
        try:
            # 2a: Klines (15m)
            klines = get_klines(sym, '15m', 200)
            if not klines:
                failed += 1
                continue

            # 2b: Indicators
            indicators = analyze_indicators(klines)
            if not indicators:
                failed += 1
                continue

            # 2c: Profit score
            profit = calculate_profit_score(indicators)

            # 2d: News — 3-tier cache (memory → DB 30-min → live API)
            news = get_news_sentiment(sym, db=db)

            # 2e: Whale detection — klines-only during main scan (FAST, no extra API calls)
            # Full 5-source whale analysis runs every 10 min via the dedicated whale_scan job.
            # This prevents scan-time bloat (4 APIs x 90 coins = 360 requests/scan = crashes).
            whale = detect_whale_activity(klines)

            # 2e2: Multi-Timeframe Confirmation (15m + 4h)
            # Uses shared cache to avoid fetching 4h for every coin independently
            mtf = get_mtf_confirmation(sym, indicators, cache=mtf_cache)

            # 2e3: News Timing Filter
            # Skip if news shows extreme fear or euphoria — high uncertainty periods
            # We still compute news for score, but flag it for the backtester
            news_extreme = False
            news_score_raw = float(news.get('news_score', 0) or 0)
            news_conf_raw  = float(news.get('confidence', 0) or 0)
            if abs(news_score_raw) > 0.75 and news_conf_raw > 70:
                # Extreme news = high volatility risk, flag it
                news_extreme = True
                logger.debug(f"[NewsFilter] {sym}: extreme news score={news_score_raw:.2f} — flagging")

            # 2f: Probability engine
            prob = calculate_probability(indicators, news, whale, regime)

            # 2g: Final AI score (tech*0.7 + news*0.3) with regime multiplier
            final = calculate_final_score(profit, news, indicators, whale, prob, regime)

            # 2g-RL: Apply RL weight adjustment on top of regime-adjusted score
            # This is a transparent multiplier — original score is preserved in score_before_rl
            score_before_rl = final['final_score']
            rl_adjusted     = round(min(100.0, max(0.0, score_before_rl * rl_weight)), 2)

            result = {
                'symbol': sym,
                'timeframe': settings.SIGNAL_TIMEFRAME,
                'timestamp': scan_time,
                'price': indicators['price'],
                'rsi': indicators['rsi'],
                'ema20': indicators['ema20'],
                'ema50': indicators['ema50'],
                'macd': indicators['macd'],
                'volume': indicators['volume'],
                'volume_spike': indicators['volume_spike'],
                'volatility': indicators['volatility'],
                'breakout_score': indicators['breakout_score'],
                # Profit Score
                'profit_score': profit['profit_score'],
                'confidence': profit['confidence'],
                'risk_level': profit['risk_level'],
                'signal': profit['signal'],
                # News
                'news_score': news['news_score'],
                'news_sentiment': news['sentiment'],
                'news_confidence': news.get('confidence', 0),
                'headline_count': news.get('headline_count', 0),
                'top_headline': news.get('top_headline', ''),
                'news_available': news.get('news_available', False),
                # Whale (v2 — full detail)
                'whale_score':          whale['whale_score'],
                'whale_signal':         whale['whale_signal'],
                'whale_buy_pressure':   whale.get('whale_buy_pressure', 0.5),
                'whale_sell_pressure':  whale.get('whale_sell_pressure', 0.5),
                'large_trade_ratio':    whale.get('large_trade_ratio', 0.0),
                'large_trade_count':    whale.get('large_trade_count', 0),
                'exchange_flow_bias':   whale.get('exchange_flow_bias', 0.0),
                'order_book_imbalance': whale.get('order_book_imbalance', 0.0),
                'whale_score_norm':     whale.get('whale_score_norm', 0.0),
                # Whale scoring transparency
                'score_before_whale':   final.get('score_before_whale', 0),
                'whale_contrib':        final.get('whale_contrib', 0),
                # Probability
                'probability_up':          prob['probability_up'],
                'probability_down':        round(100.0 - float(prob.get('probability_up', 50) or 50), 2),
                'probability_confidence':  prob['probability_confidence'],
                # Final AI Score — RL-adjusted (regime mult already applied inside final)
                'final_score':     rl_adjusted,             # RL-enhanced score
                'final_signal':    final['final_signal'],
                'technical_score': final.get('technical_score', profit['profit_score']),
                'news_impact':     final.get('news_impact', 50),
                # Indicators snapshot for backtester — now includes S/R + ATR
                'indicators': {
                    'rsi':          indicators['rsi'],
                    'ema20':        indicators['ema20'],
                    'ema50':        indicators['ema50'],
                    'volume_ratio': indicators.get('volume_spike', 1.0),
                },
                # ATR + Support/Resistance (for adaptive backtester)
                'atr':               indicators.get('atr', 0),
                'atr_pct':           indicators.get('atr_pct', 0),
                'near_support':      indicators.get('near_support', False),
                'near_resistance':   indicators.get('near_resistance', False),
                'sr_quality':        indicators.get('sr_quality', 0),
                'nearest_support':   indicators.get('nearest_support'),
                'nearest_resistance':indicators.get('nearest_resistance'),
                # Multi-Timeframe Confirmation
                'mtf_score':         mtf.get('mtf_score', 50),
                'mtf_confirmed':     mtf.get('mtf_confirmed', False),
                'mtf_bias':          mtf.get('mtf_bias', 'NEUTRAL'),
                'mtf_reason':        mtf.get('alignment_reason', ''),
                'rsi_4h':            mtf.get('rsi_4h', 50),
                # News Timing Flag
                'news_extreme':     news_extreme,
                # Market context
                'market_regime': regime['regime'],
                'regime_confidence': regime.get('confidence', 0),
                # Regime scoring transparency
                'score_before_regime': final.get('score_before_regime', final['final_score']),
                'regime_multiplier':   final.get('regime_multiplier', 1.0),
                'applied_regime':      final.get('applied_regime', 'NEUTRAL'),
                # RL scoring transparency
                'score_before_rl': score_before_rl,
                'rl_weight':       round(rl_weight, 4),
                'rl_adjusted_score': rl_adjusted,
            }

            # 2h: Risk-Adjusted Score (NEW)
            # Computes volatility_penalty, drawdown_penalty, liquidity_bonus,
            # rr_bonus, probability_weight, news_weight → risk_adjusted_score + risk_level
            apply_risk_model(result, klines)

            results.append(result)

            if i % 15 == 0 or i <= 3:
                logger.info(
                    f"[{i}/{len(symbols)}] {sym}: Score={final['final_score']}, "
                    f"RA={result.get('risk_adjusted_score',0):.1f} ({result.get('risk_level','?')}), "
                    f"News={news['news_score']:+.2f}, Whale={whale['whale_signal']}"
                )

        except Exception as e:
            logger.error(f"  Error {sym}: {e}")
            failed += 1

    logger.info(f"Scan complete: {len(results)} success, {failed} failed")

    _scan_client.close()
    return results


def save_to_mongodb(results: List[Dict], force: bool = False) -> bool:
    """
    Save AI signals to MongoDB using a dedicated short-lived connection.
    Never touches the shared mongo_client singleton (whale tracker uses that).

    Storage rule: saves every scan (should_store_signals() always returns True).
    Pass force=True to bypass the rule (used on startup or by tests).

    Returns True if signals were saved or storage was intentionally skipped.
    """
    import pymongo as _pymongo

    if not results:
        return False

    if not force and not should_store_signals():
        logger.info(f"Signal storage skipped "
                    f"(current: {datetime.utcnow().strftime('%H:%M')})")
        return True

    try:
        _client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[settings.DATABASE_NAME]
        col = _db[settings.COLLECTION_AI_SIGNALS]

        stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
        for s in results:
            try:
                filter_key = {
                    'symbol':    s['symbol'],
                    'timestamp': s.get('timestamp', datetime.utcnow()),
                    'timeframe': s.get('timeframe', settings.SIGNAL_TIMEFRAME),
                }
                doc = {k: v for k, v in s.items() if k != '_id'}
                doc.setdefault('timeframe', settings.SIGNAL_TIMEFRAME)
                result = col.update_one(
                    filter_key,
                    {'$set': doc, '$setOnInsert': {'created_at': datetime.utcnow()}},
                    upsert=True
                )
                if result.upserted_id:
                    stats['inserted'] += 1
                elif result.modified_count > 0:
                    stats['updated'] += 1
                else:
                    stats['skipped'] += 1
            except Exception:
                stats['skipped'] += 1

        total = col.count_documents({})
        _client.close()
        logger.info(f"Saved {stats['inserted']} new signals "
                    f"(+{stats['updated']} updated, ={stats['skipped']} skipped). "
                    f"Total history: {total} signals")
        return True
    except Exception as e:
        logger.error(f"Save error: {e}")
        return False


def get_latest_signals_from_db(limit: int = 90) -> List[Dict]:
    """
    Load the latest signal snapshot per symbol from DB.
    Used by dashboard and backtester for current state.
    """
    from database.mongo_client import mongo_client
    try:
        if not mongo_client.connect():
            return []
        signals = mongo_client.get_latest_signals(limit)
        mongo_client.close()
        return signals
    except Exception:
        return []


if __name__ == "__main__":
    results = scan_market(5)
    if results:
        save_to_mongodb(results, force=True)
        for r in results[:5]:
            rank = f"#{r['top_rank']}" if r.get('top_rank') else "  "
            print(f"{rank} {r['symbol']}: Score={r['final_score']}, Prob={r['probability_up']}%, "
                  f"Whale={r['whale_signal']}, News={r['news_sentiment']}, Regime={r['market_regime']}")
