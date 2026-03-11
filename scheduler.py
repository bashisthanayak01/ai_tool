"""
Scheduler v6 — 5-min scans + 10-min whale scan + hourly regime/signal storage +
               daily RL learning + weekly self-learning + weekly auto-optimization
"""

import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from services.market_pipeline import scan_market, save_to_mongodb, collect_historical_data
from services.binance_scanner import get_top_symbols
from database.mongo_client import mongo_client
from ranking_engine import rank_coins, save_rankings
from learning_engine import run_learning_cycle
from optimization.auto_optimizer import run_optimization
from ai.market_regime import detect_market_regime, save_regime_history
from ai.rl_optimizer import run_rl_learning
from ai.whale_tracker import run_whale_scan
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Track whether any signals exist in DB (for bootstrap force-save)
_first_scan_done = False


def initialize():
    """Startup: setup indexes, collect historical data"""
    logger.info("=" * 60)
    logger.info("INITIALIZATION")
    logger.info("=" * 60)

    if not mongo_client.connect():
        return False

    # Create/update indexes (safe to call multiple times)
    mongo_client.setup_indexes()

    # Show DB state
    sig_count = mongo_client.get_signal_count()
    news_count = mongo_client.get_news_count()
    logger.info(f"DB State: {sig_count} signals, {news_count} news items in store")
    mongo_client.close()

    # Historical data (incremental)
    symbols = get_top_symbols(90)
    if symbols:
        stats = collect_historical_data(symbols)
        logger.info(f"Historical: +{stats['inserted']} new candles")
    return True


def run_scan():
    """
    Execute full market scan and save signals on every run.
    Force-save only needed when DB is empty (bootstrap case).
    Uses dedicated short-lived connections — never touches the shared
    mongo_client singleton (whale_tracker uses that concurrently).
    """
    global _first_scan_done

    try:
        logger.info("=" * 60)
        logger.info(f"SCAN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        results = scan_market(limit=90)
        if not results:
            logger.warning("No results from scan")
            return

        # Check if DB has any signals at all (bootstrap case)
        # Use a fresh connection for this check to avoid interfering with the main mongo_client singleton
        # which might be used by other concurrent jobs or later parts of this function.
        from database.mongo_client import MongoClient
        temp_mongo_client = MongoClient()
        if not temp_mongo_client.connect():
            force_save = not _first_scan_done
        else:
            sig_count = temp_mongo_client.get_signal_count()
            temp_mongo_client.close() # Close the temporary connection
            force_save = (sig_count == 0)

        saved = save_to_mongodb(results, force=force_save)
        _first_scan_done = True

        # ── Ranking Engine: run after every scan — own dedicated connection ──
        try:
            import pymongo as _pymongo
            from services.market_pipeline import get_latest_signals_from_db
            _rank_client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
            _rank_db = _rank_client[settings.DATABASE_NAME]
            # Use freshly-saved DB signals (consistent field format) rather than
            # raw live scan results which may have edge-case field differences
            db_signals = get_latest_signals_from_db(limit=90)
            ranked = rank_coins(db_signals) if db_signals else rank_coins(results)
            all_symbols = [r['symbol'] for r in (db_signals or results)]
            n_saved = save_rankings(ranked, _rank_db, all_scanned_symbols=all_symbols)
            _rank_client.close()
            if ranked:
                top_r = ranked[0]
                logger.info(f"[Ranking] Top coin: {top_r['symbol']} score={top_r['rank_score']}")
            logger.info(f"[Ranking] Saved {n_saved}/{len(db_signals)} ranked coins")
        except Exception as re:
            logger.error(f"[Ranking] Error: {type(re).__name__}: {re}", exc_info=True)

        top = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)[0]
        logger.info(f"[OK] {len(results)} coins scanned. "
                    f"Top: {top['symbol']} Score={top['final_score']}, "
                    f"Prob={top['probability_up']}% | "
                    f"Stored={'YES' if saved else 'SKIPPED (hourly rule)'}")

    except Exception as e:
        logger.error(f"Scan error: {e}")


def cleanup_old_data():
    """
    Daily cleanup job: uses its OWN dedicated connection — never the shared singleton.
    - Delete AI signals older than SIGNAL_RETENTION_DAYS (90 days)
    - Delete news older than NEWS_RETENTION_DAYS (30 days)
    """
    try:
        import pymongo as _pymongo
        _client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        _db = _client[settings.DATABASE_NAME]
        from datetime import timedelta
        now = datetime.utcnow()
        # Delete old signals
        sig_cutoff = now - timedelta(days=settings.SIGNAL_RETENTION_DAYS)
        sig_del = _db['ai_signals'].delete_many({'timestamp': {'$lt': sig_cutoff}}).deleted_count
        # Delete old news
        news_cutoff = now - timedelta(days=settings.NEWS_RETENTION_DAYS)
        news_del = _db['news_data'].delete_many({'created_at': {'$lt': news_cutoff}}).deleted_count
        # Prune stale rankings older than 24 hours
        try:
            rank_del = _db['ranked_opportunities'].delete_many(
                {'created_at': {'$lt': now - timedelta(hours=24)}}
            ).deleted_count
        except Exception:
            rank_del = 0
        sig_count  = _db['ai_signals'].count_documents({})
        news_count = _db['news_data'].count_documents({})
        _client.close()  # close OUR connection, not the shared singleton
        logger.info(f"Daily cleanup: {sig_del} signals, {news_del} news, {rank_del} old rankings deleted. "
                    f"Remaining: {sig_count} signals, {news_count} news")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


def run_learning():
    """
    Weekly learning cycle: mine ai_signals history,
    compute indicator win rates, adjust model weights ≤10%.
    """
    logger.info("=" * 60)
    logger.info("[Learning] Weekly self-learning cycle started")
    logger.info("=" * 60)
    try:
        result = run_learning_cycle(lookback_days=30)
        if result['trade_count'] > 0:
            logger.info(f"[Learning] Analysed {result['trade_count']} trades over 30 days")
            for msg in result['improvements']:
                logger.info(f"[Learning]   {msg}")
        else:
            logger.info("[Learning] Insufficient data — keeping current weights")
    except Exception as e:
        logger.error(f"[Learning] Error: {e}")


def save_regime_snapshot():
    """
    Hourly job: detect current market regime and persist to regime_history.
    Also fetches BTC price for the snapshot.
    """
    try:
        regime = detect_market_regime()
        # Get BTC price for the snapshot
        btc_price = None
        try:
            from services.binance_scanner import get_klines
            klines = get_klines('BTCUSDT', '1h', 2)
            if klines:
                btc_price = float(klines[-1].get('close', 0))
        except Exception:
            pass

        saved = save_regime_history(regime, btc_price=btc_price)
        logger.info(
            f"[RegimeSnapshot] {regime['regime']} conf={regime['confidence']}% "
            f"btc=${btc_price:,.0f} | saved={saved}" if btc_price
            else f"[RegimeSnapshot] {regime['regime']} conf={regime['confidence']}% | saved={saved}"
        )
    except Exception as e:
        logger.error(f"[RegimeSnapshot] Error: {e}")


def run_optimization_job():
    """
    Weekly auto-optimization: grid search over trading parameters,
    save best config to strategy_configs collection.
    """
    logger.info("=" * 60)
    logger.info("[Optimizer] Weekly strategy optimization started")
    logger.info("=" * 60)
    try:
        result = run_optimization(lookback_days=90)
        logger.info(f"[Optimizer] Tested {result['tested_count']} configs")
        logger.info(f"[Optimizer] Best composite: {result['composite']:.4f} "
                    f"(improvement: {result['improvement_pct']:+.4f})")
        if result['applied']:
            bc = result.get('best_config', {})
            bp = result.get('best_perf', {})
            logger.info(
                f"[Optimizer] ✅ New config applied: "
                f"TP={bc.get('take_profit',0)*100:.0f}% "
                f"SL={bc.get('stop_loss',0)*100:.0f}% "
                f"Score≥{bc.get('min_score',0)} "
                f"Return={bp.get('return_pct',0):+.2f}%"
            )
        else:
            logger.info("[Optimizer] No config change (safety gate or no improvement)")
    except Exception as e:
        logger.error(f"[Optimizer] Error: {e}")


def run_rl_learning_job():
    """
    Daily RL learning cycle:
    - Requires >= 20 trade outcomes in DB
    - Runs reward-based parameter tuning
    - Saves updated rl_parameters to DB
    - Logs improvement metrics
    """
    logger.info("=" * 60)
    logger.info("[RL] Daily reinforcement learning cycle started")
    logger.info("=" * 60)
    try:
        result = run_rl_learning(lookback_days=60)
        if result.get('applied'):
            p = result.get('params', {})
            s = result.get('stats', {})
            logger.info(
                f"[RL] Episode {result['episode']} complete | "
                f"RL weight={p.get('rl_weight_adjustment',1.0):.4f} | "
                f"Reward={result['reward']:+.2f} | "
                f"WR={s.get('win_rate',0):.1f}% PF={s.get('profit_factor',0):.2f}"
            )
        elif result.get('error'):
            logger.info(f"[RL] Skipped: {result['error']}")
        else:
            logger.info("[RL] No parameter change applied")
    except Exception as e:
        logger.error(f"[RL] Learning error: {e}")


def run_whale_scan_job():
    """
    10-min whale intelligence scan — uses its OWN dedicated connection.
    Never touches the shared mongo_client singleton.
    """
    try:
        import pymongo as _pymongo
        _wh_client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        _wh_db = _wh_client[settings.DATABASE_NAME]
        n_symbols = getattr(settings, 'WHALE_SCAN_TOP_N', 30)
        symbols = get_top_symbols(n_symbols)
        results = run_whale_scan(symbols[:n_symbols], db=_wh_db)
        accu = sum(1 for r in results if r.get('whale_signal') == 'ACCUMULATION')
        dist = sum(1 for r in results if r.get('whale_signal') == 'DISTRIBUTION')
        avg_score = round(sum(r.get('whale_score', 50) for r in results) / max(len(results), 1), 1)
        _wh_client.close()  # close OUR connection only
        logger.info(
            f"[Whale] Scanned {len(results)} symbols — "
            f"ACCUM={accu} DIST={dist} avg_score={avg_score}"
        )
    except Exception as e:
        logger.error(f"[Whale] Scan error: {e}")


def print_db_summary():
    """Print current DB state — called at startup. Uses dedicated connection."""
    try:
        import pymongo as _pymongo
        _client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[settings.DATABASE_NAME]
        sig  = _db['ai_signals'].count_documents({})
        news = _db['news_data'].count_documents({})
        mkt  = _db['market_data'].count_documents({}) if 'market_data' in _db.list_collection_names() else 0
        _client.close()
        logger.info(f"DB State: {sig} signals, {news} news items in store")
    except Exception:
        pass


def main():
    logger.info("=" * 60)
    logger.info("CRYPTO AI ANALYTICS PLATFORM v6")
    logger.info("Signal history:        ENABLED (hourly snapshots, 90-day retention)")
    logger.info("News storage:          ENABLED (30-min cache, 30-day retention)")
    logger.info("AI formula:            Tech(70%) + News(30%) + Whale(10pt) x Regime x RL")
    logger.info("Regime multipliers:    BULL=1.10 | BEAR=0.80 | SIDEWAYS=0.95")
    logger.info("Whale Intelligence:    ENABLED (5-source: aggTrades+OB+ticker+klines+depth)")
    logger.info("RL Optimizer:          ENABLED (daily, clamped 0.80-1.20)")
    logger.info("Self-Learning:         ENABLED (weekly weight adaptation)")
    logger.info("Auto-Optimization:     ENABLED (weekly grid search, DB-backed)")
    logger.info("Regime history:        ENABLED (hourly snapshot to regime_history)")
    logger.info("=" * 60)

    if not initialize():
        logger.error("Init failed — check MongoDB connection")
        return

    print_db_summary()

    # Run first scan immediately (force-saves if no signals exist)
    run_scan()

    # Seed regime history immediately
    try:
        save_regime_snapshot()
    except Exception:
        pass

    scheduler = BlockingScheduler()

    # Every 5 min: full market scan
    # coalesce=True: if previous scan still running, skip redundant queued triggers
    # max_instances=1: ONLY one scan allowed at a time — prevents parallel overlapping scans
    # misfire_grace_time=120s: still fire if scheduler was briefly paused (e.g. startup delay)
    scheduler.add_job(run_scan, trigger=IntervalTrigger(minutes=5),
                      id='scan', name='Market Scanner', replace_existing=True,
                      max_instances=1, coalesce=True, misfire_grace_time=120)

    # Every hour: persist regime snapshot
    scheduler.add_job(save_regime_snapshot, trigger=IntervalTrigger(hours=1),
                      id='regime_snapshot', name='Regime History', replace_existing=True)

    # Once per day at 02:00 UTC: cleanup old records
    scheduler.add_job(cleanup_old_data, trigger=IntervalTrigger(hours=24),
                      id='cleanup', name='Data Cleanup', replace_existing=True)

    # Once per week: self-learning cycle (weight adaptation)
    scheduler.add_job(run_learning, trigger=IntervalTrigger(weeks=1),
                      id='learning', name='Self-Learning AI', replace_existing=True)

    # Once per week: strategy parameter optimization
    scheduler.add_job(run_optimization_job, trigger=IntervalTrigger(weeks=1),
                      id='optimization', name='Strategy Optimizer', replace_existing=True)

    # Once per day: RL reinforcement learning cycle
    scheduler.add_job(run_rl_learning_job, trigger=IntervalTrigger(hours=24),
                      id='rl_learning', name='RL Optimizer', replace_existing=True)

    # Every 10 minutes: whale intelligence scan (top 30 symbols)
    # coalesce=True: skip redundant queued runs (whale scan can take 2-3 min)
    scheduler.add_job(run_whale_scan_job, trigger=IntervalTrigger(minutes=10),
                      id='whale_scan', name='Whale Intelligence', replace_existing=True,
                      max_instances=1, coalesce=True, misfire_grace_time=60)

    logger.info("[OK] Scheduler running.")
    logger.info("  - Market scan:         every 5 minutes")
    logger.info("  - Whale scan:          every 10 minutes (top 30 symbols)")
    logger.info("  - Signal storage:      once per hour (at minute :00)")
    logger.info("  - Regime snapshot:     every 1 hour")
    logger.info("  - Data cleanup:        every 24 hours")
    logger.info("  - RL Learning:         every 24 hours")
    logger.info("  - Self-Learning:       every 7 days")
    logger.info("  - Strategy Optimizer:  every 7 days")
    logger.info("  Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[OK] Stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
