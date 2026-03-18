"""
patch_scheduler_fix.py — Fix the 'Database object is not callable' crash
in scheduler.run_scan() ranking block.

The bug: get_latest_signals_from_db() uses the shared mongo_client singleton
which gets closed by the concurrent whale_tracker job. This crashes the ranking
save, so batch_ts never updates → 'Last Signal' stays stale on dashboard.

Fix: replace get_latest_signals_from_db() call with direct query on _rank_client.
"""

path = 'scheduler.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

OLD = '''        # ── Ranking Engine: run after every scan — own dedicated connection ──
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
            logger.error(f"[Ranking] Error: {type(re).__name__}: {re}", exc_info=True)'''

NEW = '''        # ── Ranking Engine: run after every scan — own dedicated connection ──
        try:
            import pymongo as _pymongo
            _rank_client = _pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
            _rank_db = _rank_client[settings.DATABASE_NAME]
            # Query latest signal per symbol directly using OUR OWN connection.
            # Do NOT use get_latest_signals_from_db() — it uses the shared
            # mongo_client singleton which may be closed by concurrent whale job,
            # causing "Database object is not callable" crash.
            pipeline = [
                {'$sort': {'timestamp': -1}},
                {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
                {'$replaceRoot': {'newRoot': '$doc'}},
            ]
            db_signals = list(_rank_db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
            ranked = rank_coins(db_signals) if db_signals else rank_coins(results)
            all_symbols = [r['symbol'] for r in (db_signals or results)]
            n_saved = save_rankings(ranked, _rank_db, all_scanned_symbols=all_symbols)
            _rank_client.close()
            if ranked:
                top_r = ranked[0]
                logger.info(f"[Ranking] Top coin: {top_r['symbol']} score={top_r['rank_score']}")
            logger.info(f"[Ranking] Saved {n_saved}/{len(db_signals)} ranked coins")
        except Exception as re:
            logger.error(f"[Ranking] Error: {type(re).__name__}: {re}", exc_info=True)'''

if OLD in content:
    content = content.replace(OLD, NEW)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Patch applied: get_latest_signals_from_db() replaced with direct query")
else:
    print("⚠️  OLD block not found — checking if already patched...")
    if 'get_latest_signals_from_db' in content:
        print("   get_latest_signals_from_db still present — manual fix needed")
    else:
        print("   get_latest_signals_from_db already removed — patch may be applied")

# Verify
import ast
ast.parse(open(path, encoding='utf-8').read())
print("✅ Syntax check: OK")
