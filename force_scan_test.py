"""
force_scan_test.py — Run a mini scan (5 symbols) and force-save to DB.
Shows exact errors so we can fix the root cause.
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from datetime import datetime
import pymongo
from config import settings

print("=" * 55)
print("FORCE SCAN + SAVE TEST")
print("=" * 55)

# Step 1: Run a small scan
print("\n[1] Running mini scan (5 symbols)...")
try:
    from services.market_pipeline import scan_market, save_to_mongodb
    results = scan_market(limit=5)
    print(f"    Scan returned {len(results)} results")
    if results:
        for r in results:
            print(f"    {r['symbol']}: score={r.get('final_score',0):.1f} ts={r.get('timestamp')}")
except Exception as e:
    print(f"    SCAN ERROR: {e}")
    import traceback; traceback.print_exc()
    results = []

# Step 2: Force save
if results:
    print("\n[2] Force-saving to ai_signals...")
    try:
        ok = save_to_mongodb(results, force=True)
        print(f"    save_to_mongodb returned: {ok}")
    except Exception as e:
        print(f"    SAVE ERROR: {e}")
        import traceback; traceback.print_exc()

# Step 3: Check DB
print("\n[3] Checking DB after save...")
try:
    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.DATABASE_NAME]
    latest = db['ai_signals'].find_one({}, sort=[('timestamp',-1)],
                                        projection={'timestamp':1,'symbol':1,'_id':0})
    count   = db['ai_signals'].count_documents({})
    print(f"    Total signals: {count}")
    print(f"    Latest signal: {latest}")
    client.close()
except Exception as e:
    print(f"    DB CHECK ERROR: {e}")

# Step 4: Run ranking
if results:
    print("\n[4] Running ranking + save...")
    try:
        from ranking_engine import rank_coins, save_rankings
        _client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        _db = _client[settings.DATABASE_NAME]
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
            {'$replaceRoot': {'newRoot': '$doc'}},
        ]
        db_signals = list(_db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
        print(f"    DB signals for ranking: {len(db_signals)}")
        ranked = rank_coins(db_signals) if db_signals else rank_coins(results)
        print(f"    Ranked coins: {len(ranked)}")
        n = save_rankings(ranked, _db, all_scanned_symbols=[r['symbol'] for r in results])
        print(f"    Rankings saved: {n}")
        latest_rank = _db['ranked_opportunities'].find_one(
            {}, sort=[('batch_ts',-1)], projection={'batch_ts':1,'symbol':1,'_id':0})
        print(f"    Latest batch_ts: {latest_rank}")
        _client.close()
    except Exception as e:
        print(f"    RANKING ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n" + "="*55)
print("TEST COMPLETE")
