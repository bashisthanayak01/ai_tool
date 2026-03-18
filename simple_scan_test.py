"""
Simple scan test - scan 3 coins, show what happens step by step.
"""
import sys
import logging
# Only show WARNING+ to reduce noise
logging.basicConfig(level=logging.WARNING)

import pymongo
from datetime import datetime
from config import settings

print(f"[{datetime.now()}] Starting simple scan test...")

# STEP 1: scan_market
print("\n--- STEP 1: scan_market (3 symbols) ---")
from services.market_pipeline import scan_market, save_to_mongodb
results = scan_market(limit=3)
print(f"scan_market returned {len(results)} results")
for r in results:
    print(f"  {r['symbol']}  score={r.get('final_score',0):.1f}  signal={r.get('final_signal')}  ts={r.get('timestamp')}")

# STEP 2: save_to_mongodb
print("\n--- STEP 2: save_to_mongodb ---")
if results:
    ok = save_to_mongodb(results, force=True)
    print(f"save_to_mongodb returned: {ok}")
    
    # Verify it actually wrote
    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.DATABASE_NAME]
    ts = results[0]['timestamp']
    from datetime import timedelta
    recent = db['ai_signals'].count_documents({'timestamp': {'$gte': ts - timedelta(seconds=1)}})
    print(f"Signals with this scan_time in DB: {recent}")
    client.close()

# STEP 3: ranking
print("\n--- STEP 3: rank_coins + save_rankings ---")
if results:
    from ranking_engine import rank_coins, save_rankings
    ranked = rank_coins(results)
    print(f"rank_coins returned {len(ranked)} coins (from {len(results)} scanned)")
    if ranked:
        for r in ranked[:3]:
            print(f"  {r['symbol']} rank_score={r['rank_score']} signal={r['signal']}")
    else:
        print("  NO COINS PASSED FILTERS! batch_ts will NOT be updated!")
    
    client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.DATABASE_NAME]
    n = save_rankings(ranked, db, all_scanned_symbols=[r['symbol'] for r in results])
    print(f"save_rankings saved {n} coins")
    
    rank_doc = db['ranked_opportunities'].find_one({}, sort=[('batch_ts', -1)],
                                                    projection={'batch_ts':1,'symbol':1,'_id':0})
    print(f"Latest batch_ts in ranked_opportunities: {rank_doc}")
    client.close()

print(f"\n[{datetime.now()}] Done!")
