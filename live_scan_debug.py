"""
live_scan_debug.py — Run a real 3-symbol scan and capture every error.
"""
import logging, sys
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

import pymongo
from datetime import datetime
from config import settings
from ranking_engine import rank_coins, save_rankings

print("\n=== STEP 1: scan_market (3 symbols) ===")
try:
    from services.market_pipeline import scan_market, save_to_mongodb
    results = scan_market(limit=3)
    print(f"scan_market returned {len(results)} results")
    for r in results:
        print(f"  {r['symbol']}  score={r.get('final_score',0):.1f}  ts={r.get('timestamp')}")
except Exception as e:
    print(f"SCAN ERROR: {e}")
    import traceback; traceback.print_exc()
    results = []

print("\n=== STEP 2: save_to_mongodb (force=True) ===")
if results:
    try:
        ok = save_to_mongodb(results, force=True)
        print(f"save_to_mongodb returned: {ok}")
    except Exception as e:
        print(f"SAVE ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n=== STEP 3: rank_coins + save_rankings ===")
if results:
    try:
        ranked = rank_coins(results)
        print(f"rank_coins returned {len(ranked)} coins")
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        db = client[settings.DATABASE_NAME]
        n = save_rankings(ranked, db, all_scanned_symbols=[r['symbol'] for r in results])
        print(f"save_rankings saved {n} coins")
        r2 = db['ranked_opportunities'].find_one({}, sort=[('batch_ts',-1)],
                                                  projection={'batch_ts':1,'symbol':1,'_id':0})
        print(f"Latest batch_ts in DB: {r2}")
        client.close()
    except Exception as e:
        print(f"RANKING ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n=== DONE ===")
