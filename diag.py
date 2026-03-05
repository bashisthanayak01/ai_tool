"""
Test the full rank + save pipeline exactly as the scheduler does it.
"""
import sys
sys.path.insert(0, r'c:\Users\bashi\Desktop\crypto_tool\crypto_ai_tool')
import pymongo
from config import settings
from ranking_engine import rank_coins, save_rankings
from services.market_pipeline import get_latest_signals_from_db
from database.mongo_client import mongo_client

mongo_client.connect()

# Get signals
signals = get_latest_signals_from_db(limit=90)
print(f"Signals loaded: {len(signals)}")

# Rank them
ranked = rank_coins(signals)
print(f"Ranked: {len(ranked)} coins qualify")

# Try save exactly like scheduler does
try:
    _rank_client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    _rank_db = _rank_client[settings.DATABASE_NAME]
    all_symbols = [r['symbol'] for r in signals]
    n_saved = save_rankings(ranked, _rank_db, all_scanned_symbols=all_symbols)
    print(f"Saved {n_saved} ranked coins to DB")
    _rank_client.close()
except Exception as e:
    print(f"ERROR in save_rankings: {type(e).__name__}: {e}")

# Verify it's in DB now
try:
    _check_client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    _check_db = _check_client[settings.DATABASE_NAME]
    count = _check_db['ranked_opportunities'].count_documents({})
    print(f"ranked_opportunities count: {count}")
    top3 = list(_check_db['ranked_opportunities'].find({}, {'_id': 0, 'symbol': 1, 'entry_price': 1, 'rank_score': 1}).sort('rank_score', -1).limit(3))
    for t in top3:
        print(f"  {t['symbol']}  entry={t['entry_price']}  rank={t['rank_score']}")
    _check_client.close()
except Exception as e:
    print(f"Check error: {e}")

mongo_client.close()
