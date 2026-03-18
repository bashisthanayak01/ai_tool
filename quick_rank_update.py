"""
quick_rank_update.py — Just update batch_ts right now using existing DB signals.
Run this anytime to fix the 'Last Signal' timestamp on the dashboard.
"""
import pymongo
from config import settings
from ranking_engine import rank_coins, save_rankings
from datetime import datetime

client = pymongo.MongoClient(settings.MONGO_URI,
                             serverSelectionTimeoutMS=30000,
                             socketTimeoutMS=60000,
                             connectTimeoutMS=30000)
db = client[settings.DATABASE_NAME]

pipeline = [
    {'$sort': {'timestamp': -1}},
    {'$group': {'_id': '$symbol', 'doc': {'$first': '$$ROOT'}}},
    {'$replaceRoot': {'newRoot': '$doc'}},
]
db_signals = list(db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
print(f"Signals: {len(db_signals)}")

ranked = rank_coins(db_signals)
n = save_rankings(ranked, db, all_scanned_symbols=[d.get('symbol', d.get('_id','')) for d in db_signals])
print(f"Ranked and saved: {n}")

r = db['ranked_opportunities'].find_one({}, sort=[('batch_ts',-1)],
                                         projection={'batch_ts':1,'symbol':1,'_id':0})
print(f"New batch_ts: {r}")
client.close()
print("Done — refresh dashboard now.")
