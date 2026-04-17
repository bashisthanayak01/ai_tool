import sys, warnings
sys.path.insert(0, r'c:\Users\bashi\Desktop\crypto_tool\crypto_ai_tool')
warnings.filterwarnings('ignore')

from config import settings
import pymongo
client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[settings.DATABASE_NAME]

print('=== DB STATE AFTER FIRST SCAN ===')
for c in sorted(db.list_collection_names()):
    n = db[c].count_documents({})
    print(f'  {c}: {n} docs')

print('\n=== TOP 10 RANKED COINS ===')
ranked = list(db['ranked_opportunities'].find(
    {}, {'symbol':1,'rank_score':1,'final_signal':1,'probability_up':1,'_id':0}
).sort('rank_score',-1).limit(10))
if ranked:
    for r in ranked:
        print(f'  {r.get("symbol","?"):12s}  rank={r.get("rank_score",0):.1f}  sig={r.get("final_signal","?")}  prob={r.get("probability_up",0):.0f}%')
else:
    print('  (no ranked_opportunities yet)')

print('\n=== SIGNAL DISTRIBUTION ===')
if 'ai_signals' in db.list_collection_names():
    buy_c  = db['ai_signals'].count_documents({'final_signal':'BUY'})
    hold_c = db['ai_signals'].count_documents({'final_signal':'HOLD'})
    sell_c = db['ai_signals'].count_documents({'final_signal':'SELL'})
    total  = db['ai_signals'].count_documents({})
    print(f'  Total={total}  BUY={buy_c}  HOLD={hold_c}  SELL={sell_c}')
else:
    print('  ai_signals not yet created')

client.close()
print('DONE')
