import sys, time
sys.path.insert(0, r'c:\Users\bashi\Desktop\crypto_tool\crypto_ai_tool')
import warnings; warnings.filterwarnings('ignore')
from config import settings
import pymongo
client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
db = client[settings.DATABASE_NAME]
col = db[settings.COLLECTION_AI_SIGNALS]
print('Docs in DB:', col.count_documents({}))
pipeline = [
    {'\': {'timestamp': -1}},
    {'\': {'_id': '\', 'doc': {'\': '\$\'}}},
    {'\': {'newRoot': '\'}},
    {'\': {'final_score': -1}}
]
t0 = time.time()
r = list(col.aggregate(pipeline))
ms = round((time.time()-t0)*1000)
if r:
    print('TEST PASS:', len(r), 'coins in', ms, 'ms')
    print('Top coin:', r[0].get('symbol'), '| score:', r[0].get('final_score'), '| signal:', r[0].get('final_signal'))
else:
    print('No data returned')
client.close()
