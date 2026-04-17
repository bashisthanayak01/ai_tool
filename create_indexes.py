import sys
sys.path.insert(0, r'c:\Users\bashi\Desktop\crypto_tool\crypto_ai_tool')
import warnings; warnings.filterwarnings('ignore')

from config import settings
import pymongo

client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
db = client[settings.DATABASE_NAME]
col = db[settings.COLLECTION_AI_SIGNALS]

print("=== Existing ai_signals indexes ===")
for name, idx in col.index_information().items():
    print("  " + name + ": " + str(idx['key']))

# Create only missing indexes (skip if conflict)
needed = [
    ([('timestamp', pymongo.DESCENDING)], 'ts_desc_idx'),
    ([('final_score', pymongo.DESCENDING)], 'score_desc_idx'),
]
for key, name in needed:
    try:
        col.create_index(key, name=name, background=True)
        print("Created: " + name)
    except Exception as ex:
        print("Skipped " + name + ": " + str(ex)[:80])

# Whale indexes
wh = db['whale_data']
print("\n=== Existing whale_data indexes ===")
for name, idx in wh.index_information().items():
    print("  " + name + ": " + str(idx['key']))

try:
    wh.create_index([('timestamp', pymongo.DESCENDING)], name='wh_ts_idx', background=True)
    print("Created: wh_ts_idx")
except Exception as ex:
    print("Skipped: " + str(ex)[:80])

client.close()
print("\nDONE")
