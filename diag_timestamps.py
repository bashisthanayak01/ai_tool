import pymongo
from config import settings
from datetime import datetime, timedelta

client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
db = client[settings.DATABASE_NAME]

print("=" * 55)
print("  DB TIMESTAMP DIAGNOSTICS")
print("=" * 55)

# Latest signal
latest_sig = db['ai_signals'].find_one({}, {'timestamp':1,'symbol':1}, sort=[('timestamp',-1)])
print(f"\nLatest ai_signal:    {latest_sig.get('timestamp') if latest_sig else 'NONE'}  ({latest_sig.get('symbol','') if latest_sig else ''})")

# Latest ranked
latest_rank = db['ranked_opportunities'].find_one({}, {'batch_ts':1,'created_at':1,'symbol':1}, sort=[('batch_ts',-1)])
print(f"Latest ranked batch_ts:  {latest_rank.get('batch_ts') if latest_rank else 'NONE'}")
print(f"Latest ranked created_at:{latest_rank.get('created_at') if latest_rank else 'NONE'}")

# Signal count by date
print("\nSignals per day (last 7 days):")
for days_ago in [0,1,2,3,4,5,6]:
    d = datetime.utcnow() - timedelta(days=days_ago)
    d_start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    d_end   = d_start + timedelta(days=1)
    count   = db['ai_signals'].count_documents({'timestamp': {'$gte': d_start, '$lt': d_end}})
    print(f"  {d_start.date()}: {count} signals")

# Total counts
print(f"\nTotal ai_signals:   {db['ai_signals'].count_documents({})}")
print(f"Total news_data:    {db['news_data'].count_documents({})}")
print(f"Total ranked_opps:  {db['ranked_opportunities'].count_documents({})}")
print(f"Total whale_data:   {db['whale_data'].count_documents({}) if 'whale_data' in db.list_collection_names() else 0}")

client.close()
print("\n" + "="*55)
