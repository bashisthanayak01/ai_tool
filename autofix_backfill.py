"""
autofix_backfill.py — Backfill missing fields in existing MongoDB docs
======================================================================
1. Add probability_down = 100 - probability_up to all ai_signals
2. Re-run ranking from DB to refresh ranked_opportunities with risk fields
3. Re-run risk model on all current signals to ensure risk fields present
"""
import sys
sys.path.insert(0, '.')

import pymongo
from datetime import datetime
from config import settings
from ranking_engine import run_ranking_from_db
from risk_model import apply_risk_model, apply_risk_model_batch

print("=" * 60)
print("  AUTO-FIX BACKFILL")
print("=" * 60)

client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
db = client[settings.DATABASE_NAME]

col = db[settings.COLLECTION_AI_SIGNALS]

# ── 1. Add probability_down where missing ──────────────────────
print("\n[1] Patching probability_down in ai_signals...")
docs = list(col.find({'probability_down': {'$exists': False}},
                     {'symbol': 1, 'probability_up': 1}))
patched = 0
for doc in docs:
    prob_up   = float(doc.get('probability_up', 50) or 50)
    prob_down = round(100.0 - prob_up, 2)
    col.update_one({'_id': doc['_id']},
                   {'$set': {'probability_down': prob_down}})
    patched += 1
print(f"  Patched {patched} signals with probability_down")

# ── 2. Re-apply risk model on all signals ──────────────────────
print("\n[2] Re-applying risk model to all ai_signals...")
all_signals = list(col.find({}))
risk_updated = 0
for sig in all_signals:
    apply_risk_model(sig)          # gives sig new risk_* fields in-place
    col.update_one(
        {'_id': sig['_id']},
        {'$set': {
            'risk_adjusted_score': sig.get('risk_adjusted_score', 0),
            'risk_score':          sig.get('risk_score', 0),
            'risk_level':          sig.get('risk_level', ''),
            'volatility_penalty':  sig.get('volatility_penalty', 0),
            'drawdown_penalty':    sig.get('drawdown_penalty', 0),
            'liquidity_bonus':     sig.get('liquidity_bonus', 0),
            'risk_reward_ratio':   sig.get('risk_reward_ratio', 2.4),
        }}
    )
    risk_updated += 1
print(f"  Risk model applied to {risk_updated} signals")

# ── 3. Re-run ranking with fresh risk fields ───────────────────
print("\n[3] Refreshing ranked_opportunities with risk fields...")
ranked = run_ranking_from_db(db)
print(f"  Ranked {len(ranked)} coins → saved to ranked_opportunities")

# ── 4. Verify ──────────────────────────────────────────────────
print("\n[4] Verification:")
n_prob_down = col.count_documents({'probability_down': {'$exists': True}})
n_signals   = col.count_documents({})
n_risk      = col.count_documents({'risk_adjusted_score': {'$gt': 0}})
n_ranked    = db['ranked_opportunities'].count_documents({})

print(f"  ai_signals with probability_down : {n_prob_down}/{n_signals}")
print(f"  ai_signals with risk_adjusted_score > 0 : {n_risk}/{n_signals}")
print(f"  ranked_opportunities docs: {n_ranked}")

# Check RA values in ranked docs
sample_ranked = db['ranked_opportunities'].find_one(
    {}, sort=[('rank_score', -1)], projection={'_id': 0, 'symbol': 1,
    'rank_score': 1, 'risk_adjusted_score': 1, 'risk_level': 1, 'final_signal': 1}
)
if sample_ranked:
    print(f"  Top ranked: {sample_ranked}")

client.close()
print("\n  BACKFILL COMPLETE")
