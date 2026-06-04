"""
GitHub Actions — Daily Job
Runs every 24 hours via .github/workflows/daily.yml

What it does:
  1. Market regime snapshot (BTC trend detection)
  2. RL reinforcement learning parameter update
  3. Data cleanup (remove signals > 100 days old)
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('daily_job')

if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set.")
    sys.exit(1)

log.info("=" * 60)
log.info("GITHUB ACTIONS: Daily Job starting...")
log.info("=" * 60)

from scheduler import save_regime_snapshot, run_rl_learning_job, cleanup_old_data

try:
    log.info("[1/3] Saving market regime snapshot...")
    save_regime_snapshot()
    log.info("[1/3] Regime snapshot done.")
except Exception as e:
    log.error(f"[1/3] Regime error: {e}")

try:
    log.info("[2/3] Running RL learning cycle...")
    run_rl_learning_job()
    log.info("[2/3] RL learning done.")
except Exception as e:
    log.error(f"[2/3] RL error: {e}")

try:
    log.info("[3/3] Running data cleanup (keep 100 days)...")
    cleanup_old_data()
    log.info("[3/3] Cleanup done.")
except Exception as e:
    log.error(f"[3/3] Cleanup error: {e}")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Daily Job complete.")
log.info("=" * 60)
