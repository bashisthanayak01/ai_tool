"""
GitHub Actions — Daily Job
Runs every 24 hours via .github/workflows/daily.yml

NOTE: Only runs tasks that do NOT need Binance API.
Binance is blocked from GitHub Actions US servers (451 error).
Regime snapshot (needs Binance BTC price) runs on local laptop via scheduler.py instead.

What this job does:
  1. RL reinforcement learning (uses MongoDB trade data only)
  2. Data cleanup (deletes old signals from MongoDB only)
"""
import os, sys, logging, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('daily_job')

if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set.")
    sys.exit(1)

log.info("=" * 60)
log.info("GITHUB ACTIONS: Daily Job starting...")
log.info("(Regime snapshot skipped - needs Binance, blocked from US IPs)")
log.info("=" * 60)

# Load scheduler.py directly (avoids scheduler/ folder conflict)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("scheduler_main",
                                                os.path.join(_root, "scheduler.py"))
_sched = importlib.util.module_from_spec(_spec)
sys.modules["scheduler_main"] = _sched
_spec.loader.exec_module(_sched)

run_rl_learning_job  = _sched.run_rl_learning_job
cleanup_old_data     = _sched.cleanup_old_data

try:
    log.info("[1/2] Running RL learning cycle (MongoDB data only)...")
    run_rl_learning_job()
    log.info("[1/2] RL learning done.")
except Exception as e:
    log.error(f"[1/2] RL error: {e}")

try:
    log.info("[2/2] Running data cleanup...")
    cleanup_old_data()
    log.info("[2/2] Cleanup done.")
except Exception as e:
    log.error(f"[2/2] Cleanup error: {e}")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Daily Job complete.")
log.info("=" * 60)
