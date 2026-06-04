"""
GitHub Actions — Daily Job
Runs every 24 hours via .github/workflows/daily.yml
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
log.info("=" * 60)

# Load scheduler.py directly (avoids scheduler/ folder conflict)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("scheduler_main",
                                                os.path.join(_root, "scheduler.py"))
_sched = importlib.util.module_from_spec(_spec)
sys.modules["scheduler_main"] = _sched
_spec.loader.exec_module(_sched)

save_regime_snapshot = _sched.save_regime_snapshot
run_rl_learning_job  = _sched.run_rl_learning_job
cleanup_old_data     = _sched.cleanup_old_data

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
    log.info("[3/3] Running data cleanup...")
    cleanup_old_data()
    log.info("[3/3] Cleanup done.")
except Exception as e:
    log.error(f"[3/3] Cleanup error: {e}")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Daily Job complete.")
log.info("=" * 60)
