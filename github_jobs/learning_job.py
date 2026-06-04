"""
GitHub Actions — Learning Job
Runs every 3 days via .github/workflows/learning.yml
"""
import os, sys, logging, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('learning_job')

if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set.")
    sys.exit(1)

log.info("=" * 60)
log.info("GITHUB ACTIONS: Learning Job starting...")
log.info("=" * 60)

# Load scheduler.py directly (avoids scheduler/ folder conflict)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("scheduler_main",
                                                os.path.join(_root, "scheduler.py"))
_sched = importlib.util.module_from_spec(_spec)
sys.modules["scheduler_main"] = _sched
_spec.loader.exec_module(_sched)

run_learning         = _sched.run_learning
run_optimization_job = _sched.run_optimization_job

try:
    log.info("[1/2] Running AI self-learning cycle...")
    run_learning()
    log.info("[1/2] AI learning done.")
except Exception as e:
    log.error(f"[1/2] Learning error: {e}")

try:
    log.info("[2/2] Running strategy optimization...")
    run_optimization_job()
    log.info("[2/2] Optimization done.")
except Exception as e:
    log.error(f"[2/2] Optimization error: {e}")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Learning Job complete.")
log.info("=" * 60)
