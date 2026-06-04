"""
GitHub Actions — Learning Job
Runs every 3 days via .github/workflows/learning.yml

What it does:
  1. AI self-learning cycle (analyse trades, adjust model weights)
  2. Strategy optimization (grid search best TP/SL/score parameters)
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('learning_job')

if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set.")
    sys.exit(1)

log.info("=" * 60)
log.info("GITHUB ACTIONS: Learning Job starting...")
log.info("=" * 60)

from scheduler import run_learning, run_optimization_job

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
