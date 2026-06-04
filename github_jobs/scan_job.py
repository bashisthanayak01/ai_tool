"""
GitHub Actions — Market Scan Job
Runs every 15 minutes via .github/workflows/scan.yml

Uses importlib to load scheduler.py directly by path,
bypassing the scheduler/ folder (package naming conflict).
"""
import os, sys, logging, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('scan_job')

# ── Validate secret ──────────────────────────────────────────────────────────
if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set in GitHub Actions. Add it under Settings > Secrets.")
    sys.exit(1)

log.info("=" * 60)
log.info("GITHUB ACTIONS: Market Scan Job starting...")
log.info("=" * 60)

# ── Load scheduler.py directly (avoids scheduler/ folder conflict) ───────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("scheduler_main",
                                                os.path.join(_root, "scheduler.py"))
_sched = importlib.util.module_from_spec(_spec)
sys.modules["scheduler_main"] = _sched
_spec.loader.exec_module(_sched)

run_scan            = _sched.run_scan
run_whale_scan_job  = _sched.run_whale_scan_job
run_conviction_update = _sched.run_conviction_update

# ── Run all scan-cycle tasks ─────────────────────────────────────────────────
try:
    log.info("[1/3] Running market scan (90 coins)...")
    run_scan()
    log.info("[1/3] Market scan done.")
except Exception as e:
    log.error(f"[1/3] Scan error: {e}")

try:
    log.info("[2/3] Running whale scan (60 symbols)...")
    run_whale_scan_job()
    log.info("[2/3] Whale scan done.")
except Exception as e:
    log.error(f"[2/3] Whale error: {e}")

try:
    log.info("[3/3] Updating conviction picks...")
    run_conviction_update()
    log.info("[3/3] Conviction update done.")
except Exception as e:
    log.error(f"[3/3] Conviction error: {e}")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Scan Job complete.")
log.info("=" * 60)
