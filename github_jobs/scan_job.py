"""
GitHub Actions — Market Scan Job
Runs every 15 minutes via .github/workflows/scan.yml

What it does (same as scheduler, but single-shot):
  1. Full market scan (90 coins) → AI signals → save to MongoDB
  2. Ranking engine → update ranked_opportunities
  3. Whale intelligence scan (60 symbols)
  4. Conviction picks update
  5. Paper trading check
"""
import os, sys, logging
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

# Import after path is set — settings.py will read MONGO_URI from env
from scheduler import run_scan, run_whale_scan_job, run_conviction_update

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
