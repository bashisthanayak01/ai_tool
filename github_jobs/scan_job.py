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

# ── Test Binance connectivity first ─────────────────────────────────────────
log.info("Testing Binance API connectivity...")
try:
    import requests
    resp = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
    if resp.status_code == 200:
        log.info("Binance API: REACHABLE")
    elif resp.status_code == 451:
        log.error("Binance API: BLOCKED (451 - Geographic restriction - US IP blocked by Binance)")
        log.error("GitHub Actions uses US AWS servers which Binance blocks.")
        log.error("Signals cannot be collected. Consider switching to a non-US runner.")
        sys.exit(1)
    else:
        log.warning(f"Binance API: status={resp.status_code}")
except Exception as e:
    log.error(f"Binance API: UNREACHABLE - {e}")
    sys.exit(1)

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

# ── Check signals count before scan ─────────────────────────────────────────
import pymongo
from config import settings
from datetime import datetime, timedelta

def get_recent_signal_count(minutes=60):
    try:
        c = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        db = c[settings.DATABASE_NAME]
        cnt = db[settings.COLLECTION_AI_SIGNALS].count_documents(
            {'timestamp': {'$gte': datetime.utcnow() - timedelta(minutes=minutes)}}
        )
        c.close()
        return cnt
    except:
        return -1

before = get_recent_signal_count(60)
log.info(f"Signals in last 60 min BEFORE scan: {before}")

# ── Run all scan-cycle tasks ─────────────────────────────────────────────────
scan_success = False
try:
    log.info("[1/3] Running market scan (90 coins)...")
    run_scan()
    scan_success = True
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

# ── Verify signals were saved ────────────────────────────────────────────────
after = get_recent_signal_count(60)
log.info(f"Signals in last 60 min AFTER scan: {after}")

if after == 0 and before == 0:
    log.warning("NO signals saved in last 60 min - hourly save rule may be active (normal)")
elif after > 0:
    log.info(f"SUCCESS: {after} signals saved this scan cycle")
else:
    log.warning("Scan ran but signal count unchanged")

log.info("=" * 60)
log.info("GITHUB ACTIONS: Scan Job complete.")
log.info("=" * 60)
