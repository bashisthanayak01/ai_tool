"""
HF Spaces Scheduler Entry Point
Runs on Hugging Face Spaces (France EU servers - not blocked by Binance)

Two things run simultaneously:
1. scheduler.py main loop — scans Binance every 5 min, saves to MongoDB
2. Health check HTTP server on port 7860 — keeps HF Space alive (responds to UptimeRobot pings)
"""
import threading
import importlib.util
import os
import sys
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger('hf_scheduler')

# ── Validate MONGO_URI secret ────────────────────────────────────────────────
if not os.environ.get('MONGO_URI'):
    log.error("MONGO_URI secret not set! Go to Space Settings > Secrets and add MONGO_URI.")
    # Don't exit — keep health server running so the space stays alive
    # Scheduler will fail gracefully when it tries to connect

_start_time = datetime.utcnow()
_scan_count = 0

# ── Health check HTTP server ─────────────────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    """Simple health check — UptimeRobot pings this to keep HF Space awake."""
    def do_GET(self):
        uptime = datetime.utcnow() - _start_time
        hours = int(uptime.total_seconds() // 3600)
        mins  = int((uptime.total_seconds() % 3600) // 60)
        body = (
            f"Crypto AI Scheduler - RUNNING\n"
            f"Uptime: {hours}h {mins}m\n"
            f"Started: {_start_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Server: Hugging Face (EU - France)\n"
            f"Binance: Accessible (not blocked)\n"
        ).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs to keep logs clean


def start_health_server():
    log.info("[Health] Starting health check server on port 7860...")
    server = HTTPServer(('0.0.0.0', 7860), HealthHandler)
    log.info("[Health] Health server running — UptimeRobot can ping port 7860")
    server.serve_forever()


# ── Start health server in background ───────────────────────────────────────
health_thread = threading.Thread(target=start_health_server, daemon=True, name='HealthServer')
health_thread.start()

log.info("=" * 60)
log.info("CRYPTO AI SCHEDULER — Hugging Face Edition")
log.info("Location: EU France (Binance accessible)")
log.info("=" * 60)

# ── Load and run scheduler.py directly (bypasses scheduler/ package) ─────────
_root = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "scheduler_main",
    os.path.join(_root, "scheduler.py")
)
_sched_mod = importlib.util.module_from_spec(_spec)
sys.modules["scheduler_main"] = _sched_mod
_spec.loader.exec_module(_sched_mod)

# Run main — this blocks forever (APScheduler loop)
log.info("[Scheduler] Starting main scheduler loop...")
_sched_mod.main()
