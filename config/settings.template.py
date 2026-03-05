"""
config/settings.py — LOCAL CONFIGURATION (not committed to git)

Copy this file as `settings.py` and fill in your own credentials.
The real `settings.py` is excluded via .gitignore.
"""

# MongoDB Configuration — replace with your Atlas URI
MONGO_URI = "mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?appName=Cluster0"
DATABASE_NAME = "crypto_ai"

# Collections
COLLECTION_MARKET_DATA           = "market_data"
COLLECTION_AI_SIGNALS            = "ai_signals"
COLLECTION_NEWS_DATA             = "news_data"
COLLECTION_RANKED_OPPORTUNITIES  = "ranked_opportunities"

# Binance API (no key needed — public endpoints only)
BINANCE_API_BASE_URL     = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT  = "/api/v3/klines"

# Scanner Settings
TOP_PAIRS_LIMIT     = 90
HISTORICAL_DAYS     = 90       # 3 months
KLINE_INTERVAL      = "15m"
KLINE_LIMIT         = 200
SCHEDULE_INTERVAL_MINUTES = 5

# Signal History Settings
SIGNAL_RETENTION_DAYS       = 90
SIGNAL_STORE_INTERVAL_HOURS = 1
SIGNAL_TIMEFRAME            = "5m"

# News Cache Settings
NEWS_CACHE_MINUTES   = 30
NEWS_RETENTION_DAYS  = 30

# Logging
LOG_LEVEL = "INFO"

# Whale Intelligence Settings
WHALE_WEIGHT           = 10
COLLECTION_WHALE_DATA  = "whale_data"
WHALE_SCAN_TOP_N       = 30
WHALE_LARGE_TRADE_USDT = 50_000
