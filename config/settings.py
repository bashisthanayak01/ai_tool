"""
Configuration settings for the crypto AI tool
"""
import os

# MongoDB Configuration
# On GitHub Actions: reads from MONGO_URI secret (env var)
# On local laptop:   falls back to hardcoded value below
MONGO_URI = os.environ.get(
    'MONGO_URI',
    'mongodb+srv://bashisthanayak01_db_user:vnnX2ZFp8bJ6BJAV@cluster0.e1bkldu.mongodb.net/?appName=Cluster0'
)
DATABASE_NAME = "crypto_ai"

# Collections
COLLECTION_MARKET_DATA = "market_data"
COLLECTION_AI_SIGNALS = "ai_signals"
COLLECTION_NEWS_DATA = "news_data"
COLLECTION_RANKED_OPPORTUNITIES = "ranked_opportunities"


# Binance API
BINANCE_API_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"

# Scanner Settings
TOP_PAIRS_LIMIT = 90
HISTORICAL_DAYS = 90  # 3 months
KLINE_INTERVAL = "15m"
KLINE_LIMIT = 200
SCHEDULE_INTERVAL_MINUTES = 5

# Signal History Settings
SIGNAL_RETENTION_DAYS = 90          # Delete signals older than 90 days
SIGNAL_STORE_INTERVAL_HOURS = 1     # Store signal snapshot once per hour per coin
SIGNAL_TIMEFRAME = "5m"             # Timeframe label for stored signals

# News Cache Settings
NEWS_CACHE_MINUTES = 30             # Use DB news if fresher than 30 min
NEWS_RETENTION_DAYS = 30            # Delete news older than 30 days

# Logging
LOG_LEVEL = "INFO"

# Whale Intelligence Settings
WHALE_WEIGHT           = 10
COLLECTION_WHALE_DATA  = 'whale_data'
WHALE_SCAN_TOP_N       = 60   # expanded from 30
WHALE_LARGE_TRADE_USDT = 50_000
