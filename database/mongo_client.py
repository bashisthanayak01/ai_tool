"""
MongoDB client — production-grade with upsert, signal history, news persistence
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime, timedelta
import logging
import hashlib
from typing import Dict, List, Optional

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self) -> bool:
        try:
            self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[settings.DATABASE_NAME]
            logger.info(f"Connected to MongoDB: {settings.DATABASE_NAME}")
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"MongoDB error: {e}")
            return False

    # Alias for backward compatibility
    def connect_to_mongo(self) -> bool:
        return self.connect()

    def reset_database(self):
        """Drop market_data, ai_signals, news_data collections"""
        if self.db is None:
            logger.error("Not connected")
            return False
        try:
            for col_name in [settings.COLLECTION_MARKET_DATA,
                             settings.COLLECTION_AI_SIGNALS,
                             settings.COLLECTION_NEWS_DATA]:
                self.db.drop_collection(col_name)
                logger.info(f"Dropped collection: {col_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def setup_indexes(self):
        """
        Create/update indexes for dedup and fast queries.
        Each index is created independently — one failure won't block the others.
        """
        if self.db is None:
            return False

        created = 0
        skipped = 0

        def _safe_create(col, keys, **kwargs):
            nonlocal created, skipped
            name = kwargs.get('name', 'unnamed')
            try:
                col.create_index(keys, **kwargs)
                created += 1
            except Exception as e:
                msg = str(e)
                if 'already exists' in msg or 'IndexOptionsConflict' in msg:
                    skipped += 1  # Already exists with correct options — OK
                else:
                    logger.warning(f"Index '{name}' warning: {msg}")
                    skipped += 1

        def _safe_drop(col, name):
            try:
                col.drop_index(name)
            except Exception:
                pass

        md = self.db[settings.COLLECTION_MARKET_DATA]
        sig = self.db[settings.COLLECTION_AI_SIGNALS]
        news = self.db[settings.COLLECTION_NEWS_DATA]

        # ── market_data: unique (symbol, open_time) ──
        _safe_create(md,
            [('symbol', ASCENDING), ('open_time', ASCENDING)],
            unique=True, name='idx_symbol_opentime')

        # ── ai_signals: drop old non-unique index first ──
        _safe_drop(sig, 'idx_symbol_timestamp')

        # Unique history index: (symbol, timestamp, timeframe)
        _safe_create(sig,
            [('symbol', ASCENDING), ('timestamp', ASCENDING), ('timeframe', ASCENDING)],
            unique=True, name='idx_signal_unique')

        # Fast lookup for aggregation pipeline
        _safe_create(sig,
            [('symbol', ASCENDING), ('timestamp', DESCENDING)],
            name='idx_signal_lookup')

        # ── news_data: unique (symbol, title_hash) ──
        _safe_create(news,
            [('symbol', ASCENDING), ('title_hash', ASCENDING)],
            unique=True, name='idx_news_unique')

        # Lookup by symbol + published_at
        _safe_create(news,
            [('symbol', ASCENDING), ('published_at', DESCENDING)],
            name='idx_news_lookup')

        # TTL index on created_at — drop old conflicting names first
        _safe_drop(news, 'idx_symbol_published')
        _safe_drop(news, 'idx_news_published')
        _safe_create(news,
            [('created_at', ASCENDING)],
            expireAfterSeconds=settings.NEWS_RETENTION_DAYS * 86400,
            name='idx_news_ttl')

        logger.info(f"Indexes ready: {created} created, {skipped} already existed")
        return True


    # Alias
    def create_indexes(self):
        return self.setup_indexes()

    # ══════════════════════════════════════════════
    # MARKET DATA
    # ══════════════════════════════════════════════

    def get_last_candle_time(self, symbol: str) -> Optional[datetime]:
        """Get last stored candle timestamp for a symbol"""
        if self.db is None:
            return None
        try:
            doc = self.db[settings.COLLECTION_MARKET_DATA].find_one(
                {'symbol': symbol},
                sort=[('open_time', DESCENDING)]
            )
            return doc['open_time'] if doc else None
        except Exception as e:
            logger.error(f"Error getting last candle for {symbol}: {e}")
            return None

    def get_market_data_count(self) -> int:
        """Count total market_data documents"""
        if self.db is None:
            return 0
        try:
            return self.db[settings.COLLECTION_MARKET_DATA].count_documents({})
        except:
            return 0

    def upsert_market_data(self, candles: List[Dict]) -> Dict:
        """
        Upsert candles using (symbol, open_time) as unique key.
        Returns: {inserted, updated, skipped}
        """
        if self.db is None:
            return {'inserted': 0, 'updated': 0, 'skipped': 0}

        stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
        if not candles:
            return stats

        col = self.db[settings.COLLECTION_MARKET_DATA]
        for c in candles:
            try:
                result = col.update_one(
                    {'symbol': c['symbol'], 'open_time': c['open_time']},
                    {'$set': {**c, 'updated_at': datetime.utcnow()}},
                    upsert=True
                )
                if result.upserted_id:
                    stats['inserted'] += 1
                elif result.modified_count > 0:
                    stats['updated'] += 1
                else:
                    stats['skipped'] += 1
            except Exception:
                stats['skipped'] += 1

        logger.info(f"Upsert: +{stats['inserted']} new, ~{stats['updated']} updated, ={stats['skipped']} skipped")
        return stats

    # Backward compat
    def insert_market_data(self, data: List[Dict]) -> bool:
        r = self.upsert_market_data(data)
        return (r['inserted'] + r['updated']) > 0

    # ══════════════════════════════════════════════
    # AI SIGNALS — HISTORICAL STORAGE
    # ══════════════════════════════════════════════

    def upsert_ai_signals(self, signals: List[Dict]) -> Dict:
        """
        Upsert AI signals using (symbol, timestamp, timeframe) as unique key.
        This APPENDS history — does NOT overwrite.
        Returns: {inserted, updated, skipped}
        """
        if self.db is None or not signals:
            return {'inserted': 0, 'updated': 0, 'skipped': 0}

        stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
        col = self.db[settings.COLLECTION_AI_SIGNALS]

        for s in signals:
            try:
                # Build the unique filter key
                filter_key = {
                    'symbol': s['symbol'],
                    'timestamp': s.get('timestamp', datetime.utcnow()),
                    'timeframe': s.get('timeframe', settings.SIGNAL_TIMEFRAME),
                }
                doc = {k: v for k, v in s.items() if k != '_id'}
                doc.setdefault('timeframe', settings.SIGNAL_TIMEFRAME)
                doc.setdefault('inserted_at', datetime.utcnow())

                result = col.update_one(
                    filter_key,
                    {'$set': doc, '$setOnInsert': {'created_at': datetime.utcnow()}},
                    upsert=True
                )
                if result.upserted_id:
                    stats['inserted'] += 1
                elif result.modified_count > 0:
                    stats['updated'] += 1
                else:
                    stats['skipped'] += 1
            except DuplicateKeyError:
                stats['skipped'] += 1
            except Exception as e:
                logger.error(f"Signal upsert error for {s.get('symbol')}: {e}")
                stats['skipped'] += 1

        logger.info(f"AI Signals: +{stats['inserted']} new, ~{stats['updated']} updated, ={stats['skipped']} skipped")
        return stats

    def save_ai_signals(self, signals: List[Dict]) -> bool:
        """Backward-compatible alias — now appends history via upsert"""
        r = self.upsert_ai_signals(signals)
        return (r['inserted'] + r['updated']) > 0

    # Alias
    def insert_ai_signals(self, data: List[Dict]) -> bool:
        return self.save_ai_signals(data)

    def get_latest_signals(self, limit: int = 90) -> List[Dict]:
        """
        Get the most recent signal snapshot per symbol (latest timestamp per symbol).
        Used by dashboard to show current state.
        """
        if self.db is None:
            return []
        try:
            # idx_signal_lookup index handles sort without RAM limit
            pipeline = [
                {'$sort': {'timestamp': -1}},
                {'$group': {
                    '_id': '$symbol',
                    'doc': {'$first': '$$ROOT'}
                }},
                {'$replaceRoot': {'newRoot': '$doc'}},
                {'$sort': {'final_score': -1}},
                {'$limit': limit}
            ]
            return list(self.db[settings.COLLECTION_AI_SIGNALS].aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error getting latest signals: {e}")
            return []

    def get_top_signals(self, limit: int = 90) -> List[Dict]:
        """Get latest signals sorted by final_score — used by dashboard"""
        return self.get_latest_signals(limit)

    def get_signal_count(self) -> int:
        """Total number of stored AI signals (all history)"""
        if self.db is None:
            return 0
        try:
            return self.db[settings.COLLECTION_AI_SIGNALS].count_documents({})
        except:
            return 0

    def get_symbol_signal_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get signal history for a specific symbol over N days"""
        if self.db is None:
            return []
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            cursor = self.db[settings.COLLECTION_AI_SIGNALS].find(
                {'symbol': symbol, 'timestamp': {'$gte': cutoff}}
            ).sort('timestamp', ASCENDING)
            return list(cursor)
        except:
            return []

    def cleanup_old_signals(self, days: int = None) -> int:
        """Delete AI signals older than N days. Returns count deleted."""
        if self.db is None:
            return 0
        if days is None:
            days = settings.SIGNAL_RETENTION_DAYS
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            result = self.db[settings.COLLECTION_AI_SIGNALS].delete_many(
                {'timestamp': {'$lt': cutoff}}
            )
            deleted = result.deleted_count
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} AI signals older than {days} days")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up signals: {e}")
            return 0

    # ══════════════════════════════════════════════
    # NEWS DATA — PERSISTENT STORAGE
    # ══════════════════════════════════════════════

    def save_news_items(self, news_items: List[Dict]) -> Dict:
        """
        Upsert news items by (symbol, title_hash) to prevent duplicates.
        Returns: {inserted, skipped}
        """
        if self.db is None or not news_items:
            return {'inserted': 0, 'skipped': 0}

        stats = {'inserted': 0, 'skipped': 0}
        col = self.db[settings.COLLECTION_NEWS_DATA]
        now = datetime.utcnow()

        for item in news_items:
            try:
                # Generate hash for deduplication
                title = item.get('title', item.get('headline', ''))
                symbol = item.get('symbol', '')
                title_hash = hashlib.md5(f"{symbol}:{title.lower()}".encode()).hexdigest()

                doc = {
                    'symbol': symbol,
                    'title': title,
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'published_at': _parse_datetime(item.get('published_at')),
                    'sentiment_score': float(item.get('sentiment_score', 0.0)),
                    'sentiment_label': _sentiment_label(item.get('sentiment_score', 0.0)),
                    'keywords': item.get('keywords', []),
                    'impact_score': float(item.get('impact_score', 0.0)),
                    'title_hash': title_hash,
                    # Note: created_at is NOT here — only set on first insert via $setOnInsert
                }

                result = col.update_one(
                    {'symbol': symbol, 'title_hash': title_hash},
                    {'$set': doc, '$setOnInsert': {'created_at': now}},
                    upsert=True
                )
                if result.upserted_id:
                    stats['inserted'] += 1
                else:
                    stats['skipped'] += 1
            except DuplicateKeyError:
                stats['skipped'] += 1
            except Exception as e:
                logger.error(f"News upsert error: {e}")
                stats['skipped'] += 1

        if stats['inserted'] > 0:
            logger.info(f"News: +{stats['inserted']} new, ={stats['skipped']} duplicates skipped")
        return stats

    def save_news(self, news_items: List[Dict]) -> int:
        """Backward-compatible alias"""
        r = self.save_news_items(news_items)
        return r['inserted']

    def get_cached_news(self, symbol: str, max_age_minutes: int = None) -> Optional[List[Dict]]:
        """
        Return DB-cached news for a symbol if fresh (< max_age_minutes old).
        Returns None if cache is stale or empty.
        """
        if self.db is None:
            return None
        if max_age_minutes is None:
            max_age_minutes = settings.NEWS_CACHE_MINUTES
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
            # Check if we have any news stored within the cache window
            count = self.db[settings.COLLECTION_NEWS_DATA].count_documents(
                {'symbol': symbol, 'created_at': {'$gte': cutoff}}
            )
            if count == 0:
                return None  # Cache miss — need to fetch

            cursor = self.db[settings.COLLECTION_NEWS_DATA].find(
                {'symbol': symbol}
            ).sort('published_at', DESCENDING).limit(20)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting cached news for {symbol}: {e}")
            return None

    def get_recent_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent news for a symbol"""
        if self.db is None:
            return []
        try:
            cursor = self.db[settings.COLLECTION_NEWS_DATA].find(
                {'symbol': symbol}
            ).sort('published_at', DESCENDING).limit(limit)
            return list(cursor)
        except:
            return []

    def get_news_count(self) -> int:
        """Total number of stored news items"""
        if self.db is None:
            return 0
        try:
            return self.db[settings.COLLECTION_NEWS_DATA].count_documents({})
        except:
            return 0

    def cleanup_old_news(self, days: int = None) -> int:
        """Delete news older than N days (manual cleanup, TTL handles it automatically)"""
        if self.db is None:
            return 0
        if days is None:
            days = settings.NEWS_RETENTION_DAYS
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            result = self.db[settings.COLLECTION_NEWS_DATA].delete_many(
                {'created_at': {'$lt': cutoff}}
            )
            return result.deleted_count
        except:
            return 0

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols that have market data"""
        if self.db is None:
            return []
        try:
            return self.db[settings.COLLECTION_MARKET_DATA].distinct('symbol')
        except:
            return []

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    # Alias
    def close_connection(self):
        self.close()


def _parse_datetime(val) -> datetime:
    """Parse datetime from string or return now"""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00').rstrip('+00:00') or val)
        except:
            pass
    return datetime.utcnow()


def _sentiment_label(score) -> str:
    s = float(score or 0)
    if s >= 0.2:
        return 'POSITIVE'
    elif s <= -0.2:
        return 'NEGATIVE'
    return 'NEUTRAL'


# Singleton
mongo_client = MongoDBClient()
