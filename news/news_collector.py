"""
Multi-Source News AI — RSS feeds + CoinGecko, improved sentiment, DB-backed 30-min cache.
"""

import logging
import time
import hashlib
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import requests
import feedparser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── RSS Sources with credibility scores ──
RSS_FEEDS = {
    'CoinDesk':     ('https://www.coindesk.com/arc/outboundfeeds/rss/', 0.90),
    'CoinTelegraph': ('https://cointelegraph.com/rss', 0.85),
    'TheBlock':     ('https://www.theblock.co/rss.xml', 0.90),
    'CryptoSlate':  ('https://cryptoslate.com/feed/', 0.70),
}

# CoinGecko (free, no key)
COINGECKO_NEWS = "https://api.coingecko.com/api/v3/news"
COINGECKO_CREDIBILITY = 0.80

# ── Keyword weights (1-3 scale) ──
BULLISH_KEYWORDS = {
    'etf': 3, 'partnership': 2, 'adoption': 2, 'listing': 2, 'integration': 2,
    'institutional': 3, 'funding': 2, 'upgrade': 2, 'bullish': 2, 'breakout': 2,
    'surge': 2, 'rally': 2, 'approval': 3, 'launch': 1, 'milestone': 1,
    'record': 2, 'growth': 1, 'positive': 1, 'buy': 1, 'moon': 1,
    'all-time high': 3, 'ath': 3, 'halving': 2, 'mainnet': 2, 'staking': 1,
}

BEARISH_KEYWORDS = {
    'hack': 3, 'exploit': 3, 'ban': 3, 'lawsuit': 3, 'investigation': 3,
    'bearish': 2, 'crash': 3, 'liquidation': 2, 'bankruptcy': 3, 'fraud': 3,
    'scam': 3, 'dump': 2, 'plunge': 2, 'collapse': 3, 'decline': 1,
    'sell': 1, 'fear': 2, 'risk': 1, 'downgrade': 2, 'regulation': 1,
    'sec': 2, 'fine': 2, 'delisting': 3, 'vulnerability': 3, 'breach': 3,
}

# ── High-impact keywords that boost total score weight ──
IMPACT_KEYWORDS = {
    'etf': 1.5, 'regulation': 1.3, 'hack': 1.5, 'partnership': 1.2,
    'sec': 1.3, 'ban': 1.4, 'approval': 1.5, 'all-time high': 1.4,
    'ath': 1.4, 'bankruptcy': 1.5, 'exploit': 1.5, 'halving': 1.3,
    'institutional': 1.3, 'lawsuit': 1.4,
}

# ── Coin name mapping for filtering ──
COIN_NAMES = {
    'BTC': ['bitcoin', 'btc'], 'ETH': ['ethereum', 'eth', 'ether'],
    'SOL': ['solana', 'sol'], 'XRP': ['xrp', 'ripple'],
    'DOGE': ['dogecoin', 'doge'], 'ADA': ['cardano', 'ada'],
    'BNB': ['bnb', 'binance coin'], 'DOT': ['polkadot', 'dot'],
    'AVAX': ['avalanche', 'avax'], 'LINK': ['chainlink', 'link'],
    'MATIC': ['polygon', 'matic'], 'UNI': ['uniswap', 'uni'],
    'SUI': ['sui'], 'NEAR': ['near protocol', 'near'],
    'ATOM': ['cosmos', 'atom'], 'FIL': ['filecoin', 'fil'],
    'LTC': ['litecoin', 'ltc'], 'PEPE': ['pepe'],
    'SHIB': ['shiba', 'shib'], 'TRX': ['tron', 'trx'],
    'ARB': ['arbitrum', 'arb'], 'OP': ['optimism'],
    'FET': ['fetch.ai', 'fet'], 'RENDER': ['render'],
    'AAVE': ['aave'], 'TRUMP': ['trump'],
    'TON': ['toncoin', 'ton'], 'HBAR': ['hedera', 'hbar'],
}

# ── In-memory fast-path cache (10 min TTL) ──
_news_cache: Dict[str, dict] = {}
_MEMORY_CACHE_TTL = 600  # 10 minutes


def get_news_sentiment(symbol: str, db=None) -> Dict:
    """
    Get news sentiment for a symbol.

    Cache hierarchy:
      1. In-memory cache (10 min) — fastest path
      2. MongoDB DB cache (30 min) — avoids API calls
      3. Live RSS + CoinGecko fetch — slowest, updates DB

    Args:
        symbol: Trading pair e.g. 'BTCUSDT'
        db: MongoDB database object (optional, for DB caching)

    Returns:
        {news_score, sentiment, confidence, headline_count, news_available,
         last_news_time, top_headline, items}
    """
    currency = symbol.replace('USDT', '').replace('USD', '').upper()
    cache_key = currency

    # ── 1. In-memory cache check ──
    if cache_key in _news_cache:
        cached = _news_cache[cache_key]
        if time.time() - cached.get('_cached_at', 0) < _MEMORY_CACHE_TTL:
            return cached

    # ── 2. MongoDB DB cache check (30 min) — informational only, never blocks write ──
    # NOTE: We no longer return early from the DB cache. Instead we use it to
    # supplement results IF the live fetch yields 0 articles. This ensures new
    # articles are always written to the DB on every scheduler cycle.
    db_cached_result = None
    if db is not None:
        try:
            from config import settings as _settings
            from datetime import timedelta as _td
            cutoff = datetime.utcnow() - _td(minutes=_settings.NEWS_CACHE_MINUTES)
            fresh_count = db['news_data'].count_documents(
                {'symbol': currency, 'created_at': {'$gte': cutoff}}
            )
            if fresh_count > 0:
                cached_docs = list(db['news_data'].find(
                    {'symbol': currency}
                ).sort('published_at', -1).limit(20))
                if cached_docs:
                    db_cached_result = _build_result_from_db(cached_docs, currency)
                    db_cached_result['_source'] = 'db_cache'
                    logger.debug(f"{currency}: DB cache has {fresh_count} fresh items")
        except Exception as e:
            logger.debug(f"DB cache check failed for {currency}: {e}")

    # ── 3. Live fetch from APIs ──
    headlines = []

    for source, (url, credibility) in RSS_FEEDS.items():
        try:
            items = _fetch_rss(url, currency, source, credibility)
            headlines.extend(items)
        except Exception:
            pass

    try:
        cg_items = _fetch_coingecko(currency)
        headlines.extend(cg_items)
    except Exception as e:
        logger.debug(f"CoinGecko error: {e}")

    # Deduplicate
    headlines = _deduplicate(headlines)

    # Score
    if not headlines:
        # No live articles found — return DB cached result if available, else neutral
        if db_cached_result is not None:
            db_cached_result['_cached_at'] = time.time()
            _news_cache[cache_key] = db_cached_result
            return db_cached_result
        result = {
            'news_score': 0.0, 'sentiment': 'NEUTRAL', 'confidence': 0,
            'headline_count': 0, 'news_available': False,
            'last_news_time': None, 'top_headline': None, 'items': []
        }
    else:
        result = _score_headlines(headlines, currency)

    # ── Save live articles to MongoDB ──
    if db is not None and headlines:
        try:
            _save_news_to_db(db, headquarters=headlines)
        except Exception as e:
            logger.warning(f"News DB save failed for {currency}: {e}")

    result['_cached_at'] = time.time()
    result['_source'] = 'live'
    _news_cache[cache_key] = result
    return result


def _save_news_to_db(db, headquarters: List[Dict]):
    """Save news headlines to MongoDB news_data collection. created_at only set on insert."""
    now = datetime.utcnow()
    from config import settings as _settings
    col = db[_settings.COLLECTION_NEWS_DATA]
    inserted = 0
    skipped = 0
    for item in headquarters:
        try:
            title = item.get('headline', item.get('title', ''))
            symbol = item.get('symbol', '').replace('USDT', '')
            title_hash = hashlib.md5(f"{symbol}:{title.lower()}".encode()).hexdigest()

            pub_at = item.get('published_at', now)
            if isinstance(pub_at, str):
                try:
                    pub_at = datetime.fromisoformat(pub_at)
                except Exception:
                    pub_at = now

            doc = {
                'symbol': symbol,
                'title': title,
                'source': item.get('source', 'Unknown'),
                'url': item.get('url', ''),
                'published_at': pub_at,
                'sentiment_score': float(item.get('sentiment_score', 0.0)),
                'sentiment_label': _sentiment_label(item.get('sentiment_score', 0.0)),
                'keywords': item.get('matched_keywords', []),
                'impact_score': float(item.get('impact_score', 0.0)),
                'title_hash': title_hash,
                # created_at NOT in $set — only written once via $setOnInsert
            }
            result = col.update_one(
                {'symbol': symbol, 'title_hash': title_hash},
                {'$set': doc, '$setOnInsert': {'created_at': now}},
                upsert=True
            )
            if result.upserted_id:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"News insert error [{item.get('headline','?')[:40]}]: {e}")
            skipped += 1

    if inserted > 0:
        logger.info(f"News DB: +{inserted} new articles inserted, {skipped} duplicates skipped")
    else:
        logger.debug(f"News DB: 0 new (all {skipped} were duplicates)")


def _build_result_from_db(docs: List[Dict], currency: str) -> Dict:
    """Build sentiment result from DB-cached news documents"""
    if not docs:
        return {
            'news_score': 0.0, 'sentiment': 'NEUTRAL', 'confidence': 0,
            'headline_count': 0, 'news_available': False,
            'last_news_time': None, 'top_headline': None, 'items': []
        }

    items = []
    for doc in docs:
        items.append({
            'headline': doc.get('title', ''),
            'source': doc.get('source', ''),
            'published_at': doc.get('published_at', datetime.utcnow()).isoformat()
                            if isinstance(doc.get('published_at'), datetime)
                            else str(doc.get('published_at', '')),
            'sentiment_score': doc.get('sentiment_score', 0.0),
            'impact_score': doc.get('impact_score', 0.0),
        })

    # Weighted average of scores
    total_weight = 0.0
    total_score = 0.0
    for item in items:
        weight = max(abs(item['impact_score']), 0.1)
        total_score += item['sentiment_score'] * weight
        total_weight += weight

    avg_score = round(max(-1.0, min(1.0, total_score / total_weight if total_weight > 0 else 0.0)), 3)

    count = len(items)
    confidence = min(95, 50 + count * 5) if count >= 5 else (30 + count * 10 if count >= 2 else 15)
    sentiment = 'BULLISH' if avg_score >= 0.2 else ('BEARISH' if avg_score <= -0.2 else 'NEUTRAL')

    return {
        'news_score': avg_score,
        'sentiment': sentiment,
        'confidence': confidence,
        'headline_count': count,
        'news_available': True,
        'last_news_time': items[0].get('published_at') if items else None,
        'top_headline': items[0].get('headline', '')[:100] if items else None,
        'items': items,
    }


def _fetch_rss(url: str, currency: str, source: str, credibility: float = 0.8) -> List[Dict]:
    """Fetch and filter RSS feed headlines (with timeout)"""
    try:
        resp = requests.get(url, timeout=(2, 3), headers={'User-Agent': 'CryptoAI/1.0'})
        if resp.status_code != 200:
            return []
        feed = feedparser.parse(resp.text)
        items = []
        names = COIN_NAMES.get(currency.upper(), [currency.lower()])

        for entry in feed.entries[:30]:
            title = entry.get('title', '')
            title_lower = title.lower()

            if not any(n in title_lower for n in names):
                continue
            if _is_spam(title):
                continue

            published = entry.get('published_parsed')
            pub_time = datetime(*published[:6]) if published else datetime.utcnow()

            items.append({
                'symbol': f"{currency.upper()}USDT",
                'headline': title,
                'source': source,
                'url': entry.get('link', ''),
                'published_at': pub_time.isoformat(),
                'credibility': credibility,
            })

        return items
    except Exception:
        return []


def _fetch_coingecko(currency: str) -> List[Dict]:
    """Fetch news from CoinGecko free API"""
    try:
        resp = requests.get(COINGECKO_NEWS, timeout=5)
        if resp.status_code != 200:
            return []

        items = []
        data = resp.json().get('data', [])
        names = COIN_NAMES.get(currency.upper(), [currency.lower()])

        for article in data[:30]:
            title = article.get('title', '')
            title_lower = title.lower()

            if not any(n in title_lower for n in names):
                continue
            if _is_spam(title):
                continue

            items.append({
                'symbol': f"{currency.upper()}USDT",
                'headline': title,
                'source': 'CoinGecko',
                'url': article.get('url', ''),
                'published_at': article.get('updated_at', datetime.utcnow().isoformat()),
                'credibility': COINGECKO_CREDIBILITY,
            })

        return items
    except Exception:
        return []


def _is_spam(title: str) -> bool:
    """Filter spam/promotional headlines"""
    spam_patterns = ['sponsored', 'advertisement', 'promo', 'giveaway',
                     'airdrop claim', 'free tokens', 'click here']
    tl = title.lower()
    return any(s in tl for s in spam_patterns)


def _deduplicate(items: List[Dict]) -> List[Dict]:
    """Remove duplicate headlines by title hash"""
    seen = set()
    unique = []
    for item in items:
        h = hashlib.md5(item['headline'].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


def _sentiment_label(score) -> str:
    s = float(score or 0)
    if s >= 0.2:
        return 'POSITIVE'
    elif s <= -0.2:
        return 'NEGATIVE'
    return 'NEUTRAL'


def _score_headlines(headlines: List[Dict], currency: str) -> Dict:
    """
    Score headlines using improved formula:

        impact_score = sentiment_score*0.5 + credibility*0.2 + recency*0.2 + keyword_weight*0.1

    Returns full sentiment result dict.
    """
    now = datetime.utcnow()
    total_impact = 0.0
    total_weight = 0.0

    for item in headlines:
        text = item['headline'].lower()

        # ── Keyword sentiment (-1 to +1) ──
        bull = sum(w for kw, w in BULLISH_KEYWORDS.items() if kw in text)
        bear = sum(w for kw, w in BEARISH_KEYWORDS.items() if kw in text)
        raw_keyword_weight = max(bull, bear, 1)
        sentiment_score = (bull - bear) / max(bull + bear, 1)

        # ── Impact keyword multiplier ──
        impact_multiplier = 1.0
        matched = []
        for kw, mult in IMPACT_KEYWORDS.items():
            if kw in text:
                impact_multiplier = max(impact_multiplier, mult)
                matched.append(kw)

        # ── Source credibility (0-1) ──
        credibility = item.get('credibility', 0.75)

        # ── Recency score (0-1): decay over 48 hours ──
        pub_str = item.get('published_at', '')
        try:
            pub_dt = datetime.fromisoformat(pub_str) if pub_str else now
            hours_old = max(0, (now - pub_dt).total_seconds() / 3600)
        except:
            hours_old = 24
        recency = max(0.0, 1.0 - hours_old / 48.0)

        # ── Normalized keyword weight (0-1) ──
        kw_norm = min(1.0, raw_keyword_weight / 6.0)

        # ── Impact score formula ──
        impact_score = (
            sentiment_score * 0.5 +
            credibility * 0.2 +
            recency * 0.2 +
            kw_norm * 0.1
        ) * impact_multiplier

        # ── Update item with scores ──
        item['sentiment_score'] = round(sentiment_score, 3)
        item['impact_score'] = round(impact_score, 3)
        item['matched_keywords'] = matched

        # Weight by impact multiplier
        total_impact += sentiment_score * impact_multiplier
        total_weight += impact_multiplier

    # ── Aggregate ──
    if total_weight > 0:
        avg_score = total_impact / total_weight
    else:
        avg_score = 0.0

    news_score = round(max(-1.0, min(1.0, avg_score)), 3)

    count = len(headlines)
    confidence = min(95, 50 + count * 5) if count >= 5 else (30 + count * 10 if count >= 2 else 15)
    sentiment = 'BULLISH' if news_score >= 0.2 else ('BEARISH' if news_score <= -0.2 else 'NEUTRAL')

    headlines.sort(key=lambda x: x.get('published_at', ''), reverse=True)

    return {
        'news_score': news_score,
        'sentiment': sentiment,
        'confidence': confidence,
        'headline_count': count,
        'news_available': True,
        'last_news_time': headlines[0].get('published_at') if headlines else None,
        'top_headline': headlines[0].get('headline', '')[:100] if headlines else None,
        'items': headlines,
    }


if __name__ == "__main__":
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        r = get_news_sentiment(sym)
        print(f"\n{sym}:")
        print(f"  Score: {r['news_score']}, Sentiment: {r['sentiment']}")
        print(f"  Confidence: {r['confidence']}%, Headlines: {r['headline_count']}")
        print(f"  Source: {r.get('_source', 'live')}")
        if r.get('top_headline'):
            print(f"  Top: {r['top_headline']}")
