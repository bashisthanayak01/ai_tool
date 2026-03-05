"""
News Sentiment Service
Fetches crypto news and calculates sentiment score
Uses simple keyword-based sentiment analysis
"""

import requests
import logging
from typing import Dict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sentiment keywords
POSITIVE_WORDS = [
    'bull', 'bullish', 'surge', 'rally', 'gain', 'profit', 'up', 'high', 'rise', 'soar',
    'breakout', 'pump', 'moon', 'rocket', 'green', 'buy', 'strong', 'growth', 'increase',
    'positive', 'upgrade', 'partnership', 'adoption', 'breakthrough', 'success'
]

NEGATIVE_WORDS = [
    'bear', 'bearish', 'crash', 'dump', 'down', 'low', 'fall', 'drop', 'sell', 'weak',
    'decline', 'loss', 'plunge', 'red', 'risk', 'fear', 'liquidation', 'hack', 'scam',
    'negative', 'downgrade', 'ban', 'regulation', 'lawsuit', 'fail', 'collapse'
]

# CryptoPanic API (free tier - no key needed for basic access)
CRYPTOPANIC_API = "https://cryptopanic.com/api/free/v1/posts/"


def get_news_sentiment(symbol: str) -> Dict:
    """
    Get news sentiment score for a symbol
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        
    Returns:
        Dictionary with news_score (-10 to +10) and news_count
    """
    try:
        # Extract base currency (BTC from BTCUSDT)
        currency = symbol.replace('USDT', '').replace('USD', '')
        
        # Try to fetch news (free API, no key required)
        try:
            params = {
                'currencies': currency,
                'public': 'true'
            }
            response = requests.get(CRYPTOPANIC_API, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    return analyze_news_sentiment(results[:10], currency)
        except:
            # If API fails, use fallback
            pass
        
        # Fallback: neutral sentiment
        return {
            'news_score': 0,
            'news_count': 0,
            'sentiment': 'NEUTRAL'
        }
        
    except Exception as e:
        logger.error(f"Error getting news sentiment for {symbol}: {e}")
        return {
            'news_score': 0,
            'news_count': 0,
            'sentiment': 'NEUTRAL'
        }


def analyze_news_sentiment(news_items: list, currency: str) -> Dict:
    """
    Analyze sentiment from news items
    
    Args:
        news_items: List of news items
        currency: Currency code
        
    Returns:
        Sentiment analysis dict
    """
    try:
        sentiment_sum = 0
        count = 0
        
        for item in news_items:
            title = item.get('title', '').lower()
            
            # Count positive and negative words
            positive_count = sum(1 for word in POSITIVE_WORDS if word in title)
            negative_count = sum(1 for word in NEGATIVE_WORDS if word in title)
            
            # Calculate sentiment for this item
            item_sentiment = positive_count - negative_count
            sentiment_sum += item_sentiment
            count += 1
        
        if count > 0:
            # Average sentiment
            avg_sentiment = sentiment_sum / count
            
            # Normalize to -10 to +10 scale
            news_score = max(-10, min(10, avg_sentiment * 3))
            news_score = round(news_score, 1)
            
            # Determine sentiment label
            if news_score >= 3:
                sentiment = 'POSITIVE'
            elif news_score <= -3:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'news_score': news_score,
                'news_count': count,
                'sentiment': sentiment
            }
        
        return {
            'news_score': 0,
            'news_count': 0,
            'sentiment': 'NEUTRAL'
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {
            'news_score': 0,
            'news_count': 0,
            'sentiment': 'NEUTRAL'
        }


if __name__ == "__main__":
    # Test
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in symbols:
        sentiment = get_news_sentiment(symbol)
        print(f"\n{symbol}:")
        print(f"  News Score: {sentiment['news_score']}/10")
        print(f"  Sentiment: {sentiment['sentiment']}")
        print(f"  News Count: {sentiment['news_count']}")
