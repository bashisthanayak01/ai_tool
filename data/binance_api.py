"""
Binance API module for fetching market data
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceAPI:
    """Binance API client for fetching cryptocurrency market data"""
    
    def __init__(self):
        self.base_url = settings.BINANCE_API_BASE_URL
        self.klines_endpoint = settings.BINANCE_KLINES_ENDPOINT
    
    def fetch_klines(
        self, 
        symbol: str = "BTCUSDT", 
        interval: str = "5m", 
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Fetch OHLCV (Kline/Candlestick) data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
            limit: Number of klines to fetch (max 1000)
            
        Returns:
            List of dictionaries containing OHLCV data, or None if error
        """
        try:
            url = f"{self.base_url}{self.klines_endpoint}"
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            logger.info(f"Fetching {limit} klines for {symbol} at {interval} interval...")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines_raw = response.json()
            
            # Parse klines data
            klines_data = []
            for kline in klines_raw:
                klines_data.append({
                    'symbol': symbol,
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'quote_asset_volume': float(kline[7]),
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10])
                })
            
            logger.info(f"Successfully fetched {len(klines_data)} klines for {symbol}")
            return klines_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        interval: str = "5m", 
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch klines data for multiple symbols
        
        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            limit: Number of klines to fetch per symbol
            
        Returns:
            Combined list of OHLCV data for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            data = self.fetch_klines(symbol, interval, limit)
            if data:
                all_data.extend(data)
        
        logger.info(f"Fetched total of {len(all_data)} klines for {len(symbols)} symbols")
        return all_data
    
    def fetch_top_usdt_pairs(self, limit=90):
        """
        Fetch top USDT trading pairs sorted by 24hr volume
        
        Args:
            limit: Number of top pairs to return (default: 90)
            
        Returns:
            List of trading pair symbols sorted by volume
        """
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            
            logger.info(f"Fetching top {limit} USDT pairs by volume...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            tickers = response.json()
            
            # Filter USDT pairs only
            usdt_pairs = [
                ticker for ticker in tickers 
                if ticker['symbol'].endswith('USDT') and 
                not any(excluded in ticker['symbol'] for excluded in ['UP', 'DOWN', 'BULL', 'BEAR'])
            ]
            
            # Sort by quote volume (descending)
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            
            # Get top N symbols
            top_pairs = [ticker['symbol'] for ticker in usdt_pairs[:limit]]
            
            logger.info(f"Successfully fetched {len(top_pairs)} top USDT pairs")
            logger.info(f"Top 5: {', '.join(top_pairs[:5])}")
            
            return top_pairs
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top USDT pairs: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching top USDT pairs: {e}")
            return []
    
    def fetch_historical_klines(self, symbol, interval='1d', days=180):
        """
        Fetch historical klines data (up to 6 months)
        Handles Binance 1000-record limit with pagination
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (default: '1d' for daily)
            days: Number of days of historical data (default: 180)
            
        Returns:
            List of historical OHLCV data
        """
        try:
            url = f"{self.base_url}{self.klines_endpoint}"
            
            # Calculate timestamps
            from datetime import datetime, timedelta
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            logger.info(f"Fetching {days} days of historical data for {symbol}...")
            
            all_klines = []
            current_start = start_ms
            
            # Binance limit is 1000 records per request
            while current_start < end_ms:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 1000
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                klines_raw = response.json()
                
                if not klines_raw:
                    break
                
                # Parse klines
                for kline in klines_raw:
                    all_klines.append({
                        'symbol': symbol,
                        'open_time': datetime.fromtimestamp(kline[0] / 1000),
                        'close_time': datetime.fromtimestamp(kline[6] / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'quote_asset_volume': float(kline[7]),
                        'number_of_trades': int(kline[8]),
                        'taker_buy_base_volume': float(kline[9]),
                        'taker_buy_quote_volume': float(kline[10])
                    })
                
                # Move to next batch
                current_start = klines_raw[-1][6] + 1  # Last close time + 1ms
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.1)
            
            logger.info(f"Successfully fetched {len(all_klines)} historical candles for {symbol}")
            return all_klines
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {symbol}: {e}")
            return []


# Singleton instance
binance_api = BinanceAPI()
