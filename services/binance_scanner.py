"""
Binance Scanner — fetches top 90 USDT pairs and klines with incremental support
"""

import requests
import logging
import time as _time
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"

EXCLUDED_TOKENS = ['UP', 'DOWN', 'BULL', 'BEAR']


def get_top_symbols(limit: int = 90) -> List[str]:
    """Fetch top USDT pairs by 24h quote volume"""
    try:
        resp = requests.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=10)
        resp.raise_for_status()

        pairs = [
            t for t in resp.json()
            if t['symbol'].endswith('USDT')
            and not any(x in t['symbol'] for x in EXCLUDED_TOKENS)
        ]
        pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        symbols = [t['symbol'] for t in pairs[:limit]]
        logger.info(f"Fetched top {len(symbols)} USDT pairs")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching top symbols: {e}")
        return []


def get_klines(symbol: str, interval: str = '15m', limit: int = 200) -> Optional[List[Dict]]:
    """Fetch latest klines for a symbol"""
    try:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={'symbol': symbol, 'interval': interval, 'limit': limit},
            timeout=10
        )
        resp.raise_for_status()
        return [_parse_kline(symbol, k) for k in resp.json()]
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return None


def get_klines_since(symbol: str, start_time: datetime, interval: str = '1d') -> List[Dict]:
    """
    Fetch klines from start_time to now (paginated).
    Used for incremental historical data collection.
    """
    try:
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(datetime.utcnow().timestamp() * 1000)
        all_klines = []

        while start_ms < end_ms:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={
                    'symbol': symbol, 'interval': interval,
                    'startTime': start_ms, 'endTime': end_ms, 'limit': 1000
                },
                timeout=10
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break

            all_klines.extend([_parse_kline(symbol, k) for k in raw])
            start_ms = raw[-1][6] + 1  # past last close_time
            _time.sleep(0.1)

        return all_klines
    except Exception as e:
        logger.error(f"Error fetching incremental klines for {symbol}: {e}")
        return []


def _parse_kline(symbol: str, k: list) -> Dict:
    """Parse raw Binance kline array into dict"""
    return {
        'symbol': symbol,
        'open_time': datetime.fromtimestamp(k[0] / 1000),
        'close_time': datetime.fromtimestamp(k[6] / 1000),
        'open': float(k[1]),
        'high': float(k[2]),
        'low': float(k[3]),
        'close': float(k[4]),
        'volume': float(k[5]),
        'quote_volume': float(k[7]),
        'trades': int(k[8])
    }


if __name__ == "__main__":
    symbols = get_top_symbols(5)
    print(f"Top 5: {symbols}")
    if symbols:
        klines = get_klines(symbols[0])
        print(f"Klines: {len(klines)} candles for {symbols[0]}")
