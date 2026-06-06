"""
Market Scanner — CoinGecko Edition
Replaces Binance API (blocked from cloud servers) with CoinGecko API.

CoinGecko advantages:
  - Works from ANY server (GitHub Actions, HF Spaces, any cloud)
  - No geographic restrictions
  - Free tier: no API key needed
  - Reliable uptime

Function signatures are IDENTICAL to the old Binance version,
so no other file needs to change.

Interval mapping (Binance → CoinGecko days):
  1m, 5m     → 1 day  (30-min candles)
  15m, 30m   → 7 days (4-hour candles)
  1h, 2h     → 14 days (4-hour candles)
  4h         → 30 days (daily candles)
  1d, 3d, 1w → 90 days (daily candles)
"""

import requests
import logging
import time as _time
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COINGECKO_API  = "https://api.coingecko.com/api/v3"
REQUEST_DELAY  = 3.5   # seconds between calls (~17 calls/min, well under 30/min limit)

# Stablecoins to exclude from scan
_STABLES = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FRAX', 'GUSD',
             'USDD', 'LUSD', 'SUSD', 'CUSD', 'ALUSD', 'FDUSD', 'PYUSD'}

# ── Caches ────────────────────────────────────────────────────────────────────
_symbol_cache: List[str] = []
_symbol_cache_ts: float  = 0.0
_SYMBOL_CACHE_TTL        = 6 * 3600   # 6 hours

_id_map:      Dict[str, str]  = {}   # "BTCUSDT"  → "bitcoin"
_market_data: Dict[str, Dict] = {}   # "BTCUSDT"  → {price, volume, change…}


# ── Low-level HTTP helper ─────────────────────────────────────────────────────
def _cg_get(endpoint: str, params: dict = None, retries: int = 3) -> Optional[any]:
    """Rate-limited GET request to CoinGecko. Handles 429 automatically."""
    url = f"{COINGECKO_API}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, timeout=30)
            if resp.status_code == 429:
                wait = 65
                logger.warning(f"CoinGecko rate-limited — waiting {wait}s…")
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko request error ({endpoint}): {e}")
            if attempt < retries - 1:
                _time.sleep(10)
    return None


def _interval_to_days(interval: str) -> int:
    """Map Binance-style interval string to CoinGecko days parameter."""
    mapping = {
        '1m': 1,  '3m': 1,  '5m': 1,
        '15m': 7, '30m': 7,
        '1h': 14, '2h': 14,
        '4h': 30,
        '6h': 30, '8h': 30, '12h': 30,
        '1d': 90, '3d': 90, '1w': 365,
    }
    return mapping.get(interval, 90)


# ── Public API ────────────────────────────────────────────────────────────────
def get_top_symbols(limit: int = 90) -> List[str]:
    """
    Fetch top coins by 24h volume — cached for 6 hours.
    Returns Binance-style USDT pairs e.g. ['BTCUSDT', 'ETHUSDT', …]
    ONE API call fetches all coin data at once (very efficient).
    """
    global _symbol_cache, _symbol_cache_ts, _id_map, _market_data

    now       = _time.time()
    cache_age = now - _symbol_cache_ts

    if _symbol_cache and cache_age < _SYMBOL_CACHE_TTL:
        return _symbol_cache

    try:
        # Single call — returns top 250 coins with full market data
        data = _cg_get('/coins/markets', {
            'vs_currency': 'usd',
            'order': 'volume_desc',
            'per_page': 250,
            'page': 1,
            'sparkline': 'false',
            'price_change_percentage': '1h,24h,7d',
        })

        if not data:
            logger.warning("[Scanner] CoinGecko returned no data — using stale cache")
            return _symbol_cache or []

        symbols, new_id_map, new_market = [], {}, {}

        for coin in data:
            sym_raw = coin.get('symbol', '').upper()
            if not sym_raw or sym_raw in _STABLES:
                continue

            pair = sym_raw + 'USDT'
            symbols.append(pair)
            new_id_map[pair] = coin['id']
            new_market[pair] = {
                'price':      float(coin.get('current_price') or 0),
                'volume':     float(coin.get('total_volume')  or 0),
                'change_24h': float(coin.get('price_change_percentage_24h') or 0),
                'change_1h':  float(coin.get('price_change_percentage_1h_in_currency') or 0),
                'change_7d':  float(coin.get('price_change_percentage_7d_in_currency') or 0),
                'market_cap': float(coin.get('market_cap') or 0),
                'high_24h':   float(coin.get('high_24h') or 0),
                'low_24h':    float(coin.get('low_24h')  or 0),
            }

            if len(symbols) >= limit:
                break

        _symbol_cache    = symbols
        _symbol_cache_ts = now
        _id_map.update(new_id_map)
        _market_data.update(new_market)

        age_h = round(cache_age / 3600, 1)
        logger.info(
            f"[Scanner] Symbol list refreshed — {len(symbols)} coins "
            f"(previous list was {age_h}h old). Next refresh in 6h."
        )
        return symbols

    except Exception as e:
        logger.error(f"[Scanner] Error fetching top symbols: {e}")
        return _symbol_cache or []


def get_klines(symbol: str, interval: str = '1d', limit: int = 200) -> Optional[List[Dict]]:
    """
    Fetch OHLC candles from CoinGecko.
    Return format is IDENTICAL to the old Binance version so all
    downstream code (indicators, AI engine) works unchanged.
    """
    try:
        # Make sure we have the CoinGecko ID for this symbol
        if symbol not in _id_map:
            get_top_symbols()
        coin_id = _id_map.get(symbol)
        if not coin_id:
            logger.error(f"[Scanner] Unknown symbol {symbol} — not in CoinGecko map")
            return None

        days = _interval_to_days(interval)
        data = _cg_get(f'/coins/{coin_id}/ohlc', {'vs_currency': 'usd', 'days': days})

        _time.sleep(REQUEST_DELAY)   # Respect rate limit

        if not data:
            return None

        # CoinGecko OHLC format: [timestamp_ms, open, high, low, close]
        market = _market_data.get(symbol, {})
        volume = market.get('volume', 0)
        vol_per_candle = volume / max(len(data), 1)

        klines = []
        for candle in data:
            open_time = datetime.utcfromtimestamp(candle[0] / 1000)
            klines.append({
                'symbol':      symbol,
                'open_time':   open_time,
                'close_time':  open_time,
                'open':        float(candle[1]),
                'high':        float(candle[2]),
                'low':         float(candle[3]),
                'close':       float(candle[4]),
                'volume':      vol_per_candle,
                'quote_volume': volume,
                'trades':      0,
            })

        # Take the most recent `limit` candles
        klines = klines[-limit:]

        # Override last candle close with live price so dashboard shows current price
        live_price = market.get('price')
        if klines and live_price:
            klines[-1]['close'] = live_price
            klines[-1]['high']  = max(klines[-1]['high'], live_price)
            klines[-1]['low']   = min(klines[-1]['low'],  live_price)

        return klines

    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return None


def get_klines_since(symbol: str, start_time: datetime, interval: str = '1d') -> List[Dict]:
    """
    Fetch klines from start_time to now.
    Used for incremental historical data collection.
    """
    klines = get_klines(symbol, interval, limit=1000)
    if not klines:
        return []
    return [k for k in klines if k['open_time'] >= start_time]


def _parse_kline(symbol: str, k: list) -> Dict:
    """Parse a raw CoinGecko OHLC array into standard dict format."""
    ts = datetime.utcfromtimestamp(k[0] / 1000)
    return {
        'symbol':      symbol,
        'open_time':   ts,
        'close_time':  ts,
        'open':        float(k[1]),
        'high':        float(k[2]),
        'low':         float(k[3]),
        'close':       float(k[4]),
        'volume':      0,
        'quote_volume': 0,
        'trades':      0,
    }


if __name__ == "__main__":
    print("Testing CoinGecko scanner...")
    symbols = get_top_symbols(5)
    print(f"Top 5: {symbols}")
    if symbols:
        klines = get_klines(symbols[1], '1d', 10)
        if klines:
            print(f"Klines for {symbols[1]}: {len(klines)} candles")
            print(f"Latest: open={klines[-1]['open']:.4f} close={klines[-1]['close']:.4f}")
