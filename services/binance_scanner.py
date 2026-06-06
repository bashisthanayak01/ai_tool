"""
Market Scanner v3 — Hybrid CoinGecko + Yahoo Finance
=====================================================

Data sources (both work from ANY cloud server):
  - CoinGecko /coins/markets : top coins list + current price/volume (1 API call)
  - Yahoo Finance (yfinance)  : OHLCV historical data (no rate limits, unlimited)

Why this combo:
  - CoinGecko OHLCV endpoint has very strict rate limits (~4 calls/min free)
    → 90 coins would require 22+ minutes with delays
  - Yahoo Finance has no documented rate limit and supports batch downloads
    → 90 coins in ~10 seconds

Function signatures are IDENTICAL to the old Binance version.
No other file needs to change.
"""

import requests
import logging
import time as _time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COINGECKO_API = "https://api.coingecko.com/api/v3"

# Stablecoins to exclude
_STABLES = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FRAX', 'GUSD',
             'USDD', 'LUSD', 'SUSD', 'CUSD', 'ALUSD', 'FDUSD', 'PYUSD'}

# ── Caches ────────────────────────────────────────────────────────────────────
_symbol_cache: List[str] = []
_symbol_cache_ts: float  = 0.0
_SYMBOL_CACHE_TTL        = 6 * 3600   # 6 hours

_id_map:      Dict[str, str]  = {}    # "BTCUSDT" → "bitcoin"
_market_data: Dict[str, Dict] = {}    # "BTCUSDT" → {price, volume, change…}

_ohlcv_cache: Dict[str, Dict] = {}    # "BTCUSDT|1d" → {data, ts}
_OHLCV_CACHE_TTL = 60 * 60            # 60 minutes (daily candles barely change)


# ── CoinGecko helper ──────────────────────────────────────────────────────────
def _cg_get(endpoint: str, params: dict = None, retries: int = 3) -> Optional[any]:
    """Rate-limited GET to CoinGecko. Handles 429 automatically."""
    url = f"{COINGECKO_API}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, timeout=30)
            if resp.status_code == 429:
                logger.warning("CoinGecko rate-limited — waiting 65s…")
                _time.sleep(65)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko request error ({endpoint}): {e}")
            if attempt < retries - 1:
                _time.sleep(10)
    return None


# ── Yahoo Finance OHLCV ───────────────────────────────────────────────────────
def _interval_to_yf(interval: str) -> tuple:
    """
    Map Binance interval to (yfinance_interval, yfinance_period).
    Returns (yf_interval, yf_period) suitable for yf.download().
    """
    mapping = {
        '1m':  ('1m',  '7d'),
        '5m':  ('5m',  '60d'),
        '15m': ('15m', '60d'),
        '30m': ('30m', '60d'),
        '1h':  ('1h',  '730d'),
        '2h':  ('1h',  '730d'),   # yf has no 2h; use 1h
        '4h':  ('1h',  '730d'),   # yf has no 4h; use 1h
        '1d':  ('1d',  '730d'),
        '3d':  ('1d',  '730d'),
        '1w':  ('1wk', '730d'),
    }
    return mapping.get(interval, ('1d', '730d'))


def _symbol_to_yf(symbol: str) -> str:
    """Convert 'BTCUSDT' → 'BTC-USD' for Yahoo Finance."""
    base = symbol.replace('USDT', '').replace('BUSD', '')

    return f"{base}-USD"


def _fetch_yf_batch(symbols: List[str], interval: str = '1d',
                    limit: int = 200) -> Dict[str, List[Dict]]:
    """
    Batch-download OHLCV for multiple symbols from Yahoo Finance.
    Handles yfinance's MultiIndex column format (new in 0.2.x).
    Returns {symbol: [kline_dicts]}
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        logger.error("[YF] yfinance not installed — run: pip install yfinance")
        return {}

    yf_interval, yf_period = _interval_to_yf(interval)
    yf_tickers = [_symbol_to_yf(s) for s in symbols]
    yf_map = {_symbol_to_yf(s): s for s in symbols}  # "BTC-USD" → "BTCUSDT"

    try:
        data = yf.download(
            tickers=yf_tickers,
            period=yf_period,
            interval=yf_interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            logger.warning("[YF] Empty response from Yahoo Finance")
            return {}

        result = {}

        for ticker in yf_tickers:
            sym = yf_map.get(ticker)
            if not sym:
                continue
            try:
                # yfinance 0.2.x always uses MultiIndex columns: ('Close', 'BTC-USD')
                # Extract per-ticker slice and flatten to simple column names
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-ticker download: data has ('Close','BTC-USD'), ('High','BTC-USD'), ...
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1).copy()
                    else:
                        logger.debug(f"[YF] Ticker {ticker} not in download result")
                        continue
                else:
                    # Single ticker (shouldn't happen with new yfinance but handle anyway)
                    df = data.copy()
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

                if df is None or df.empty:
                    continue

                market = _market_data.get(sym, {})
                vol = market.get('volume', 0)
                klines = _parse_yf_df(df, sym, vol, limit)
                if klines:
                    result[sym] = klines
                    logger.debug(f"[YF] {sym}: {len(klines)} candles, close={klines[-1]['close']:.4f}")

            except Exception as e:
                logger.debug(f"[YF] parse error for {sym}: {e}")

        logger.info(f"[YF] Fetched OHLCV for {len(result)}/{len(symbols)} symbols")
        return result

    except Exception as e:
        logger.error(f"[YF] Batch download error: {e}")
        return {}


def _parse_yf_df(df, symbol: str, volume_24h: float, limit: int) -> List[Dict]:
    """Parse a yfinance DataFrame into our standard kline dict list."""
    try:
        rows = df.dropna(subset=['Close']).tail(limit)
        klines = []
        vol_per_candle = volume_24h / max(len(rows), 1)

        for ts, row in rows.iterrows():
            open_time = ts.to_pydatetime().replace(tzinfo=None)
            klines.append({
                'symbol':      symbol,
                'open_time':   open_time,
                'close_time':  open_time,
                'open':        float(row.get('Open',  row.get('Close', 0))),
                'high':        float(row.get('High',  row.get('Close', 0))),
                'low':         float(row.get('Low',   row.get('Close', 0))),
                'close':       float(row['Close']),
                'volume':      float(row.get('Volume', vol_per_candle)),
                'quote_volume': float(row.get('Volume', 0)),
                'trades':      0,
            })
        return klines
    except Exception as e:
        logger.debug(f"[YF] DataFrame parse error ({symbol}): {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────
def get_top_symbols(limit: int = 90) -> List[str]:
    """
    Fetch top coins by 24h volume from CoinGecko — cached 6 hours.
    Returns Binance-style USDT pairs: ['BTCUSDT', 'ETHUSDT', …]
    ONE API call fetches data for all coins at once.
    """
    global _symbol_cache, _symbol_cache_ts, _id_map, _market_data

    now       = _time.time()
    cache_age = now - _symbol_cache_ts

    if _symbol_cache and cache_age < _SYMBOL_CACHE_TTL:
        return _symbol_cache

    try:
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
    Fetch OHLCV candles via Yahoo Finance (no rate limits, instant).
    Uses 60-minute cache — candles are refetched only once per hour.
    Return format identical to old Binance version.
    """
    cache_key = f"{symbol}|{interval}"
    now = _time.time()

    # ── OHLCV cache hit ────────────────────────────────────────────────────────
    cached = _ohlcv_cache.get(cache_key)
    if cached and (now - cached['ts']) < _OHLCV_CACHE_TTL:
        klines = list(cached['data'])   # shallow copy
        # Inject fresh live price into last candle (always up-to-date)
        live_price = _market_data.get(symbol, {}).get('price')
        if klines and live_price:
            klines[-1] = dict(klines[-1])  # copy before mutating
            klines[-1]['close'] = live_price
            klines[-1]['high']  = max(klines[-1]['high'], live_price)
            klines[-1]['low']   = min(klines[-1]['low'],  live_price)
        return klines[-limit:] if len(klines) > limit else klines

    # ── Fetch from Yahoo Finance ───────────────────────────────────────────────
    batch = _fetch_yf_batch([symbol], interval, limit)
    klines = batch.get(symbol)

    if not klines:
        logger.warning(f"[Scanner] No YF data for {symbol}, interval={interval}")
        return None

    # Store clean data in cache (no live-price injection)
    _ohlcv_cache[cache_key] = {'data': klines, 'ts': now}

    # Trim to limit and inject live price
    klines = klines[-limit:]
    live_price = _market_data.get(symbol, {}).get('price')
    if klines and live_price:
        klines[-1] = dict(klines[-1])
        klines[-1]['close'] = live_price
        klines[-1]['high']  = max(klines[-1]['high'], live_price)
        klines[-1]['low']   = min(klines[-1]['low'],  live_price)

    return klines


def get_klines_since(symbol: str, start_time: datetime, interval: str = '1d') -> List[Dict]:
    """
    Fetch klines from start_time to now (for incremental updates).
    """
    klines = get_klines(symbol, interval, limit=1000)
    if not klines:
        return []
    return [k for k in klines if k['open_time'] >= start_time]


def _parse_kline(symbol: str, k: list) -> Dict:
    """Compatibility stub — not used in YF version."""
    ts = datetime.utcfromtimestamp(k[0] / 1000)
    return {
        'symbol': symbol, 'open_time': ts, 'close_time': ts,
        'open': float(k[1]), 'high': float(k[2]),
        'low': float(k[3]), 'close': float(k[4]),
        'volume': 0, 'quote_volume': 0, 'trades': 0,
    }


if __name__ == "__main__":
    print("Testing hybrid CoinGecko + Yahoo Finance scanner...")
    symbols = get_top_symbols(5)
    print(f"Top 5: {symbols}")
    if symbols:
        k = get_klines(symbols[0], '1d', 10)
        if k:
            print(f"{symbols[0]}: {len(k)} candles, close={k[-1]['close']:.4f}")
        k2 = get_klines(symbols[0], '1d', 10)  # should be instant (cache)
        print(f"Cache hit: {len(k2)} candles (should be instant)")
