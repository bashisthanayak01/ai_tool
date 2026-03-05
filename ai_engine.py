"""
AI Trading Engine - Technical Analysis & Signal Generation
Analyzes cryptocurrency market data and generates trading signals
"""

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: Pandas Series of closing prices
        period: RSI period (default: 14)
    
    Returns:
        Pandas Series with RSI values
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: Pandas Series of closing prices
        period: EMA period (e.g., 20, 50)
    
    Returns:
        Pandas Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_volume_ma(data, period=20):
    """
    Calculate Volume Moving Average
    
    Args:
        data: Pandas Series of volume data
        period: Moving average period (default: 20)
    
    Returns:
        Pandas Series with volume moving average
    """
    return data.rolling(window=period).mean()


def generate_signal(row):
    """
    Generate trading signal based on technical indicators
    
    Signal Logic:
    - STRONG BUY: EMA20 > EMA50 AND RSI > 70 AND strong volume spike
    - BUY: EMA20 > EMA50 AND RSI between 50-70 AND volume spike
    - STRONG SELL: EMA20 < EMA50 AND RSI < 30
    - SELL: EMA20 < EMA50 AND RSI between 30-50
    - HOLD: All other conditions
    
    Args:
        row: DataFrame row with indicators
    
    Returns:
        str: Signal (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
    """
    ema20 = row['ema_20']
    ema50 = row['ema_50']
    rsi = row['rsi']
    volume = row['volume']
    volume_ma = row['volume_ma']
    
    # Check for valid data
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(rsi) or pd.isna(volume_ma):
        return 'HOLD'
    
    # Calculate volume spike (volume is 1.5x or 2x the average)
    volume_spike = volume > (volume_ma * 1.5)
    strong_volume_spike = volume > (volume_ma * 2.0)
    
    # STRONG BUY conditions
    if ema20 > ema50 and rsi > 70 and strong_volume_spike:
        return 'STRONG BUY'
    
    # BUY conditions
    if ema20 > ema50 and 50 <= rsi <= 70 and volume_spike:
        return 'BUY'
    
    # STRONG SELL conditions
    if ema20 < ema50 and rsi < 30:
        return 'STRONG SELL'
    
    # SELL conditions
    if ema20 < ema50 and 30 <= rsi <= 50:
        return 'SELL'
    
    # Default to HOLD
    return 'HOLD'


def analyze_symbol(df_symbol, symbol):
    """
    Analyze a single symbol and generate signals
    
    Args:
        df_symbol: DataFrame with market data for one symbol
        symbol: Trading pair symbol
    
    Returns:
        dict: Analysis results with latest signal
    """
    if df_symbol.empty:
        logger.warning(f"No data available for {symbol}")
        return None
    
    # Sort by time
    df_symbol = df_symbol.sort_values('open_time')
    
    # Calculate technical indicators
    df_symbol['rsi'] = calculate_rsi(df_symbol['close'], period=14)
    df_symbol['ema_20'] = calculate_ema(df_symbol['close'], period=20)
    df_symbol['ema_50'] = calculate_ema(df_symbol['close'], period=50)
    df_symbol['volume_ma'] = calculate_volume_ma(df_symbol['volume'], period=20)
    
    # Generate signals
    df_symbol['signal'] = df_symbol.apply(generate_signal, axis=1)
    
    # Get latest data
    latest = df_symbol.iloc[-1]
    
    result = {
        'symbol': symbol,
        'timestamp': latest['close_time'],
        'close_price': latest['close'],
        'rsi': round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else None,
        'ema_20': round(latest['ema_20'], 2) if not pd.isna(latest['ema_20']) else None,
        'ema_50': round(latest['ema_50'], 2) if not pd.isna(latest['ema_50']) else None,
        'volume': latest['volume'],
        'volume_ma': round(latest['volume_ma'], 2) if not pd.isna(latest['volume_ma']) else None,
        'signal': latest['signal']
    }
    
    return result


def fetch_market_data():
    """
    Fetch market data from MongoDB
    
    Returns:
        DataFrame: Market data from all symbols
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[settings.DATABASE_NAME]
        collection = db[settings.COLLECTION_MARKET_DATA]
        
        logger.info("Connected to MongoDB successfully")
        
        # Fetch all market data
        cursor = collection.find({})
        data = list(cursor)
        
        if not data:
            logger.warning("No market data found in database")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Fetched {len(df)} records from market_data collection")
        
        # Close connection
        client.close()
        
        return df
        
    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()


def main():
    """
    Main function to run AI trading analysis
    """
    logger.info("=" * 70)
    logger.info("AI TRADING ENGINE - Technical Analysis")
    logger.info("=" * 70)
    
    # Fetch market data from MongoDB
    df = fetch_market_data()
    
    if df.empty:
        logger.error("No data available for analysis. Please run main.py first to collect data.")
        return
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    logger.info(f"Analyzing {len(symbols)} symbols: {', '.join(symbols)}")
    logger.info("-" * 70)
    
    # Analyze each symbol
    results = []
    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()
        result = analyze_symbol(df_symbol, symbol)
        
        if result:
            results.append(result)
    
    # Safe formatter for potentially None numeric values
    def safe_fmt(val, fmt=",.2f"):
        if val is None:
            return "N/A"
        try:
            return format(val, fmt)
        except (TypeError, ValueError):
            return str(val)

    # Display results
    if results:
        logger.info("\n" + "=" * 70)
        logger.info("TRADING SIGNALS")
        logger.info("=" * 70)
        
        for result in results:
            # Warn if any indicator is missing
            missing = [k for k in ('rsi', 'ema_20', 'ema_50', 'volume_ma') if result.get(k) is None]
            if missing:
                logger.warning(f"Missing indicator data for {result['symbol']}: {', '.join(missing)}")

            signal_color = {
                'STRONG BUY': '🚀',
                'BUY': '📈',
                'HOLD': '⏸️',
                'SELL': '📉',
                'STRONG SELL': '🔻'
            }
            
            icon = signal_color.get(result['signal'], '⏸️')
            
            print(f"\n{icon} {result['symbol']}")
            print(f"   Price: ${safe_fmt(result.get('close_price'))}")
            print(f"   RSI (14): {safe_fmt(result.get('rsi'))}")
            print(f"   EMA 20: {safe_fmt(result.get('ema_20'))}")
            print(f"   EMA 50: {safe_fmt(result.get('ema_50'))}")
            print(f"   Volume: {safe_fmt(result.get('volume'))}")
            print(f"   Volume MA: {safe_fmt(result.get('volume_ma'))}")
            print(f"   ► SIGNAL: {result['signal']} ◄")
            print(f"   Time: {result.get('timestamp', 'N/A')}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Analysis complete!")
    else:
        logger.warning("No analysis results generated")


if __name__ == "__main__":
    main()
