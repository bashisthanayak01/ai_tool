"""
Advanced AI Engine - Multi-Coin Profit Scoring System
Analyzes top cryptocurrencies and generates profit scores (0-100)
"""

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_macd(data):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns MACD line, signal line, and histogram
    """
    ema_12 = data.ewm(span=12, adjust=False).mean()
    ema_26 = data.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (volatility indicator)
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_volume_change(data):
    """Calculate volume percentage change"""
    return data.pct_change() * 100


def calculate_trend_score(row):
    """
    Calculate trend score (0-25) based on EMA alignment
    """
    ema20 = row['ema_20']
    ema50 = row['ema_50']
    price = row['close']
    
    if pd.isna(ema20) or pd.isna(ema50):
        return 0
    
    score = 0
    
    # Strong bullish trend
    if price > ema20 > ema50:
        score = 25
    # Bullish trend
    elif ema20 > ema50:
        score = 18
    # Weak bullish
    elif price > ema20:
        score = 12
    # Weak bearish
    elif price < ema20 and ema20 < ema50:
        score = 5
    # Bearish
    else:
        score = 0
    
    return score


def calculate_momentum_score(row):
    """
    Calculate momentum score (0-25) based on RSI
    """
    rsi = row['rsi']
    
    if pd.isna(rsi):
        return 0
    
    score = 0
    
    # Overbought but strong momentum
    if rsi >= 70:
        score = 22
    # Strong momentum
    elif 60 <= rsi < 70:
        score = 25
    # Good momentum
    elif 55 <= rsi < 60:
        score = 20
    # Neutral momentum
    elif 45 <= rsi < 55:
        score = 12
    # Weak momentum
    elif 35 <= rsi < 45:
        score = 8
    # Oversold (potential reversal)
    elif 25 <= rsi < 35:
        score = 5
    # Very oversold
    else:
        score = 3
    
    return score


def calculate_volume_score(row):
    """
    Calculate volume score (0-20) based on volume spike
    """
    volume = row['volume']
    volume_ma = row['volume_ma']
    
    if pd.isna(volume_ma) or volume_ma == 0:
        return 0
    
    volume_ratio = volume / volume_ma
    
    score = 0
    
    # Massive volume spike
    if volume_ratio >= 3.0:
        score = 20
    # Strong volume spike
    elif volume_ratio >= 2.0:
        score = 18
    # Good volume spike
    elif volume_ratio >= 1.5:
        score = 15
    # Above average
    elif volume_ratio >= 1.2:
        score = 10
    # Normal volume
    elif volume_ratio >= 0.8:
        score = 5
    # Low volume
    else:
        score = 2
    
    return score


def calculate_macd_score(row):
    """
    Calculate MACD score (0-15) based on crossover
    """
    macd = row['macd']
    macd_signal = row['macd_signal']
    macd_hist = row['macd_hist']
    
    if pd.isna(macd) or pd.isna(macd_signal):
        return 0
    
    score = 0
    
    # Bullish crossover with positive histogram
    if macd > macd_signal and macd_hist > 0:
        score = 15
    # Recent bullish crossover
    elif macd > macd_signal:
        score = 12
    # Approaching bullish crossover
    elif macd > 0 and macd_hist > 0:
        score = 8
    # Neutral
    elif macd > macd_signal - 0.01:
        score = 5
    # Bearish
    else:
        score = 0
    
    return score


def calculate_volatility_score(row):
    """
    Calculate volatility score (0-15) based on ATR
    Higher volatility = more trading opportunity
    """
    atr = row['atr']
    price = row['close']
    
    if pd.isna(atr) or price == 0:
        return 0
    
    # ATR as percentage of price
    atr_percent = (atr / price) * 100
    
    score = 0
    
    # High volatility (good for trading)
    if atr_percent >= 5:
        score = 15
    elif atr_percent >= 3:
        score = 12
    elif atr_percent >= 2:
        score = 9
    elif atr_percent >= 1:
        score = 6
    # Low volatility
    else:
        score = 3
    
    return score


def calculate_profit_score(row):
    """
    Calculate overall profit score (0-100)
    Weighted combination of all indicators
    """
    trend = calculate_trend_score(row)
    momentum = calculate_momentum_score(row)
    volume = calculate_volume_score(row)
    macd = calculate_macd_score(row)
    volatility = calculate_volatility_score(row)
    
    # Weighted total (all weights sum to 100)
    profit_score = trend + momentum + volume + macd + volatility
    
    return round(profit_score, 2)


def determine_confidence(score):
    """
    Determine confidence level based on profit score
    """
    if score >= 85:
        return "Very High"
    elif score >= 75:
        return "High"
    elif score >= 60:
        return "Medium-High"
    elif score >= 40:
        return "Medium"
    elif score >= 25:
        return "Medium-Low"
    elif score >= 15:
        return "Low"
    else:
        return "Very Low"


def generate_advanced_signal(row):
    """
    Generate trading signal based on profit score
    BUY: score >= 70
    SELL: score <= 30
    HOLD: otherwise
    """
    score = row['profit_score']
    
    if pd.isna(score):
        return 'HOLD'
    
    if score >= 70:
        return 'BUY'
    elif score <= 30:
        return 'SELL'
    else:
        return 'HOLD'


def analyze_symbol_advanced(df_symbol, symbol):
    """
    Perform advanced analysis on a single symbol
    
    Args:
        df_symbol: DataFrame with market data for one symbol
        symbol: Trading pair symbol
    
    Returns:
        dict: Analysis results with profit score and signal
    """
    if df_symbol.empty:
        logger.warning(f"No data available for {symbol}")
        return None
    
    # Sort by time
    df_symbol = df_symbol.sort_values('open_time')
    
    # Calculate all technical indicators
    df_symbol['rsi'] = calculate_rsi(df_symbol['close'], period=14)
    df_symbol['ema_20'] = calculate_ema(df_symbol['close'], period=20)
    df_symbol['ema_50'] = calculate_ema(df_symbol['close'], period=50)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(df_symbol['close'])
    df_symbol['macd'] = macd_line
    df_symbol['macd_signal'] = signal_line
    df_symbol['macd_hist'] = histogram
    
    # Volatility
    df_symbol['atr'] = calculate_atr(df_symbol, period=14)
    
    # Volume
    df_symbol['volume_ma'] = df_symbol['volume'].rolling(window=20).mean()
    
    # Calculate profit score
    df_symbol['profit_score'] = df_symbol.apply(calculate_profit_score, axis=1)
    
    # Generate signal
    df_symbol['signal'] = df_symbol.apply(generate_advanced_signal, axis=1)
    
    # Get latest data
    latest = df_symbol.iloc[-1]
    
    # Determine confidence
    confidence = determine_confidence(latest['profit_score'])
    
    result = {
        'symbol': symbol,
        'price': latest['close'],
        'signal': latest['signal'],
        'profit_score': round(latest['profit_score'], 2) if not pd.isna(latest['profit_score']) else 0,
        'confidence_level': confidence,
        'timestamp': latest['close_time'],
        # Additional metrics for dashboard
        'rsi': round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else None,
        'ema_20': round(latest['ema_20'], 2) if not pd.isna(latest['ema_20']) else None,
        'ema_50': round(latest['ema_50'], 2) if not pd.isna(latest['ema_50']) else None,
        'volume': latest['volume'],
        'atr': round(latest['atr'], 2) if not pd.isna(latest['atr']) else None
    }
    
    return result


def fetch_market_data():
    """Fetch market data from MongoDB"""
    try:
        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[settings.DATABASE_NAME]
        collection = db[settings.COLLECTION_MARKET_DATA]
        
        logger.info("Connected to MongoDB successfully")
        
        cursor = collection.find({})
        data = list(cursor)
        
        if not data:
            logger.warning("No market data found in database")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.info(f"Fetched {len(df)} records from market_data collection")
        
        client.close()
        return df
        
    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()


def save_ai_signals(signals):
    """Save AI signals to MongoDB"""
    try:
        from database import mongo_client
        
        if not mongo_client.connect_to_mongo():
            logger.error("Failed to connect to MongoDB")
            return False
        
        success = mongo_client.insert_ai_signals(signals)
        mongo_client.close_connection()
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving AI signals: {e}")
        return False


def main():
    """Main function to run advanced AI analysis"""
    logger.info("=" * 70)
    logger.info("ADVANCED AI ENGINE - Multi-Coin Profit Scoring")
    logger.info("=" * 70)
    
    # Fetch market data from MongoDB
    df = fetch_market_data()
    
    if df.empty:
        logger.error("No data available for analysis. Please run data collection first.")
        return
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    logger.info(f"Analyzing {len(symbols)} symbols...")
    logger.info("-" * 70)
    
    # Analyze each symbol
    all_signals = []
    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()
        result = analyze_symbol_advanced(df_symbol, symbol)
        
        if result:
            all_signals.append(result)
    
    if not all_signals:
        logger.warning("No analysis results generated")
        return
    
    # Sort by profit score
    all_signals.sort(key=lambda x: x['profit_score'], reverse=True)
    
    # Save to MongoDB
    logger.info(f"Saving {len(all_signals)} AI signals to MongoDB...")
    if save_ai_signals(all_signals):
        logger.info("✓ AI signals saved successfully")
    else:
        logger.error("✗ Failed to save AI signals")
    
    # Display Top 10
    logger.info("\n" + "=" * 70)
    logger.info("TOP 10 PROFIT SCORES")
    logger.info("=" * 70)
    
    for i, result in enumerate(all_signals[:10], 1):
        signal_icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(result['signal'], '⚪')
        
        print(f"\n#{i} {signal_icon} {result['symbol']}")
        print(f"   Price: ${result['price']:,.2f}")
        print(f"   Profit Score: {result['profit_score']}/100 ⭐")
        print(f"   Signal: {result['signal']}")
        print(f"   Confidence: {result['confidence_level']}")
        print(f"   RSI: {result['rsi']}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Analysis complete! {len(all_signals)} signals generated.")


if __name__ == "__main__":
    main()
