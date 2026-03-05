"""
Scheduler module for automated data collection
"""

import schedule
import time
import logging
from datetime import datetime

from config import settings
from data import binance_api
from database import mongo_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Scheduler for automated crypto data collection"""
    
    def __init__(self):
        self.symbols = settings.TRADING_PAIRS
        self.interval = settings.KLINE_INTERVAL
        self.limit = settings.KLINE_LIMIT
        self.schedule_interval = settings.SCHEDULE_INTERVAL_MINUTES
    
    def collect_and_store_data(self):
        """
        Collect data from Binance and store in MongoDB
        This is the main job that runs on schedule
        """
        try:
            logger.info("=" * 60)
            logger.info(f"Starting data collection at {datetime.now()}")
            logger.info(f"Collecting data for: {', '.join(self.symbols)}")
            
            # Fetch data for all symbols
            market_data = binance_api.fetch_multiple_symbols(
                symbols=self.symbols,
                interval=self.interval,
                limit=self.limit
            )
            
            if not market_data:
                logger.warning("No market data fetched")
                return
            
            # Store data in MongoDB
            success = mongo_client.insert_market_data(market_data)
            
            if success:
                logger.info(f"✓ Successfully stored {len(market_data)} records to MongoDB")
                logger.info(f"Next collection scheduled in {self.schedule_interval} minutes")
            else:
                logger.error("✗ Failed to store market data")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in data collection job: {e}")
    
    def start_scheduler(self):
        """
        Start the scheduler to run data collection every N minutes
        """
        logger.info("Initializing Data Collector Scheduler...")
        logger.info(f"Schedule: Every {self.schedule_interval} minutes")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Interval: {self.interval}")
        
        # Connect to MongoDB
        if not mongo_client.connect_to_mongo():
            logger.error("Failed to connect to MongoDB. Exiting...")
            return
        
        # Run immediately on start
        logger.info("Running initial data collection...")
        self.collect_and_store_data()
        
        # Schedule regular runs
        schedule.every(self.schedule_interval).minutes.do(self.collect_and_store_data)
        
        logger.info(f"Scheduler started. Running every {self.schedule_interval} minutes.")
        logger.info("Press Ctrl+C to stop")
        
        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            mongo_client.close_connection()


# Singleton instance
data_collector = DataCollector()
