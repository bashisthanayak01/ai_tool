"""
Crypto AI Tool - Main Entry Point
Automated cryptocurrency data collector with MongoDB storage
"""

import logging
from scheduler import data_collector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application"""
    logger.info("=" * 60)
    logger.info("CRYPTO AI TOOL - Data Collector")
    logger.info("=" * 60)
    logger.info("Starting automated cryptocurrency data collection...")
    
    # Start the scheduler
    data_collector.start_scheduler()


if __name__ == "__main__":
    main()
