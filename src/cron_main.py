"""
Standalone entry point for running the daily news agent update via external schedulers (like Fly.io).
"""

import asyncio
import os
import logging
import sys
from src.database.manager import DatabaseManager
from src.bot.telegram_bot import MultiTenantBot
from src.bot.scheduler import DailyScheduler
from src.utils.logging import AgentLogger

logger = AgentLogger(__name__)

async def run_cron():
    """Run the daily update for all groups and exit."""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    database_url = os.getenv('DATABASE_URL')
    
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
        sys.exit(1)
        
    db = DatabaseManager(database_url)
    
    try:
        logger.info("Initializing database for cron run...")
        await db.initialize()
        
        # Initialize bot without starting polling
        bot = MultiTenantBot(bot_token, db)
        await bot.initialize()
        
        # We need the bot to be 'started' to have its internal application ready, 
        # but we don't start the updater (polling).
        await bot.application.initialize()
        await bot.application.start()
        
        scheduler = DailyScheduler(bot, db)
        
        logger.info("Starting scheduled run for all groups...")
        await scheduler.run_now()
        logger.info("Cron run completed successfully.")
        
    except Exception as e:
        logger.error(f"Fatal error in cron run: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'bot' in locals() and bot.application:
            await bot.application.stop()
            await bot.application.shutdown()
        await db.close()

if __name__ == "__main__":
    asyncio.run(run_cron())
