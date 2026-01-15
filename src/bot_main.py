"""
Main entry point for the multi-tenant Telegram bot.
"""

import asyncio
import os
import signal
import sys
import logging
from pathlib import Path

from src.database.manager import DatabaseManager
from src.bot.telegram_bot import MultiTenantBot
from src.bot.scheduler import DailyScheduler
from src.utils.logging import AgentLogger

logger = AgentLogger(__name__)


class BotService:
    """Main service for running the multi-tenant bot."""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.database_url = os.getenv('DATABASE_URL')
        self.schedule_time = os.getenv('SCHEDULE_TIME', '09:00')
        self.enable_scheduler = os.getenv('ENABLE_INTERNAL_SCHEDULER', 'true').lower() == 'true'
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        self.db = DatabaseManager(self.database_url)
        self.bot = None
        self.scheduler = None
        self.running = False
    
    async def initialize(self):
        """Initialize the bot service."""
        logger.info("Initializing bot service...")
        
        try:
            await self.db.initialize()
            
            self.bot = MultiTenantBot(self.bot_token, self.db)
            await self.bot.initialize()
            
            if self.enable_scheduler:
                self.scheduler = DailyScheduler(
                    bot=self.bot,
                    db=self.db,
                    schedule_time=self.schedule_time
                )
            else:
                logger.info("Internal scheduler disabled by configuration")
            
            logger.info("Bot service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot service: {e}")
            raise
    
    async def start(self):
        """Start the bot service."""
        logger.info("Starting bot service...")
        
        try:
            self.running = True
            
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            await self.bot.start()
            
            if self.scheduler:
                await self.scheduler.start()
                logger.info(f"Daily runs scheduled at {self.schedule_time} UTC")
            
            logger.info("Bot service started successfully")
            
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error running bot service: {e}")
            raise
    
    async def stop(self):
        """Stop the bot service."""
        logger.info("Stopping bot service...")
        
        self.running = False
        
        try:
            if self.scheduler:
                await self.scheduler.stop()
            
            if self.bot:
                await self.bot.stop()
            
            if self.db:
                await self.db.close()
            
            logger.info("Bot service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bot service: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())


async def main():
    """Main entry point."""
    logger.info("Starting AI News Agent Multi-Tenant Bot")
    
    service = BotService()
    
    try:
        await service.initialize()
        
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
