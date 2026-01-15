"""
Daily scheduler for running AI news agent for all active groups.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional
# import aioschedule as schedule  # Removed to fix coroutine error

from ..database.manager import DatabaseManager
from .telegram_bot import MultiTenantBot

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Scheduler for running the agent daily for all active groups."""
    
    def __init__(self, bot: MultiTenantBot, db: DatabaseManager, schedule_time: str = "09:00"):
        self.bot = bot
        self.db = db
        self.schedule_time = schedule_time
        self.running = False
        self._task = None
    
    async def start(self):
        """Start the scheduler."""
        logger.info(f"Starting daily scheduler (runs at {self.schedule_time} UTC)")
        
        self.running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Daily scheduler started successfully")
    
    async def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping daily scheduler...")
        
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Daily scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info(f"Scheduler loop started. Schedule time: {self.schedule_time}")
        
        while self.running:
            try:
                # Calculate time until next run
                now = datetime.now()
                # Assuming schedule_time is HH:MM
                h, m = map(int, self.schedule_time.split(':'))
                
                # Create target time for today
                target = now.replace(hour=h, minute=m, second=0, microsecond=0)
                
                # If target is in the past, schedule for tomorrow
                if target <= now:
                    target = target + timedelta(days=1)
                
                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next run scheduled in {wait_seconds:.0f} seconds (at {target})")
                
                # Sleep until the target time
                # Check running flag periodically
                while wait_seconds > 0 and self.running:
                    sleep_time = min(wait_seconds, 60)
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time
                
                if not self.running:
                    break
                
                # Run the job
                await self._run_all_groups()
                
                # Wait a bit to avoid double execution
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_all_groups(self):
        """Run the agent for all active groups."""
        logger.info("Starting scheduled run for all active groups")
        
        try:
            groups = await self.db.get_all_active_groups()
            
            if not groups:
                logger.info("No active groups found")
                return
            
            logger.info(f"Found {len(groups)} active groups")
            
            for group in groups:
                try:
                    logger.info(f"Running agent for group: {group.chat_id} ({group.chat_title})")
                    
                    sources = await self.db.get_group_sources(group.chat_id)
                    
                    if not sources:
                        logger.warning(f"No sources configured for group {group.chat_id}, skipping")
                        continue
                    
                    result = await self.bot._run_agent_for_group(group, group.chat_id)
                    
                    if result['success']:
                        logger.info(
                            f"Successfully ran agent for {group.chat_id}: "
                            f"{result.get('articles_extracted', 0)} articles, "
                            f"{result.get('execution_time', 0):.1f}s"
                        )
                        
                        await self.bot.application.bot.send_message(
                            chat_id=group.chat_id,
                            text=(
                                f"✅ <b>Daily News Update Complete!</b>\n\n"
                                f"📊 Processed {result.get('articles_extracted', 0)} articles\n"
                                f"📝 Generated {result.get('top_documents_selected', 0)} top stories\n"
                                f"⏱ Completed in {result.get('execution_time', 0):.1f}s"
                            ),
                            parse_mode='HTML'
                        )
                    else:
                        logger.error(f"Failed to run agent for {group.chat_id}: {result.get('error')}")
                        
                        await self.bot.application.bot.send_message(
                            chat_id=group.chat_id,
                            text=(
                                f"❌ <b>Daily News Update Failed</b>\n\n"
                                f"Error: {result.get('error', 'Unknown error')}\n\n"
                                f"Please check your configuration with /config"
                            ),
                            parse_mode='HTML'
                        )
                    
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error running agent for group {group.chat_id}: {e}")
                    continue
            
            logger.info("Completed scheduled run for all groups")
            
        except Exception as e:
            logger.error(f"Error in scheduled run: {e}")
    
    async def run_now(self):
        """Manually trigger a run for all groups (for testing)."""
        logger.info("Manually triggering run for all groups")
        await self._run_all_groups()
