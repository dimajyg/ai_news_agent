"""
Scheduler utility for managing periodic tasks and cron-like scheduling.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime, timedelta
import aioschedule as schedule
from dataclasses import dataclass

try:
    from .logging import AgentLogger
except ImportError:
    from logging import AgentLogger


logger = AgentLogger(__name__)


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    name: str
    function: Callable
    schedule_type: str  # 'interval', 'cron', 'daily', 'weekly'
    schedule_value: Any
    args: tuple = ()
    kwargs: dict = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0


class TaskScheduler:
    """Advanced task scheduler with error handling and monitoring."""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._scheduler_task = None
        self._lock = asyncio.Lock()
    
    async def add_interval_task(
        self, 
        name: str, 
        function: Callable, 
        interval_seconds: int,
        args: tuple = (),
        kwargs: dict = None,
        enabled: bool = True
    ) -> str:
        """
        Add a task that runs at regular intervals.
        
        Args:
            name: Task name
            function: Function to execute
            interval_seconds: Interval in seconds
            args: Function arguments
            kwargs: Function keyword arguments
            enabled: Whether task is enabled
            
        Returns:
            Task ID
        """
        task = ScheduledTask(
            name=name,
            function=function,
            schedule_type='interval',
            schedule_value=interval_seconds,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled
        )
        
        async with self._lock:
            self.tasks[name] = task
        
        # Schedule the task
        job = schedule.every(interval_seconds).seconds.do(
            self._execute_task, task
        )
        
        logger.info(f"Added interval task '{name}' running every {interval_seconds} seconds")
        return name
    
    async def add_daily_task(
        self, 
        name: str, 
        function: Callable, 
        time_str: str,
        args: tuple = (),
        kwargs: dict = None,
        enabled: bool = True
    ) -> str:
        """
        Add a task that runs daily at a specific time.
        
        Args:
            name: Task name
            function: Function to execute
            time_str: Time in HH:MM format
            args: Function arguments
            kwargs: Function keyword arguments
            enabled: Whether task is enabled
            
        Returns:
            Task ID
        """
        task = ScheduledTask(
            name=name,
            function=function,
            schedule_type='daily',
            schedule_value=time_str,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled
        )
        
        async with self._lock:
            self.tasks[name] = task
        
        # Schedule the task
        job = schedule.every().day.at(time_str).do(
            self._execute_task, task
        )
        
        logger.info(f"Added daily task '{name}' running at {time_str}")
        return name
    
    async def add_cron_task(
        self, 
        name: str, 
        function: Callable, 
        cron_expression: str,
        args: tuple = (),
        kwargs: dict = None,
        enabled: bool = True
    ) -> str:
        """
        Add a task using cron-like expression (simplified).
        
        Args:
            name: Task name
            function: Function to execute
            cron_expression: Simplified cron expression
            args: Function arguments
            kwargs: Function keyword arguments
            enabled: Whether task is enabled
            
        Returns:
            Task ID
        """
        task = ScheduledTask(
            name=name,
            function=function,
            schedule_type='cron',
            schedule_value=cron_expression,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled
        )
        
        async with self._lock:
            self.tasks[name] = task
        
        # Parse simplified cron expression and schedule
        # Format: "minute hour day month weekday"
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts: minute hour day month weekday")
        
        minute, hour, day, month, weekday = parts
        
        # Create schedule job (simplified implementation)
        job = schedule.every().day
        if hour != '*':
            job = job.at(f"{hour}:{minute if minute != '*' else '00'}")
        
        job.do(self._execute_task, task)
        
        logger.info(f"Added cron task '{name}' with expression '{cron_expression}'")
        return name
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task with error handling."""
        if not task.enabled:
            return
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing task: {task.name}")
            
            # Execute the function
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            # Update task statistics
            task.last_run = start_time
            task.run_count += 1
            task.next_run = self._calculate_next_run(task)
            
            logger.info(f"Task '{task.name}' completed successfully in {(datetime.now() - start_time).total_seconds():.2f}s")
            
            return result
            
        except Exception as e:
            task.error_count += 1
            logger.error(f"Task '{task.name}' failed: {e}")
            
            # Log detailed error information
            logger.error(f"Task error details - Name: {task.name}, Args: {task.args}, Kwargs: {task.kwargs}")
            
            # Optionally disable task after too many errors
            if task.error_count >= 5:
                logger.warning(f"Disabling task '{task.name}' after {task.error_count} errors")
                task.enabled = False
            
            raise
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task."""
        if not task.enabled:
            return None
        
        now = datetime.now()
        
        if task.schedule_type == 'interval':
            return now + timedelta(seconds=task.schedule_value)
        elif task.schedule_type == 'daily':
            # Next day at the same time
            return now + timedelta(days=1)
        else:
            return None
    
    async def remove_task(self, task_name: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_name: Name of the task to remove
            
        Returns:
            True if task was removed, False if not found
        """
        async with self._lock:
            if task_name in self.tasks:
                del self.tasks[task_name]
                logger.info(f"Removed task: {task_name}")
                return True
            return False
    
    async def enable_task(self, task_name: str) -> bool:
        """
        Enable a disabled task.
        
        Args:
            task_name: Name of the task to enable
            
        Returns:
            True if task was enabled, False if not found
        """
        async with self._lock:
            if task_name in self.tasks:
                self.tasks[task_name].enabled = True
                logger.info(f"Enabled task: {task_name}")
                return True
            return False
    
    async def disable_task(self, task_name: str) -> bool:
        """
        Disable a task without removing it.
        
        Args:
            task_name: Name of the task to disable
            
        Returns:
            True if task was disabled, False if not found
        """
        async with self._lock:
            if task_name in self.tasks:
                self.tasks[task_name].enabled = False
                logger.info(f"Disabled task: {task_name}")
                return True
            return False
    
    async def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task status information or None if not found
        """
        async with self._lock:
            if task_name in self.tasks:
                task = self.tasks[task_name]
                return {
                    "name": task.name,
                    "enabled": task.enabled,
                    "schedule_type": task.schedule_type,
                    "schedule_value": task.schedule_value,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "run_count": task.run_count,
                    "error_count": task.error_count
                }
            return None
    
    async def get_all_tasks_status(self) -> List[Dict[str, Any]]:
        """
        Get status information for all tasks.
        
        Returns:
            List of task status information
        """
        async with self._lock:
            return [await self.get_task_status(task_name) for task_name in self.tasks.keys()]
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task scheduler stopped")
    
    async def _run_scheduler(self):
        """Main scheduler loop."""
        logger.info("Starting scheduler loop")
        
        while self.running:
            try:
                # Run pending scheduled tasks
                result = schedule.run_pending()
                if asyncio.iscoroutine(result):
                    await result
                
                # Wait a bit before checking again
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
        
        logger.info("Scheduler loop ended")
    
    async def run_once(self, task_name: str) -> Any:
        """
        Manually run a task once, regardless of schedule.
        
        Args:
            task_name: Name of the task to run
            
        Returns:
            Task execution result
        """
        async with self._lock:
            if task_name not in self.tasks:
                raise ValueError(f"Task '{task_name}' not found")
            
            task = self.tasks[task_name]
        
        return await self._execute_task(task)


# Global scheduler instance
_global_scheduler = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = TaskScheduler()
    return _global_scheduler


async def initialize_scheduler():
    """Initialize and start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()
    return scheduler


async def shutdown_scheduler():
    """Shutdown the global scheduler."""
    global _global_scheduler
    if _global_scheduler:
        await _global_scheduler.stop()
        _global_scheduler = None


# Example usage and predefined tasks
async def create_news_extraction_task(extraction_function: Callable, interval_hours: int = 6) -> str:
    """
    Create a scheduled task for news extraction.
    
    Args:
        extraction_function: Function to extract news
        interval_hours: Interval in hours
        
    Returns:
        Task ID
    """
    scheduler = get_scheduler()
    
    task_id = await scheduler.add_interval_task(
        name="news_extraction",
        function=extraction_function,
        interval_seconds=interval_hours * 3600,  # Convert hours to seconds
        enabled=True
    )
    
    return task_id


async def create_daily_posting_task(posting_function: Callable, time_str: str = "09:00") -> str:
    """
    Create a scheduled task for daily posting.
    
    Args:
        posting_function: Function to post to Telegram
        time_str: Time to post (HH:MM format)
        
    Returns:
        Task ID
    """
    scheduler = get_scheduler()
    
    task_id = await scheduler.add_daily_task(
        name="daily_posting",
        function=posting_function,
        time_str=time_str,
        enabled=True
    )
    
    return task_id