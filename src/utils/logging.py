# Logging Configuration

import logging
import logging.handlers
from pathlib import Path
from loguru import logger
import sys

def setup_logging(log_level: str = "INFO", log_file: str = "./logs/agent.log"):
    """Setup structured logging for the application."""
    
    # Remove default logger
    logger.remove()
    
    # Console logging with colors
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logging with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True  # Async logging
    )
    
    # Error log file
    error_log_file = log_file.replace('.log', '_errors.log')
    logger.add(
        error_log_file,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry.opentelemetry").setLevel(logging.ERROR)

    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")

class AgentLogger:
    """Custom logger for AI News Agent with structured logging."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(component=name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with full traceback."""
        self.logger.exception(message, **kwargs)
    
    def log_extraction_start(self, source: str):
        """Log extraction start."""
        self.logger.info(f"Starting extraction from {source}", source=source, event="extraction_start")
    
    def log_extraction_complete(self, source: str, article_count: int, duration: float):
        """Log extraction completion."""
        self.logger.info(
            f"Completed extraction from {source}",
            source=source,
            article_count=article_count,
            duration=duration,
            event="extraction_complete"
        )
    
    def log_clustering_complete(self, total_articles: int, cluster_count: int, duration: float):
        """Log clustering completion."""
        self.logger.info(
            f"Clustering complete",
            total_articles=total_articles,
            cluster_count=cluster_count,
            duration=duration,
            event="clustering_complete"
        )
    
    def log_workflow_step(self, step_name: str, state: dict):
        """Log workflow step execution."""
        self.logger.info(
            f"Executing workflow step: {step_name}",
            step=step_name,
            state_summary=self._summarize_state(state),
            event="workflow_step"
        )
    
    def log_workflow_complete(self, execution_time: float, results: dict):
        """Log workflow completion."""
        self.logger.info(
            f"Workflow completed",
            execution_time=execution_time,
            articles_processed=results.get('articles_extracted', 0),
            clusters_created=results.get('clusters_created', 0),
            post_sent=results.get('post_sent', False),
            event="workflow_complete"
        )
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log structured error information."""
        self.logger.error(
            f"Error occurred: {error_type}",
            error_type=error_type,
            error_message=error_message,
            context=context or {},
            event="error"
        )
    
    def _summarize_state(self, state: dict) -> dict:
        """Create a summary of the agent state for logging."""
        summary = {}
        
        if 'raw_articles' in state:
            summary['raw_articles'] = len(state['raw_articles'])
        
        if 'clustered_articles' in state:
            cluster_ids = set(a.get('cluster_id') for a in state['clustered_articles'])
            summary['clusters'] = len(cluster_ids)
        
        if 'filtered_clusters' in state:
            summary['filtered_clusters'] = len(state['filtered_clusters'])
        
        if 'top_clusters' in state:
            summary['top_clusters'] = len(state['top_clusters'])
        
        if 'error_log' in state:
            summary['errors'] = len(state['error_log'])
        
        return summary
