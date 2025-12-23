# Metrics Collection and Monitoring

import time
import logging
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and exposes metrics for the AI News Agent."""
    
    def __init__(self):
        # Check if metrics already exist to avoid duplicate registration
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup metrics, checking for existing ones."""
        def get_or_create_counter(name, description, labels=None):
            try:
                if labels:
                    return Counter(name, description, labels)
                else:
                    return Counter(name, description)
            except ValueError:
                # Metric already exists, get existing one
                return REGISTRY._names_to_collectors[name]
        
        def get_or_create_histogram(name, description, labels=None):
            try:
                if labels:
                    return Histogram(name, description, labels)
                else:
                    return Histogram(name, description)
            except ValueError:
                # Metric already exists, get existing one
                return REGISTRY._names_to_collectors[name]
        
        def get_or_create_gauge(name, description, labels=None):
            try:
                if labels:
                    return Gauge(name, description, labels)
                else:
                    return Gauge(name, description)
            except ValueError:
                # Metric already exists, get existing one
                return REGISTRY._names_to_collectors[name]
        
        # Counters
        self.articles_extracted = get_or_create_counter(
            'ai_news_articles_extracted_total',
            'Total number of articles extracted',
            ['source']
        )
        
        self.clusters_created = get_or_create_counter(
            'ai_news_clusters_created_total',
            'Total number of clusters created'
        )
        
        self.duplicates_removed = get_or_create_counter(
            'ai_news_duplicates_removed_total',
            'Total number of duplicate articles removed'
        )
        
        self.posts_generated = get_or_create_counter(
            'ai_news_posts_generated_total',
            'Total number of Telegram posts generated'
        )
        
        self.posts_sent = get_or_create_counter(
            'ai_news_posts_sent_total',
            'Total number of Telegram posts successfully sent',
            ['status']  # success, failed
        )
        
        self.workflow_executions = get_or_create_counter(
            'ai_news_workflow_executions_total',
            'Total number of workflow executions',
            ['status']  # success, failed
        )
        
        self.node_executions = get_or_create_counter(
            'ai_news_node_executions_total',
            'Total number of node executions',
            ['node_name', 'status']  # success, failed
        )
        
        # Histograms
        self.extraction_duration = get_or_create_histogram(
            'ai_news_extraction_duration_seconds',
            'Time spent extracting articles from sources',
            ['source']
        )
        
        self.clustering_duration = get_or_create_histogram(
            'ai_news_clustering_duration_seconds',
            'Time spent clustering articles'
        )
        
        self.workflow_duration = get_or_create_histogram(
            'ai_news_workflow_duration_seconds',
            'Total time spent in workflow execution'
        )
        
        self.vector_search_duration = get_or_create_histogram(
            'ai_news_vector_search_duration_seconds',
            'Time spent on vector similarity searches'
        )
        
        # Gauges
        self.vector_store_size = get_or_create_gauge(
            'ai_news_vector_store_size',
            'Number of documents in vector store'
        )
        
        self.last_execution_time = get_or_create_gauge(
            'ai_news_last_execution_timestamp',
            'Timestamp of last workflow execution'
        )
        
        self.active_errors = get_or_create_gauge(
            'ai_news_active_errors',
            'Number of active errors in the system'
        )
        
        # Internal metrics storage
        self._execution_metrics = []
        self._error_log = []
        self._start_time = datetime.now()
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
    
    def record_extraction(self, source: str, article_count: int, duration: float):
        """Record article extraction metrics."""
        self.articles_extracted.labels(source=source).inc(article_count)
        self.extraction_duration.labels(source=source).observe(duration)
        logger.debug(f"Recorded extraction: {source} - {article_count} articles in {duration:.2f}s")
    
    def record_clustering(self, cluster_count: int, duration: float):
        """Record clustering metrics."""
        self.clusters_created.inc(cluster_count)
        self.clustering_duration.observe(duration)
        logger.debug(f"Recorded clustering: {cluster_count} clusters in {duration:.2f}s")
    
    def record_duplicates_removed(self, count: int):
        """Record duplicate removal metrics."""
        self.duplicates_removed.inc(count)
        logger.debug(f"Recorded duplicates removed: {count}")
    
    def record_post_generation(self, sent: bool = True, error: str = None):
        """Record post generation and sending metrics."""
        self.posts_generated.inc()
        
        if sent:
            self.posts_sent.labels(status='success').inc()
        else:
            self.posts_sent.labels(status='failed').inc()
            
        if error:
            self.record_error(error)
        
        logger.debug(f"Recorded post generation: sent={sent}")
    
    def record_workflow_execution(self, success: bool, duration: float, error: str = None):
        """Record workflow execution metrics."""
        status = 'success' if success else 'failed'
        self.workflow_executions.labels(status=status).inc()
        self.workflow_duration.observe(duration)
        self.last_execution_time.set_to_current_time()
        
        if error:
            self.record_error(error)
        
        logger.debug(f"Recorded workflow execution: {status} in {duration:.2f}s")
    
    def record_node_execution(self, node_name: str, success: bool, error: str = None):
        """Record individual node execution metrics."""
        status = 'success' if success else 'failed'
        self.node_executions.labels(node_name=node_name, status=status).inc()
        
        if error:
            self.record_error(f"Node {node_name}: {error}")
        
        logger.debug(f"Recorded node execution: {node_name} - {status}")
    
    def record_vector_search(self, duration: float):
        """Record vector search performance."""
        self.vector_search_duration.observe(duration)
        logger.debug(f"Recorded vector search: {duration:.4f}s")
    
    def record_vector_store_size(self, size: int):
        """Update vector store size gauge."""
        self.vector_store_size.set(size)
        logger.debug(f"Updated vector store size: {size}")
    
    def record_error(self, error: str):
        """Record error in internal log."""
        timestamp = datetime.now()
        self._error_log.append({
            'timestamp': timestamp,
            'error': error
        })
        
        # Keep only recent errors (last 100)
        if len(self._error_log) > 100:
            self._error_log = self._error_log[-100:]
        
        self.active_errors.set(len(self._error_log))
        logger.error(f"Recorded error: {error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_articles_extracted': sum(
                self.articles_extracted.labels(source=source)._value.get() 
                for source in ['arxiv', 'websites', 'telegram']
            ),
            'total_clusters_created': self.clusters_created._value.get(),
            'total_duplicates_removed': self.duplicates_removed._value.get(),
            'total_posts_generated': self.posts_generated._value.get(),
            'successful_posts': self.posts_sent.labels(status='success')._value.get(),
            'failed_posts': self.posts_sent.labels(status='failed')._value.get(),
            'successful_workflows': self.workflow_executions.labels(status='success')._value.get(),
            'failed_workflows': self.workflow_executions.labels(status='failed')._value.get(),
            'vector_store_size': self.vector_store_size._value.get(),
            'active_errors': len(self._error_log),
            'recent_errors': self._error_log[-5:] if self._error_log else [],
            'last_execution': self.last_execution_time._value.get()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        metrics = self.get_metrics()
        
        # Calculate rates
        uptime_hours = metrics['uptime_seconds'] / 3600
        
        return {
            'articles_per_hour': metrics['total_articles_extracted'] / max(uptime_hours, 0.001),
            'clusters_per_article': metrics['total_clusters_created'] / max(metrics['total_articles_extracted'], 1),
            'duplicates_per_article': metrics['total_duplicates_removed'] / max(metrics['total_articles_extracted'], 1),
            'post_success_rate': metrics['successful_posts'] / max(metrics['successful_posts'] + metrics['failed_posts'], 1),
            'workflow_success_rate': metrics['successful_workflows'] / max(metrics['successful_workflows'] + metrics['failed_workflows'], 1),
            'error_rate_per_hour': metrics['active_errors'] / max(uptime_hours, 0.001)
        }