# Deduplication Node Implementation

import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    from ..agent_state import AgentState
except ImportError:
    from agents.agent_state import AgentState

logger = logging.getLogger(__name__)

class DeduplicationNode:
    """Node responsible for removing duplicate articles using vector similarity."""
    
    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.similarity_threshold = config.get('deduplication.DEDUP_THRESHOLD', 0.90)
    
    async def execute(self, state: AgentState) -> AgentState:
        """Remove duplicate articles from clustered articles."""
        logger.info("Starting deduplication process")
        
        clustered_articles = state['clustered_articles']
        if not clustered_articles:
            logger.warning("No articles to deduplicate")
            state['filtered_clusters'] = []
            return state
        
        try:
            # Group articles by cluster
            cluster_groups = self._group_by_clusters(clustered_articles)
            
            # Filter out duplicate clusters
            unique_clusters = []
            duplicates_removed = 0
            
            for cluster_id, cluster_articles in cluster_groups.items():
                # Check if this cluster is already in vector store
                is_duplicate = await self._check_cluster_duplicate(cluster_articles)
                
                if not is_duplicate:
                    unique_clusters.extend(cluster_articles)
                else:
                    duplicates_removed += len(cluster_articles)
                    logger.info(f"Removed duplicate cluster {cluster_id} with {len(cluster_articles)} articles")
            
            logger.info(f"Deduplication complete: {duplicates_removed} articles removed, {len(unique_clusters)} kept")
            
            state['filtered_clusters'] = unique_clusters
            state.setdefault('metrics', {})['duplicates_removed'] = duplicates_removed
            
            return state
            
        except Exception as e:
            logger.error(f"Deduplication failed: {str(e)}")
            state['error_log'].append(f"Deduplication error: {str(e)}")
            # Fallback: return all clusters
            state['filtered_clusters'] = clustered_articles
            return state
    
    def _group_by_clusters(self, articles: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group articles by cluster ID."""
        cluster_groups = {}
        
        for article in articles:
            cluster_id = article.get('cluster_id')
            if cluster_id is not None:
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(article)
        
        return cluster_groups
    
    async def _check_cluster_duplicate(self, cluster_articles: List[Dict[str, Any]]) -> bool:
        """Check if a cluster of articles is already in the vector store."""
        if not cluster_articles:
            return False
        
        # Select representative article from cluster
        representative = self._select_representative_article(cluster_articles)
        
        # Check if representative is duplicate
        duplicate = await self.vector_store.check_duplicate(representative, self.similarity_threshold)
        
        return duplicate is not None and duplicate.get('is_duplicate', False)
    
    def _select_representative_article(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the most representative article from a cluster."""
        # Prefer articles with more complete metadata and from reliable sources
        scored_articles = []
        
        for article in articles:
            score = 0
            
            # Content completeness
            if article.get('title'):
                score += 2
            if article.get('abstract'):
                score += 3
            if article.get('content'):
                score += 2
            if article.get('authors'):
                score += 1
            if article.get('published_date'):
                score += 1
            
            # Source reliability
            source_scores = {
                'arxiv': 5,
                'mit_technology_review': 4,
                'openai_blog': 4,
                'deepmind_blog': 4,
                'towards_data_science': 3,
                'default': 2
            }
            
            source = article.get('source', 'default')
            score += source_scores.get(source, source_scores['default'])
            
            # Length bonus (but not too long)
            content_length = len(article.get('content', article.get('abstract', '')))
            if 500 <= content_length <= 2000:
                score += 2
            elif content_length > 2000:
                score += 1
            
            scored_articles.append((score, article))
        
        # Return highest scored article
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        return scored_articles[0][1] if scored_articles else articles[0]