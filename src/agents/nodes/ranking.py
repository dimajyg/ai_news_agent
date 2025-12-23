# Ranking Node Implementation

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

try:
    from ..agent_state import AgentState
except ImportError:
    from agents.agent_state import AgentState

logger = logging.getLogger(__name__)

class RankingNode:
    """Node responsible for ranking clusters by relevance and importance."""
    
    def __init__(self, config):
        self.config = config
        self.weights = config.get('ranking.weights', {
            'cluster_size': 0.4,
            'source_diversity': 0.2,
            'recency': 0.2,
            'author_reputation': 0.2
        })
        # Use post_size if available, otherwise fall back to max_top_clusters, default to 10
        self.max_top_clusters = config.get('ranking.post_size', 
                                          config.get('ranking.max_top_clusters', 10))
    
    async def execute(self, state: AgentState) -> AgentState:
        """Rank clusters and select top ones."""
        logger.info("Starting cluster ranking")
        
        filtered_clusters = state['filtered_clusters']
        if not filtered_clusters:
            logger.warning("No clusters to rank")
            state['top_clusters'] = []
            return state
        
        try:
            # Group clusters
            cluster_groups = self._group_by_clusters(filtered_clusters)
            
            # Score each cluster
            cluster_scores = {}
            for cluster_id, cluster_articles in cluster_groups.items():
                score = self._calculate_cluster_score(cluster_articles)
                cluster_scores[cluster_id] = score
            
            # Sort by score and select top clusters
            sorted_clusters = sorted(
                cluster_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters[:self.max_top_clusters]]
            
            # Get articles from top clusters
            top_clusters = [
                article for article in filtered_clusters
                if article.get('cluster_id') in top_cluster_ids
            ]
            
            logger.info(f"Selected {len(top_clusters)} articles from {len(top_cluster_ids)} top clusters")
            
            state['top_clusters'] = top_clusters
            state.setdefault('metrics', {})['ranking_stats'] = {
                'total_clusters': len(cluster_groups),
                'top_clusters_selected': len(top_cluster_ids),
                'avg_score': np.mean(list(cluster_scores.values())),
                'max_score': max(cluster_scores.values()) if cluster_scores else 0,
                'min_score': min(cluster_scores.values()) if cluster_scores else 0
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Ranking failed: {str(e)}")
            state['error_log'].append(f"Ranking error: {str(e)}")
            # Fallback: return all clusters
            state['top_clusters'] = filtered_clusters[:self.max_top_clusters]
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
    
    def _calculate_cluster_score(self, cluster_articles: List[Dict[str, Any]]) -> float:
        """Calculate relevance score for a cluster."""
        if not cluster_articles:
            return 0.0
        
        scores = {}
        
        # 1. Cluster size score
        cluster_size = len(cluster_articles)
        scores['cluster_size'] = self._normalize_cluster_size(cluster_size)
        
        # 2. Source diversity score
        scores['source_diversity'] = self._calculate_source_diversity(cluster_articles)
        
        # 3. Recency score
        scores['recency'] = self._calculate_recency_score(cluster_articles)
        
        # 4. Author reputation score (mainly for arXiv)
        scores['author_reputation'] = self._calculate_author_reputation(cluster_articles)
        
        # Calculate weighted total score
        total_score = 0.0
        for criterion, weight in self.weights.items():
            total_score += scores.get(criterion, 0.0) * weight
        
        return total_score
    
    def _normalize_cluster_size(self, size: int) -> float:
        """Normalize cluster size to 0-1 range."""
        # Logarithmic scaling: more articles = higher score, but with diminishing returns
        import math
        return min(1.0, math.log(size + 1) / math.log(21))  # log(21) ≈ 3, so size=20 gives score=1
    
    def _calculate_source_diversity(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate source diversity score (0-1)."""
        sources = [article.get('source', 'unknown') for article in articles]
        unique_sources = len(set(sources))
        total_articles = len(articles)
        
        # Diversity score: more unique sources = higher score
        return min(1.0, unique_sources / max(3, total_articles * 0.5))
    
    def _calculate_recency_score(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate recency score based on publication dates."""
        try:
            dates = []
            for article in articles:
                pub_date = article.get('published_date')
                if pub_date:
                    if isinstance(pub_date, str):
                        try:
                            date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            dates.append(date)
                        except:
                            continue
                    elif isinstance(pub_date, datetime):
                        dates.append(pub_date)
            
            if not dates:
                return 0.5  # Neutral score if no dates
            
            # Use the most recent date in the cluster
            most_recent = max(dates)
            days_old = (datetime.now(most_recent.tzinfo) - most_recent).days
            
            # Score: 1.0 for today, 0.8 for yesterday, 0.5 for 1 week ago, 0.1 for 1 month ago
            if days_old <= 0:
                return 1.0
            elif days_old == 1:
                return 0.8
            elif days_old <= 7:
                return 0.5 + (7 - days_old) * 0.3 / 7
            elif days_old <= 30:
                return 0.1 + (30 - days_old) * 0.4 / 23
            else:
                return 0.1
                
        except Exception as e:
            logger.warning(f"Error calculating recency score: {str(e)}")
            return 0.5
    
    def _calculate_author_reputation(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate author reputation score (mainly for arXiv)."""
        try:
            # For now, just check if articles are from arXiv
            # In a more sophisticated implementation, we could:
            # - Track author publication history
            # - Check citation counts
            # - Consider institutional affiliations
            
            arxiv_count = sum(1 for article in articles if article.get('source') == 'arxiv')
            total_count = len(articles)
            
            if total_count == 0:
                return 0.0
            
            # Simple scoring: more arXiv articles = higher reputation score
            return arxiv_count / total_count
            
        except Exception as e:
            logger.warning(f"Error calculating author reputation: {str(e)}")
            return 0.0