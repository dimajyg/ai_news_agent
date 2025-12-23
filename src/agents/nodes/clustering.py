# Clustering Node Implementation

import logging
from datetime import datetime
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from ...core.llm_provider import create_embeddings_provider

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
try:
    from ..agent_state import AgentState
except ImportError:
    from agents.agent_state import AgentState

logger = logging.getLogger(__name__)

class ClusteringNode:
    """Node responsible for clustering similar articles together."""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.similarity_threshold = config.get('clustering.similarity_threshold', 0.85)
        self.min_cluster_size = config.get('clustering.min_cluster_size', 2)
        self.max_cluster_size = config.get('clustering.max_cluster_size', 10)
        self.method = 'dbscan'
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on provider."""
        provider = self.config.get('llm.provider', 'openai')
        model_name = self.config.get('clustering.embedding_model', 'text-embedding-ada-002')
        api_key = ''
        if provider == 'openai':
            api_key = self.config.get('OPENAI_API_KEY', '')
        elif provider == 'anthropic':
            api_key = self.config.get('ANTHROPIC_API_KEY', '')
        elif provider == 'google':
            api_key = self.config.get('GOOGLE_API_KEY', '')
        elif provider == 'deepseek':
            api_key = self.config.get('DEEPSEEK_API_KEY', '')
        elif provider == 'openrouter':
            api_key = self.config.get('OPENROUTER_API_KEY', '')
        try:
            if api_key:
                self.embedding_model = create_embeddings_provider(
                    provider,
                    api_key,
                    additional_params={"model": model_name}
                )
                logger.info(f"Initialized embeddings via provider '{provider}' model='{model_name}'")
            else:
                logger.warning(f"No API key for provider {provider}, using simple text similarity")
                self.embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize provider embeddings for {provider}: {e}; using simple text similarity")
            self.embedding_model = None
    
    async def execute(self, state: AgentState) -> AgentState:
        """Cluster similar articles together."""
        logger.info("Starting article clustering")
        
        articles = state.get('raw_articles', [])
        xml_docs = state.get('xml_documents', [])
        if not articles:
            logger.warning("No articles to cluster")
            state['clustered_articles'] = []
            return state
        
        try:
            # Generate embeddings for all articles
            if xml_docs:
                embeddings = await self._generate_embeddings_from_xml(xml_docs)
                items_for_assignment = [{'title': '', 'abstract': d} for d in xml_docs]
            else:
                embeddings = await self._generate_embeddings(articles)
                items_for_assignment = articles
            
            # Perform clustering
            clusters = self._perform_clustering(embeddings)
            
            # Assign cluster information to articles
            clustered_articles = self._assign_clusters(items_for_assignment, clusters)
            
            # Log clustering results
            cluster_stats = self._calculate_cluster_stats(clustered_articles)
            logger.info(f"Clustering complete: {cluster_stats}")
            # Log up to 3 cluster examples (largest clusters, excluding noise -1)
            examples = []
            by_cluster: Dict[int, List[Dict[str, Any]]] = {}
            for a in clustered_articles:
                cid = int(a.get('cluster_id', -1))
                by_cluster.setdefault(cid, []).append(a)
            # Sort clusters by size desc
            ordered = sorted([(cid, arts) for cid, arts in by_cluster.items() if cid != -1], key=lambda x: len(x[1]), reverse=True)
            for cid, arts in ordered[:5]:
                titles = [art.get('title') or (art.get('abstract','')[:60]) for art in arts[:5]]
                examples.append({"cluster_id": cid, "size": len(arts), "examples": titles})
                logger.info(f"Cluster {cid} size={len(arts)} examples={titles}")
            
            state['clustered_articles'] = clustered_articles
            state.setdefault('metrics', {})['clustering_stats'] = cluster_stats
            state.setdefault('metrics', {})['cluster_examples'] = examples
            
            return state
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            state['error_log'].append(f"Clustering error: {str(e)}")
            state['clustered_articles'] = articles  # Fallback to unclustered
            return state
    
    async def _generate_embeddings(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for all articles."""
        logger.info(f"Generating embeddings for {len(articles)} articles")
        
        # Prepare texts for embedding
        texts = []
        for article in articles:
            title = article.get('title', '')
            abstract = article.get('abstract', article.get('content', ''))
            text = f"{title} {abstract}".strip()
            texts.append(text[:8000])  # OpenAI limit
        
        # Generate embeddings based on available model
        if self.embedding_model:
            # Use proper embedding model
            batch_size = self.config.get('performance.embedding_batch_size', 32)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self.embedding_model.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            arr = np.array(all_embeddings)
            # Validate embeddings: check for zero vectors
            norms = np.linalg.norm(arr, axis=1)
            zero_count = int(np.sum(norms == 0))
            if zero_count > 0:
                logger.warning(f"Embeddings contain {zero_count} zero vectors; falling back to simple embeddings")
                return self._generate_simple_embeddings(texts)
            return arr
        else:
            # Fallback: Use simple TF-IDF based embeddings
            logger.info("Using simple TF-IDF embeddings as fallback")
        return self._generate_simple_embeddings(texts)

    async def _generate_embeddings_from_xml(self, xml_docs: List[str]) -> np.ndarray:
        texts = []
        from lxml import etree
        for xml in xml_docs:
            try:
                root = etree.fromstring(xml.encode('utf-8'))
                title = (root.findtext('Title') or '')
                main = (root.findtext('MainIdea') or '')
                uniq = (root.findtext('Uniqueness') or '')
                text = f"{title} {main} {uniq}".strip()
                texts.append(text[:8000])
            except Exception:
                texts.append(xml[:2000])
        return self._generate_simple_embeddings(texts)
    
    def _generate_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple embeddings using TF-IDF (fallback method)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Use TF-IDF to create simple embeddings
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Convert to dense array and return
        return tfidf_matrix.toarray()
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering on embeddings."""
        logger.info("Performing hierarchical clustering")
        # Validate embedding matrix
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Empty embeddings array")
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("Embeddings contain NaN/Inf")
        if np.allclose(np.linalg.norm(embeddings, axis=1), 0):
            raise ValueError("All-zero embeddings")
        # DBSCAN only
        eps = float(self.config.get('clustering.dbscan_eps', 0.25))
        min_samples = int(self.config.get('clustering.dbscan_min_samples', 2))
        metric = self.config.get('clustering.metric', 'cosine')
        clusters = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(embeddings)
        logger.info(f"DBSCAN clusters={len(set(clusters))} eps={eps} min_samples={min_samples} metric={metric}")
        return clusters
    
    def _post_process_clusters(self, embeddings: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Post-process clusters to enforce size constraints."""
        unique_clusters = np.unique(clusters)
        processed_clusters = clusters.copy()
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # No post-processing for DBSCAN; respect discovered clusters
        
        return processed_clusters
    
    def _merge_small_cluster(self, embeddings: np.ndarray, clusters: np.ndarray, 
                           cluster_id: int, cluster_indices: np.ndarray) -> np.ndarray:
        """Merge a small cluster with the most similar larger cluster."""
        if len(cluster_indices) == 0:
            return clusters
        
        # Calculate cluster centroid
        cluster_centroid = np.mean(embeddings[cluster_indices], axis=0)
        
        # Find most similar larger cluster
        unique_clusters = np.unique(clusters)
        best_target_cluster = None
        best_similarity = -1
        
        for target_id in unique_clusters:
            if target_id == cluster_id:
                continue
                
            target_indices = np.where(clusters == target_id)[0]
            if len(target_indices) < self.min_cluster_size:
                continue
            
            target_centroid = np.mean(embeddings[target_indices], axis=0)
            similarity = cosine_similarity([cluster_centroid], [target_centroid])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_target_cluster = target_id
        
        # Merge if suitable target found
        if best_target_cluster is not None and best_similarity > 0.65:
            clusters[clusters == cluster_id] = best_target_cluster
            logger.debug(f"Merged small cluster {cluster_id} into {best_target_cluster} (similarity: {best_similarity:.3f})")
        
        return clusters
    
    def _split_large_cluster(self, embeddings: np.ndarray, clusters: np.ndarray,
                           cluster_id: int, cluster_indices: np.ndarray) -> np.ndarray:
        """Split a large cluster into smaller sub-clusters."""
        if len(cluster_indices) <= self.max_cluster_size:
            return clusters
        
        # Perform sub-clustering within the large cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Determine number of sub-clusters needed
        n_sub_clusters = max(2, len(cluster_indices) // self.max_cluster_size)
        
        sub_clustering = AgglomerativeClustering(
            n_clusters=n_sub_clusters,
            linkage='average',
            metric='cosine'
        )
        
        sub_clusters = sub_clustering.fit_predict(cluster_embeddings)
        
        # Assign new cluster IDs
        max_cluster_id = np.max(clusters)
        for i, sub_cluster_id in enumerate(sub_clusters):
            original_idx = cluster_indices[i]
            clusters[original_idx] = max_cluster_id + sub_cluster_id + 1
        
        logger.debug(f"Split large cluster {cluster_id} into {n_sub_clusters} sub-clusters")
        
        return clusters
    
    def _assign_clusters(self, articles: List[Dict[str, Any]], clusters: np.ndarray) -> List[Dict[str, Any]]:
        """Assign cluster information to articles."""
        clustered_articles = []
        
        for i, article in enumerate(articles):
            cluster_id = int(clusters[i])
            
            # Create enhanced article with cluster information
            enhanced_article = article.copy()
            enhanced_article['cluster_id'] = cluster_id
            enhanced_article['clustering_timestamp'] = datetime.now()
            
            clustered_articles.append(enhanced_article)
        
        # Add cluster size information
        cluster_sizes = {}
        for cluster_id in clusters:
            cluster_sizes[int(cluster_id)] = cluster_sizes.get(int(cluster_id), 0) + 1
        
        for article in clustered_articles:
            article['cluster_size'] = cluster_sizes[article['cluster_id']]
        
        return clustered_articles
    
    def _calculate_cluster_stats(self, clustered_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        if not clustered_articles:
            return {'total_clusters': 0, 'total_articles': 0}
        
        cluster_ids = [article['cluster_id'] for article in clustered_articles]
        unique_clusters = set(cluster_ids)
        
        cluster_sizes = {}
        for cluster_id in cluster_ids:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        sizes = list(cluster_sizes.values())
        
        return {
            'total_clusters': len(unique_clusters),
            'total_articles': len(clustered_articles),
            'avg_cluster_size': np.mean(sizes) if sizes else 0,
            'max_cluster_size': max(sizes) if sizes else 0,
            'min_cluster_size': min(sizes) if sizes else 0,
            'singleton_clusters': sum(1 for size in sizes if size == 1)
        }
