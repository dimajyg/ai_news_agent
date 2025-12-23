"""
Vector update node for storing processed content in ChromaDB.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json

try:
    from ...utils.config import Config
    from ...vector_store.chroma_manager import ChromaManager
    from ...utils.logging import AgentLogger
except ImportError:
    from src.utils.config import Config
    from src.vector_store.chroma_manager import ChromaManager
    from src.utils.logging import AgentLogger


logger = AgentLogger(__name__)


class VectorUpdateNode:
    """Node for updating vector store with processed content."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = None
    
    async def _get_vector_store(self) -> ChromaManager:
        """Get or create vector store instance."""
        if not self.vector_store:
            self.vector_store = ChromaManager(self.config)
            await self.vector_store.initialize()
        return self.vector_store
    
    async def store_article(
        self, 
        article_data: Dict[str, Any], 
        analysis_data: Dict[str, Any],
        cluster_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a single article with its analysis in the vector store.
        
        Args:
            article_data: Original article data
            analysis_data: Analysis results
            cluster_id: Cluster ID if article belongs to a cluster
            context: Optional context information
            
        Returns:
            Storage result with metadata
        """
        try:
            vector_store = await self._get_vector_store()
            
            # Generate unique ID for the article
            article_id = self._generate_article_id(article_data)
            
            # Prepare content for embedding
            content_text = self._prepare_content_text(article_data, analysis_data)
            
            # Prepare metadata
            metadata = self._prepare_metadata(article_data, analysis_data, cluster_id, context)
            
            logger.info(f"Storing article in vector store: {article_id}")
            
            # Store in vector store
            await vector_store.add_document(
                document_id=article_id,
                content=content_text,
                metadata=metadata
            )
            
            result = {
                "success": True,
                "article_id": article_id,
                "stored_at": datetime.now().isoformat(),
                "content_length": len(content_text),
                "metadata_keys": list(metadata.keys()),
                "cluster_id": cluster_id
            }
            
            logger.info(f"Article stored successfully: {article_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing article {article_data.get('id', 'Unknown')}: {e}")
            return {
                "success": False,
                "error": str(e),
                "article_id": article_data.get("id"),
                "stored_at": datetime.now().isoformat()
            }
    
    async def store_batch(
        self, 
        articles_data: List[Dict[str, Any]], 
        analyses_data: List[Dict[str, Any]],
        cluster_assignments: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Store a batch of articles with their analyses.
        
        Args:
            articles_data: List of article data
            analyses_data: List of analysis results
            cluster_assignments: Mapping of article IDs to cluster IDs
            context: Optional context information
            
        Returns:
            List of storage results
        """
        results = []
        
        # Ensure we have analyses for all articles
        if len(articles_data) != len(analyses_data):
            logger.warning(f"Mismatch in articles ({len(articles_data)}) and analyses ({len(analyses_data)})")
        
        for i, article_data in enumerate(articles_data):
            try:
                # Get corresponding analysis (if available)
                analysis_data = analyses_data[i] if i < len(analyses_data) else {}
                
                # Get cluster ID if available
                article_id = article_data.get("id", "")
                cluster_id = cluster_assignments.get(article_id) if cluster_assignments else None
                
                # Store the article
                result = await self.store_article(
                    article_data, 
                    analysis_data, 
                    cluster_id, 
                    context
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch storage for article {article_data.get('id')}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "article_id": article_data.get("id"),
                    "stored_at": datetime.now().isoformat()
                })
        
        return results
    
    async def update_cluster_assignments(
        self, 
        cluster_results: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update cluster assignments for articles.
        
        Args:
            cluster_results: Clustering results with article assignments
            context: Optional context information
            
        Returns:
            Update result
        """
        try:
            vector_store = await self._get_vector_store()
            
            clusters = cluster_results.get("clusters", [])
            updated_count = 0
            
            logger.info(f"Updating cluster assignments for {len(clusters)} clusters")
            
            for cluster in clusters:
                cluster_id = cluster.get("cluster_id")
                article_ids = cluster.get("article_ids", [])
                
                for article_id in article_ids:
                    try:
                        # Update cluster ID in metadata
                        await vector_store.update_metadata(
                            document_id=article_id,
                            updates={"cluster_id": cluster_id}
                        )
                        updated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error updating cluster for article {article_id}: {e}")
                        continue
            
            result = {
                "success": True,
                "clusters_updated": len(clusters),
                "articles_updated": updated_count,
                "updated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Cluster assignments updated: {updated_count} articles in {len(clusters)} clusters")
            return result
            
        except Exception as e:
            logger.error(f"Error updating cluster assignments: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
    
    def _generate_article_id(self, article_data: Dict[str, Any]) -> str:
        """Generate unique ID for article."""
        # Use URL as primary identifier, fallback to title + source
        url = article_data.get("url", "")
        if url:
            # Create hash of URL
            return hashlib.md5(url.encode()).hexdigest()
        else:
            # Fallback to title + source hash
            title = article_data.get("title", "")
            source = article_data.get("source", "")
            content = f"{title}|{source}|{article_data.get('published_at', '')}"
            return hashlib.md5(content.encode()).hexdigest()
    
    def _prepare_content_text(self, article_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
        """Prepare content text for embedding."""
        title = article_data.get("title", "")
        content = article_data.get("content", "")
        
        # Add analysis insights
        key_insights = analysis_data.get("key_insights", [])
        summary = analysis_data.get("summary", "")
        
        # Combine content
        combined_content = f"Title: {title}\n\n"
        combined_content += f"Content: {content}\n\n"
        
        if key_insights:
            combined_content += f"Key Insights: {' '.join(key_insights)}\n\n"
        
        if summary:
            combined_content += f"Summary: {summary}"
        
        return combined_content.strip()
    
    def _prepare_metadata(
        self, 
        article_data: Dict[str, Any], 
        analysis_data: Dict[str, Any], 
        cluster_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare metadata for storage."""
        metadata = {
            # Basic article info
            "title": article_data.get("title", ""),
            "source": article_data.get("source", ""),
            "url": article_data.get("url", ""),
            "authors": ", ".join(article_data.get("authors", [])),
            "published_at": article_data.get("published_at", ""),
            "content_type": article_data.get("content_type", "article"),
            
            # Analysis results
            "technical_level": analysis_data.get("technical_level", "intermediate"),
            "relevance_score": analysis_data.get("relevance_score", 0.5),
            "novelty_score": analysis_data.get("novelty_score", 0.5),
            "target_audience": analysis_data.get("target_audience", "AI/ML community"),
            
            # Processing metadata
            "processed_at": datetime.now().isoformat(),
            "cluster_id": cluster_id,
            "has_analysis": bool(analysis_data),
            "has_post": False,  # Will be updated when post is sent
            
            # Additional context
            "context": json.dumps(context) if context else "{}",
            "version": "1.0"
        }
        
        # Add key insights as separate metadata
        key_insights = analysis_data.get("key_insights", [])
        if key_insights:
            metadata["key_insights"] = " | ".join(key_insights[:3])  # Top 3 insights
        
        practical_applications = analysis_data.get("practical_applications", [])
        if practical_applications:
            metadata["practical_applications"] = " | ".join(practical_applications[:3])
        
        return metadata
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored content."""
        try:
            vector_store = await self._get_vector_store()
            stats = await vector_store.get_collection_stats()
            
            return {
                "success": True,
                "total_documents": stats.get("total_documents", 0),
                "unique_sources": stats.get("unique_sources", 0),
                "date_range": stats.get("date_range", {}),
                "cluster_count": stats.get("cluster_count", 0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector update node (LangGraph compatible).
        Saves top N XML documents to vector store for deduplication.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        try:
            logger.info("VectorUpdateNode: executing")
            
            # Get top XML documents (not the final post)
            top_xml_docs = state.get('top_10_xml_documents', [])
            
            if not top_xml_docs:
                logger.warning("VectorUpdateNode: No top_10_xml_documents found")
                state['vector_updated'] = False
                return state
            
            logger.info(f"VectorUpdateNode: Saving {len(top_xml_docs)} XML documents to vector store")
            
            # Get vector store
            vector_store = await self._get_vector_store()
            
            # Save each XML document
            saved_count = 0
            for i, doc_item in enumerate(top_xml_docs):
                try:
                    xml_content = doc_item.get('xml', '')
                    cluster_id = doc_item.get('cluster_id', i)
                    metadata = doc_item.get('metadata', {})
                    
                    if not xml_content:
                        logger.warning(f"VectorUpdateNode: Empty XML for document {i}")
                        continue
                    
                    # Extract title from XML for ID generation
                    from lxml import etree
                    try:
                        root = etree.fromstring(xml_content.encode('utf-8'))
                        title = root.findtext('Title') or f'doc_{cluster_id}'
                        main_idea = root.findtext('MainIdea') or ''
                    except Exception as e:
                        logger.warning(f"VectorUpdateNode: Failed to parse XML {i}: {e}")
                        title = f'doc_{cluster_id}'
                        main_idea = xml_content[:500]  # Use first 500 chars
                    
                    # Generate document ID
                    doc_id = hashlib.md5(f"{title}_{cluster_id}".encode()).hexdigest()
                    
                    # Prepare metadata
                    doc_metadata = {
                        'cluster_id': str(cluster_id),
                        'title': title,
                        'stored_at': datetime.now().isoformat(),
                        'document_type': 'research_xml',
                        'source': metadata.get('source', 'unknown')
                    }
                    
                    # Store in vector store (use MainIdea for embedding)
                    await vector_store.add_document(
                        document_id=doc_id,
                        content=main_idea,  # Use MainIdea for similarity search
                        metadata=doc_metadata
                    )
                    
                    saved_count += 1
                    logger.info(f"VectorUpdateNode: Saved document {i+1}/{len(top_xml_docs)}: {title[:50]}")
                    
                except Exception as e:
                    logger.error(f"VectorUpdateNode: Failed to save document {i}: {e}")
                    continue
            
            state['vector_updated'] = True
            state.setdefault('metrics', {})['vector_update'] = {
                'updated_at': datetime.now().isoformat(),
                'success': True,
                'documents_saved': saved_count,
                'total_documents': len(top_xml_docs)
            }
            
            logger.info(f"VectorUpdateNode: completed successfully ({saved_count}/{len(top_xml_docs)} saved)")
            return state
            
        except Exception as e:
            logger.error(f"VectorUpdateNode: failed - {e}")
            state['vector_updated'] = False
            state.setdefault('metrics', {})['vector_update'] = {
                'updated_at': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            return state
    
    async def close(self):
        """Close vector store connection."""
        if self.vector_store:
            await self.vector_store.close()
            self.vector_store = None


def create_vector_update_node(config: Config) -> VectorUpdateNode:
    """Factory function to create vector update node."""
    return VectorUpdateNode(config)


async def vector_update_node(state: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    LangGraph node function for vector store updates.
    
    Args:
        state: Current graph state
        config: Application configuration
        
    Returns:
        Updated state with storage results
    """
    try:
        node = create_vector_update_node(config)
        
        # Get data from state
        articles = state.get("articles", [])
        analysis_results = state.get("analysis_results", [])
        cluster_results = state.get("cluster_results", {})
        
        if not articles:
            logger.warning("No articles found in state for vector storage")
            return {**state, "storage_results": []}
        
        logger.info(f"Storing {len(articles)} articles in vector store")
        
        # Store articles
        storage_results = await node.store_batch(
            articles, 
            analysis_results,
            cluster_assignments=cluster_results.get("assignments"),
            context=state.get("context", {})
        )
        
        # Update cluster assignments if available
        if cluster_results:
            cluster_update_result = await node.update_cluster_assignments(
                cluster_results, 
                context=state.get("context", {})
            )
        else:
            cluster_update_result = {"success": False, "error": "No cluster results"}
        
        # Get storage statistics
        storage_stats = await node.get_storage_stats()
        
        # Update state
        return {
            **state,
            "storage_results": storage_results,
            "cluster_update_result": cluster_update_result,
            "storage_stats": storage_stats,
            "storage_completed": True,
            "total_stored": sum(1 for result in storage_results if result.get("success", False))
        }
        
    except Exception as e:
        logger.error(f"Error in vector update node: {e}")
        return {
            **state,
            "storage_results": [],
            "cluster_update_result": {"success": False, "error": str(e)},
            "storage_stats": {},
            "storage_completed": False,
            "total_stored": 0,
            "error": str(e)
        }
    finally:
        # Clean up
        if 'node' in locals():
            await node.close()
