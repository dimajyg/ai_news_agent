# ChromaDB Vector Store Manager

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB vector store for article deduplication and storage."""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.collection = None
        self.vector_store = None
        self.embedding_function = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize ChromaDB connection and setup."""
        logger.info("Initializing ChromaDB vector store")
        
        try:
            # Suppress Chromadb telemetry warnings BEFORE importing chromadb
            import os
            os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
            os.environ.setdefault('CHROMADB_ANONYMIZED_TELEMETRY', 'False')
            os.environ.setdefault('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS', 'chromadb')
            # Lazy import chromadb and related modules after env setup
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
            # Best-effort telemetry capture no-op
            try:
                import chromadb.telemetry.opentelemetry as chroma_otel
                chroma_otel.capture = lambda *args, **kwargs: None
            except Exception:
                pass
            # As a last resort, suppress direct print spam emitted by telemetry wrappers
            import builtins
            _original_print = builtins.print
            def _filtered_print(*args, **kwargs):
                if args and isinstance(args[0], str):
                    msg = args[0]
                    if (
                        'Failed to send telemetry event' in msg or
                        'ClientStartEvent' in msg or
                        'ClientCreateCollectionEvent' in msg
                    ):
                        return
                return _original_print(*args, **kwargs)
            builtins.print = _filtered_print
            
            # Setup ChromaDB client
            persist_directory = self.config.get(
                'CHROMA_PERSIST_DIRECTORY', 
                './data/chroma_db'
            )
            
            # Redirect noisy stdout/stderr during client initialization
            import io
            import contextlib
            _sink_out, _sink_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(_sink_out), contextlib.redirect_stderr(_sink_err):
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            # Setup embedding function based on provider
            provider = self.config.get('llm.provider', 'openai')
            
            if provider == 'google':
                # For Google provider, use default embedding function (no API key required)
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                logger.info("Using default ChromaDB embeddings for Google provider")
            elif provider == 'anthropic':
                # For Anthropic provider, use default embedding function
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                logger.info("Using default ChromaDB embeddings for Anthropic provider")
            else:  # OpenAI default
                openai_api_key = self.config.get('OPENAI_API_KEY')
                if not openai_api_key:
                    # Use default embedding if no OpenAI key
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    logger.info("No OpenAI API key, using default ChromaDB embeddings")
                else:
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        model_name=self.config.get('clustering.embedding_model', 'text-embedding-ada-002'),
                        api_key=openai_api_key
                    )
                    logger.info("Using OpenAI embeddings")
            
            # Get or create collection
            collection_name = self.config.get(
                'CHROMA_COLLECTION_NAME', 
                'ai_news_articles'
            )
            
            with contextlib.redirect_stdout(_sink_out), contextlib.redirect_stderr(_sink_err):
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Skip LangChain Chroma wrapper to reduce optional dependency requirements
            self.vector_store = None
            
            self._initialized = True
            logger.info(f"ChromaDB initialized with collection: {collection_name}")
            
            # Log collection stats
            stats = await self.get_collection_stats()
            logger.info(f"Collection stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
        finally:
            # Restore original print
            try:
                import builtins
                if ' _filtered_print' in dir():  # no-op, ensure scope exists
                    pass
                builtins.print = _original_print
            except Exception:
                pass
    
    async def check_duplicate(self, article: Dict[str, Any], threshold: float = 0.90) -> Optional[Dict[str, Any]]:
        """Check if an article is a duplicate of existing content."""
        if not self._initialized:
            raise RuntimeError("ChromaManager not initialized")
        
        try:
            # Prepare search content
            title = article.get('title', '')
            abstract = article.get('abstract', article.get('content', ''))
            search_text = f"{title} {abstract}".strip()
            
            if not search_text:
                return None
            
            # Query similar documents
            results = self.collection.query(
                query_texts=[search_text],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            # Check for duplicates
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        duplicate = {
                            'similarity': similarity,
                            'existing_content': results['documents'][0][i],
                            'existing_metadata': results['metadatas'][0][i],
                            'is_duplicate': True
                        }
                        
                        logger.info(f"Found duplicate with similarity {similarity:.3f}")
                        return duplicate
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return None
    
    async def check_cluster_duplicate(self, cluster_articles: List[Dict[str, Any]], threshold: float = 0.85) -> bool:
        """Check if a cluster of articles is already in the vector store."""
        if not cluster_articles:
            return False
        
        try:
            # Use the main/representative article for checking
            main_article = self._select_representative_article(cluster_articles)
            
            # Check if this article (and thus cluster) exists
            duplicate = await self.check_duplicate(main_article, threshold)
            
            if duplicate and duplicate['is_duplicate']:
                logger.info(f"Cluster already exists in vector store (similarity: {duplicate['similarity']:.3f})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cluster duplicate: {str(e)}")
            return False
    
    async def add_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a single document to the vector store."""
        if not self._initialized:
            raise RuntimeError("ChromaManager not initialized")
        
        try:
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            logger.debug(f"Added document {document_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document_id}: {str(e)}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        if not self._initialized:
            raise RuntimeError("ChromaManager not initialized")
        
        try:
            # Accept generic dictionaries or LangChain Document instances
            texts = []
            metadatas = []
            for doc in documents:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    texts.append(getattr(doc, 'page_content'))
                    metadatas.append(getattr(doc, 'metadata'))
                elif isinstance(doc, dict):
                    texts.append(doc.get('page_content', ''))
                    metadatas.append(doc.get('metadata', {}))
                else:
                    continue
            ids = [f"article_{datetime.now().timestamp()}_{i}" for i in range(len(documents))]
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    async def search_similar(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if not self._initialized:
            raise RuntimeError("ChromaManager not initialized")
        
        try:
            # Prepare search parameters
            search_params = {
                'query_texts': [query],
                'n_results': k,
                'include': ["documents", "metadatas", "distances"]
            }
            
            if filter_metadata:
                search_params['where'] = filter_metadata
            
            # Execute search
            results = self.collection.query(**search_params)
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else 0
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        try:
            count = self.collection.count()
            
            # Get sample of recent documents for analysis
            recent_docs = self.collection.peek(limit=min(10, count))
            
            # Analyze date range
            dates = []
            for metadata in recent_docs.get('metadatas', []):
                if 'processed_date' in metadata:
                    try:
                        date = datetime.fromisoformat(metadata['processed_date'])
                        dates.append(date)
                    except:
                        continue
            
            stats = {
                'total_documents': count,
                'status': 'active',
                'collection_name': self.collection.name,
                'recent_sample_size': len(recent_docs.get('documents', [])),
                'date_range': {
                    'oldest': min(dates).isoformat() if dates else None,
                    'newest': max(dates).isoformat() if dates else None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def cleanup_old_documents(self, days_to_keep: int = 90) -> int:
        """Remove documents older than specified days."""
        if not self._initialized:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Get all documents older than cutoff
            all_docs = self.collection.get()
            ids_to_delete = []
            
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if 'processed_date' in metadata:
                    try:
                        doc_date = datetime.fromisoformat(metadata['processed_date'])
                        if doc_date < cutoff_date:
                            ids_to_delete.append(all_docs['ids'][i])
                    except:
                        continue
            
            # Delete old documents
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old documents")
            
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Error cleaning up old documents: {str(e)}")
            return 0
    
    def _select_representative_article(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the most representative article from a cluster."""
        # Prefer articles with more complete metadata
        scored_articles = []
        
        for article in articles:
            score = 0
            
            # Score based on content completeness
            if article.get('title'):
                score += 1
            if article.get('abstract'):
                score += 2
            if article.get('authors'):
                score += 1
            if article.get('published_date'):
                score += 1
            if article.get('source') == 'arxiv':
                score += 2  # Prefer arXiv articles
            
            scored_articles.append((score, article))
        
        # Return highest scored article
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        return scored_articles[0][1] if scored_articles else articles[0]
    
    async def close(self):
        """Close ChromaDB connections."""
        if self.client:
            # ChromaDB persistent client doesn't need explicit closing
            pass
        
        self._initialized = False
        logger.info("ChromaDB manager closed")
    
    def is_connected(self) -> bool:
        """Check if connected to ChromaDB."""
        return self._initialized and self.client is not None
