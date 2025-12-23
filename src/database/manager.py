"""
Database manager for multi-tenant operations.
"""

import os
import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.orm import sessionmaker, Session, attributes
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from .models import Base, TelegramGroup, GroupSource, AgentRunHistory, BotSettings

logger = logging.getLogger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """Convert datetime and other non-JSON-serializable objects to strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


class DatabaseManager:
    """Manager for database operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://ai_news_user:ai_news_password@localhost:5432/ai_news_agent'
        )
        
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and create tables."""
        logger.info(f"Initializing database connection...")
        
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
            
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            
            Base.metadata.create_all(self.engine)
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.session_factory()
    
    @asynccontextmanager
    async def session_scope(self):
        """Provide a transactional scope for database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def create_or_get_group(
        self,
        chat_id: str,
        chat_type: str,
        chat_title: Optional[str] = None
    ) -> TelegramGroup:
        """Create or get a Telegram group configuration."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            
            if not group:
                chroma_collection = f"ai_news_{chat_id.replace('-', '_')}"
                
                group = TelegramGroup(
                    chat_id=chat_id,
                    chat_type=chat_type,
                    chat_title=chat_title,
                    chroma_collection=chroma_collection,
                    config=self._get_default_config()
                )
                session.add(group)
                session.flush()
                logger.info(f"Created new group: {chat_id} ({chat_title})")
            elif chat_title and group.chat_title != chat_title:
                group.chat_title = chat_title
                group.updated_at = datetime.utcnow()
                logger.info(f"Updated group title: {chat_id} -> {chat_title}")
            
            return group
    
    async def get_group(self, chat_id: str) -> Optional[TelegramGroup]:
        """Get a Telegram group by chat_id."""
        async with self.session_scope() as session:
            return session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
    
    async def get_all_active_groups(self) -> List[TelegramGroup]:
        """Get all active groups."""
        async with self.session_scope() as session:
            return session.query(TelegramGroup).filter_by(is_active=True).all()
    
    async def add_source(
        self,
        chat_id: str,
        source_type: str,
        source_identifier: str,
        source_config: Optional[Dict[str, Any]] = None
    ) -> GroupSource:
        """Add a news source to a group."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                raise ValueError(f"Group not found: {chat_id}")
            
            existing = session.query(GroupSource).filter_by(
                group_id=group.id,
                source_type=source_type,
                source_identifier=source_identifier
            ).first()
            
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                    existing.updated_at = datetime.utcnow()
                    logger.info(f"Reactivated source: {source_type}:{source_identifier}")
                return existing
            
            source = GroupSource(
                group_id=group.id,
                source_type=source_type,
                source_identifier=source_identifier,
                source_config=source_config or {}
            )
            session.add(source)
            logger.info(f"Added source to {chat_id}: {source_type}:{source_identifier}")
            
            return source
    
    async def remove_source(
        self,
        chat_id: str,
        source_type: str,
        source_identifier: str
    ) -> bool:
        """Remove (deactivate) a news source from a group."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                return False
            
            source = session.query(GroupSource).filter_by(
                group_id=group.id,
                source_type=source_type,
                source_identifier=source_identifier
            ).first()
            
            if source:
                source.is_active = False
                source.updated_at = datetime.utcnow()
                logger.info(f"Removed source from {chat_id}: {source_type}:{source_identifier}")
                return True
            
            return False
    
    async def get_group_sources(self, chat_id: str) -> List[GroupSource]:
        """Get all active sources for a group."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                return []
            
            return session.query(GroupSource).filter_by(
                group_id=group.id,
                is_active=True
            ).all()
    
    async def update_group_config(
        self,
        chat_id: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """Update group configuration."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                return False
            
            group.update_config(config_updates)
            logger.info(f"Updated config for {chat_id}")
            return True
    
    async def create_run_history(
        self,
        chat_id: str,
        status: str = 'running'
    ) -> AgentRunHistory:
        """Create a new run history entry."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                raise ValueError(f"Group not found: {chat_id}")
            
            run = AgentRunHistory(
                group_id=group.id,
                status=status,
                started_at=datetime.utcnow()
            )
            session.add(run)
            session.flush()
            
            return run
    
    async def update_run_history(
        self,
        run_id: int,
        status: Optional[str] = None,
        articles_extracted: Optional[int] = None,
        documents_merged: Optional[int] = None,
        top_documents_selected: Optional[int] = None,
        post_sent: Optional[bool] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        execution_time_seconds: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update run history entry."""
        async with self.session_scope() as session:
            run = session.query(AgentRunHistory).filter_by(id=run_id).first()
            if not run:
                return
            
            if status:
                run.status = status
            if status in ['completed', 'failed']:
                run.completed_at = datetime.utcnow()
            if articles_extracted is not None:
                run.articles_extracted = articles_extracted
            if documents_merged is not None:
                run.documents_merged = documents_merged
            if top_documents_selected is not None:
                run.top_documents_selected = top_documents_selected
            if post_sent is not None:
                run.post_sent = post_sent
            if error_message:
                run.error_message = error_message
            if error_traceback:
                run.error_traceback = error_traceback
            if execution_time_seconds is not None:
                run.execution_time_seconds = execution_time_seconds
            if metrics:
                run.metrics = serialize_for_json(metrics)
    
    async def get_group_run_history(
        self,
        chat_id: str,
        limit: int = 10
    ) -> List[AgentRunHistory]:
        """Get run history for a group."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if not group:
                return []
            
            return session.query(AgentRunHistory).filter_by(
                group_id=group.id
            ).order_by(AgentRunHistory.started_at.desc()).limit(limit).all()
    
    async def update_last_run(self, chat_id: str):
        """Update the last run timestamp for a group."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if group:
                group.last_run = datetime.utcnow()
    
    async def set_language(self, chat_id: str, language: str) -> bool:
        """Set the language preference for a group."""
        if language not in ['en', 'ru']:
            return False
        
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if group:
                group.language = language
                group.updated_at = datetime.utcnow()
                logger.info(f"Set language for {chat_id} to {language}")
                return True
            return False
    
    async def set_keywords(self, chat_id: str, keywords: str) -> bool:
        """Set the keywords for content filtering."""
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if group:
                group.keywords = keywords
                group.updated_at = datetime.utcnow()
                logger.info(f"Set keywords for {chat_id}: {keywords}")
                return True
            return False
    
    async def set_post_size(self, chat_id: str, post_size: int) -> bool:
        """Set the post size (number of top documents)."""
        if post_size < 1 or post_size > 50:
            return False
        
        async with self.session_scope() as session:
            group = session.query(TelegramGroup).filter_by(chat_id=chat_id).first()
            if group:
                config = group.get_config_dict()
                if 'ranking' not in config:
                    config['ranking'] = {}
                config['ranking']['post_size'] = post_size
                group.config = config
                # Mark the config field as modified so SQLAlchemy detects the change
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(group, 'config')
                group.updated_at = datetime.utcnow()
                logger.info(f"Set post_size for {chat_id} to {post_size}")
                return True
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for a new group."""
        return {
            "clustering": {
                "algorithm": "dbscan",
                "metric": "cosine",
                "dbscan_eps": 0.2,
                "dbscan_min_samples": 2,
                "min_cluster_size": 1,
                "max_cluster_size": 10,
                "embedding_model": "qwen/qwen3-embedding-8b"
            },
            "ranking": {
                "weights": {
                    "cluster_size": 0.4,
                    "source_diversity": 0.2,
                    "recency": 0.2,
                    "author_reputation": 0.2
                },
                "max_top_clusters": 10,
                "post_size": 10  # Number of top documents to include in the post
            },
            "llm": {
                "provider": "openrouter"
            },
            "content_analysis": {
                "llm_model": "google/gemini-2.5-flash",
                "max_tokens": 10000,
                "temperature": 0.3
            },
            "post_generation": {
                "llm_model": "google/gemini-2.5-flash",
                "max_tokens": 20000,
                "temperature": 0.7
            },
            "vector_store": {
                "similarity_threshold": 0.90,
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "processing": {
                "xml": {
                    "use_research_agent": True,
                    "translate_to_english": True,
                    "web_enrich_if_empty": True,
                    "web_enrich_if_short": True,
                    "min_main_idea_words": 120,
                    "min_uniqueness_words": 60
                }
            }
        }
    
    async def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
