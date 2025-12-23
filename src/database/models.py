"""
Database models for multi-tenant configuration storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json

Base = declarative_base()


class TelegramGroup(Base):
    """Model for storing Telegram group/channel configurations."""
    
    __tablename__ = 'telegram_groups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String(255), unique=True, nullable=False, index=True)
    chat_type = Column(String(50), nullable=False)  # 'group', 'supergroup', 'channel'
    chat_title = Column(String(500))
    
    # Configuration
    config = Column(JSON, nullable=False, default=dict)
    
    # ChromaDB collection name (unique per group)
    chroma_collection = Column(String(255), unique=True, nullable=False)
    
    # User preferences
    language = Column(String(10), default='en', nullable=False)  # 'en' or 'ru'
    keywords = Column(Text, nullable=True)  # User-defined keywords for filtering
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_run = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    sources = relationship("GroupSource", back_populates="group", cascade="all, delete-orphan")
    run_history = relationship("AgentRunHistory", back_populates="group", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TelegramGroup(chat_id={self.chat_id}, title={self.chat_title})>"
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        if isinstance(self.config, str):
            return json.loads(self.config)
        return self.config or {}
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        current_config = self.get_config_dict()
        current_config.update(updates)
        self.config = current_config
        self.updated_at = datetime.utcnow()


class GroupSource(Base):
    """Model for storing news sources per group."""
    
    __tablename__ = 'group_sources'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey('telegram_groups.id', ondelete='CASCADE'), nullable=False)
    
    source_type = Column(String(50), nullable=False)  # 'arxiv', 'telegram', 'website'
    source_identifier = Column(String(500), nullable=False)  # category, channel, URL
    source_config = Column(JSON, default=dict)  # Additional config per source
    
    is_active = Column(Boolean, default=True, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    group = relationship("TelegramGroup", back_populates="sources")
    
    __table_args__ = (
        UniqueConstraint('group_id', 'source_type', 'source_identifier', name='uix_group_source'),
    )
    
    def __repr__(self):
        return f"<GroupSource(type={self.source_type}, identifier={self.source_identifier})>"


class AgentRunHistory(Base):
    """Model for tracking agent execution history per group."""
    
    __tablename__ = 'agent_run_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey('telegram_groups.id', ondelete='CASCADE'), nullable=False)
    
    # Run details
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False)  # 'running', 'completed', 'failed'
    
    # Results
    articles_extracted = Column(Integer, default=0)
    documents_merged = Column(Integer, default=0)
    top_documents_selected = Column(Integer, default=0)
    post_sent = Column(Boolean, default=False)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Metrics
    execution_time_seconds = Column(Integer, nullable=True)
    metrics = Column(JSON, default=dict)
    
    # Relationships
    group = relationship("TelegramGroup", back_populates="run_history")
    
    def __repr__(self):
        return f"<AgentRunHistory(group_id={self.group_id}, status={self.status}, started_at={self.started_at})>"


class BotSettings(Base):
    """Global bot settings and configuration."""
    
    __tablename__ = 'bot_settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<BotSettings(key={self.key})>"
