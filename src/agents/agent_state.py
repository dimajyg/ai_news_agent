# Agent State Definition
# This module defines the shared state structure used by all agent nodes
# to avoid circular imports

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class AgentState(TypedDict):
    """Shared state structure for the news agent workflow."""
    
    # Input parameters
    query: str
    date_range: str
    sources: List[str]
    
    # Processing state
    raw_articles: List[Dict[str, Any]]
    clustered_articles: List[Dict[str, Any]]
    unique_articles: List[Dict[str, Any]]
    ranked_articles: List[Dict[str, Any]]
    selected_articles: List[Dict[str, Any]]
    
    # Analysis results
    content_analysis: List[Dict[str, Any]]
    generated_posts: List[Dict[str, Any]]
    
    # Output tracking
    posted_articles: List[Dict[str, Any]]
    errors: List[str]
    
    # Metadata
    timestamp: str
    session_id: str
    llm_provider: str