# News Extractor Node

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

try:
    from ...extractors.arxiv_extractor import ArxivExtractor
    from ...extractors.website_extractor import WebsiteExtractor
    from ...extractors.telegram_extractor import TelegramExtractor
    from ..agent_state import AgentState
except ImportError:
    from extractors.arxiv_extractor import ArxivExtractor
    from extractors.website_extractor import WebsiteExtractor
    from extractors.telegram_extractor import TelegramExtractor
    from agents.agent_state import AgentState

logger = logging.getLogger(__name__)

class NewsExtractorNode:
    """Node responsible for extracting news articles from multiple sources."""
    
    def __init__(self, config):
        self.config = config
        self.extractors = {}
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize all news extractors."""
        # arXiv extractor
        if self.config.get('sources.arxiv.enabled', True):
            self.extractors['arxiv'] = ArxivExtractor(self.config)
        
        # Website extractor
        if self.config.get('sources.websites.enabled', True):
            self.extractors['websites'] = WebsiteExtractor(self.config)
        
        # Telegram extractor
        if self.config.get('sources.telegram.enabled', True):
            self.extractors['telegram'] = TelegramExtractor(self.config)
        
        logger.info(f"Initialized {len(self.extractors)} extractors: {list(self.extractors.keys())}")
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute news extraction from all configured sources."""
        logger.info("Starting news extraction from all sources")
        
        all_articles = []
        extraction_results = {}
        
        # Extract from all sources concurrently
        tasks = []
        for source_name, extractor in self.extractors.items():
            task = asyncio.create_task(
                self._extract_with_retry(source_name, extractor),
                name=f"extract_{source_name}"
            )
            tasks.append((source_name, task))
        
        # Wait for all extractions to complete
        for source_name, task in tasks:
            try:
                articles = await task
                all_articles.extend(articles)
                extraction_results[source_name] = {
                    'success': True,
                    'article_count': len(articles),
                    'timestamp': datetime.now()
                }
                logger.info(f"Extracted {len(articles)} articles from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to extract from {source_name}: {str(e)}")
                extraction_results[source_name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                state.setdefault('error_log', []).append(f"Extraction failed for {source_name}: {str(e)}")
        
        # Add metadata to articles
        for article in all_articles:
            article['extracted_at'] = datetime.now()
            article['processing_status'] = 'extracted'
        
        # Update state
        state['raw_articles'] = all_articles
        state.setdefault('metrics', {})['extraction_results'] = extraction_results
        
        # Optional: save extracted texts to debug file
        try:
            if self.config.get('debug.save_extracted_texts', False):
                import json
                debug_path = self.config.get('debug.extracted_texts_path', './logs/extraction_debug.jsonl')
                with open(debug_path, 'a', encoding='utf-8') as f:
                    for a in all_articles:
                        record = {
                            'title': a.get('title'),
                            'content': a.get('content') or a.get('abstract'),
                            'abstract': a.get('abstract'),
                            'url': a.get('url') or a.get('arxiv_url'),
                            'source': a.get('source'),
                            'source_name': a.get('source_name'),
                            'published_date': a.get('published_date'),
                        }
                        try:
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        except Exception:
                            continue
                logger.info(f"Saved {len(all_articles)} extracted items to {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to write extraction debug file: {e}")

        logger.info(f"Total articles extracted: {len(all_articles)}")
        return state
    
    async def _extract_with_retry(self, source_name: str, extractor) -> List[Dict[str, Any]]:
        """Extract articles with retry logic."""
        max_retries = self.config.get('performance.max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Extracting from {source_name} (attempt {attempt + 1})")
                articles = await extractor.extract_latest()
                
                # Validate extracted articles
                validated_articles = self._validate_articles(articles)
                
                logger.info(f"Successfully extracted {len(validated_articles)} valid articles from {source_name}")
                return validated_articles
                
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed for {source_name}: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"All extraction attempts failed for {source_name}")
                    raise
        
        return []
    
    def _validate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter extracted articles."""
        validated = []
        
        for article in articles:
            try:
                # Required fields
                if not article.get('title'):
                    logger.warning("Article missing title, skipping")
                    continue
                
                if not article.get('content') and not article.get('abstract'):
                    logger.warning("Article missing content/abstract, skipping")
                    continue
                
                # Date validation
                pub_date = article.get('published_date')
                if pub_date:
                    if isinstance(pub_date, str):
                        try:
                            # Try to parse date
                            datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        except ValueError:
                            logger.warning(f"Invalid date format: {pub_date}")
                            continue
                
                # Content length validation
                content = article.get('content', article.get('abstract', ''))
                if len(content) < self.config.get('content_filtering.min_content_length', 100):
                    logger.warning("Article content too short, skipping")
                    continue
                
                # Add validation metadata
                article['validation_status'] = 'valid'
                article['validation_timestamp'] = datetime.now()
                
                validated.append(article)
                
            except Exception as e:
                logger.error(f"Error validating article: {str(e)}")
                continue
        
        return validated
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction capabilities."""
        return {
            'configured_sources': list(self.extractors.keys()),
            'source_configs': {
                name: extractor.get_config_summary() 
                for name, extractor in self.extractors.items()
            }
        }
