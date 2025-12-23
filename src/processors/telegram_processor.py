import re
from typing import Dict, Any, List
from .base_processor import BaseProcessor, _build_xml_document
from ..utils.config import Config

def _extract_links(text: str) -> List[str]:
    return re.findall(r'https?://[^\s]+', text or '')

class TelegramProcessor(BaseProcessor):
    def __init__(self, config: Config):
        self.config = config
    
    def _is_research(self, title: str, content: str) -> bool:
        keywords = ['arxiv', 'doi', 'paper', 'dataset', 'benchmark', 'research', 'study', 'algorithm', 'model', 'neural', 'learning']
        text = f"{title} {content}".lower()
        return any(k in text for k in keywords)

    async def process(self, item: Dict[str, Any]) -> str:
        title = item.get('title') or (item.get('content', '').split('\n')[0][:120])
        content = item.get('content', item.get('abstract', ''))
        url = item.get('url', '')
        
        if not self._is_research(title or '', content or ''):
            return ''
        
        main_idea = (content or '')[:500] if content else 'No content available'
        uniqueness = f"Telegram community update on {(title or '')[:100]}"
        citations: List[str] = []
        
        links = [url] if url else []
        extracted_links = _extract_links(content or '')
        links += extracted_links
        
        return _build_xml_document(title or '', main_idea, uniqueness, citations, links, {
            'source': 'telegram',
            'source_name': item.get('source_name', ''),
            'published_date': item.get('published_date', '')
        })
