from typing import Dict, Any, List
from .base_processor import BaseProcessor, _build_xml_document
from ..utils.config import Config

class ArxivProcessor(BaseProcessor):
    def __init__(self, config: Config):
        self.config = config

    async def process(self, item: Dict[str, Any]) -> str:
        title = item.get('title', '')
        content = item.get('abstract', '')
        url = item.get('arxiv_url') or item.get('url', '')
        
        links: List[str] = []
        if url:
            links.append(url)
        if item.get('doi'):
            links.append(f"https://doi.org/{item['doi']}")
        
        main_idea = content[:500] if content else 'No abstract available'
        uniqueness = f"arXiv preprint on {title[:100]}"
        citations: List[str] = []
        
        return _build_xml_document(title, main_idea, uniqueness, citations, links, {
            'source': 'arxiv',
            'source_name': 'arXiv',
            'published_date': item.get('published_date', '')
        })
