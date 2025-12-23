import asyncio
from pathlib import Path
from typing import Dict, Any, List
from ..agent_state import AgentState
from ...utils.logging import AgentLogger
from ...utils.config import Config
from ...processors.arxiv_processor import ArxivProcessor
from ...processors.website_processor import WebsiteProcessor
from ...processors.telegram_processor import TelegramProcessor
from ...processors.xml_schema import validate_xml

logger = AgentLogger(__name__)

class XMLUnificationNode:
    def __init__(self, config: Config):
        self.config = config
        self.proc_map = {
            'arxiv': ArxivProcessor(config),
            'website': WebsiteProcessor(config),
            'telegram': TelegramProcessor(config),
        }
        self.use_agent = bool(self.config.get('processing.xml.use_research_agent', True))
        if self.use_agent:
            from .xml_research_agent import XMLResearchAgentNode  # lazy import
            self.agent = XMLResearchAgentNode(config)

    async def _process_single_item(self, item: Dict[str, Any]) -> str:
        """Process a single item to XML, using agent or processor with retry logic."""
        src = item.get('source', '')
        key = 'website' if src == 'website' else ('arxiv' if src == 'arxiv' else ('telegram' if src == 'telegram' else 'website'))
        title_preview = item.get('title', '')[:60]
        
        xml = None
        
        # Try agent first if enabled
        if self.use_agent:
            try:
                xml = await self.agent.execute(item)
                if xml and validate_xml(xml):
                    logger.info(f"Agent successfully produced XML for {title_preview}")
                    return xml
                elif not xml:
                    logger.warning(f"Agent returned empty XML for {title_preview}")
                else:
                    logger.warning(f"Agent returned invalid XML for {title_preview}")
            except Exception as e:
                logger.error(f"Agent failed for {title_preview}: {type(e).__name__}: {str(e)}")
        
        # Fallback to processor with retry
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Processor attempt {attempt + 1}/{max_retries} for {title_preview}")
                xml = await self.proc_map[key].process(item)
                
                if xml and validate_xml(xml):
                    logger.info(f"Processor successfully produced XML for {title_preview} on attempt {attempt + 1}")
                    return xml
                elif not xml:
                    logger.warning(f"Processor returned empty XML for {title_preview} on attempt {attempt + 1}")
                else:
                    logger.warning(f"Processor returned invalid XML for {title_preview} on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Processor attempt {attempt + 1} failed for {title_preview}: {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
        
        logger.error(f"All attempts failed for {title_preview}")
        return None

    async def execute(self, state: AgentState) -> AgentState:
        items: List[Dict[str, Any]] = state.get('raw_articles', [])
        
        # Process items with concurrency limit to avoid rate limiting
        max_concurrent = self.config.get('processing.xml.max_concurrent', 5)
        logger.info(f"XMLUnificationNode: processing {len(items)} items with max_concurrent={max_concurrent}")
        
        xml_docs = []
        for i in range(0, len(items), max_concurrent):
            batch = items[i:i + max_concurrent]
            logger.info(f"XMLUnificationNode: processing batch {i//max_concurrent + 1}/{(len(items) + max_concurrent - 1)//max_concurrent} ({len(batch)} items)")
            tasks = [self._process_single_item(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"XMLUnificationNode: batch item {j} failed: {type(result).__name__}: {str(result)}")
                elif result is not None:
                    xml_docs.append(result)
            
            # Delay between batches to avoid rate limiting
            if i + max_concurrent < len(items):
                await asyncio.sleep(1.0)
        
        logger.info(f"XMLUnificationNode: produced {len(xml_docs)} valid XML documents from {len(items)} items")
        state['xml_documents'] = xml_docs
        state['xml_unified'] = True
        
        try:
            if self.config.get('debug.save_xml_documents', False):
                out_dir = Path(self.config.get('debug.xml_output_dir', './logs/xml_docs'))
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, xml in enumerate(xml_docs):
                    fname = out_dir / f'doc_{i+1}.xml'
                    with open(fname, 'w', encoding='utf-8') as f:
                        f.write(xml)
        except Exception:
            pass
        return state
