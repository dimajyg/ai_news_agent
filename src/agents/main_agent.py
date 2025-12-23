# Main LangGraph Agent Implementation

import logging
from typing import Dict, Any
from datetime import datetime

try:
    from ..vector_store.chroma_manager import ChromaManager
    from ..utils.config import Config
    from ..utils.metrics import MetricsCollector
    from .graphs.research_workflow import ResearchWorkflow
except ImportError:
    from src.vector_store.chroma_manager import ChromaManager
    from src.utils.config import Config
    from src.utils.metrics import MetricsCollector
    from src.agents.graphs.research_workflow import ResearchWorkflow

logger = logging.getLogger(__name__)

class AINewsAgent:
    """Main AI News Agent using LangGraph for orchestration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = None
        self.workflow = None
        self.metrics = MetricsCollector()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the agent components."""
        logger.info("Initializing AI News Agent")
        
        self.vector_store = ChromaManager(self.config)
        await self.vector_store.initialize()
        
        self.workflow = ResearchWorkflow(self.config, self.vector_store)
        await self.workflow.build()
        
        self._initialized = True
        logger.info("AI News Agent initialization complete")
    

    
    async def run_workflow(self, language: str = 'en', keywords: str = None) -> Dict[str, Any]:
        """Execute the complete agent workflow.
        
        Args:
            language: Target language for the post ('en' or 'ru')
            keywords: Optional keywords for content filtering
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        logger.info(f"Starting agent workflow execution (language={language}, keywords={keywords})")
        start_time = datetime.now()
        
        try:
            final_state = await self.workflow.run(language=language, keywords=keywords)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'execution_time': execution_time,
                'articles_extracted': len(final_state.get('raw_articles', [])),
                'merged_documents': len(final_state.get('merged_xml_documents', [])),
                'top_documents_selected': len(final_state.get('top_10_xml_documents', [])),
                'post_sent': final_state.get('post_sent', False),
                'vector_updated': final_state.get('vector_updated', False),
                'errors': final_state.get('error_log', []),
                'metrics': final_state.get('metrics', {})
            }
            
            logger.info(f"Workflow completed successfully in {execution_time:.2f}s")
            logger.info(
                f"Processed {result['articles_extracted']} articles, "
                f"merged into {result['merged_documents']} documents, "
                f"selected top {result['top_documents_selected']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            self.metrics.record_workflow_failure(str(e))
            raise
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current agent state and metrics."""
        return {
            'initialized': self._initialized,
            'vector_store_connected': self.vector_store.is_connected() if self.vector_store else False,
            'metrics': self.metrics.get_metrics(),
            'config': self.config.get_summary()
        }
    
    async def cleanup(self):
        """Cleanup agent resources."""
        logger.info("Cleaning up AI News Agent")
        await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the agent."""
        logger.info("Shutting down AI News Agent")
        
        if self.vector_store:
            await self.vector_store.close()
        
        self._initialized = False
        logger.info("AI News Agent shutdown complete")
