import logging
import asyncio
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import operator

try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = None

from ..nodes.news_extractor import NewsExtractorNode
from ..nodes.source_verification import SourceVerificationNode
from ..nodes.xml_unification import XMLUnificationNode
from ..nodes.clustering import ClusteringNode
from ..nodes.deduplication import DeduplicationNode
from ..nodes.ranking import RankingNode
from ..nodes.xml_merger import XMLMergerNode
from ..nodes.tournament_ranker import TournamentRankerNode
from ..nodes.post_processor import PostProcessorNode
from ..nodes.telegram_poster import TelegramPosterNode
from ..nodes.vector_update import VectorUpdateNode

try:
    from ...utils.logging import AgentLogger
    logger = AgentLogger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    """Unified state for complete research pipeline graph."""
    raw_articles: List[Dict[str, Any]]
    clustered_articles: List[Dict[str, Any]]
    filtered_clusters: List[Dict[str, Any]]
    top_clusters: List[Dict[str, Any]]
    xml_documents: List[str]
    xml_clusters: List[Dict[str, Any]]
    merged_xml_documents: List[str]
    top_10_xml_documents: List[str]
    telegram_post: str
    research_post: str  # English version for vector store
    final_post: str  # Translated version for Telegram
    post_sent: bool
    vector_updated: bool
    execution_start_time: Any
    current_step: str
    error_log: List[str]
    retry_count: int
    metrics: Dict[str, Any]
    post_result: Dict[str, Any]
    language: str  # Target language ('en' or 'ru')
    keywords: str  # Optional keywords for filtering


class ResearchWorkflow:
    """
    Unified LangGraph workflow for research pipeline:
    
    1. Extract content from sources (arxiv, websites, telegram)
    2. Verify sources are research-related
    3. Convert each article to XML using XMLResearchAgent
    4. Cluster similar articles
    5. Deduplicate against vector store
    6. Rank clusters
    7. Merge XML documents within clusters
    8. Tournament ranking to select top 10
    9. Generate comprehensive Telegram post
    10. Post to Telegram
    11. Update vector store
    """

    def __init__(self, config, vector_store=None):
        self.config = config
        self.vector_store = vector_store
        self.graph = None

    async def build(self):
        """Build the complete research workflow graph."""
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed")

        extractor = NewsExtractorNode(self.config)
        verifier = SourceVerificationNode(self.config)
        xml_unifier = XMLUnificationNode(self.config)
        clusterer = ClusteringNode(self.config)
        deduplicator = DeduplicationNode(self.config, self.vector_store)
        ranker = RankingNode(self.config)
        merger = XMLMergerNode(self.config)
        tournament = TournamentRankerNode(self.config, self.vector_store)
        post_processor = PostProcessorNode(self.config)
        poster = TelegramPosterNode(self.config)
        vector_updater = VectorUpdateNode(self.config)

        def wrap_node(name: str, func):
            async def wrapped(state: ResearchState) -> ResearchState:
                logger.info(f"ResearchWorkflow: {name}")
                try:
                    result = await func(state)
                    if result is None:
                        logger.error(f"{name} returned None, using original state")
                        state.setdefault('error_log', []).append(f"{name}: returned None")
                        return state
                    return result
                except Exception as e:
                    logger.error(f"{name} failed: {e}")
                    state.setdefault('error_log', []).append(f"{name}: {str(e)}")
                    return state
            return wrapped

        async def extract_impl(state: ResearchState) -> ResearchState:
            logger.info(f"Extract: input state id = {id(state)}")
            result = await extractor.execute(state)
            logger.info(f"Extract: output state id = {id(result)}")
            logger.info(f"Extract: {len(result.get('raw_articles', []))} articles extracted")
            logger.info(f"Extract: result is state? {result is state}")
            return result

        async def verify_impl(state: ResearchState) -> ResearchState:
            raw_articles = state.get('raw_articles', [])
            keywords = state.get('keywords')
            logger.info(f"Verify: input {len(raw_articles)} articles, keywords={keywords}")
            logger.info(f"Verify: state type = {type(state)}, keys = {list(state.keys())[:5]}")
            ver_result = await verifier.execute(raw_articles, keywords=keywords)
            state['raw_articles'] = ver_result['verified']
            state.setdefault('metrics', {})['verification'] = {
                'verified_count': len(ver_result['verified']),
                'rejected_count': len(ver_result['rejected'])
            }
            state.setdefault('error_log', [])
            for r in ver_result['rejected']:
                state['error_log'].append(
                    f"Rejected non-research: {r.get('title','')} reason={r.get('verification_reason','')}"
                )
            logger.info(f"Verify: {len(ver_result['verified'])} verified, {len(ver_result['rejected'])} rejected")
            return state

        async def xml_unify_impl(state: ResearchState) -> ResearchState:
            raw_articles = state.get('raw_articles', [])
            logger.info(f"XML Unify: input {len(raw_articles)} articles")
            result = await xml_unifier.execute(state)
            xml_docs = result.get('xml_documents', [])
            logger.info(f"XML Unify: output {len(xml_docs)} XML documents")
            if len(xml_docs) == 0 and len(raw_articles) > 0:
                logger.warning(f"XML Unify: Failed to convert {len(raw_articles)} articles to XML!")
            return result

        async def cluster_impl(state: ResearchState) -> ResearchState:
            raw_articles = state.get('raw_articles', [])
            xml_docs = state.get('xml_documents', [])
            logger.info(f"Cluster: input {len(raw_articles)} raw articles, {len(xml_docs)} XML documents")
            result = await clusterer.execute(state)
            output = result.get('clustered_articles', [])
            logger.info(f"Cluster: output {len(output)} clustered articles")
            return result

        async def dedup_impl(state: ResearchState) -> ResearchState:
            clustered = state.get('clustered_articles', [])
            logger.info(f"Dedup: input {len(clustered)} articles")
            result = await deduplicator.execute(state)
            filtered = result.get('filtered_clusters', [])
            logger.info(f"Dedup: output {len(filtered)} filtered articles")
            return result

        async def rank_impl(state: ResearchState) -> ResearchState:
            filtered = state.get('filtered_clusters', [])
            logger.info(f"Rank: input {len(filtered)} articles")
            result = await ranker.execute(state)
            top = result.get('top_clusters', [])
            logger.info(f"Rank: output {len(top)} top articles")
            return result

        async def build_clusters_impl(state: ResearchState) -> ResearchState:
            top_clusters = state.get('top_clusters', [])
            logger.info(f"Build Clusters: input {len(top_clusters)} top clusters")
            state['xml_clusters'] = self._build_xml_clusters_from_state(state)
            logger.info(f"Build Clusters: output {len(state.get('xml_clusters', []))} XML clusters")
            return state

        async def merge_impl(state: ResearchState) -> ResearchState:
            xml_clusters = state.get('xml_clusters', [])
            total_docs = sum(len(c.get('documents', [])) for c in xml_clusters)
            logger.info(f"Merge: input {len(xml_clusters)} clusters with {total_docs} documents")
            result = await merger.execute(state)
            merged = result.get('merged_xml_documents', [])
            logger.info(f"Merge: output {len(merged)} merged documents")
            return result

        async def tournament_impl(state: ResearchState) -> ResearchState:
            merged = state.get('merged_xml_documents', [])
            logger.info(f"Tournament: input {len(merged)} merged documents")
            result = await tournament.execute(state)
            top10 = result.get('top_10_xml_documents', [])
            logger.info(f"Tournament: output {len(top10)} top documents")
            return result

        async def post_processor_impl(state: ResearchState) -> ResearchState:
            """Process articles through write → translate → format → validate → fix pipeline."""
            top10 = state.get('top_10_xml_documents', [])
            logger.info(f"Post Processor: input {len(top10)} documents")
            result = await post_processor.execute(state)
            post = result.get('final_post', '')
            logger.info(f"Post Processor: generated post with {len(post)} characters")
            return result

        async def poster_impl(state: ResearchState) -> ResearchState:
            # Use final_post (translated) instead of telegram_post
            telegram_post = state.get('final_post', state.get('telegram_post', ''))
            logger.info(f"Poster: post length {len(telegram_post)} characters")
            if telegram_post:
                post_data = {
                    "title": "AI Research Highlights",
                    "content": telegram_post,
                    "hashtags": ["#AI", "#Research"],
                    "link_preview": False
                }
                result = await poster.send_post(post_data)
                state['post_sent'] = result.get('success', False)
                state['post_result'] = result
                logger.info(f"Poster: post sent = {state['post_sent']}")
            else:
                state['post_sent'] = False
                state['post_result'] = {'success': False, 'error': 'No post content'}
                logger.warning("Poster: No post content to send")
            return state

        async def vector_update_impl(state: ResearchState) -> ResearchState:
            logger.info("Vector Update: starting")
            result = await vector_updater.execute(state)
            logger.info(f"Vector Update: completed, updated = {result.get('vector_updated', False)}")
            return result

        workflow = StateGraph(ResearchState)
        
        workflow.add_node("extract", wrap_node("extract_node", extract_impl))
        workflow.add_node("verify", wrap_node("verify_node", verify_impl))
        workflow.add_node("xml_unify", wrap_node("xml_unify_node", xml_unify_impl))
        workflow.add_node("cluster", wrap_node("cluster_node", cluster_impl))
        workflow.add_node("dedup", wrap_node("dedup_node", dedup_impl))
        workflow.add_node("rank", wrap_node("rank_node", rank_impl))
        workflow.add_node("build_clusters", wrap_node("build_clusters_node", build_clusters_impl))
        workflow.add_node("merge", wrap_node("merge_node", merge_impl))
        workflow.add_node("tournament", wrap_node("tournament_node", tournament_impl))
        workflow.add_node("post_processor", wrap_node("post_processor_node", post_processor_impl))
        workflow.add_node("poster", wrap_node("poster_node", poster_impl))
        workflow.add_node("vector_update", wrap_node("vector_update_node", vector_update_impl))
        
        workflow.add_edge("extract", "verify")
        workflow.add_edge("verify", "xml_unify")
        workflow.add_edge("xml_unify", "cluster")
        workflow.add_edge("cluster", "dedup")
        workflow.add_edge("dedup", "rank")
        workflow.add_edge("rank", "build_clusters")
        workflow.add_edge("build_clusters", "merge")
        workflow.add_edge("merge", "tournament")
        workflow.add_edge("tournament", "post_processor")
        workflow.add_edge("post_processor", "poster")
        workflow.add_edge("poster", "vector_update")
        workflow.add_edge("vector_update", END)
        
        workflow.set_entry_point("extract")
        
        self.graph = workflow.compile()
        logger.info("ResearchWorkflow graph compiled successfully")
        return self.graph

    def _build_xml_clusters_from_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build xml_clusters from state for merger node."""
        logger.info("Building xml_clusters from state")
        from ...processors.xml_schema import validate_xml_detailed

        clustered = state.get('clustered_articles', [])
        xml_clusters: Dict[int, List[str]] = {}

        for item in clustered:
            cid = int(item.get('cluster_id', -1))
            xml_str = item.get('abstract', '')
            if not xml_str or '<Document' not in xml_str:
                continue
            ok, _ = validate_xml_detailed(xml_str)
            if not ok:
                continue
            if cid == -1:
                new_id = -100000 - len(xml_clusters)
                xml_clusters.setdefault(new_id, []).append(xml_str)
            else:
                xml_clusters.setdefault(cid, []).append(xml_str)

        if not xml_clusters:
            logger.warning("No XML in clustered_articles; falling back to xml_documents")
            xml_docs: List[str] = state.get('xml_documents', [])
            batch = []
            cid_counter = 1
            for xml in xml_docs:
                ok, _ = validate_xml_detailed(xml)
                if not ok:
                    continue
                batch.append(xml)
                if len(batch) == 5:
                    xml_clusters[cid_counter] = batch[:]
                    cid_counter += 1
                    batch.clear()
            if batch:
                xml_clusters[cid_counter] = batch[:]

        clusters = [{"cluster_id": cid, "documents": docs} for cid, docs in xml_clusters.items()]
        logger.info(
            f"xml_clusters built: clusters={len(clusters)} "
            f"total_docs={sum(len(d) for d in xml_clusters.values())}"
        )
        return clusters

    async def run(self, initial_state: Dict[str, Any] = None, language: str = 'en', keywords: str = None) -> Dict[str, Any]:
        """
        Execute the complete research workflow.
        
        Args:
            initial_state: Optional initial state. If not provided, creates default state.
            language: Target language for the post ('en' or 'ru')
            keywords: Optional keywords for content filtering
            
        Returns:
            Final state after workflow execution
        """
        if self.graph is None:
            await self.build()

        if initial_state is None:
            initial_state = self._create_initial_state()
        
        # Add language and keywords to state
        initial_state['language'] = language
        initial_state['keywords'] = keywords

        logger.info(f"Starting ResearchWorkflow execution (language={language}, keywords={keywords})")
        logger.info(f"Initial state keys: {list(initial_state.keys())}")
        start_time = datetime.now()

        try:
            logger.info("Calling graph.ainvoke...")
            final_state = await self.graph.ainvoke(initial_state)
            logger.info(f"graph.ainvoke returned: {type(final_state)}")
            
            if final_state is None:
                logger.error("Graph returned None instead of state!")
                return initial_state
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"ResearchWorkflow completed in {execution_time:.2f}s: "
                f"articles={len(final_state.get('raw_articles', []))} "
                f"merged={len(final_state.get('merged_xml_documents', []))} "
                f"top10={len(final_state.get('top_10_xml_documents', []))} "
                f"post_sent={final_state.get('post_sent', False)}"
            )
            
            return final_state
            
        except Exception as e:
            logger.error(f"ResearchWorkflow execution failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _create_initial_state(self) -> ResearchState:
        """Create initial state for workflow."""
        state = ResearchState()
        state['raw_articles'] = []
        state['clustered_articles'] = []
        state['filtered_clusters'] = []
        state['top_clusters'] = []
        state['xml_documents'] = []
        state['xml_clusters'] = []
        state['merged_xml_documents'] = []
        state['top_10_xml_documents'] = []
        state['telegram_post'] = ""
        state['post_sent'] = False
        state['vector_updated'] = False
        state['execution_start_time'] = datetime.now()
        state['current_step'] = ""
        state['error_log'] = []
        state['retry_count'] = 0
        state['metrics'] = {}
        return state
