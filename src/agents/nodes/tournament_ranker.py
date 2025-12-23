from ...utils.logging import AgentLogger
from typing import List, Dict, Any, Tuple
from datetime import datetime
from lxml import etree

try:
    from ..agent_state import AgentState
except ImportError:
    from agents.agent_state import AgentState

logger = AgentLogger(__name__)

class TournamentRankerNode:
    """Ranks merged XML documents via a tournament, outputs top 10 and saves to vector store.

    Input state keys:
      - `merged_xml_documents`: List[{"cluster_id": int, "xml": str, "metadata": Dict[str, Any]}]

    Output state keys:
      - `top_10_xml_documents`: same structure as input, limited to 10
      - `tournament_bracket`: List[List[Tuple[int,int]]] of indices per round
      - `metrics['tournament_ranker']`: stats
    """

    def __init__(self, config: Dict[str, Any], vector_store: Any):
        self.config = config
        self.vector_store = vector_store
        self.dup_threshold = config.get('ranker.duplicate_threshold', 0.92)
        # Use post_size if available, otherwise fall back to max_top, default to 10
        self.max_top = config.get('ranking.post_size',
                                  config.get('ranker.max_top', 10))

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Starting TournamentRankerNode")

        merged = state.get('merged_xml_documents', [])
        if not merged:
            logger.warning("No merged_xml_documents provided; nothing to rank")
            state['top_10_xml_documents'] = []
            state['tournament_bracket'] = []
            return state

        # First round: remove duplicates via vector store similarity
        unique_items, removed_dups = await self._dedup_first_round(merged)
        logger.info(f"First-round dedup: kept {len(unique_items)}, removed {removed_dups}")

        if not unique_items:
            state['top_10_xml_documents'] = []
            state['tournament_bracket'] = []
            state.setdefault('metrics', {})['tournament_ranker'] = {
                'input': len(merged), 'after_dedup': 0, 'rounds': 0
            }
            return state

        # Seed scores for pairing
        scores = [self._score_document(item['xml']) for item in unique_items]
        order = sorted(range(len(unique_items)), key=lambda i: scores[i], reverse=True)
        items_ordered = [unique_items[i] for i in order]
        scores_ordered = [scores[i] for i in order]

        # Tournament rounds
        bracket: List[List[Tuple[int, int]]] = []
        current_indices = list(range(len(items_ordered)))

        round_num = 0
        while len(current_indices) > self.max_top:
            round_num += 1
            pairs = self._make_pairs(current_indices)
            logger.info(f"Tournament round={round_num} pairs={len(pairs)} candidates={len(current_indices)}")
            bracket.append(pairs)
            winners = []

            for a_idx, b_idx in pairs:
                a_item = items_ordered[a_idx]
                b_item = items_ordered[b_idx]
                a_score = scores_ordered[a_idx]
                b_score = scores_ordered[b_idx]

                winner = self._compare_pair(a_item['xml'], b_item['xml'], a_score, b_score, a_idx, b_idx)
                winners.append(winner)
                logger.debug(f"Match a_idx={a_idx} b_idx={b_idx} a_score={a_score:.3f} b_score={b_score:.3f} winner_idx={winner}")

            # Promote winners to next round set
            current_indices = sorted(set(winners))
            logger.info(f"Tournament round={round_num} winners={len(current_indices)}")

            # If odd item left without a pair, auto-advance
            leftover = [i for i in current_indices if i >= len(items_ordered)]
            if leftover:
                logger.warning(f"Invalid indices in tournament winners: {leftover}")

            if len(current_indices) == len(pairs):
                # No reduction happened; limit to top by score
                current_indices = current_indices[:self.max_top]
                break

        final_indices = current_indices[:self.max_top]
        top_docs = [items_ordered[i] for i in final_indices]

        # Save to vector store when enabled
        if bool(self.config.get('ranker.save_to_vector_store', True)):
            await self._save_to_vector_store(top_docs)

        try:
            titles = []
            for it in top_docs:
                root = etree.fromstring(it['xml'].encode('utf-8'))
                titles.append((root.findtext('Title') or '')[:90])
            logger.info(f"TournamentRankerNode: selected_top={len(top_docs)} titles={titles[:5]}")
        except Exception:
            logger.info(f"TournamentRankerNode: selected_top={len(top_docs)}")
        state['top_10_xml_documents'] = top_docs
        state['tournament_bracket'] = bracket
        state.setdefault('metrics', {})['tournament_ranker'] = {
            'input': len(merged),
            'after_dedup': len(unique_items),
            'rounds': round_num,
            'top_selected': len(top_docs)
        }

        logger.info(f"Tournament ranking complete: selected {len(top_docs)} documents in {round_num} rounds")
        return state

    async def _dedup_first_round(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        kept: List[Dict[str, Any]] = []
        removed = 0
        for item in items:
            title, main_idea = self._extract_title_main(item['xml'])
            query = f"{title} {main_idea}".strip()
            if not query:
                kept.append(item)
                continue
            try:
                results = await self.vector_store.search_similar(query, k=3)
                is_dup = False
                for r in results:
                    if r.get('similarity', 0) >= self.dup_threshold:
                        is_dup = True
                        break
                if is_dup:
                    removed += 1
                    logger.info(f"Dedup: status=duplicate title_len={len(title)}")
                else:
                    kept.append(item)
                    logger.info(f"Dedup: status=unique title_len={len(title)}")
            except Exception as ex:
                logger.warning(f"Vector store check failed, keeping item: {str(ex)}")
                kept.append(item)
                logger.warning(f"Dedup: status=unknown title_len={len(title)} reason={str(ex)}")
        return kept, removed

    def _extract_title_main(self, xml: str) -> Tuple[str, str]:
        try:
            doc = etree.fromstring(xml.encode('utf-8'))
            return (doc.findtext('Title') or ''), (doc.findtext('MainIdea') or '')
        except Exception:
            return '', ''

    def _score_document(self, xml: str) -> float:
        """Compute aggregate score across importance, groundbreaking, novelty, rigor."""
        try:
            doc = etree.fromstring(xml.encode('utf-8'))
        except Exception:
            return 0.0

        title = (doc.findtext('Title') or '')
        main = (doc.findtext('MainIdea') or '')
        uniq = (doc.findtext('Uniqueness') or '')
        citations = doc.find('RelevantWorks')
        links = doc.find('UsefulLinks')

        num_citations = len(citations.findall('Citation')) if citations is not None else 0
        num_links = len(links.findall('Link')) if links is not None else 0

        # Heuristics
        importance = min(1.0, (len(title) * 0.002) + (len(main) * 0.0005))
        groundbreaking = min(1.0, (len(uniq) * 0.0008) + (self._keyword_boost(uniq) * 0.1))
        novelty = min(1.0, 0.6 + (self._keyword_boost(uniq) * 0.3) - (num_citations * 0.01))
        rigor = min(1.0, 0.3 + (num_citations * 0.05) + (num_links * 0.02))

        # Weighted sum
        weights = self.config.get('ranker.weights', {
            'importance': 0.35,
            'groundbreaking': 0.30,
            'novelty': 0.20,
            'rigor': 0.15
        })

        score = (
            importance * weights.get('importance', 0.35) +
            groundbreaking * weights.get('groundbreaking', 0.30) +
            novelty * weights.get('novelty', 0.20) +
            rigor * weights.get('rigor', 0.15)
        )
        return score

    def _keyword_boost(self, text: str) -> float:
        keywords = [
            'state-of-the-art', 'SOTA', 'novel', 'first', 'breakthrough',
            'scalable', 'efficient', 'robust', 'provable', 'theoretical', 'empirical'
        ]
        t = text.lower()
        hits = sum(1 for k in keywords if k in t)
        return min(1.0, hits * 0.15)

    def _make_pairs(self, indices: List[int]) -> List[Tuple[int, int]]:
        pairs = []
        i = 0
        while i < len(indices) - 1:
            pairs.append((indices[i], indices[i+1]))
            i += 2
        # If odd count, last element advances without pairing by returning self-pair
        if i == len(indices) - 1:
            pairs.append((indices[i], indices[i]))
        return pairs

    def _compare_pair(self, a_xml: str, b_xml: str, a_score: float, b_score: float, a_idx: int, b_idx: int) -> int:
        # Tie-breakers: rigor then importance via recomputation
        if a_score > b_score:
            return a_idx
        if b_score > a_score:
            return b_idx
        # Recompute individual dimensions
        def dims(xml: str):
            try:
                doc = etree.fromstring(xml.encode('utf-8'))
                title = (doc.findtext('Title') or '')
                main = (doc.findtext('MainIdea') or '')
                uniq = (doc.findtext('Uniqueness') or '')
                citations = doc.find('RelevantWorks')
                links = doc.find('UsefulLinks')
                num_citations = len(citations.findall('Citation')) if citations is not None else 0
                num_links = len(links.findall('Link')) if links is not None else 0
                rigor = min(1.0, 0.3 + (num_citations * 0.05) + (num_links * 0.02))
                importance = min(1.0, (len(title) * 0.002) + (len(main) * 0.0005))
                return rigor, importance
            except Exception:
                return 0.0, 0.0
        a_r, a_i = dims(a_xml)
        b_r, b_i = dims(b_xml)
        if a_r > b_r:
            return a_idx
        if b_r > a_r:
            return b_idx
        if a_i >= b_i:
            return a_idx
        return b_idx

    async def _save_to_vector_store(self, items: List[Dict[str, Any]]):
        try:
            from langchain_core.documents import Document
            docs = []
            for item in items:
                title, main_idea = self._extract_title_main(item['xml'])
                page_content = f"{title}\n\n{main_idea}".strip()
                metadata = {
                    **(item.get('metadata') or {}),
                    'processed_date': datetime.utcnow().isoformat(),
                    'type': 'research_xml',
                }
                docs.append(Document(page_content=page_content, metadata=metadata))
            if docs:
                await self.vector_store.add_documents(docs)
        except Exception as ex:
            logger.error(f"Failed to save top documents to vector store: {str(ex)}")
