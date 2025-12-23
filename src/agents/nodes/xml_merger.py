from ...utils.logging import AgentLogger
from typing import List, Dict, Any
from datetime import datetime
from lxml import etree

try:
    from ..agent_state import AgentState
except ImportError:
    from agents.agent_state import AgentState

try:
    from ...processors.xml_schema import validate_xml_detailed
except ImportError:
    from processors.xml_schema import validate_xml_detailed

logger = AgentLogger(__name__)

class XMLMergerNode:
    """Merges clustered XML-formatted research documents into unified XML per cluster.

    Input state keys:
      - `xml_clusters`: List[{"cluster_id": int, "documents": List[str]}]

    Output state keys:
      - `merged_xml_documents`: List[{"cluster_id": int, "xml": str, "metadata": Dict[str, Any]}]
      - `metrics['xml_merger']`: dict with counts and validation stats
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Starting XML merger for clustered research documents")

        clusters = state.get('xml_clusters', [])
        if not clusters:
            logger.warning("No xml_clusters provided; skipping merge")
            state['merged_xml_documents'] = []
            return state

        logger.info(f"XMLMergerNode: input_clusters={len(clusters)} total_docs={sum(len(e.get('documents', [])) for e in clusters)}")
        merged_docs: List[Dict[str, Any]] = []
        validation_failures: List[Dict[str, Any]] = []

        for entry in clusters:
            cluster_id = entry.get('cluster_id')
            docs: List[str] = entry.get('documents', [])

            if not docs:
                continue

            try:
                merged_xml = self._merge_cluster_documents(docs)

                # Validate merged output against schema
                is_valid, errors = validate_xml_detailed(merged_xml)
                if not is_valid:
                    logger.warning(f"Merged XML failed validation for cluster {cluster_id}: {errors}")
                    validation_failures.append({'cluster_id': cluster_id, 'errors': errors})
                    # Skip invalid merge but keep original docs for downstream if configured
                    if self.config.get('xml_merger.keep_on_validation_failure', False):
                        merged_docs.append({
                            'cluster_id': cluster_id,
                            'xml': merged_xml,
                            'metadata': {
                                'cluster_id': cluster_id,
                                'validated': False,
                                'validation_errors': errors,
                                'merged_at': datetime.utcnow().isoformat(),
                            }
                        })
                    continue

                try:
                    root = etree.fromstring(merged_xml.encode('utf-8'))
                    title = (root.findtext('Title') or '')
                    citations = root.find('RelevantWorks')
                    links = root.find('UsefulLinks')
                    c_count = len(citations.findall('Citation')) if citations is not None else 0
                    l_count = len(links.findall('Link')) if links is not None else 0
                    logger.info(f"XMLMergerNode: merged cluster_id={cluster_id} title_len={len(title)} citations={c_count} links={l_count}")
                except Exception:
                    logger.info(f"XMLMergerNode: merged cluster_id={cluster_id} (summary unavailable)")
                logger.info(f"XMLMergerNode: status=success cluster_id={cluster_id} docs_input={len(docs)}")
                merged_docs.append({
                    'cluster_id': cluster_id,
                    'xml': merged_xml,
                    'metadata': {
                        'cluster_id': cluster_id,
                        'validated': True,
                        'merged_at': datetime.utcnow().isoformat(),
                    }
                })
            except Exception as ex:
                logger.error(f"Error merging cluster {cluster_id}: {str(ex)}")
                state.setdefault('error_log', []).append(f"xml_merger cluster {cluster_id}: {str(ex)}")
                logger.error(f"XMLMergerNode: status=failure cluster_id={cluster_id} reason={str(ex)}")

        state['merged_xml_documents'] = merged_docs
        state.setdefault('metrics', {})['xml_merger'] = {
            'clusters_input': len(clusters),
            'clusters_merged': len(merged_docs),
            'validation_failures': len(validation_failures)
        }

        logger.info(
            f"XML merger complete: merged {len(merged_docs)}/{len(clusters)} clusters, "
            f"validation failures: {len(validation_failures)}"
        )

        return state

    def _merge_cluster_documents(self, docs: List[str]) -> str:
        """Merge multiple XML `Document` elements into a single unified `Document`.

        Strategy:
          - Title: choose the longest non-empty title; fallback to first.
          - MainIdea: concatenate distinct summaries with separators.
          - Uniqueness: aggregate distinct points, deduplicated by content.
          - RelevantWorks/Citation: union unique citations (string match).
          - UsefulLinks/Link: union unique links (URI).
          - Attributes:
            * source: use majority source if consistent; else 'merged'
            * sourceName: "merge-of:<n>" indicating count (avoid exceeding schema)
            * publishedDate: most recent if any
        """

        parsed = []
        auto_fix = bool(self.config.get('xml_merger.auto_fix_common_entities', False))
        for xml in docs:
            try:
                to_parse = xml
                if auto_fix:
                    # Replace bare ampersands that are not part of entity refs
                    import re
                    to_parse = re.sub(r"&(?![a-zA-Z#0-9]+;)", "&amp;", to_parse)
                parsed.append(etree.fromstring(to_parse.encode('utf-8')))
            except Exception as ex:
                logger.warning(f"Skipping invalid XML during merge: {str(ex)}")

        if not parsed:
            raise ValueError("No valid XML documents to merge")

        # Collect fields
        titles: List[str] = []
        main_ideas: List[str] = []
        uniqueness_points: List[str] = []
        further_ideas: List[str] = []
        citations: List[str] = []
        links: List[str] = []
        sources: List[str] = []
        source_names: List[str] = []
        pub_dates: List[str] = []

        for doc in parsed:
            t = doc.findtext('Title') or ''
            mi = doc.findtext('MainIdea') or ''
            uq = doc.findtext('Uniqueness') or ''
            titles.append(t.strip())
            if mi.strip():
                main_ideas.append(mi.strip())
            if uq.strip():
                uniqueness_points.append(uq.strip())

            # Collect FurtherResearch ideas
            fr = doc.find('FurtherResearch')
            if fr is not None:
                for idea in fr.findall('Idea'):
                    val = (idea.text or '').strip()
                    if val:
                        further_ideas.append(val)

            rw = doc.find('RelevantWorks')
            if rw is not None:
                for c in rw.findall('Citation'):
                    val = (c.text or '').strip()
                    if val:
                        citations.append(val)

            ul = doc.find('UsefulLinks')
            if ul is not None:
                for l in ul.findall('Link'):
                    val = (l.text or '').strip()
                    if val:
                        links.append(val)

            src = doc.get('source')
            if src:
                sources.append(src)
            sn = doc.get('sourceName')
            if sn:
                source_names.append(sn)
            pd = doc.get('publishedDate')
            if pd:
                pub_dates.append(pd)

        # Title selection: longest non-empty
        title = max(titles, key=lambda s: len(s)) if any(titles) else (titles[0] if titles else '')

        # MainIdea: concatenate distinct with separators
        def _distinct(seq: List[str]) -> List[str]:
            seen = set()
            result = []
            for s in seq:
                k = s.lower()
                if k not in seen:
                    seen.add(k)
                    result.append(s)
            return result

        main_idea = '\n\n'.join(_distinct(main_ideas))
        uniqueness = '\n\n'.join(_distinct(uniqueness_points))
        further_ideas_unique = _distinct(further_ideas)
        citations_unique = _distinct(citations)
        links_unique = _distinct(links)

        # Attributes resolution
        source = 'merged'
        if sources:
            # majority vote if a single winner with >50%
            from collections import Counter
            cnt = Counter(sources)
            winner, freq = cnt.most_common(1)[0]
            if freq > len(sources) / 2:
                source = winner

        source_name = f"merge-of:{len(source_names) or len(parsed)}"

        # publishedDate: choose most recent ISO date-like string
        def _parse_dt(s: str):
            try:
                return datetime.fromisoformat(s.replace('Z', '+00:00'))
            except Exception:
                return None

        parsed_dates = [(_parse_dt(d), d) for d in pub_dates if d]
        published_date = ''
        if parsed_dates:
            parsed_dates = [p for p in parsed_dates if p[0] is not None]
            if parsed_dates:
                published_date = max(parsed_dates, key=lambda x: x[0])[1]

        # Build merged XML
        root = etree.Element('Document')
        if source:
            root.set('source', source)
        if source_name:
            root.set('sourceName', source_name)
        if published_date:
            root.set('publishedDate', published_date)

        title_el = etree.SubElement(root, 'Title')
        title_el.text = title or 'Untitled'

        mi_el = etree.SubElement(root, 'MainIdea')
        mi_el.text = main_idea or 'No summary available.'

        uq_el = etree.SubElement(root, 'Uniqueness')
        uq_el.text = uniqueness or 'Not specified.'

        # Add FurtherResearch element
        fr_el = etree.SubElement(root, 'FurtherResearch')
        if further_ideas_unique:
            for idea in further_ideas_unique[:5]:  # Limit to 5 ideas
                idea_el = etree.SubElement(fr_el, 'Idea')
                idea_el.text = idea
        else:
            # Add empty element to satisfy schema
            pass

        rw_el = etree.SubElement(root, 'RelevantWorks')
        for c in citations_unique:
            c_el = etree.SubElement(rw_el, 'Citation')
            c_el.text = c

        ul_el = etree.SubElement(root, 'UsefulLinks')
        for l in links_unique:
            l_el = etree.SubElement(ul_el, 'Link')
            l_el.text = l

        return etree.tostring(root, encoding='unicode')
