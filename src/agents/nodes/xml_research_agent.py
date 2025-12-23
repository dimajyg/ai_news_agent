from typing import Dict, Any, List, TypedDict
from datetime import datetime
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.config import Config
from ...core.llm_provider import create_llm_provider, LLMProviderType
from ...utils.logging import AgentLogger
from ...processors.xml_schema import validate_xml, XML_SCHEMA, validate_xml_detailed
from ..tools.research_tools import tools as research_tools

logger = AgentLogger(__name__)

class XMLResearchAgentNode:
    def __init__(self, config: Config, llm_provider_type: LLMProviderType = None):
        self.config = config
        if llm_provider_type is None:
            provider_name = config.get('llm.provider', 'openai')
            try:
                llm_provider_type = LLMProviderType(provider_name)
            except ValueError:
                llm_provider_type = LLMProviderType.OPENAI
        model_name = config.get('content_analysis.llm_model', 'gpt-4')
        api_key = ''
        if llm_provider_type == LLMProviderType.OPENAI:
            api_key = config.get('OPENAI_API_KEY', '')
        elif llm_provider_type == LLMProviderType.ANTHROPIC:
            api_key = config.get('ANTHROPIC_API_KEY', '')
        elif llm_provider_type == LLMProviderType.GOOGLE:
            api_key = config.get('GOOGLE_API_KEY', '')
        elif llm_provider_type == LLMProviderType.DEEPSEEK:
            api_key = config.get('DEEPSEEK_API_KEY', '')
        elif llm_provider_type == LLMProviderType.OPENROUTER:
            api_key = config.get('OPENROUTER_API_KEY', '')
        self.llm_provider = create_llm_provider(llm_provider_type.value, model_name, api_key, temperature=0.2, max_tokens=config.get('content_analysis.max_tokens', 1000))
        self.llm = self.llm_provider.llm
        self.llm_with_tools = self.llm.bind_tools(research_tools)
        self.max_retries = self.config.get('processing.xml.max_agent_retries', 3)
        self.max_cycles = self.config.get('processing.xml.agent_max_cycles', 5)

    async def execute(self, item: Dict[str, Any]) -> str:
        logger.info(f"XMLResearchAgent: start execute for source={item.get('source')} title={item.get('title','')[:80]}")
        
        class State(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            valid: bool
            errors: List[str]
            retry_count: int
            restart_count: int

        graph = StateGraph(State)

        async def analyze(s: State):
            logger.info("XMLResearchAgent: analyze enter")
            mi_min = int(self.config.get('processing.xml.main_idea_target_min', 200))
            mi_max = int(self.config.get('processing.xml.main_idea_target_max', 300))
            uq_min = int(self.config.get('processing.xml.uniqueness_target_min', 100))
            uq_max = int(self.config.get('processing.xml.uniqueness_target_max', 150))
            system_prompt = (
                f"You are a research agent with access to tools. Your workflow:\n\n"
                f"1. **REQUIRED**: Use fetch_web_page_content tool to get the full article from the provided URL\n"
                f"2. If needed, use web_search tool to find related research or context\n"
                f"3. Synthesize all gathered information into a comprehensive XML document\n\n"
                f"**CRITICAL**: The initial content snippet is incomplete. You MUST fetch the full page to write a quality summary.\n\n"
                f"Available tools:\n"
                f"- fetch_web_page_content(url: str) - Fetch full article content\n"
                f"- web_search(query: str, max_results: int) - Search for related information\n\n"
                f"Field Requirements:\n"
                f"- MainIdea ({mi_min}-{mi_max} words): Comprehensive summary covering the research problem, methodology, key findings, and results. Include technical details, experimental setup, and quantitative outcomes.\n"
                f"- Uniqueness ({uq_min}-{uq_max} words): Detailed explanation of what makes this research novel - new approaches, improvements over existing methods, unique insights, or innovative applications. Compare with prior work.\n"
                f"- FurtherResearch: List of exactly 5 short ideas (1-2 sentences each) on how to improve the results, extend the research, or apply it to new domains.\n\n"
                f"- RelevantWorks: List of 0-3 <Work> elements with titles, urls, and short descriptions of relevant prior research.\n"
                f"- UsefulLinks: 1-5 <Link> elements with valid URLs (at least one url for article itself)\n"
                f"XML Schema:\n{XML_SCHEMA}\n\n"
                f"After gathering information with tools, return ONLY the XML document (no markdown, no explanations).\n"
            )
            human_prompt = (
                f"Source: {item.get('source')}\nTitle: {item.get('title','')}\nURL: {item.get('url', item.get('arxiv_url',''))}\n"
                f"Content (initial article):\n{(item.get('content') or item.get('abstract') or '')}\n\n"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            try:
                response = await self.llm_with_tools.ainvoke(messages + (s.get("messages", [])))
                content_len = len(getattr(response, 'content', '') or '')
                tool_calls_count = len(getattr(response, 'tool_calls', []) or [])
                logger.info(f"XMLResearchAgent: analyze response content_len={content_len} tool_calls_count={tool_calls_count}")
                if content_len:
                    logger.info(f"XMLResearchAgent: analyze content_head={getattr(response, 'content', '')[:160]}")
                return {"messages": [response]}
            except Exception as e:
                logger.error(f"XMLResearchAgent: analyze error={type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"XMLResearchAgent: analyze traceback={traceback.format_exc()[:500]}")
                raise

        async def validate(s: State):
            logger.info("XMLResearchAgent: validate enter")
            last = s.get("messages", [])[-1] if s.get("messages") else None
            xml = getattr(last, 'content', '') if last else ''
            if not xml:
                s['valid'] = False
                s['errors'] = ["empty xml"]
                logger.warning("XMLResearchAgent: validate empty xml from last message")
            else:
                ok, errs = validate_xml_detailed(xml)
                # Be less strict - only fail on critical schema errors, not minor formatting
                critical_errors = [e for e in errs if 'element' in e.lower() or 'missing' in e.lower()]
                
                # Check if UsefulLinks is empty
                try:
                    from lxml import etree
                    root = etree.fromstring(xml.encode('utf-8'))
                    links_node = root.find('UsefulLinks')
                    if links_node is not None:
                        links = [l.text.strip() for l in links_node.findall('Link') if (l.text or '').strip()]
                        if len(links) == 0:
                            critical_errors.append("UsefulLinks is empty - must include at least the article URL")
                            logger.warning("XMLResearchAgent: validate found empty UsefulLinks")
                except Exception as e:
                    logger.warning(f"XMLResearchAgent: validate failed to check UsefulLinks: {e}")
                
                if ok or len(critical_errors) == 0:
                    s['valid'] = True
                    s['errors'] = []
                    logger.info(f"XMLResearchAgent: validate accepted (strict_valid={ok}, critical_errors={len(critical_errors)})")
                else:
                    s['valid'] = False
                    s['errors'] = critical_errors[:3]  # Limit to top 3 errors
                    logger.info(f"XMLResearchAgent: validate rejected errors_count={len(critical_errors)}")
                    if critical_errors:
                        logger.info(f"XMLResearchAgent: validate first_error={critical_errors[0]}")
            return s

        async def fix(s: State):
            logger.info("XMLResearchAgent: fix enter")
            mi_min = int(self.config.get('processing.xml.main_idea_target_min', 200))
            mi_max = int(self.config.get('processing.xml.main_idea_target_max', 300))
            uq_min = int(self.config.get('processing.xml.uniqueness_target_min', 100))
            uq_max = int(self.config.get('processing.xml.uniqueness_target_max', 150))
            
            errors_str = '\n'.join(s.get('errors', []))
            
            # Check if UsefulLinks is empty and provide the article URL
            article_url_hint = ""
            if "UsefulLinks is empty" in errors_str:
                article_url = item.get('url', item.get('arxiv_url', ''))
                if article_url:
                    article_url_hint = f"\n\nIMPORTANT: UsefulLinks is empty. Add the article URL to UsefulLinks:\n<UsefulLinks>\n  <Link>{article_url}</Link>\n</UsefulLinks>\n"
            
            system_prompt = (
                f"Fix the XML below to match the schema. Focus on correcting the specific errors listed.\n\n"
                f"Common fixes:\n"
                f"- Ensure all required elements are present: Title, MainIdea, Uniqueness, FurtherResearch, RelevantWorks, UsefulLinks\n"
                f"- FurtherResearch must contain 5 <Idea> elements\n"
                f"- UsefulLinks must contain at least 1 <Link> element (the article URL)\n"
                f"- Wrap text content properly in CDATA if it contains special characters\n"
                f"- Ensure proper XML structure with opening and closing tags\n\n"
                f"Field Requirements:\n"
                f"- MainIdea: {mi_min}-{mi_max} words (comprehensive summary)\n"
                f"- Uniqueness: {uq_min}-{uq_max} words (what makes it novel)\n"
                f"- FurtherResearch: Exactly 5 <Idea> elements with continuation ideas\n"
                f"- UsefulLinks: At least 1 <Link> element (must include article URL)\n\n"
                f"Schema:\n{XML_SCHEMA}\n\n"
                f"Return ONLY the corrected XML (no explanations).\n"
            )
            human_prompt = (
                f"Previous XML:\n{getattr(s.get('messages', [])[-1], 'content', '')}\n\n"
                f"Errors to fix:\n{errors_str}\n"
                f"{article_url_hint}"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            try:
                response = await self.llm.ainvoke(messages)
                fix_content_len = len(getattr(response, 'content', '') or '')
                logger.info(f"XMLResearchAgent: fix produced content_len={fix_content_len}")
                if fix_content_len:
                    logger.info(f"XMLResearchAgent: fix content_head={getattr(response, 'content', '')[:160]}")
                return {"messages": (s.get("messages", []) + [response])}
            except Exception as e:
                logger.error(f"XMLResearchAgent: fix error={type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"XMLResearchAgent: fix traceback={traceback.format_exc()[:500]}")
                raise
        
        tool_node = ToolNode(research_tools)

        def should_continue_tools(s: State):
            last_message = s.get("messages", [])[-1] if s.get("messages") else None
            if last_message is not None and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"XMLResearchAgent: route tools tool_calls_count={len(last_message.tool_calls)}")
                return "tools"
            logger.info("XMLResearchAgent: route validate (no tool_calls)")
            return "validate"
        
        def should_continue_validate(s: State):
            if s["valid"]:
                logger.info("XMLResearchAgent: route END (xml valid)")
                return "Accept"
            else:
                # Check retry limits
                retry_count = s.get("retry_count", 0)
                restart_count = s.get("restart_count", 0)
                
                # If too many retries, give up
                if retry_count >= 10 or restart_count >= 3:
                    logger.warning(f"XMLResearchAgent: giving up after {retry_count} retries and {restart_count} restarts")
                    return "Accept"  # Accept whatever we have
                
                # Check if XML is empty - if so, restart from analyze instead of fix
                last = s.get("messages", [])[-1] if s.get("messages") else None
                xml = getattr(last, 'content', '') if last else ''
                if not xml or len(xml.strip()) < 50:
                    logger.warning(f"XMLResearchAgent: route analyze (empty/too short XML, restart {restart_count + 1})")
                    # Clear messages to restart fresh
                    s["messages"] = []
                    s["restart_count"] = restart_count + 1
                    return "Restart"
                
                logger.info(f"XMLResearchAgent: route fix (xml invalid, retry {retry_count + 1})")
                s["retry_count"] = retry_count + 1
                return "Reject"

        graph.add_node('analyze', analyze)
        graph.add_node('fix', fix)
        graph.add_node('tools', tool_node)
        graph.add_node('validate', validate)

        graph.add_edge(START, 'analyze')
        graph.add_edge('tools', 'analyze')
        graph.add_edge('fix', 'validate')
        graph.add_conditional_edges(
            'analyze',
            should_continue_tools,
            {"tools": 'tools', "validate": 'validate'}
        )
        graph.add_conditional_edges(
            'validate',
            should_continue_validate,
            {"Accept": END, "Reject": 'fix', "Restart": 'analyze'}
        )
        
        # Compile with increased recursion limit to handle fix loops
        app = graph.compile()
        
        # Configure recursion limit to prevent infinite loops (increased to 100)
        run_config = {"recursion_limit": 100}
        initial_state: State = {
            "messages": [],
            "valid": False,
            "errors": [],
            "retry_count": 0,
            "restart_count": 0
        }
        
        try:
            final = await app.ainvoke(initial_state, config=run_config)
            last = final.get('messages', [])[-1] if final.get('messages') else None
            last_len = len(getattr(last, 'content', '') or '') if last else 0
            logger.info(f"XMLResearchAgent: end execute valid={final.get('valid', False)} last_content_len={last_len}")
            return getattr(last, 'content', '') if last else ''
        except Exception as e:
            logger.error(f"XMLResearchAgent: execute failed for {item.get('title', '')[:60]}: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"XMLResearchAgent: traceback={traceback.format_exc()[:800]}")
            return ''
