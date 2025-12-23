"""
Post Processor Node - Orchestrates writing, translation, formatting, and validation.
Processes each article separately through a pipeline.
"""

import asyncio
from typing import Dict, Any, List, TypedDict
from lxml import etree
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from ...utils.logging import AgentLogger
from ...core.llm_provider import create_llm_provider, LLMProviderType

logger = AgentLogger(__name__)


class ArticleState(TypedDict):
    """State for processing a single article."""
    messages: Annotated[List[BaseMessage], add_messages]
    cluster_id: int
    xml_content: str
    title: str
    main_idea: str
    uniqueness: str
    further_research: List[str]  # Future research directions
    links: List[str]
    language: str  # 'en' or 'ru'
    
    # Processing stages
    raw_text: str  # Plain text summary (no formatting)
    translated_text: str  # Translated if needed
    formatted_text: str  # With Telegram HTML formatting
    
    # Validation
    is_valid: bool
    validation_errors: List[str]
    retry_count: int


class PostProcessorNode:
    """
    Processes articles through: Write → Translate → Format → Validate → Fix
    Similar to XMLResearchAgent but for post generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize LLM
        provider_name = config.get('llm.provider', 'openrouter')
        try:
            llm_provider_type = LLMProviderType(provider_name)
        except ValueError:
            llm_provider_type = LLMProviderType.OPENROUTER
        
        model_name = config.get('post_generation.llm_model', 'google/gemini-2.0-flash-exp:free')
        
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
        
        self.llm_provider = create_llm_provider(
            llm_provider_type.value,
            model_name,
            api_key,
            temperature=0.7,
            max_tokens=2000
        )
        self.llm = self.llm_provider.llm
        
        self.max_retries = 3
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all top documents through the pipeline."""
        items = state.get('top_10_xml_documents', [])
        language = state.get('language', 'en')
        
        if not items:
            logger.warning("No top_10_xml_documents; post generation skipped")
            state['telegram_post'] = ""
            state['final_post'] = ""
            return state
        
        logger.info(f"PostProcessor: processing {len(items)} articles in language={language}")
        
        # Process each article through the pipeline
        processed_articles = []
        for i, item in enumerate(items):
            logger.info(f"PostProcessor: processing article {i+1}/{len(items)}")
            try:
                article_text = await self._process_single_article(item, language)
                if article_text:
                    processed_articles.append(article_text)
            except Exception as e:
                logger.error(f"PostProcessor: failed to process article {i+1}: {e}")
        
        # Join all articles
        final_post = "\n\n".join(processed_articles)
        
        state['telegram_post'] = final_post  # English version for vector store
        state['final_post'] = final_post  # Translated version for Telegram
        
        logger.info(f"PostProcessor: completed {len(processed_articles)}/{len(items)} articles")
        return state
    
    async def _process_single_article(self, item: Dict[str, Any], language: str) -> str:
        """Process a single article through the complete pipeline."""
        
        # Extract XML content
        try:
            root = etree.fromstring(item['xml'].encode('utf-8'))
            title = root.findtext('Title') or 'Untitled'
            main_idea = root.findtext('MainIdea') or ''
            uniqueness = root.findtext('Uniqueness') or ''
            
            # Extract FurtherResearch ideas
            further_research = []
            further_node = root.find('FurtherResearch')
            if further_node is not None:
                ideas = further_node.findall('Idea')
                further_research = [idea.text.strip() for idea in ideas if idea.text and idea.text.strip()]
            
            links = []
            links_node = root.find('UsefulLinks')
            if links_node is not None:
                links = [l.text.strip() for l in links_node.findall('Link') if (l.text or '').strip()]
        except Exception as e:
            logger.error(f"PostProcessor: failed to parse XML: {e}")
            return ""
        
        # Build the processing graph
        graph = StateGraph(ArticleState)
        
        # Define nodes
        async def write_node(s: ArticleState) -> ArticleState:
            """Write plain text summary without formatting."""
            logger.info(f"PostProcessor: write_node for '{s['title'][:50]}'")
            
            # Format further research ideas
            further_research_text = ""
            if s.get('further_research'):
                ideas = s['further_research'][:3]  # Use top 3 ideas
                further_research_text = "\n".join([f"- {idea}" for idea in ideas])
            
            prompt = f"""Write a concise research summary (5-10 sentences) for this article.

Title: {s['title']}
Main Idea: {s['main_idea']}
Uniqueness: {s['uniqueness']}"""
            
            if further_research_text:
                prompt += f"""
Future Research Directions:
{further_research_text}"""
            
            prompt += """

Requirements:
1. Core idea - what problem it solves or what it achieves
2. Uniqueness - what makes it novel or different
3. Connections - how it relates to other research or builds on existing work
4. Future directions - potential applications, improvements, or extensions (use the Future Research Directions if provided)
5. Impact - why this matters for the field

Write in plain text (no HTML, no markdown, no formatting).
Be direct and concise. Professional but accessible.
DO NOT include title, emojis, or links - just the summary text.
Focus on making the research accessible while maintaining technical accuracy."""
            
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            s['raw_text'] = response.content.strip()
            logger.info(f"PostProcessor: write_node produced {len(s['raw_text'])} chars")
            return s
        
        async def translate_node(s: ArticleState) -> ArticleState:
            """Translate if needed."""
            if s['language'] == 'en':
                s['translated_text'] = s['raw_text']
                logger.info("PostProcessor: translate_node skipped (language=en)")
                return s
            
            logger.info(f"PostProcessor: translate_node to {s['language']}")
            
            language_names = {'ru': 'Russian (Русский)'}
            target_lang = language_names.get(s['language'], s['language'])
            
            prompt = f"""Translate this AI/ML research summary to {target_lang}.

IMPORTANT:
1. Keep all technical terms unchanged
2. Maintain professional tone
3. Be natural and fluent in {target_lang}
4. DO NOT add any formatting, emojis, or links
5. Return ONLY the translated text

Original text:
{s['raw_text']}

Translated text in {target_lang}:"""
            
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            s['translated_text'] = response.content.strip()
            logger.info(f"PostProcessor: translate_node produced {len(s['translated_text'])} chars")
            return s
        
        async def format_node(s: ArticleState) -> ArticleState:
            """Add Telegram HTML formatting."""
            logger.info("PostProcessor: format_node")
            
            # Determine "Read more" text based on language
            read_more_text = "Подробнее" if s['language'] == 'ru' else "Read more"
            
            # Select emoji based on content (simple heuristic)
            emoji = "🔬"  # Default
            title_lower = s['title'].lower()
            if any(word in title_lower for word in ['llm', 'language model', 'gpt', 'transformer']):
                emoji = "🤖"
            elif any(word in title_lower for word in ['vision', 'image', 'video']):
                emoji = "👁️"
            elif any(word in title_lower for word in ['agent', 'reasoning']):
                emoji = "🧠"
            elif any(word in title_lower for word in ['benchmark', 'evaluation']):
                emoji = "📊"
            
            # Build formatted text
            formatted_parts = []
            
            # Add emoji and bold title
            formatted_parts.append(f"{emoji} <b>{s['title']}</b>")
            
            # Add translated content
            formatted_parts.append(s['translated_text'])
            
            # Add "Read more" link - ALWAYS include if we have a link
            if s['links'] and len(s['links']) > 0:
                link = s['links'][0]  # Use first link
                formatted_parts.append(f'<a href="{link}">{read_more_text}</a>')
            else:
                # No link available - log warning
                logger.warning(f"PostProcessor: No link available for article '{s['title'][:50]}'")
            
            s['formatted_text'] = "\n\n".join(formatted_parts)
            logger.info(f"PostProcessor: format_node produced {len(s['formatted_text'])} chars")
            return s
        
        async def validate_node(s: ArticleState) -> ArticleState:
            """Validate Telegram HTML formatting."""
            logger.info("PostProcessor: validate_node")
            
            text = s['formatted_text']
            errors = []
            
            # Check for unsupported tags
            unsupported_tags = ['<img', '<div', '<span', '<br>', '<br/>', '<p>']
            for tag in unsupported_tags:
                if tag in text:
                    errors.append(f"Unsupported tag found: {tag}")
            
            # Check for proper link format
            if '<a href=' in text:
                if not ('</a>' in text):
                    errors.append("Unclosed <a> tag")
                # Check if link is at the end
                if not text.strip().endswith('</a>'):
                    errors.append("Link should be at the end")
            else:
                # No link found - check if we have links available
                if s.get('links') and len(s['links']) > 0:
                    errors.append("Missing 'Read more' link at the end")
            
            # Check for bold title at start
            if not text.strip().startswith(('🔬 <b>', '🤖 <b>', '👁️ <b>', '🧠 <b>', '📊 <b>')):
                errors.append("Should start with emoji and bold title")
            
            
            s['validation_errors'] = errors
            s['is_valid'] = len(errors) == 0
            
            logger.info(f"PostProcessor: validate_node valid={s['is_valid']} errors={errors}")
            return s
        
        async def fix_node(s: ArticleState) -> ArticleState:
            """Fix validation errors."""
            logger.info(f"PostProcessor: fix_node errors={len(s['validation_errors'])}")
            
            errors_str = '\n'.join(s['validation_errors'])
            
            # Determine "Read more" text based on language
            read_more_text = "Подробнее" if s['language'] == 'ru' else "Read more"
            
            # Get the link if available
            link_info = ""
            if s.get('links') and len(s['links']) > 0:
                link = s['links'][0]
                link_info = f"\nAvailable link: {link}\nUse this link for the 'Read more' section: <a href=\"{link}\">{read_more_text}</a>"
            
            prompt = f"""Fix the following Telegram HTML formatted text to resolve these errors:

Errors:
{errors_str}

Current text:
{s['formatted_text']}
{link_info}

Requirements:
1. Start with emoji and <b>bold title</b>
2. Use only supported tags: <b>, <i>, <u>, <s>, <code>, <a>
3. NO <img>, <div>, <span>, <br>, <p> tags
4. Link MUST be at the end: <a href="URL">{read_more_text}</a>
5. No duplicate content
6. Use double newlines (\\n\\n) for spacing
7. If missing the 'Read more' link, add it at the end using the provided link

Return ONLY the corrected HTML text (no explanations)."""
            
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            s['formatted_text'] = response.content.strip()
            s['retry_count'] = s.get('retry_count', 0) + 1
            
            logger.info(f"PostProcessor: fix_node retry={s['retry_count']}")
            return s
        
        # Add nodes to graph
        graph.add_node('write', write_node)
        graph.add_node('translate', translate_node)
        graph.add_node('format', format_node)
        graph.add_node('validate', validate_node)
        graph.add_node('fix', fix_node)
        
        # Define edges
        graph.add_edge(START, 'write')
        graph.add_edge('write', 'translate')
        graph.add_edge('translate', 'format')
        graph.add_edge('format', 'validate')
        graph.add_edge('fix', 'validate')
        
        # Conditional edge from validate
        def should_fix(s: ArticleState) -> str:
            if s['is_valid']:
                return "accept"
            elif s.get('retry_count', 0) >= self.max_retries:
                logger.warning(f"PostProcessor: max retries reached, accepting anyway")
                return "accept"
            else:
                return "fix"
        
        graph.add_conditional_edges(
            'validate',
            should_fix,
            {"accept": END, "fix": 'fix'}
        )
        
        # Compile and run
        app = graph.compile()
        
        initial_state: ArticleState = {
            "messages": [],
            "cluster_id": item.get('cluster_id', 0),
            "xml_content": item.get('xml', ''),
            "title": title,
            "main_idea": main_idea,
            "uniqueness": uniqueness,
            "further_research": further_research,
            "links": links,
            "language": language,
            "raw_text": "",
            "translated_text": "",
            "formatted_text": "",
            "is_valid": False,
            "validation_errors": [],
            "retry_count": 0
        }
        
        try:
            final_state = await app.ainvoke(initial_state, config={"recursion_limit": 50})
            return final_state['formatted_text']
        except Exception as e:
            logger.error(f"PostProcessor: pipeline failed for '{title[:50]}': {e}")
            return ""
