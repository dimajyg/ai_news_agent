import json
import asyncio
import logging
import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from ...core.llm_provider import create_llm_provider, LLMProviderType
    from ...utils.logging import AgentLogger
except ImportError:
    from core.llm_provider import create_llm_provider, LLMProviderType
    from utils.logging import AgentLogger

logger = AgentLogger(__name__)

class SourceVerificationNode:
    """
    Node responsible for verifying if articles are relevant AI breakthroughs/research
    and filtering out tutorials, opinions, etc.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize LLM
        provider_name = config.get('llm.provider', 'openai')
        try:
            self.provider_type = LLMProviderType(provider_name)
        except ValueError:
            self.provider_type = LLMProviderType.OPENAI
            
        model_name = config.get('verification.llm_model', config.get('llm.model_name', 'gpt-4o'))
        
        # Get API key based on provider
        api_key = ""
        if self.provider_type == LLMProviderType.OPENAI:
            api_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY', '')
        elif self.provider_type == LLMProviderType.ANTHROPIC:
            api_key = config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY', '')
        elif self.provider_type == LLMProviderType.GOOGLE:
            api_key = config.get('google_api_key') or os.getenv('GOOGLE_API_KEY', '')
        elif self.provider_type == LLMProviderType.DEEPSEEK:
            api_key = config.get('deepseek_api_key') or os.getenv('DEEPSEEK_API_KEY', '')
        elif self.provider_type == LLMProviderType.OPENROUTER:
            api_key = config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY', '')
            
        self.llm_provider = create_llm_provider(
            self.provider_type.value, 
            model_name, 
            api_key, 
            temperature=0.1, # Low temperature for classification
            max_tokens=config.get('verification.max_tokens', 500)
        )
        self.llm = self.llm_provider.llm
        
        self.concurrency_limit = config.get('processing.verification.concurrency', 5)

    async def execute(self, articles: List[Dict[str, Any]], keywords: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Verify a list of articles using LLM.
        
        Args:
            articles: List of article dictionaries
            keywords: Optional keywords to focus on (passed to LLM context)
            
        Returns:
            Dictionary with 'verified' and 'rejected' lists of articles
        """
        logger.info(f"Starting source verification for {len(articles)} articles")
        
        verified = []
        rejected = []
        
        if not articles:
            return {'verified': [], 'rejected': []}
        
        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        async def verify_article(article):
            async with semaphore:
                try:
                    result = await self._verify_single_article(article, keywords)
                    return result
                except Exception as e:
                    logger.error(f"Error verifying article '{article.get('title', 'unknown')}': {e}")
                    # Default to rejected on error to be safe
                    return {
                        'article': article,
                        'is_valid': False,
                        'reason': f"Verification error: {str(e)}"
                    }

        # Create tasks
        tasks = [verify_article(article) for article in articles]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            article = res['article']
            if res['is_valid']:
                article['verification_reason'] = res['reason']
                verified.append(article)
            else:
                article['verification_reason'] = res['reason']
                rejected.append(article)
                
        return {
            'verified': verified,
            'rejected': rejected
        }

    async def _verify_single_article(self, article: Dict[str, Any], keywords: str = None) -> Dict[str, Any]:
        """Verify a single article using LLM."""
        
        title = article.get('title', '')
        content = article.get('abstract') or article.get('content') or ''
        # Truncate content to avoid token limits, focus on beginning which usually has the main point
        content = content[:2000] 
        source = article.get('source', 'unknown')
        
        system_prompt = (
            "You are an expert AI research analyst. Your task is to evaluate if an article "
            "is about a significant breakthrough in Artificial Intelligence, new research, "
            "or a substantial review of AI technology.\n"
            "Criteria for VALID articles:\n"
            "- Discusses new AI research, models, architectures, or algorithms.\n"
            "- Reports on significant benchmarks or performance improvements.\n"
            "- Reviews specific AI technologies or papers.\n"
            "- Is a technical announcement from a major AI lab.\n\n"
            "Criteria for INVALID articles:\n"
            "- Simple tutorials (e.g., 'How to use ChatGPT', 'Python basics').\n"
            "- Purely opinion pieces or speculation without technical substance.\n"
            "- General news not related to AI technology advancements (e.g. stock prices, policy only).\n"
            "- Marketing fluff without technical details.\n\n"
            "Respond ONLY with a JSON object in the following format:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "reason": "short explanation of why it was accepted or rejected"\n'
            "}"
        )
        
        user_content = f"Title: {title}\nSource: {source}\n\nContent/Abstract:\n{content}"
        
        if keywords:
            user_content += f"\n\nFocus Keywords: {keywords}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        try:
            # Use JSON mode if supported by provider, but since we use generic BaseLanguageModel, 
            # we rely on the prompt. Some providers might need specific flags.
            # For now, we'll try to parse the string output.
            response = await self.llm.ainvoke(messages)
            content_str = response.content.strip()
            
            # Clean up potential markdown code blocks
            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.startswith("```"):
                content_str = content_str[3:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            
            content_str = content_str.strip()
            
            result = json.loads(content_str)
            
            return {
                'article': article,
                'is_valid': result.get('is_valid', False),
                'reason': result.get('reason', 'No reason provided')
            }
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for article: {title}")
            return {
                'article': article,
                'is_valid': False, # Conservative fail
                'reason': "LLM response format error"
            }
        except Exception as e:
            raise e
