"""
LLM Provider abstraction module for easy switching between different LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

try:
    from ..utils.config import Config
    from ..utils.logging import AgentLogger
except ImportError:
    from utils.config import Config
    from utils.logging import AgentLogger


logger = AgentLogger(__name__)


class LLMProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProviderType
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: int = 3
    additional_params: Dict[str, Any] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm: Optional[BaseLanguageModel] = None
        self._embeddings: Optional[Embeddings] = None
    
    @abstractmethod
    def create_llm(self) -> BaseLanguageModel:
        """Create and return the language model instance."""
        pass
    
    @abstractmethod
    def create_embeddings(self) -> Embeddings:
        """Create and return the embeddings instance."""
        pass
    
    @property
    def llm(self) -> BaseLanguageModel:
        """Get the language model instance."""
        if self._llm is None:
            self._llm = self.create_llm()
        return self._llm
    
    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        if self._embeddings is None:
            self._embeddings = self.create_embeddings()
        return self._embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries
        }


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def create_llm(self) -> BaseLanguageModel:
        """Create OpenAI language model."""
        params = {
            "model": self.config.model_name,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        
        # Add any additional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)
        
        logger.info(f"Creating OpenAI model: {self.config.model_name}")
        from langchain_openai import ChatOpenAI  # lazy import
        return ChatOpenAI(**params)
    
    def create_embeddings(self) -> Embeddings:
        """Create OpenAI embeddings."""
        params = {
            "api_key": self.config.api_key,
        }
        
        # Add any additional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)
        
        logger.info("Creating OpenAI embeddings")
        from langchain_openai import OpenAIEmbeddings  # lazy import
        return OpenAIEmbeddings(**params)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    def create_llm(self) -> BaseLanguageModel:
        """Create Anthropic language model."""
        params = {
            "model": self.config.model_name,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        
        # Add any additional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)
        
        logger.info(f"Creating Anthropic model: {self.config.model_name}")
        from langchain_anthropic import ChatAnthropic  # lazy import
        return ChatAnthropic(**params)
    
    def create_embeddings(self) -> Embeddings:
        """Create Anthropic embeddings (fallback to OpenAI for now)."""
        logger.warning("Anthropic embeddings not available, falling back to OpenAI")
        # For now, we'll use OpenAI embeddings as fallback
        # In a real implementation, you might want to use a different embeddings provider
        from langchain_openai import OpenAIEmbeddings  # lazy import
        return OpenAIEmbeddings(api_key=self.config.api_key)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def create_llm(self) -> BaseLanguageModel:
        """Create Google language model."""
        params = {
            "model": self.config.model_name,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.max_tokens:
            params["max_output_tokens"] = self.config.max_tokens
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        
        # Add any additional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)
        
        logger.info(f"Creating Google model: {self.config.model_name}")
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import
        return ChatGoogleGenerativeAI(**params)
    
    def create_embeddings(self) -> Embeddings:
        """Create Google embeddings."""
        # Note: Google embeddings might require different setup
        logger.warning("Google embeddings not fully implemented, falling back to OpenAI")
        from langchain_openai import OpenAIEmbeddings  # lazy import
        return OpenAIEmbeddings(api_key=self.config.api_key)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider (OpenAI-compatible API)."""
    def create_llm(self) -> BaseLanguageModel:
        params = {
            "model": self.config.model_name,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
            "base_url": "https://api.deepseek.com"
        }
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        if self.config.additional_params:
            params.update(self.config.additional_params)
        logger.info(f"Creating DeepSeek model: {self.config.model_name}")
        from langchain_openai import ChatOpenAI  # lazy import
        return ChatOpenAI(**params)

    def create_embeddings(self) -> Embeddings:
        logger.warning("DeepSeek embeddings not available; falling back to OpenAI embeddings")
        return OpenAIEmbeddings(api_key=self.config.api_key)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider using OpenAI-compatible interfaces."""
    BASE_URL = "https://openrouter.ai/api/v1"

    def create_llm(self) -> BaseLanguageModel:
        params = {
            "model_name": self.config.model_name,
            "openai_api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
            "openai_api_base": self.BASE_URL,
        }
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        if self.config.additional_params:
            params.update(self.config.additional_params)
        logger.info(f"Creating OpenRouter model: {self.config.model_name}")
        from langchain_openai import ChatOpenAI  # lazy import
        return ChatOpenAI(**params)

    def create_embeddings(self) -> Embeddings:
        # Prefer explicit model provided; fall back to default qwen embeddings
        model = "qwen/qwen3-embedding-8b"
        if self.config.additional_params:
            model = self.config.additional_params.get("model") or self.config.additional_params.get("embeddings_model") or model
        params = {
            "api_key": self.config.api_key,
            "base_url": self.BASE_URL,
            "model": model,
        }
        logger.info(f"Creating OpenRouter embeddings: {model}")
        from langchain_openai import OpenAIEmbeddings  # lazy import
        return OpenAIEmbeddings(**params)


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        LLMProviderType.OPENAI: OpenAIProvider,
        LLMProviderType.ANTHROPIC: AnthropicProvider,
        LLMProviderType.GOOGLE: GoogleProvider,
        LLMProviderType.DEEPSEEK: DeepSeekProvider,
        LLMProviderType.OPENROUTER: OpenRouterProvider,
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create an LLM provider based on configuration."""
        if config.provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        provider_class = cls._providers[config.provider]
        return provider_class(config)
    
    @classmethod
    def create_from_config(cls, config: Config) -> BaseLLMProvider:
        """Create provider from application configuration."""
        # Map config to LLMConfig
        llm_config = LLMConfig(
            provider=LLMProviderType(config.llm.provider),
            model_name=config.llm.model_name,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens if hasattr(config.llm, 'max_tokens') else None,
            timeout=config.llm.timeout if hasattr(config.llm, 'timeout') else None,
            max_retries=config.llm.max_retries if hasattr(config.llm, 'max_retries') else 3,
            additional_params=config.llm.additional_params if hasattr(config.llm, 'additional_params') else None
        )
        
        return cls.create_provider(llm_config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return [provider.value for provider in cls._providers.keys()]
    
    @classmethod
    def get_provider_models(cls, provider: str) -> List[str]:
        """Get available models for a provider."""
        # This is a simplified version - in a real implementation,
        # you might want to fetch this from the provider's API
        models = {
            "openai": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4o",
                "gpt-4o-mini"
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "google": [
                "gemini-pro",
                "gemini-pro-vision",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            "deepseek": [
                "deepseek-chat"
            ],
            "openrouter": [
                "x-ai/grok-4.1-fast:free",
                "qwen/qwen3-embedding-8b",
                "google/gemini-2.5-flash-lite"
            ]
        }
        return models.get(provider, [])


# Convenience functions for easy provider switching
def create_llm_provider(
    provider: str,
    model_name: str,
    api_key: str,
    **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider with the given configuration.
    
    Args:
        provider: Provider name (openai, anthropic, google)
        model_name: Model name
        api_key: API key
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM provider instance
    """
    config = LLMConfig(
        provider=LLMProviderType(provider),
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return LLMProviderFactory.create_provider(config)


def create_embeddings_provider(
    provider: str,
    api_key: str,
    **kwargs
) -> Embeddings:
    """
    Create an embeddings provider.
    
    Args:
        provider: Provider name
        api_key: API key
        **kwargs: Additional configuration parameters
        
    Returns:
        Embeddings instance
    """
    config = LLMConfig(
        provider=LLMProviderType(provider),
        model_name="",  # Not needed for embeddings
        api_key=api_key,
        **kwargs
    )
    provider_instance = LLMProviderFactory.create_provider(config)
    return provider_instance.embeddings
