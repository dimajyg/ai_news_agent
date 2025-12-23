# Configuration Management

import os
import yaml
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings from environment variables."""
    # API Keys (optional - can be set at runtime)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_channel_id: str = ""
    
    # Application settings
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "ai_news_articles"
    log_level: str = "INFO"
    log_file: str = "./logs/agent.log"
    max_retries: int = 3
    batch_size: int = 50
    similarity_threshold: float = 0.85
    dedup_threshold: float = 0.90
    metrics_port: int = 8000
    health_check_interval: int = 30
    debug: bool = False
    test_mode: bool = False
    mock_external_apis: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class Config:
    """Main configuration manager."""
    
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)
        self.settings = Settings()
        self._agent_config = None
        self._sources_config = None
        self._custom_config_file = None
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files."""
        # Check if config_path is a file (custom config)
        if self.config_path.is_file():
            # Load the single config file
            with open(self.config_path, 'r') as f:
                self._custom_config_file = yaml.safe_load(f)
                # Extract agent and sources config if they exist
                if 'sources' in self._custom_config_file:
                    self._sources_config = {'sources': self._custom_config_file['sources']}
                # The whole file is the agent config
                self._agent_config = self._custom_config_file
        else:
            # Load from directory (default behavior)
            # Load agent configuration
            agent_config_path = self.config_path / "agent_config.yaml"
            if agent_config_path.exists():
                with open(agent_config_path, 'r') as f:
                    self._agent_config = yaml.safe_load(f)
            
            # Load sources configuration
            sources_config_path = self.config_path / "sources.yaml"
            if sources_config_path.exists():
                with open(sources_config_path, 'r') as f:
                    self._sources_config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        # First check environment settings
        env_key = key.replace('.', '_').lower()
        if hasattr(self.settings, env_key):
            return getattr(self.settings, env_key)
        
        # Then check YAML configs
        if self._agent_config:
            value = self._get_nested_value(self._agent_config, key)
            if value is not None:
                return value
        
        if self._sources_config:
            value = self._get_nested_value(self._sources_config, key)
            if value is not None:
                return value
        
        return default
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def get_sources_config(self) -> Dict[str, Any]:
        """Get sources configuration."""
        return self._sources_config or {}
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self._agent_config or {}
    
    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI configuration."""
        return {
            'api_key': self.settings.openai_api_key,
            'model': self.get('clustering.embedding_model', 'text-embedding-ada-002')
        }
    
    def get_anthropic_config(self) -> Dict[str, str]:
        """Get Anthropic configuration."""
        return {
            'api_key': self.settings.anthropic_api_key,
            'model': self.get('clustering.embedding_model', 'claude-3-sonnet-20240229')
        }
    
    def get_google_config(self) -> Dict[str, str]:
        """Get Google configuration."""
        return {
            'api_key': self.settings.google_api_key,
            'model': self.get('clustering.embedding_model', 'gemini-pro')
        }
    
    def get_telegram_config(self) -> Dict[str, str]:
        """Get Telegram configuration."""
        return {
            'bot_token': self.settings.telegram_bot_token,
            'channel_id': self.settings.telegram_channel_id
        }
    
    def get_chroma_config(self) -> Dict[str, str]:
        """Get ChromaDB configuration."""
        return {
            'persist_directory': self.settings.chroma_persist_directory,
            'collection_name': self.settings.chroma_collection_name
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'sources_enabled': list(self.get_sources_config().keys()),
            'agent_features': list(self.get_agent_config().keys()),
            'log_level': self.settings.log_level,
            'debug_mode': self.settings.debug,
            'test_mode': self.settings.test_mode
        }
    
    def validate(self) -> bool:
        """Validate required configuration based on selected provider."""
        # Get the current LLM provider
        current_provider = self.get('llm.provider', 'openai')
        
        # Check required API key for the selected provider
        provider_key_map = {
            'openai': 'openai_api_key',
            'anthropic': 'anthropic_api_key',
            'google': 'google_api_key',
            'deepseek': 'deepseek_api_key',
            'openrouter': 'openrouter_api_key'
        }
        
        missing_keys = []
        
        # Check API key for current provider
        if current_provider in provider_key_map:
            api_key_field = provider_key_map[current_provider]
            if not getattr(self.settings, api_key_field, None):
                missing_keys.append(f"{current_provider.upper()}_API_KEY")
        
        # Check Telegram configuration (required for posting)
        if not self.settings.telegram_bot_token:
            missing_keys.append('TELEGRAM_BOT_TOKEN')
        if not self.settings.telegram_channel_id:
            missing_keys.append('TELEGRAM_CHANNEL_ID')
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        return True

# Global settings instance for easy access
settings = Settings()
