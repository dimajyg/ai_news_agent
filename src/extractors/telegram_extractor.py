# Telegram Extractor Implementation

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from telegram import Bot
from telegram.constants import ParseMode
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class TelegramExtractor:
    """Extractor for Telegram channels and groups."""
    
    def __init__(self, config):
        self.config = config
        self.bot_token = config.get('TELEGRAM_BOT_TOKEN')
        self.channels = config.get('sources.telegram.channels', []) or config.get('sources.telegram.channels_to_monitor', [])
        self.max_messages_per_channel = config.get('sources.telegram.max_messages_per_channel', 50)
        self.message_age_limit = config.get('sources.telegram.message_age_limit', 24)  # hours
        self.rate_limit = config.get('sources.telegram.rate_limit', 20)  # messages per minute
        self.bot = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def extract_latest(self) -> List[Dict[str, Any]]:
        """Extract latest messages from Telegram channels."""
        logger.info("Extracting latest messages from Telegram channels")
        
        if not self.bot_token:
            logger.warning("No Telegram bot token configured")
            return []
        
        try:
            if not self.channels:
                logger.info("No Telegram channels configured for monitoring")
                return []
            
            # Prefer web scraping fallback for public channels
            all_messages: List[Dict[str, Any]] = []
            for channel in self.channels:
                try:
                    messages = await self._extract_channel_web(channel)
                    all_messages.extend(messages[: self.max_messages_per_channel])
                    # Rate limiting
                    await asyncio.sleep(max(1, int(60 / max(1, self.rate_limit))))
                except Exception as e:
                    logger.error(f"Failed to extract from channel {channel}: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(all_messages)} messages from Telegram (web mode)")
            return all_messages
            
        except Exception as e:
            logger.error(f"Telegram extraction failed: {str(e)}")
            return []
        finally:
            if self.bot:
                await self.bot.close()
            if self.session:
                await self.session.close()
                self.session = None
    
    async def _extract_channel_messages(self, channel: str) -> List[Dict[str, Any]]:
        """Extract messages from a specific channel."""
        logger.info(f"Extracting messages from channel: {channel}")
        
        try:
            # Ensure bot is active
            if self.bot is None:
                self.bot = Bot(token=self.bot_token)
            
            # Get channel info
            chat = await self.bot.get_chat(channel)
            
            # Calculate message age cutoff
            cutoff_time = datetime.now() - timedelta(hours=self.message_age_limit)
            
            # Get recent messages
            messages = []
            try:
                async for message in self.bot.get_chat_history(
                    chat_id=channel,
                    limit=self.max_messages_per_channel
                ):
                    if message.date.replace(tzinfo=None) < cutoff_time.replace(tzinfo=None):
                        break
                    processed_message = self._process_message(message, channel)
                    if processed_message:
                        messages.append(processed_message)
            except Exception as te:
                # Retry once if bot was closed
                if 'closed' in str(te).lower():
                    self.bot = Bot(token=self.bot_token)
                    async for message in self.bot.get_chat_history(
                        chat_id=channel,
                        limit=self.max_messages_per_channel
                    ):
                        if message.date.replace(tzinfo=None) < cutoff_time.replace(tzinfo=None):
                            break
                        processed_message = self._process_message(message, channel)
                        if processed_message:
                            messages.append(processed_message)
                else:
                    raise te
            
            logger.info(f"Extracted {len(messages)} messages from {channel}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to extract from channel {channel}: {str(e)}")
            return []

    async def _extract_channel_web(self, channel: str) -> List[Dict[str, Any]]:
        """Extract public channel messages via web.telegram.org static view (t.me/s/<channel>)."""
        logger.info(f"Extracting messages via web for channel: {channel}")
        
        # Normalize channel name
        channel_slug = channel.lstrip('@')
        url = f"https://t.me/s/{channel_slug}"
        
        # Prepare session
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': self.config.get('sources.websites.user_agent', 'Mozilla/5.0 (TelegramWebScraper/1.0)'),
                'Accept-Language': 'en-US,en;q=0.9',
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        
        try:
            async with self.session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Telegram web returned status {response.status}: {url}")
                    return []
                html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            message_nodes = soup.select('.tgme_widget_message_wrap')
            messages: List[Dict[str, Any]] = []
            cutoff_time = datetime.now() - timedelta(hours=self.message_age_limit)
            
            for node in message_nodes[: self.max_messages_per_channel]:
                try:
                    text_el = node.select_one('.tgme_widget_message_text')
                    if not text_el:
                        continue
                    content = text_el.get_text(separator='\n').strip()
                    if not content or len(content) < 50:
                        continue
                    
                    date_el = node.select_one('.tgme_widget_message_date time')
                    published_iso = date_el.get('datetime') if date_el else None
                    if published_iso:
                        try:
                            published_dt = datetime.fromisoformat(published_iso.replace('Z', '+00:00'))
                        except Exception:
                            published_dt = datetime.now()
                    else:
                        published_dt = datetime.now()
                    
                    if published_dt.replace(tzinfo=None) < cutoff_time.replace(tzinfo=None):
                        continue
                    
                    link_el = node.select_one('.tgme_widget_message_date')
                    post_url = link_el.get('href') if link_el else ''
                    # Extract message id if present
                    message_id = None
                    if post_url and '/'+channel_slug+'/' in post_url:
                        try:
                            message_id = post_url.rstrip('/').split('/')[-1]
                        except Exception:
                            message_id = None
                    
                    article = {
                        'title': self._extract_title_from_message(content),
                        'abstract': content[:1000],
                        'content': content,
                        'url': post_url,
                        'authors': [],
                        'published_date': published_dt.isoformat(),
                        'source': 'telegram',
                        'source_name': channel,
                        'source_category': 'social',
                        'reliability_score': 0.6,
                        'telegram_metadata': {
                            'message_id': message_id,
                            'channel': channel,
                            'date': published_dt.isoformat(),
                            'hashtags': self._extract_hashtags(content),
                            'mentions': self._extract_mentions(content),
                        },
                        'extraction_metadata': {
                            'extracted_at': datetime.now().isoformat(),
                            'method': 'web_scrape',
                        }
                    }
                    if self._validate_message(article):
                        messages.append(article)
                except Exception as e:
                    logger.warning(f"Failed to process web message: {e}")
                    continue
            
            logger.info(f"Extracted {len(messages)} messages via web for {channel}")
            return messages
        except Exception as e:
            logger.error(f"Telegram web extraction failed for {channel}: {e}")
            return []
        
        finally:
            # Do not close session here; allow reuse and close in extract_latest finally
            ...
    
    def _process_message(self, message, channel: str) -> Dict[str, Any]:
        """Process Telegram message."""
        try:
            # Skip messages without text or with media only
            if not message.text:
                return None
            
            # Apply content filtering
            if not self._filter_message(message.text):
                return None
            
            # Extract message information
            message_data = {
                'title': self._extract_title_from_message(message.text),
                'abstract': message.text[:1000],  # Limit content length
                'content': message.text,
                'url': self._extract_url_from_message(message.text),
                'authors': [message.from_user.username] if message.from_user and message.from_user.username else [],
                'published_date': message.date.isoformat(),
                'source': 'telegram',
                'source_name': channel,
                'source_category': 'social',
                'reliability_score': 0.6,  # Lower reliability for social media
                'telegram_metadata': {
                    'message_id': message.message_id,
                    'chat_id': message.chat_id,
                    'date': message.date.isoformat(),
                    'hashtags': self._extract_hashtags(message.text),
                    'mentions': self._extract_mentions(message.text),
                    'forwarded_from': message.forward_from.username if message.forward_from else None,
                    'reply_to_message': message.reply_to_message.message_id if message.reply_to_message else None
                },
                'extraction_metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'channel': channel
                }
            }
            
            return message_data if self._validate_message(message_data) else None
            
        except Exception as e:
            logger.warning(f"Failed to process Telegram message: {str(e)}")
            return None
    
    def _filter_message(self, text: str) -> bool:
        """Filter messages based on content."""
        if not text or len(text) < 30:  # Minimum length
            return False
        
        # Check for required keywords
        must_contain = self.config.get('sources.telegram.message_filtering.keywords.must_contain_one', [])
        if must_contain:
            if not any(keyword.lower() in text.lower() for keyword in must_contain):
                return False
        
        # Check for excluded keywords
        exclude_keywords = self.config.get('sources.telegram.message_filtering.keywords.exclude_if_contains', [])
        if exclude_keywords:
            if any(keyword.lower() in text.lower() for keyword in exclude_keywords):
                return False
        
        return True
    
    def _extract_title_from_message(self, text: str) -> str:
        """Extract title from message text."""
        lines = text.strip().split('\n')
        
        # Look for first line that looks like a title
        for line in lines:
            line = line.strip()
            if len(line) > 20 and len(line) < 150:  # Reasonable title length
                # Remove common prefixes
                prefixes = ['📄', '📊', '🤖', '🧠', '🔬', '💡', '🚀', '📢']
                for prefix in prefixes:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                return line
        
        # Fallback: use first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                return line[:100]  # Limit length
        
        return "AI/ML Research Update"  # Default title
    
    def _extract_url_from_message(self, text: str) -> str:
        """Extract URL from message text."""
        import re
        
        # Look for URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .-]*/?'
        urls = re.findall(url_pattern, text)
        
        # Prefer arXiv URLs
        for url in urls:
            if 'arxiv.org' in url:
                return url
        
        # Return first URL found
        return urls[0] if urls else ''
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from message."""
        import re
        return re.findall(r'#\w+', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from message."""
        import re
        return re.findall(r'@\w+', text)
    
    def _validate_message(self, message_data: Dict[str, Any]) -> bool:
        """Validate extracted message data."""
        # Check required fields
        if not message_data.get('title') or not message_data.get('content'):
            return False
        
        # Check content quality
        content = message_data.get('content', '')
        if len(content) < 50:  # Minimum content length
            return False
        
        # Check for spam indicators
        content_lower = content.lower()
        spam_keywords = ['buy now', 'click here', 'subscribe', 'promotion', 'advertisement']
        if any(keyword in content_lower for keyword in spam_keywords):
            return False
        
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'source': 'telegram',
            'channels': len(self.channels),
            'max_messages_per_channel': self.max_messages_per_channel,
            'message_age_limit_hours': self.message_age_limit
        }
