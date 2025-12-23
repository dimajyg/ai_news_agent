"""
Telegram poster node for sending posts to Telegram channels.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp

try:
    from ...utils.config import Config
    from ...utils.logging import AgentLogger
except ImportError:
    from src.utils.config import Config
    from src.utils.logging import AgentLogger


logger = AgentLogger(__name__)


class TelegramPosterNode:
    """Node for posting content to Telegram channels."""
    
    def __init__(self, config: Config):
        self.config = config
        self.bot_token = config.get('telegram_bot_token', '')
        self.channel_ids = config.get('telegram_channel_id', '').split(',') if config.get('telegram_channel_id') else []
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def send_post(
        self, 
        post_data: Dict[str, Any], 
        channel_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a single post to Telegram.
        
        Args:
            post_data: Post data including title, content, etc.
            channel_id: Specific channel ID (uses config default if None)
            context: Optional context information
            
        Returns:
            Post result with status and metadata
        """
        try:
            target_channel = channel_id or self.channel_ids[0] if self.channel_ids else None
            if not target_channel:
                raise ValueError("No Telegram channel ID configured")
            
            logger.info(f"Sending post to Telegram channel: {target_channel}")
            
            # Format the post content
            formatted_content = self._format_post_content(post_data)
            
            # Send the message
            message_id = await self._send_message(target_channel, formatted_content, post_data)
            
            result = {
                "success": True,
                "message_id": message_id,
                "channel_id": target_channel,
                "post_data": post_data,
                "sent_at": datetime.now().isoformat(),
                "content_length": len(formatted_content),
                "context": context or {}
            }
            
            logger.info(f"Post sent successfully to {target_channel}, message ID: {message_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error sending post to Telegram: {e}")
            return {
                "success": False,
                "error": str(e),
                "post_data": post_data,
                "channel_id": channel_id,
                "context": context or {},
                "sent_at": datetime.now().isoformat()
            }
    
    async def send_posts_batch(
        self, 
        posts_data: List[Dict[str, Any]], 
        channel_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Send multiple posts to Telegram channels.
        
        Args:
            posts_data: List of post data
            channel_ids: List of channel IDs (uses config default if None)
            context: Optional context information
            
        Returns:
            List of post results
        """
        results = []
        target_channels = channel_ids or self.channel_ids
        
        if not target_channels:
            logger.error("No Telegram channels configured")
            return []
        
        # Send posts with rate limiting
        for i, post_data in enumerate(posts_data):
            try:
                # Distribute posts across channels
                channel_index = i % len(target_channels)
                target_channel = target_channels[channel_index]
                
                result = await self.send_post(post_data, target_channel, context)
                results.append(result)
                
                # Rate limiting: wait between posts to avoid spam detection
                if i < len(posts_data) - 1:  # Don't wait after the last post
                    await asyncio.sleep(2)  # 2 second delay between posts
                    
            except Exception as e:
                logger.error(f"Error in batch posting for post {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "post_data": post_data,
                    "index": i,
                    "context": context or {}
                })
        
        return results
    
    def _clean_html_for_telegram(self, text: str) -> str:
        """
        Clean HTML to only include Telegram-supported tags.
        Telegram supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
        """
        import re
        
        logger.info(f"TelegramPoster: _clean_html_for_telegram input length={len(text)}")
        logger.info(f"TelegramPoster: Has <a href tags: {'<a href' in text}")
        
        # Remove unsupported tags like <img>, <div>, <span>, etc.
        # Keep only Telegram-supported tags
        supported_tags = ['b', 'i', 'u', 's', 'code', 'pre', 'a', 'strong', 'em']
        
        # Remove <img> tags completely
        text = re.sub(r'<img[^>]*>', '', text)
        
        # Remove other unsupported tags but keep their content
        # This regex finds tags that are NOT in the supported list
        def replace_unsupported_tag(match):
            tag_name = match.group(1).lower().split()[0]  # Get tag name without attributes
            if tag_name not in supported_tags:
                return ''  # Remove the tag but keep content
            return match.group(0)  # Keep supported tags
        
        # Remove opening tags - FIXED: preserve <a> tags with attributes
        text = re.sub(
            r'<(/?)(\w+)([^>]*)>', 
            lambda m: f'<{m.group(1)}{m.group(2)}{m.group(3) if m.group(2).lower() == "a" else ""}>' 
            if m.group(2).lower() in supported_tags else '', 
            text
        )
        
        # Convert <strong> to <b> and <em> to <i> for consistency
        text = text.replace('<strong>', '<b>').replace('</strong>', '</b>')
        text = text.replace('<em>', '<i>').replace('</em>', '</i>')
        
        logger.info(f"TelegramPoster: _clean_html_for_telegram output length={len(text)}")
        logger.info(f"TelegramPoster: Output has <a href tags: {'<a href' in text}")
        
        return text
    
    def _format_post_content(self, post_data: Dict[str, Any]) -> str:
        """Format post content for Telegram."""
        title = post_data.get("title", "")
        content = post_data.get("content", "")
        hashtags = post_data.get("hashtags", [])
        emojis = post_data.get("emojis", [])
        
        # Clean HTML content for Telegram
        if content:
            content = self._clean_html_for_telegram(content)
        
        # Build the message
        message_parts = []
        
        # Add title with emojis
        if title:
            emoji_prefix = " ".join(emojis[:2]) if emojis else "🔬"
            message_parts.append(f"{emoji_prefix} {title}")
        
        # Add main content
        if content:
            message_parts.append(content)
        
        # Add hashtags
        if hashtags:
            hashtag_string = " ".join(hashtags[:10])  # Limit hashtags
            message_parts.append(hashtag_string)
        
        # Join with appropriate spacing
        return "\n\n".join(message_parts)
    
    async def _send_message(
        self, 
        chat_id: str, 
        text: str, 
        post_data: Dict[str, Any]
    ) -> Optional[str]:
        """Send message via Telegram Bot API."""
        if not self.session:
            self._setup_session()
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        # Prepare request data
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",  # Use HTML formatting
            "disable_web_page_preview": not post_data.get("link_preview", True),
            "disable_notification": False
        }
        
        # If message is too long, split it
        if len(text) > 4096:  # Telegram message limit
            return await self._send_long_message(chat_id, text, post_data)
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        return str(result["result"]["message_id"])
                    else:
                        raise Exception(f"Telegram API error: {result.get('description')}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            raise
    
    async def _send_long_message(
        self, 
        chat_id: str, 
        text: str, 
        post_data: Dict[str, Any]
    ) -> Optional[str]:
        """Send long message by splitting it into parts."""
        if not self.session:
            self._setup_session()
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        # Split the message into parts
        message_parts = self._split_long_message(text)
        message_ids = []
        
        for i, part in enumerate(message_parts):
            # For the first part, include title and hashtags
            if i == 0:
                formatted_part = part
            else:
                # For subsequent parts, just send the content
                formatted_part = part
            
            payload = {
                "chat_id": chat_id,
                "text": formatted_part,
                "parse_mode": "HTML",
                "disable_web_page_preview": not post_data.get("link_preview", True) if i == 0 else True,
                "disable_notification": False
            }
            
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            message_ids.append(str(result["result"]["message_id"]))
                        else:
                            logger.error(f"Telegram API error in part {i+1}: {result.get('description')}")
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP {response.status} in part {i+1}: {error_text}")
                        
            except Exception as e:
                logger.error(f"Error sending message part {i+1}: {e}")
                continue
            
            # Small delay between parts
            if i < len(message_parts) - 1:
                await asyncio.sleep(1)
        
        # Return the ID of the first message
        return message_ids[0] if message_ids else None
    
    def _split_long_message(self, text: str, max_length: int = 4000) -> List[str]:
        """
        Split long message into parts, respecting article block boundaries.
        Each article block (marked by emoji + bold title) should stay together.
        """
        if len(text) <= max_length:
            return [text]
        
        parts = []
        current_part = ""
        
        # Split by double newlines to get blocks
        blocks = text.split("\n\n")
        
        # Try to identify article blocks (they start with emoji or have <b> tags)
        import re
        article_pattern = re.compile(r'^[🔬📊🤖💡🚀⚡🎯🔥✨🌟💻🧠🎨🔮🎭🎪🎬🎮🎯🎲🎰🎳]|^<b>')
        
        i = 0
        while i < len(blocks):
            block = blocks[i]
            
            # Check if this is the start of an article block
            is_article_start = article_pattern.match(block.strip())
            
            if is_article_start:
                # This is an article block - try to keep it together with next block (content)
                article_block = block
                
                # Include the next block (article content) if it exists
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    # Check if next block is content (not another article start)
                    if not article_pattern.match(next_block.strip()):
                        article_block = block + "\n\n" + next_block
                        i += 1  # Skip the next block since we included it
                
                # Check if adding this article block would exceed limit
                if current_part and len(current_part) + len(article_block) + 2 > max_length:
                    # Current part is full, save it and start new part
                    parts.append(current_part)
                    current_part = article_block
                elif not current_part:
                    # First block in new part
                    current_part = article_block
                else:
                    # Add to current part
                    current_part += "\n\n" + article_block
                
                # If the article block itself is too long, we need to split it
                if len(current_part) > max_length:
                    # Save what we have and continue
                    if len(article_block) > max_length:
                        # Article block is too long, split it carefully
                        logger.warning(f"Article block too long ({len(article_block)} chars), splitting carefully")
                        # Split by sentences within the block
                        sentences = article_block.split(". ")
                        temp_part = ""
                        for sentence in sentences:
                            if len(temp_part) + len(sentence) + 2 <= max_length:
                                temp_part += sentence + ". " if temp_part else sentence
                            else:
                                if temp_part:
                                    parts.append(temp_part)
                                temp_part = sentence
                        if temp_part:
                            current_part = temp_part
                    else:
                        parts.append(current_part)
                        current_part = ""
            else:
                # Regular block (hashtags, etc.)
                if current_part and len(current_part) + len(block) + 2 <= max_length:
                    current_part += "\n\n" + block
                elif not current_part:
                    current_part = block
                else:
                    # Save current part and start new one
                    parts.append(current_part)
                    current_part = block
            
            i += 1
        
        # Add the last part if it exists
        if current_part:
            parts.append(current_part)
        
        logger.info(f"Split message into {len(parts)} parts (original: {len(text)} chars)")
        return parts
    
    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get information about a Telegram channel."""
        if not self.session:
            self._setup_session()
        
        url = f"https://api.telegram.org/bot{self.bot_token}/getChat"
        payload = {"chat_id": channel_id}
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        return result["result"]
                    else:
                        raise Exception(f"Telegram API error: {result.get('description')}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_id}: {e}")
            raise


def create_telegram_poster_node(config: Config) -> TelegramPosterNode:
    """Factory function to create Telegram poster node."""
    return TelegramPosterNode(config)


async def telegram_poster_node(state: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    LangGraph node function for Telegram posting.
    
    Args:
        state: Current graph state
        config: Application configuration
        
    Returns:
        Updated state with posting results
    """
    try:
        node = create_telegram_poster_node(config)
        
        # Get generated posts from state
        generated_posts = state.get("generated_posts", [])
        if not generated_posts:
            logger.warning("No generated posts found in state for Telegram posting")
            return {**state, "posting_results": []}
        
        logger.info(f"Sending {len(generated_posts)} posts to Telegram")
        
        # Send posts
        posting_results = await node.send_posts_batch(
            generated_posts, 
            context=state.get("context", {})
        )
        
        # Calculate success metrics
        successful_posts = sum(1 for result in posting_results if result.get("success", False))
        total_posts = len(posting_results)
        
        # Update state
        return {
            **state,
            "posting_results": posting_results,
            "posts_sent": successful_posts,
            "total_posts_attempted": total_posts,
            "posting_completed": True,
            "posting_success_rate": successful_posts / total_posts if total_posts > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error in Telegram poster node: {e}")
        return {
            **state,
            "posting_results": [],
            "posts_sent": 0,
            "total_posts_attempted": 0,
            "posting_completed": False,
            "posting_success_rate": 0.0,
            "error": str(e)
        }
    finally:
        # Clean up session
        if 'node' in locals():
            await node.close_session()
