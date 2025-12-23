"""
Multi-tenant Telegram bot for AI News Agent.
"""

import os
import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

from ..database.manager import DatabaseManager
from ..agents.main_agent import AINewsAgent
from ..utils.config import Config
from ..vector_store.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class MultiTenantBot:
    """Multi-tenant Telegram bot for managing AI news agents per group."""
    
    def __init__(self, token: str, database_manager: DatabaseManager):
        self.token = token
        self.db = database_manager
        self.application = None
        self.running_agents: Dict[str, AINewsAgent] = {}
    
    async def initialize(self):
        """Initialize the bot."""
        logger.info("Initializing Telegram bot...")
        
        self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("add_telegram", self.cmd_add_telegram))
        self.application.add_handler(CommandHandler("add_arxiv", self.cmd_add_arxiv))
        self.application.add_handler(CommandHandler("add_website", self.cmd_add_website))
        self.application.add_handler(CommandHandler("remove_source", self.cmd_remove_source))
        self.application.add_handler(CommandHandler("sources", self.cmd_sources))
        self.application.add_handler(CommandHandler("run", self.cmd_run))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("config", self.cmd_config))
        self.application.add_handler(CommandHandler("set_language", self.cmd_set_language))
        self.application.add_handler(CommandHandler("add_keywords", self.cmd_add_keywords))
        self.application.add_handler(CommandHandler("set_post_size", self.cmd_set_post_size))
        
        await self.application.bot.set_my_commands([
            BotCommand("start", "Initialize bot for this group"),
            BotCommand("help", "Show help message"),
            BotCommand("add_telegram", "Add Telegram channel source (@channel_name)"),
            BotCommand("add_arxiv", "Add arXiv category (e.g., cs.AI)"),
            BotCommand("add_website", "Add website RSS feed (URL)"),
            BotCommand("remove_source", "Remove a source"),
            BotCommand("sources", "List all configured sources"),
            BotCommand("run", "Run the news agent now"),
            BotCommand("status", "Show agent status"),
            BotCommand("history", "Show run history"),
            BotCommand("config", "Show current configuration"),
            BotCommand("set_language", "Set language (en/ru)"),
            BotCommand("add_keywords", "Add keywords for filtering"),
            BotCommand("set_post_size", "Set number of articles in post (1-50)"),
        ])
        
        logger.info("Telegram bot initialized successfully")
    
    async def start(self):
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Telegram bot started successfully")
    
    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping Telegram bot...")
        
        for agent in self.running_agents.values():
            await agent.cleanup()
        self.running_agents.clear()
        
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        
        logger.info("Telegram bot stopped")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        chat = update.effective_chat
        
        try:
            await self.db.create_or_get_group(
                chat_id=str(chat.id),
                chat_type=chat.type,
                chat_title=chat.title or chat.first_name
            )
            
            welcome_message = (
                f"🤖 <b>AI News Agent Initialized!</b>\n\n"
                f"Welcome to the AI News Agent for <b>{chat.title or 'this chat'}</b>!\n\n"
                f"I'll help you aggregate and analyze AI/ML news from multiple sources.\n\n"
                f"<b>Quick Start:</b>\n"
                f"1. Add sources using /add_telegram, /add_arxiv, or /add_website\n"
                f"2. View your sources with /sources\n"
                f"3. Run the agent manually with /run\n"
                f"4. The agent will also run automatically once per day\n\n"
                f"Use /help to see all available commands."
            )
            
            await update.message.reply_text(welcome_message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /start: {e}")
            await update.message.reply_text(
                f"❌ Error initializing bot: {str(e)}"
            )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = (
            "<b>🤖 AI News Agent - Command Reference</b>\n\n"
            
            "<b>📰 Source Management:</b>\n"
            "/add_telegram @channel - Add Telegram channel\n"
            "  Example: /add_telegram @gonzo_ML\n\n"
            
            "/add_arxiv category - Add arXiv category\n"
            "  Example: /add_arxiv cs.AI\n"
            "  Categories: cs.AI, cs.LG, cs.CL, cs.CV, cs.RO, cs.MA, cs.NE\n\n"
            
            "/add_website URL - Add RSS feed\n"
            "  Example: /add_website https://example.com/feed\n\n"
            
            "/remove_source type identifier - Remove a source\n"
            "  Example: /remove_source telegram @gonzo_ML\n\n"
            
            "/sources - List all configured sources\n\n"
            
            "<b>🚀 Agent Operations:</b>\n"
            "/run - Run the news agent immediately\n"
            "/status - Show current agent status\n"
            "/history - Show recent run history\n"
            "/config - Show current configuration\n\n"
            
            "<b>ℹ️ General:</b>\n"
            "/start - Initialize bot for this group\n"
            "/help - Show this help message\n\n"
            
            "<b>📅 Automatic Runs:</b>\n"
            "The agent runs automatically once per day at 09:00 UTC.\n"
            "Results are posted directly to this chat."
        )
        
        await update.message.reply_text(help_message, parse_mode='HTML')
    
    async def cmd_add_telegram(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_telegram command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /add_telegram @channel_name\n"
                "Example: /add_telegram @gonzo_ML"
            )
            return
        
        channel = context.args[0]
        if not channel.startswith('@'):
            channel = f"@{channel}"
        
        try:
            await self.db.add_source(
                chat_id=chat_id,
                source_type='telegram',
                source_identifier=channel
            )
            
            await update.message.reply_text(
                f"✅ Added Telegram channel: {channel}\n\n"
                f"Use /sources to see all configured sources."
            )
            
        except Exception as e:
            logger.error(f"Error adding Telegram source: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_add_arxiv(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_arxiv command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /add_arxiv category\n"
                "Example: /add_arxiv cs.AI\n\n"
                "Available categories:\n"
                "• cs.AI - Artificial Intelligence\n"
                "• cs.LG - Machine Learning\n"
                "• cs.CL - Computation and Language\n"
                "• cs.CV - Computer Vision\n"
                "• cs.RO - Robotics\n"
                "• cs.MA - Multiagent Systems\n"
                "• cs.NE - Neural and Evolutionary Computing"
            )
            return
        
        category = context.args[0]
        
        valid_categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO', 'cs.MA', 'cs.NE']
        if category not in valid_categories:
            await update.message.reply_text(
                f"❌ Invalid category: {category}\n\n"
                f"Valid categories: {', '.join(valid_categories)}"
            )
            return
        
        try:
            await self.db.add_source(
                chat_id=chat_id,
                source_type='arxiv',
                source_identifier=category
            )
            
            await update.message.reply_text(
                f"✅ Added arXiv category: {category}\n\n"
                f"Use /sources to see all configured sources."
            )
            
        except Exception as e:
            logger.error(f"Error adding arXiv source: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_add_website(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_website command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /add_website URL\n"
                "Example: /add_website https://www.artificialintelligence-news.com/feed/"
            )
            return
        
        url = context.args[0]
        
        if not url.startswith('http'):
            await update.message.reply_text(
                "❌ Invalid URL. Must start with http:// or https://"
            )
            return
        
        try:
            await self.db.add_source(
                chat_id=chat_id,
                source_type='website',
                source_identifier=url
            )
            
            await update.message.reply_text(
                f"✅ Added website: {url}\n\n"
                f"Use /sources to see all configured sources."
            )
            
        except Exception as e:
            logger.error(f"Error adding website source: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_remove_source(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove_source command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "❌ Usage: /remove_source type identifier\n"
                "Examples:\n"
                "  /remove_source telegram @gonzo_ML\n"
                "  /remove_source arxiv cs.AI\n"
                "  /remove_source website https://example.com/feed"
            )
            return
        
        source_type = context.args[0].lower()
        source_identifier = ' '.join(context.args[1:])
        
        if source_type not in ['telegram', 'arxiv', 'website']:
            await update.message.reply_text(
                "❌ Invalid source type. Must be: telegram, arxiv, or website"
            )
            return
        
        try:
            success = await self.db.remove_source(
                chat_id=chat_id,
                source_type=source_type,
                source_identifier=source_identifier
            )
            
            if success:
                await update.message.reply_text(
                    f"✅ Removed {source_type} source: {source_identifier}"
                )
            else:
                await update.message.reply_text(
                    f"❌ Source not found: {source_type} {source_identifier}"
                )
            
        except Exception as e:
            logger.error(f"Error removing source: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_sources(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sources command."""
        chat_id = str(update.effective_chat.id)
        
        try:
            sources = await self.db.get_group_sources(chat_id)
            
            if not sources:
                await update.message.reply_text(
                    "📭 No sources configured yet.\n\n"
                    "Add sources using:\n"
                    "• /add_telegram @channel\n"
                    "• /add_arxiv category\n"
                    "• /add_website URL"
                )
                return
            
            telegram_sources = [s for s in sources if s.source_type == 'telegram']
            arxiv_sources = [s for s in sources if s.source_type == 'arxiv']
            website_sources = [s for s in sources if s.source_type == 'website']
            
            message = "<b>📰 Configured News Sources</b>\n\n"
            
            if telegram_sources:
                message += "<b>Telegram Channels:</b>\n"
                for source in telegram_sources:
                    message += f"  • {source.source_identifier}\n"
                message += "\n"
            
            if arxiv_sources:
                message += "<b>arXiv Categories:</b>\n"
                for source in arxiv_sources:
                    message += f"  • {source.source_identifier}\n"
                message += "\n"
            
            if website_sources:
                message += "<b>Websites:</b>\n"
                for source in website_sources:
                    message += f"  • {source.source_identifier}\n"
                message += "\n"
            
            message += f"<b>Total:</b> {len(sources)} sources"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /run command."""
        chat_id = str(update.effective_chat.id)
        
        try:
            group = await self.db.get_group(chat_id)
            if not group:
                await update.message.reply_text(
                    "❌ Group not initialized. Use /start first."
                )
                return
            
            sources = await self.db.get_group_sources(chat_id)
            if not sources:
                await update.message.reply_text(
                    "❌ No sources configured. Add sources first using:\n"
                    "• /add_telegram @channel\n"
                    "• /add_arxiv category\n"
                    "• /add_website URL"
                )
                return
            
            status_message = await update.message.reply_text(
                "🚀 Starting AI News Agent...\n"
                "This may take a few minutes."
            )
            
            result = await self._run_agent_for_group(group, chat_id)
            
            if result['success']:
                await status_message.edit_text(
                    f"✅ <b>Agent Run Completed!</b>\n\n"
                    f"📊 <b>Results:</b>\n"
                    f"  • Articles extracted: {result.get('articles_extracted', 0)}\n"
                    f"  • Documents merged: {result.get('documents_merged', 0)}\n"
                    f"  • Top documents: {result.get('top_documents_selected', 0)}\n"
                    f"  • Execution time: {result.get('execution_time', 0):.1f}s\n\n"
                    f"📬 News post has been sent to this chat!",
                    parse_mode='HTML'
                )
            else:
                await status_message.edit_text(
                    f"❌ <b>Agent Run Failed</b>\n\n"
                    f"Error: {result.get('error', 'Unknown error')}",
                    parse_mode='HTML'
                )
            
        except Exception as e:
            logger.error(f"Error in /run command: {e}\n{traceback.format_exc()}")
            await update.message.reply_text(
                f"❌ Error running agent: {str(e)}"
            )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        chat_id = str(update.effective_chat.id)
        
        try:
            group = await self.db.get_group(chat_id)
            if not group:
                await update.message.reply_text(
                    "❌ Group not initialized. Use /start first."
                )
                return
            
            sources = await self.db.get_group_sources(chat_id)
            history = await self.db.get_group_run_history(chat_id, limit=1)
            
            message = f"<b>📊 Agent Status</b>\n\n"
            message += f"<b>Group:</b> {group.chat_title}\n"
            message += f"<b>Status:</b> {'🟢 Active' if group.is_active else '🔴 Inactive'}\n"
            message += f"<b>Sources:</b> {len(sources)}\n"
            
            if group.last_run:
                message += f"<b>Last Run:</b> {group.last_run.strftime('%Y-%m-%d %H:%M UTC')}\n"
            else:
                message += f"<b>Last Run:</b> Never\n"
            
            if history:
                last_run = history[0]
                message += f"\n<b>Last Run Details:</b>\n"
                message += f"  • Status: {last_run.status}\n"
                message += f"  • Articles: {last_run.articles_extracted}\n"
                message += f"  • Documents: {last_run.documents_merged}\n"
                if last_run.execution_time_seconds:
                    message += f"  • Time: {last_run.execution_time_seconds}s\n"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command."""
        chat_id = str(update.effective_chat.id)
        
        try:
            history = await self.db.get_group_run_history(chat_id, limit=10)
            
            if not history:
                await update.message.reply_text(
                    "📭 No run history yet.\n\n"
                    "Use /run to execute the agent manually."
                )
                return
            
            message = "<b>📜 Run History (Last 10)</b>\n\n"
            
            for run in history:
                status_emoji = {
                    'completed': '✅',
                    'failed': '❌',
                    'running': '🔄'
                }.get(run.status, '❓')
                
                message += f"{status_emoji} <b>{run.started_at.strftime('%Y-%m-%d %H:%M')}</b>\n"
                message += f"   Status: {run.status}\n"
                
                if run.status == 'completed':
                    message += f"   Articles: {run.articles_extracted}, "
                    message += f"Docs: {run.documents_merged}\n"
                elif run.status == 'failed' and run.error_message:
                    error_short = run.error_message[:50] + '...' if len(run.error_message) > 50 else run.error_message
                    message += f"   Error: {error_short}\n"
                
                message += "\n"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /history: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command."""
        chat_id = str(update.effective_chat.id)
        
        try:
            group = await self.db.get_group(chat_id)
            if not group:
                await update.message.reply_text(
                    "❌ Group not initialized. Use /start first."
                )
                return
            
            config = group.get_config_dict()
            
            # Get post_size (prioritize post_size over max_top_clusters)
            post_size = config.get('ranking', {}).get('post_size', 
                                                     config.get('ranking', {}).get('max_top_clusters', 10))
            
            # Get language and keywords
            language = group.language if hasattr(group, 'language') else 'en'
            keywords = group.keywords if hasattr(group, 'keywords') else None
            
            message = "<b>⚙️ Current Configuration</b>\n\n"
            
            # User Preferences
            message += "<b>📝 User Preferences:</b>\n"
            message += f"  • Language: {language.upper()}\n"
            message += f"  • Post Size: {post_size} articles\n"
            if keywords:
                message += f"  • Keywords: {keywords}\n"
            else:
                message += f"  • Keywords: Not set\n"
            
            # LLM Configuration
            message += f"\n<b>🤖 LLM Configuration:</b>\n"
            message += f"  • Provider: {config.get('llm', {}).get('provider', 'openrouter')}\n"
            message += f"  • Embedding: {config.get('clustering', {}).get('embedding_model', 'N/A')}\n"
            message += f"  • Analysis: {config.get('content_analysis', {}).get('llm_model', 'N/A')}\n"
            message += f"  • Post Gen: {config.get('post_generation', {}).get('llm_model', 'N/A')}\n"
            
            # Processing Configuration
            message += f"\n<b>⚙️ Processing:</b>\n"
            message += f"  • Clustering: {config.get('clustering', {}).get('algorithm', 'dbscan')}\n"
            message += f"  • Collection: {group.chroma_collection}\n"
            message += f"  • Similarity: {config.get('vector_store', {}).get('similarity_threshold', 0.90)}\n"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /config: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_set_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /set_language command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /set_language <language>\n"
                "Available languages:\n"
                "  • en - English\n"
                "  • ru - Russian (Русский)\n\n"
                "Example: /set_language ru"
            )
            return
        
        language = context.args[0].lower()
        
        try:
            success = await self.db.set_language(chat_id, language)
            
            if success:
                lang_name = "English" if language == "en" else "Русский"
                await update.message.reply_text(
                    f"✅ Language set to: {lang_name}\n\n"
                    f"Posts will be translated to {lang_name}.\n"
                    f"Note: English version is always saved to vector store."
                )
            else:
                await update.message.reply_text(
                    f"❌ Invalid language: {language}\n"
                    f"Use 'en' or 'ru'"
                )
            
        except Exception as e:
            logger.error(f"Error setting language: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_add_keywords(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_keywords command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /add_keywords <keywords or phrase>\n\n"
                "Examples:\n"
                "  /add_keywords machine learning, neural networks\n"
                "  /add_keywords LLM agents and reasoning\n\n"
                "The agent will filter content based on relevance to these keywords."
            )
            return
        
        keywords = ' '.join(context.args)
        
        try:
            success = await self.db.set_keywords(chat_id, keywords)
            
            if success:
                await update.message.reply_text(
                    f"✅ Keywords set:\n{keywords}\n\n"
                    f"Content will be filtered based on relevance to these keywords."
                )
            else:
                await update.message.reply_text("❌ Failed to set keywords")
            
        except Exception as e:
            logger.error(f"Error setting keywords: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_set_post_size(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /set_post_size command."""
        chat_id = str(update.effective_chat.id)
        
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /set_post_size <number>\n\n"
                "Set the number of top articles to include in each post.\n"
                "Range: 1-50\n\n"
                "Example: /set_post_size 15"
            )
            return
        
        try:
            post_size = int(context.args[0])
            
            success = await self.db.set_post_size(chat_id, post_size)
            
            if success:
                await update.message.reply_text(
                    f"✅ Post size set to: {post_size}\n\n"
                    f"Each post will include up to {post_size} top articles."
                )
            else:
                await update.message.reply_text(
                    f"❌ Invalid post size: {post_size}\n"
                    f"Must be between 1 and 50"
                )
            
        except ValueError:
            await update.message.reply_text("❌ Post size must be a number")
        except Exception as e:
            logger.error(f"Error setting post size: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _run_agent_for_group(self, group, chat_id: str) -> Dict[str, Any]:
        """Run the AI news agent for a specific group."""
        run_history = None
        start_time = datetime.now()
        
        try:
            run_history = await self.db.create_run_history(chat_id, status='running')
            
            config_dict = group.get_config_dict()
            sources = await self.db.get_group_sources(chat_id)
            
            config_dict['sources'] = self._build_sources_config(sources)
            config_dict['telegram'] = {
                'bot_token': self.token,
                'channel_id': chat_id
            }
            config_dict['vector_store']['collection_name'] = group.chroma_collection
            
            # Add language and keywords to config
            config_dict['language'] = group.language
            config_dict['keywords'] = group.keywords
            
            config = Config.__new__(Config)
            config._agent_config = config_dict
            config._sources_config = {'sources': config_dict['sources']}
            config.settings = Config("config").settings
            
            # Set telegram channel ID for the poster node
            config.settings.telegram_bot_token = self.token
            config.settings.telegram_channel_id = chat_id
            
            agent = AINewsAgent(config)
            await agent.initialize()
            
            result = await agent.run_workflow(
                language=group.language,
                keywords=group.keywords
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if run_history:
                await self.db.update_run_history(
                    run_id=run_history.id,
                    status='completed',
                    articles_extracted=result.get('articles_extracted', 0),
                    documents_merged=result.get('merged_documents', 0),
                    top_documents_selected=result.get('top_documents_selected', 0),
                    post_sent=result.get('post_sent', False),
                    execution_time_seconds=int(execution_time),
                    metrics=result.get('metrics', {})
                )
            
            await self.db.update_last_run(chat_id)
            await agent.cleanup()
            
            return {
                'success': True,
                'articles_extracted': result.get('articles_extracted', 0),
                'documents_merged': result.get('merged_documents', 0),
                'top_documents_selected': result.get('top_documents_selected', 0),
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Error running agent for group {chat_id}: {e}\n{traceback.format_exc()}")
            
            if run_history:
                await self.db.update_run_history(
                    run_id=run_history.id,
                    status='failed',
                    error_message=str(e),
                    error_traceback=traceback.format_exc()
                )
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_sources_config(self, sources) -> Dict[str, Any]:
        """Build sources configuration from database sources."""
        config = {
            'arxiv': {
                'enabled': False,
                'categories': [],
                'max_results': 50,
                'date_range': 5,
                'sort_by': 'submittedDate'
            },
            'telegram': {
                'enabled': False,
                'channels': [],
                'max_messages_per_channel': 50,
                'message_age_limit': 24,
                'rate_limit': 20
            },
            'websites': {
                'enabled': False,
                'sources': [],
                'scrape_delay': 1.0,
                'max_articles_per_source': 20
            }
        }
        
        for source in sources:
            if source.source_type == 'arxiv':
                config['arxiv']['enabled'] = True
                config['arxiv']['categories'].append(source.source_identifier)
            elif source.source_type == 'telegram':
                config['telegram']['enabled'] = True
                config['telegram']['channels'].append(source.source_identifier)
            elif source.source_type == 'website':
                config['websites']['enabled'] = True
                config['websites']['sources'].append({
                    'name': source.source_identifier,
                    'url': source.source_identifier,
                    'type': 'rss',
                    'reliability_score': 0.8
                })
        
        return config
