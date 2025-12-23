# Website Extractor Implementation

import asyncio
import logging
import aiohttp
import feedparser
from typing import Dict, Any, List
from datetime import datetime
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class WebsiteExtractor:
    """Extractor for AI news websites and RSS feeds."""
    
    def __init__(self, config):
        self.config = config
        self.rss_feeds = self._load_rss_feeds()
        self.web_sources = self._load_web_sources()
        self.session = None
    
    def _load_rss_feeds(self) -> List[Dict[str, Any]]:
        """Load RSS feed configurations."""
        feeds = self.config.get('sources.websites.rss_feeds', [])
        alt_sources = self.config.get('sources.websites.sources', [])
        if alt_sources:
            for src in alt_sources:
                if src.get('type') == 'rss' and 'url' in src:
                    feeds.append({
                        'name': src.get('name', 'Unknown'),
                        'url': src['url'],
                        'category': src.get('category', 'general'),
                        'reliability_score': src.get('reliability_score', 0.5),
                        'max_articles': src.get('max_articles', 20)
                    })
        return [
            {
                'name': feed.get('name', 'Unknown'),
                'url': feed['url'],
                'category': feed.get('category', 'general'),
                'reliability_score': feed.get('reliability_score', 0.5),
                'max_articles': feed.get('max_articles', 20)
            }
            for feed in feeds
            if 'url' in feed
        ]
    
    def _load_web_sources(self) -> List[Dict[str, Any]]:
        """Load web scraping source configurations."""
        sources = self.config.get('sources.websites.web_scraping', [])
        return [
            {
                'name': source.get('name', 'Unknown'),
                'url': source['url'],
                'selector': source.get('selector', 'article'),
                'category': source.get('category', 'company'),
                'reliability_score': source.get('reliability_score', 0.8),
                'max_articles': source.get('max_articles', 15)
            }
            for source in sources
            if 'url' in source
        ]
    
    async def extract_latest(self) -> List[Dict[str, Any]]:
        """Extract latest articles from websites."""
        logger.info("Extracting latest articles from websites")
        
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            # Extract from RSS feeds and websites concurrently
            tasks = []
            
            # RSS feed extraction
            for feed in self.rss_feeds:
                task = asyncio.create_task(self._extract_rss_feed(feed))
                tasks.append(task)
            
            # Web scraping (with rate limiting)
            for source in self.web_sources:
                task = asyncio.create_task(self._scrape_website(source))
                tasks.append(task)
            
            # Execute all extractions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_articles = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Extraction task failed: {str(result)}")
                elif result:
                    all_articles.extend(result)
            
            logger.info(f"Successfully extracted {len(all_articles)} articles from websites")
            return all_articles
            
        finally:
            if self.session:
                await self.session.close()
    
    async def _extract_rss_feed(self, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract articles from RSS feed."""
        logger.info(f"Extracting RSS feed: {feed_config['name']}")
        
        try:
            # Fetch RSS feed with browser-like headers
            headers = {
                'User-Agent': self.config.get('sources.websites.user_agent', 'Mozilla/5.0 (AI-News-Agent/1.0)'),
                'Accept': 'application/rss+xml, application/xml;q=0.9, */*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': feed_config.get('url', '')
            }
            async with self.session.get(feed_config['url'], headers=headers, allow_redirects=True) as response:
                if response.status in (403, 404):
                    # Try fallbacks for known hosts
                    fallback_url = self._fallback_rss_url(feed_config['url'])
                    if fallback_url:
                        logger.info(f"Trying RSS fallback URL: {fallback_url}")
                        async with self.session.get(fallback_url, headers=headers, allow_redirects=True) as resp2:
                            if resp2.status != 200:
                                logger.warning(f"RSS fallback returned status {resp2.status}: {fallback_url}")
                                return []
                            content = await resp2.text()
                    else:
                        logger.warning(f"RSS feed returned status {response.status}: {feed_config['url']}")
                        return []
                elif response.status != 200:
                    logger.warning(f"RSS feed returned status {response.status}: {feed_config['url']}")
                    return []
                else:
                    content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(content)
            
            articles = []
            for entry in feed.entries[:feed_config['max_articles']]:
                article = self._process_rss_entry(entry, feed_config)
                if article:
                    # If content is very short, attempt to fetch full article page
                    if len(article.get('content', '')) < max(50, self.config.get('content_filtering.min_content_length', 100)) and article.get('url'):
                        full_text = await self._fetch_full_article(article['url'])
                        if full_text and len(full_text) > len(article['content']):
                            article['content'] = full_text
                            article['abstract'] = full_text[:1000]
                    if self._validate_article(article):
                        articles.append(article)
            
            logger.info(f"Extracted {len(articles)} articles from RSS feed: {feed_config['name']}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to extract RSS feed {feed_config['name']}: {str(e)}")
            return []

    def _fallback_rss_url(self, url: str) -> str:
        """Return a known fallback RSS URL for certain hosts."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.netloc
            # VentureBeat AI feed moved under /category
            if 'venturebeat.com' in host:
                if '/ai/' in parsed.path:
                    return 'https://venturebeat.com/category/ai/feed/'
                return 'https://venturebeat.com/?feed=rss2'
            # Medium/Towards Data Science requires medium.com feed
            if 'towardsdatascience.com' in host:
                return 'https://medium.com/feed/towards-data-science'
            return ''
        except Exception:
            return ''
    
    def _process_rss_entry(self, entry, feed_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS feed entry."""
        try:
            # Extract basic information
            title = entry.get('title', '').strip()
            link = entry.get('link', '')
            
            # Extract content/summary
            content = entry.get('summary', entry.get('description', ''))
            if not content and hasattr(entry, 'content'):
                content = entry.content[0].value if entry.content else ''
            
            # Extract publication date
            published = entry.get('published_parsed')
            if published:
                published_date = datetime(*published[:6]).isoformat()
            else:
                published_date = datetime.now().isoformat()
            
            # Extract author
            author = entry.get('author', '')
            
            # Create article
            article = {
                'title': title,
                'abstract': content[:1000],  # Limit content length
                'content': content,
                'url': link,
                'authors': [author] if author else [],
                'published_date': published_date,
                'source': 'website',
                'source_name': feed_config['name'],
                'source_category': feed_config['category'],
                'reliability_score': feed_config['reliability_score'],
                'extraction_metadata': {
                    'feed_type': 'rss',
                    'extracted_at': datetime.now().isoformat()
                }
            }
            
            return article if self._validate_article(article) else None
            
        except Exception as e:
            logger.warning(f"Failed to process RSS entry: {str(e)}")
            return None
    
    async def _scrape_website(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape articles from website."""
        logger.info(f"Scraping website: {source_config['name']}")
        
        try:
            # Add delay to respect rate limits
            await asyncio.sleep(self.config.get('sources.websites.scrape_delay', 1.0))
            
            # Fetch website
            headers = {
                'User-Agent': self.config.get('sources.websites.user_agent', 'AI-News-Agent/1.0')
            }
            
            async with self.session.get(source_config['url'], headers=headers, allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Website returned status {response.status}: {source_config['url']}")
                    return []
                
                html = await response.text()
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract articles using selector
            articles = []
            article_elements = soup.select(source_config['selector'])
            
            for element in article_elements[:source_config['max_articles']]:
                article = self._process_article_element(element, source_config)
                if article:
                    # If content is short, try fetching full article
                    if len(article.get('content', '')) < max(50, self.config.get('content_filtering.min_content_length', 100)) and article.get('url'):
                        full_text = await self._fetch_full_article(article['url'])
                        if full_text and len(full_text) > len(article['content']):
                            article['content'] = full_text
                            article['abstract'] = full_text[:1000]
                    if self._validate_article(article):
                        articles.append(article)
            
            logger.info(f"Scraped {len(articles)} articles from website: {source_config['name']}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to scrape website {source_config['name']}: {str(e)}")
            return []

    async def _fetch_full_article(self, url: str) -> str:
        """Fetch full article content from a URL using simple heuristics."""
        try:
            headers = {
                'User-Agent': self.config.get('sources.websites.user_agent', 'Mozilla/5.0 (AI-News-Agent/1.0)'),
                'Accept-Language': 'en-US,en;q=0.9',
            }
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status != 200:
                    return ''
                html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            # Try common containers
            candidates = []
            candidates += soup.select('article')
            candidates += soup.select('div[class*="content"], div[id*="content"], div[class*="post"], section[class*="content"]')
            text_chunks = []
            for node in candidates or [soup]:
                for p in node.select('p'):
                    txt = p.get_text().strip()
                    if len(txt) > 0:
                        text_chunks.append(txt)
                if len(' '.join(text_chunks)) > 300:
                    break
            full_text = '\n'.join(text_chunks).strip()
            return full_text
        except Exception:
            return ''
    
    def _process_article_element(self, element, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process HTML article element."""
        try:
            # Extract title
            title_elem = element.find(['h1', 'h2', 'h3', 'h4']) or element.find('a')
            title = title_elem.get_text().strip() if title_elem else ''
            
            # Extract content
            content_elem = element.find(['p', 'div', 'section']) or element
            content = content_elem.get_text().strip()
            
            # Extract link
            link_elem = element.find('a', href=True)
            url = link_elem['href'] if link_elem else source_config['url']
            
            # Make relative URLs absolute
            if url.startswith('/'):
                from urllib.parse import urljoin
                url = urljoin(source_config['url'], url)
            
            # Extract date (try multiple selectors)
            date_elem = element.find(['time', 'span', 'div'], class_=lambda x: x and 'date' in str(x).lower())
            published_date = None
            if date_elem:
                date_text = date_elem.get('datetime', date_elem.get_text())
                published_date = self._parse_date(date_text)
            
            if not published_date:
                published_date = datetime.now().isoformat()
            
            # Extract author
            author_elem = element.find(['span', 'div'], class_=lambda x: x and 'author' in str(x).lower())
            author = author_elem.get_text().strip() if author_elem else ''
            
            # Create article
            article = {
                'title': title,
                'abstract': content[:1000],  # Limit abstract length
                'content': content,
                'url': url,
                'authors': [author] if author else [],
                'published_date': published_date,
                'source': 'website',
                'source_name': source_config['name'],
                'source_category': source_config['category'],
                'reliability_score': source_config['reliability_score'],
                'extraction_metadata': {
                    'extraction_method': 'web_scraping',
                    'extracted_at': datetime.now().isoformat()
                }
            }
            
            return article if self._validate_article(article) else None
            
        except Exception as e:
            logger.warning(f"Failed to process article element: {str(e)}")
            return None
    
    def _parse_date(self, date_text: str) -> str:
        """Parse date from various formats."""
        try:
            # Try parsing as ISO format
            if 'T' in date_text:
                return datetime.fromisoformat(date_text.replace('Z', '+00:00')).isoformat()
            
            # Try common date formats
            date_formats = [
                '%Y-%m-%d',
                '%B %d, %Y',
                '%b %d, %Y',
                '%d %B %Y',
                '%d %b %Y'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_text.strip(), fmt).isoformat()
                except ValueError:
                    continue
            
            return datetime.now().isoformat()
            
        except Exception:
            return datetime.now().isoformat()
    
    def _validate_article(self, article: Dict[str, Any]) -> bool:
        """Validate extracted article."""
        # Check required fields
        if not article.get('title') or len(article['title']) < 10:
            return False
        
        if not article.get('content') or len(article['content']) < 50:
            return False
        
        # Check for spam indicators
        content_lower = article.get('content', '').lower()
        spam_keywords = ['click here', 'buy now', 'subscribe', 'advertisement']
        if any(keyword in content_lower for keyword in spam_keywords):
            return False
        
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'source': 'websites',
            'rss_feeds': len(self.rss_feeds),
            'web_sources': len(self.web_sources),
            'total_sources': len(self.rss_feeds) + len(self.web_sources)
        }
