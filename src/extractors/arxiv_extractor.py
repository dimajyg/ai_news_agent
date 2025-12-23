# arXiv Extractor Implementation

import asyncio
import logging
import arxiv
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class ArxivConfig:
    """Configuration for arXiv extraction."""
    categories: List[str]
    max_results: int
    date_range: int  # days
    sort_by: str
    min_abstract_length: int
    exclude_keywords: List[str]
    include_keywords: List[str]

class ArxivExtractor:
    """Extractor for arXiv research papers."""
    
    def __init__(self, config):
        self.config = config
        self.arxiv_config = self._parse_config()
        # Initialize arXiv client with retries and pacing
        try:
            self.client = arxiv.Client(
                page_size=max(25, min(100, self.config.get('sources.arxiv.max_results', 50))),
                delay_seconds=2,
                num_retries=3
            )
        except Exception:
            self.client = arxiv.Client()
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _parse_config(self) -> ArxivConfig:
        """Parse configuration for arXiv extraction."""
        return ArxivConfig(
            categories=self.config.get('sources.arxiv.categories', ['cs.AI', 'cs.LG', 'cs.CL']),
            max_results=self.config.get('sources.arxiv.max_results', 50),
            date_range=self.config.get('sources.arxiv.date_range', 5),
            sort_by=self.config.get('sources.arxiv.sort_by', 'submittedDate'),
            min_abstract_length=self.config.get('sources.arxiv.min_abstract_length', 200),
            exclude_keywords=self.config.get('sources.arxiv.exclude_keywords', []),
            include_keywords=self.config.get('sources.arxiv.include_keywords', [])
        )
    
    async def extract_latest(self) -> List[Dict[str, Any]]:
        """Extract latest papers from arXiv."""
        logger.info("Extracting latest papers from arXiv")
        
        try:
            # Build one or more search queries to avoid overly long strings
            queries = self._build_chunked_queries()
            papers: List[Dict[str, Any]] = []
            for q in queries:
                search = arxiv.Search(
                    query=q,
                    max_results=self.arxiv_config.max_results // max(1, len(queries)),
                    sort_by=self._get_sort_criteria(),
                    sort_order=arxiv.SortOrder.Descending
                )
                async for paper in self._fetch_papers_async(search):
                    processed_paper = self._process_paper(paper)
                    if processed_paper:
                        papers.append(processed_paper)
                await asyncio.sleep(1)
            
            logger.info(f"Successfully extracted {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to extract papers from arXiv: {str(e)}")
            # Fallback: query per-category without keyword filters and smaller batch
            try:
                fallback_papers: List[Dict[str, Any]] = []
                for cat in self.arxiv_config.categories:
                    simple_query = self._build_simple_query(cat)
                    search = arxiv.Search(
                        query=simple_query,
                        max_results=max(10, min(25, self.arxiv_config.max_results // max(1, len(self.arxiv_config.categories))))
                    )
                    async for paper in self._fetch_papers_async(search):
                        processed = self._process_paper(paper)
                        if processed:
                            fallback_papers.append(processed)
                    await asyncio.sleep(1)
                logger.info(f"Fallback arXiv extraction returned {len(fallback_papers)} papers")
                if fallback_papers:
                    return fallback_papers
                logger.info("No results via API/fallback, attempting web scraping mode")
                web_results = await self._extract_via_web()
                return web_results
            except Exception as fe:
                logger.error(f"Fallback arXiv extraction failed: {str(fe)}")
                try:
                    logger.info("Attempting web scraping mode after fallback failure")
                    web_results = await self._extract_via_web()
                    return web_results
                except Exception as we:
                    logger.error(f"Web scraping mode failed: {we}")
                    return []
    
    def _build_search_query(self) -> str:
        """Build arXiv search query."""
        query_parts = []
        
        # Add categories
        if self.arxiv_config.categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in self.arxiv_config.categories])
            query_parts.append(f"({category_query})")
        
        # Add date range filter
        cutoff_date = datetime.now() - timedelta(days=self.arxiv_config.date_range)
        date_filter = f"submittedDate:[{cutoff_date.strftime('%Y%m%d')}0000 TO *]"
        query_parts.append(date_filter)
        
        # Add include keywords
        if self.arxiv_config.include_keywords:
            keyword_query = " OR ".join([kw for kw in self.arxiv_config.include_keywords if kw and len(kw) <= 20])
            if keyword_query:
                query_parts.append(f"(ti:({keyword_query}) OR abs:({keyword_query}))")
        
        # Add exclude keywords
        if self.arxiv_config.exclude_keywords:
            exclude_query = " OR ".join([kw for kw in self.arxiv_config.exclude_keywords if kw and len(kw) <= 20])
            if exclude_query:
                query_parts.append(f"(-ti:({exclude_query}) -abs:({exclude_query}))")
        
        return " AND ".join(query_parts)

    def _build_chunked_queries(self) -> List[str]:
        """Build one or more queries by chunking include keywords to avoid API errors."""
        base = []
        # Categories and date filter
        if self.arxiv_config.categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in self.arxiv_config.categories])
            base.append(f"({category_query})")
        cutoff_date = datetime.now() - timedelta(days=self.arxiv_config.date_range)
        base.append(f"submittedDate:[{cutoff_date.strftime('%Y%m%d')}0000 TO *]")
        exclude = []
        if self.arxiv_config.exclude_keywords:
            exclude_query = " OR ".join([kw for kw in self.arxiv_config.exclude_keywords if kw and len(kw) <= 20])
            if exclude_query:
                exclude.append(f"(-ti:({exclude_query}) -abs:({exclude_query}))")
        include = self.arxiv_config.include_keywords or []
        if not include:
            # No include keywords; single query
            return [" AND ".join(base + exclude)]
        # Chunk include keywords
        chunk_size = 5
        queries: List[str] = []
        for i in range(0, len(include), chunk_size):
            chunk = include[i:i+chunk_size]
            kw = " OR ".join([kw for kw in chunk if kw and len(kw) <= 30])
            if not kw:
                continue
            queries.append(" AND ".join(base + [f"(ti:({kw}) OR abs:({kw}))"] + exclude))
        return queries or [" AND ".join(base + exclude)]

    def _build_simple_query(self, category: str) -> str:
        cutoff_date = datetime.now() - timedelta(days=self.arxiv_config.date_range)
        date_filter = f"submittedDate:[{cutoff_date.strftime('%Y%m%d')}0000 TO *]"
        return f"cat:{category} AND {date_filter}"
    
    def _get_sort_criteria(self):
        """Get arXiv sort criteria."""
        sort_map = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
            'submittedDate': arxiv.SortCriterion.SubmittedDate
        }
        return sort_map.get(self.arxiv_config.sort_by, arxiv.SortCriterion.SubmittedDate)
    
    async def _fetch_papers_async(self, search):
        """Fetch papers asynchronously."""
        loop = asyncio.get_event_loop()
        
        def fetch_results():
            try:
                return list(self.client.results(search))
            except Exception:
                return list(search.results())
        
        papers = await loop.run_in_executor(None, fetch_results)
        
        for paper in papers:
            yield paper

    async def _extract_via_web(self) -> List[Dict[str, Any]]:
        headers = {
            'User-Agent': self.config.get('sources.websites.user_agent', 'Mozilla/5.0 (AI-News-Agent/1.0)'),
            'Accept-Language': 'en-US,en;q=0.9',
        }
        timeout = aiohttp.ClientTimeout(total=self.config.get('performance.timeout', 30))
        connector = aiohttp.TCPConnector(limit=self.config.get('performance.max_concurrent_requests', 10))
        past_scope = self._compute_past_scope()
        page_size = max(25, min(100, self.config.get('sources.arxiv.web_page_size', 25)))
        max_results = self.arxiv_config.max_results
        try:
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers)
            ids: List[str] = []
            for cat in self.arxiv_config.categories:
                cat_ids = await self._collect_ids_for_category(cat, past_scope, page_size, max_results)
                ids.extend(cat_ids)
                await asyncio.sleep(1)
            seen = set()
            dedup_ids = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    dedup_ids.append(i)
                if len(dedup_ids) >= max_results:
                    break
            sem = asyncio.Semaphore(self.config.get('performance.max_concurrent_requests', 10))
            async def fetch_one(arxiv_id: str) -> Optional[Dict[str, Any]]:
                async with sem:
                    return await self._fetch_abs_page(arxiv_id)
            tasks = [asyncio.create_task(fetch_one(i)) for i in dedup_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            papers: List[Dict[str, Any]] = []
            for r in results:
                if isinstance(r, dict) and r:
                    if self._validate_paper(r):
                        papers.append(r)
            logger.info(f"Extracted {len(papers)} papers via web scraping")
            return papers
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def _collect_ids_for_category(self, category: str, past_scope: str, page_size: int, max_results: int) -> List[str]:
        base = f"https://arxiv.org/list/{category}/{past_scope}"
        ids: List[str] = []
        skip = 0
        while skip < max_results:
            url = f"{base}?skip={skip}&show={page_size}"
            try:
                async with self.session.get(url, allow_redirects=True) as resp:
                    if resp.status != 200:
                        break
                    html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                anchors = soup.select('a[href^="/abs/"]')
                found = 0
                for a in anchors:
                    href = a.get('href', '')
                    if '/abs/' in href:
                        arxiv_id = href.split('/abs/')[-1]
                        if arxiv_id and arxiv_id not in ids:
                            ids.append(arxiv_id)
                            found += 1
                if found == 0:
                    break
                skip += page_size
                if len(ids) >= max_results:
                    break
                await asyncio.sleep(0.5)
            except Exception:
                break
        return ids

    async def _fetch_abs_page(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            async with self.session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            title = self._extract_meta(soup, 'citation_title') or self._extract_title_html(soup)
            authors = self._extract_authors_meta(soup)
            published_date = self._extract_date_meta(soup) or datetime.now().isoformat()
            abstract = self._extract_abstract_html(soup)
            pdf_url = self._extract_pdf_url(soup, arxiv_id)
            categories, primary_category = self._extract_categories_html(soup)
            paper_data = {
                'title': (title or '').strip(),
                'abstract': (abstract or '').strip(),
                'authors': authors,
                'published_date': published_date,
                'arxiv_id': arxiv_id,
                'arxiv_url': url,
                'pdf_url': pdf_url,
                'categories': categories,
                'primary_category': primary_category,
                'source': 'arxiv',
                'extraction_metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'method': 'web_scrape'
                }
            }
            return paper_data
        except Exception:
            return None

    def _extract_meta(self, soup: BeautifulSoup, name: str) -> Optional[str]:
        m = soup.select_one(f'meta[name="{name}"]')
        return m.get('content') if m else None

    def _extract_title_html(self, soup: BeautifulSoup) -> Optional[str]:
        h = soup.select_one('#abs .title')
        if h:
            txt = h.get_text().strip()
            return txt.replace('Title:', '').strip()
        return None

    def _extract_authors_meta(self, soup: BeautifulSoup) -> List[str]:
        metas = soup.select('meta[name="citation_author"]')
        if metas:
            return [m.get('content').strip() for m in metas if m.get('content')]
        authors_div = soup.select_one('#abs .authors')
        if authors_div:
            txt = authors_div.get_text().replace('Authors:', '').strip()
            parts = [p.strip() for p in txt.split(',') if p.strip()]
            return parts
        return []

    def _extract_date_meta(self, soup: BeautifulSoup) -> Optional[str]:
        d = self._extract_meta(soup, 'citation_date')
        if d:
            try:
                return datetime.strptime(d.strip(), '%Y/%m/%d').isoformat()
            except Exception:
                return d
        dateline = soup.select_one('#abs .dateline')
        if dateline:
            txt = dateline.get_text().strip()
            return datetime.now().isoformat()
        return None

    def _extract_abstract_html(self, soup: BeautifulSoup) -> Optional[str]:
        bq = soup.select_one('blockquote.abstract')
        if bq:
            txt = bq.get_text().strip()
            return txt.replace('Abstract:', '').strip()
        span = soup.select_one('#abs .abstract-full')
        if span:
            return span.get_text().strip()
        return None

    def _extract_pdf_url(self, soup: BeautifulSoup, arxiv_id: str) -> Optional[str]:
        m = self._extract_meta(soup, 'citation_pdf_url')
        if m:
            return m
        a = soup.select_one(f'a[href^="/pdf/{arxiv_id}"]')
        if a:
            href = a.get('href')
            return f"https://arxiv.org{href}" if href.startswith('/') else href
        return None

    def _extract_categories_html(self, soup: BeautifulSoup) -> Tuple[List[str], Optional[str]]:
        cats = []
        primary = None
        subj = soup.select_one('#abs .subheader')
        if subj:
            txt = subj.get_text().strip()
            if '(' in txt and ')' in txt:
                inside = txt[txt.find('(')+1:txt.find(')')]
                parts = [p.strip() for p in inside.split(';') if p.strip()]
                cats = parts
                primary = parts[0] if parts else None
        return cats, primary

    def _compute_past_scope(self) -> str:
        days = self.arxiv_config.date_range
        if days <= 1:
            return 'pastday'
        if days <= 7:
            return 'pastweek'
        return 'pastmonth'
    
    def _process_paper(self, paper) -> Dict[str, Any]:
        """Process arXiv paper data."""
        try:
            # Extract basic information
            paper_data = {
                'title': paper.title.strip(),
                'abstract': paper.summary.strip(),
                'authors': [str(author) for author in paper.authors],
                'published_date': paper.published.isoformat(),
                'arxiv_id': paper.entry_id.split('/')[-1],
                'arxiv_url': paper.entry_id,
                'pdf_url': paper.pdf_url,
                'categories': paper.categories,
                'primary_category': paper.primary_category,
                'source': 'arxiv',
                'extraction_metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'comment': paper.comment if hasattr(paper, 'comment') else None,
                    'journal_ref': paper.journal_ref if hasattr(paper, 'journal_ref') else None,
                    'doi': paper.doi if hasattr(paper, 'doi') else None
                }
            }
            
            # Validate paper
            if not self._validate_paper(paper_data):
                return None
            
            return paper_data
            
        except Exception as e:
            logger.warning(f"Failed to process arXiv paper: {str(e)}")
            return None
    
    def _validate_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Validate extracted paper data."""
        # Check required fields
        if not paper_data.get('title') or not paper_data.get('abstract'):
            logger.warning("Paper missing title or abstract")
            return False
        
        # Check abstract length
        if len(paper_data['abstract']) < self.arxiv_config.min_abstract_length:
            logger.warning(f"Abstract too short: {len(paper_data['abstract'])} characters")
            return False
        
        # Check title quality
        title = paper_data['title'].lower()
        if any(keyword in title for keyword in ['withdrawn', 'retracted', 'erratum']):
            logger.warning(f"Paper appears to be withdrawn or retracted: {paper_data['title']}")
            return False
        
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'source': 'arxiv',
            'categories': self.arxiv_config.categories,
            'max_results': self.arxiv_config.max_results,
            'date_range_days': self.arxiv_config.date_range,
            'sort_by': self.arxiv_config.sort_by,
            'min_abstract_length': self.arxiv_config.min_abstract_length
        }
