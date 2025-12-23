import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Tuple
import re
from .logging import AgentLogger

logger = AgentLogger(__name__)

async def fetch_url_text(url: str, timeout_sec: int = 30) -> Tuple[str, str]:
    """Fetch URL and return (title, text).
    Strategy:
    - PDFs: download and parse via pdfminer.six
    - arXiv abs pages: parse meta tags and abstract blockquote
    - GitHub: parse .markdown-body (README)
    - Hugging Face: parse main/article sections
    - Generic: newspaper3k then BeautifulSoup fallback with Readability (if available)
    """
    if url.lower().endswith('.pdf'):
        try:
            from pdfminer.high_level import extract_text
            import tempfile, os
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                pdf_url = url if url.lower().endswith('.pdf') else (url + '.pdf')
                data = None
                for attempt in range(3):
                    try:
                        async with session.get(pdf_url, allow_redirects=True, headers={
                            'User-Agent': 'Mozilla/5.0 (AI-News-Agent/1.0)',
                            'Accept': 'application/pdf,*/*'
                        }) as resp:
                            if resp.status != 200:
                                logger.warning(f"fetch_url_text: pdf GET status={resp.status} attempt={attempt+1} url='{pdf_url}'")
                            else:
                                data = await resp.read()
                                break
                    except Exception as ex:
                        logger.warning(f"fetch_url_text: pdf GET exception attempt={attempt+1} url='{pdf_url}' err={ex}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                if not data:
                    return '', ''
            fd, tmp = tempfile.mkstemp(suffix='.pdf')
            try:
                with open(tmp, 'wb') as f:
                    f.write(data)
                text = extract_text(tmp) or ''
                logger.info(f"fetch_url_text: pdf extracted length={len(text or '')} url='{pdf_url}'")
                return '', text.strip()
            finally:
                os.close(fd)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            return '', ''
    # Special handling: arXiv abs pages
    try:
        from urllib.parse import urlparse
        pu = urlparse(url)
        if 'arxiv.org' in pu.netloc and ('/abs/' in pu.path or '/html/' in pu.path):
            # Normalize html route to abs
            arxiv_url = url
            if '/html/' in pu.path and '/abs/' not in pu.path:
                arxiv_id = pu.path.split('/html/')[-1]
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec), headers={
                'User-Agent': 'Mozilla/5.0 (AI-News-Agent/1.0)',
                'Accept-Language': 'en-US,en;q=0.9'
            }) as session:
                html = ''
                for attempt in range(3):
                    try:
                        async with session.get(arxiv_url, allow_redirects=True) as resp:
                            if resp.status != 200:
                                logger.warning(f"fetch_url_text: arxiv abs GET status={resp.status} attempt={attempt+1} url='{arxiv_url}'")
                            else:
                                html = await resp.text()
                                break
                    except Exception as ex:
                        logger.warning(f"fetch_url_text: arxiv abs GET exception attempt={attempt+1} url='{arxiv_url}' err={ex}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                if not html:
                    return '', ''
            soup = BeautifulSoup(html, 'html.parser')
            title = None
            mt = soup.select_one('meta[name="citation_title"]')
            if mt:
                title = mt.get('content')
            if not title:
                ht = soup.select_one('#abs .title')
                title = ht.get_text().replace('Title:', '').strip() if ht else ''
            bq = soup.select_one('blockquote.abstract')
            abstract = ''
            if bq:
                abstract = bq.get_text().replace('Abstract:', '').strip()
            logger.info(f"fetch_url_text: arxiv abs parsed title_len={len(title or '')} text_len={len(abstract or '')}")
            return title or '', abstract or ''
    except Exception:
        pass

    # arXiv library fallback: when URL is arXiv and above failed
    try:
        from urllib.parse import urlparse
        pu = urlparse(url)
        if 'arxiv.org' in pu.netloc:
            # /abs/ handler via arxiv library
            if '/abs/' in pu.path:
                import arxiv
                arxiv_id = pu.path.split('/abs/')[-1]
                search = arxiv.Search(id_list=[arxiv_id])
                results = list(search.results())
                if results:
                    paper = results[0]
                    logger.info(f"fetch_url_text: arxiv library fallback for id='{arxiv_id}'")
                    return paper.title.strip(), paper.summary.strip()
            # /pdf/ handler to read PDF content
            if '/pdf/' in pu.path:
                import io
                import requests
                from pdfminer.high_level import extract_text
                pdf_url = url
                if not pdf_url.endswith('.pdf'):
                    pdf_url = pdf_url + '.pdf'
                try:
                    resp = requests.get(pdf_url, timeout=20)
                    if resp.status_code == 200 and resp.content:
                        text = extract_text(io.BytesIO(resp.content))
                        # Heuristic title: first non-empty line
                        title_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), 'arXiv PDF')
                        logger.info("fetch_url_text: extracted text from arXiv PDF")
                        return title_line, text
                except Exception as ex:
                    logger.warning(f"fetch_url_text: arXiv PDF extraction failed: {ex}")
    except Exception:
        pass

    # Try newspaper3k first
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        title = art.title or ''
        text = art.text or ''
        if text and len(text.strip()) >= 50:
            logger.info(f"fetch_url_text: newspaper3k success title_len={len(title or '')} text_len={len(text or '')} url='{url}'")
            return title.strip(), text.strip()
    except Exception:
        pass
    # Fallback simple fetch
    # Domain-specific: Hugging Face pages via Jina Reader (handles JS)
    try:
        from urllib.parse import urlparse
        pu = urlparse(url)
        # Prefer proxy for JS-heavy or gated domains
        proxy_domains = [
            'huggingface.co',
            'dl.acm.org',
            'researchgate.net',
            'pubmed.ncbi.nlm.nih.gov',
            'journals.lww.com',
            'medium.com'
        ]
        if any(d in pu.netloc for d in proxy_domains):
            proxy_url = f"https://r.jina.ai/http://{pu.netloc}{pu.path}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec), headers={
                'User-Agent': 'Mozilla/5.0 (AI-News-Agent/1.0)'
            }) as session:
                async with session.get(proxy_url, allow_redirects=True) as resp:
                    if resp.status == 200:
                        proxy_text = await resp.text()
                        if proxy_text and len(proxy_text.strip()) > 100:
                            # Title fallback from original page
                            title = pu.path.strip('/').split('/')[-1]
                            logger.info(f"fetch_url_text: proxy via Jina length={len(proxy_text.strip())} domain='{pu.netloc}' url='{url}'")
                            return title, proxy_text.strip()
    except Exception:
        pass

    # Generic HTML fetch with robust headers
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec), headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }) as session:
            html = ''
            for attempt in range(3):
                try:
                    async with session.get(url, allow_redirects=True) as resp:
                        if resp.status != 200:
                            logger.warning(f"fetch_url_text: generic GET status={resp.status} attempt={attempt+1} url='{url}'")
                        else:
                            html = await resp.text()
                            break
                except Exception as ex:
                    logger.warning(f"fetch_url_text: generic GET exception attempt={attempt+1} url='{url}' err={ex}")
                await asyncio.sleep(0.5 * (attempt + 1))
            if not html:
                return '', ''
        soup = BeautifulSoup(html, 'html.parser')
        title = (soup.title.get_text().strip() if soup.title else '')

        # Platform-specific selectors
        text_blocks = []
        if 'github.com' in (pu.netloc if 'pu' in locals() else url):
            for node in soup.select('.markdown-body'):
                node_text = node.get_text(separator='\n').strip()
                if node_text:
                    text_blocks.append(node_text)
            # Try raw README if markdown-body was empty
            if not text_blocks:
                try:
                    parts = pu.path.strip('/').split('/') if 'pu' in locals() else []
                    if len(parts) >= 2:
                        owner, repo = parts[0], parts[1]
                        for branch in ['main', 'master']:
                            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
                            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                                async with session.get(raw_url, allow_redirects=True) as resp3:
                                    if resp3.status == 200:
                                        md = await resp3.text()
                                        if md and len(md.strip()) > 80:
                                            text_blocks.append(md.strip())
                                            break
                        if not text_blocks:
                            # Try README.rst
                            for branch in ['main', 'master']:
                                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.rst"
                                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                                    async with session.get(raw_url, allow_redirects=True) as resp4:
                                        if resp4.status == 200:
                                            rst = await resp4.text()
                                            if rst and len(rst.strip()) > 80:
                                                text_blocks.append(rst.strip())
                                                break
                except Exception:
                    pass
        elif 'huggingface.co' in (pu.netloc if 'pu' in locals() else url):
            for sel in ['article', 'main', '[data-target="markdown.body"]']:
                for node in soup.select(sel):
                    node_text = node.get_text(separator='\n').strip()
                    if node_text:
                        text_blocks.append(node_text)
        else:
            # Generic candidates
            for sel in ['article', 'main', 'div[class*="content"]', 'div[id*="content"]', 'section']:
                for node in soup.select(sel):
                    node_text = node.get_text(separator='\n').strip()
                    if node_text:
                        text_blocks.append(node_text)

        # Fallback to all paragraphs if blocks are small
        if sum(len(tb) for tb in text_blocks) < 300:
            texts = [p.get_text().strip() for p in soup.select('p')]
            text = '\n'.join([t for t in texts if t])
        else:
            # Choose the longest block as main content
            text = max(text_blocks, key=len) if text_blocks else ''

        # Readability fallback
        if len(text.strip()) < 100:
            try:
                from readability import Document
                doc = Document(html)
                readable_html = doc.summary()
                rsoup = BeautifulSoup(readable_html, 'html.parser')
                text = rsoup.get_text(separator='\n').strip()
                if not title:
                    title = doc.short_title() or title
                logger.info(f"fetch_url_text: readability length={len(text or '')} url='{url}'")
            except Exception:
                pass

        # Jina Reader fallback (generic, handles JS-heavy sites)
        if len(text.strip()) < 100:
            try:
                proxy_url = f"https://r.jina.ai/http://{url}"
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec), headers={
                    'User-Agent': 'Mozilla/5.0 (AI-News-Agent/1.0)'
                }) as session:
                    async with session.get(proxy_url, allow_redirects=True) as resp2:
                        if resp2.status == 200:
                            proxy_text = await resp2.text()
                            if proxy_text and len(proxy_text.strip()) > 100:
                                text = proxy_text.strip()
                                logger.info(f"fetch_url_text: Jina fallback length={len(text or '')} url='{url}'")
            except Exception:
                pass

        return title.strip(), text.strip()
    except Exception as ex:
        logger.error(f"fetch_url_text: exception for url='{url}' err={ex}")
        return '', ''
