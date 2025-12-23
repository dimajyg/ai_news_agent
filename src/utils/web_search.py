import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urlencode, urlparse, parse_qs

async def search(query: str, max_results: int = 5, timeout_sec: int = 20) -> List[Dict[str, str]]:
    url = 'https://html.duckduckgo.com/html/?' + urlencode({'q': query})
    headers = {
        'User-Agent': 'Mozilla/5.0 (AI-News-Agent/1.0)'
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec), headers=headers) as session:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()
    soup = BeautifulSoup(html, 'html.parser')
    results: List[Dict[str, str]] = []
    for a in soup.select('a.result__a')[:max_results * 2]:
        title = a.get_text().strip()
        href = a.get('href', '')
        if 'uddg=' in href:
            qs = parse_qs(urlparse(href).query)
            real = qs.get('uddg', [''])[0]
            href = real or href
        if href and title:
            results.append({'title': title, 'url': href})
        if len(results) >= max_results:
            break
    return results
