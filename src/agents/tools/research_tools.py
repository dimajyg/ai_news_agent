from typing import List, Dict
import json
import asyncio
from langchain.tools import tool

from ...utils.web_search import search as ddg_search
from ...utils.web_content import fetch_url_text
from ...utils.logging import AgentLogger

logger = AgentLogger(__name__)

TOOL_TIMEOUT_SECONDS = 10

@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return a JSON string with a list of {title,url}."""
    logger.info(f"ToolCall: web_search query='{query}' max_results={max_results}")
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            results: List[Dict[str, str]] = await ddg_search(query, max_results=max_results)
            logger.info(f"ToolResult: web_search results_count={len(results)}")
            return json.dumps(results, ensure_ascii=False)
    except asyncio.TimeoutError:
        logger.warning(f"ToolTimeout: web_search timed out after {TOOL_TIMEOUT_SECONDS}s query='{query}'")
        return json.dumps([], ensure_ascii=False)
    except Exception as e:
        logger.error(f"ToolError: web_search failed query='{query}' error={type(e).__name__}: {str(e)}")
        return json.dumps([], ensure_ascii=False)

@tool
async def fetch_web_page_content(url: str) -> str:
    """Fetch a web page or PDF and return JSON string with {title,text}."""
    logger.info(f"ToolCall: fetch_web_page_content url='{url}'")
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            title, text = await fetch_url_text(url)
            logger.info(f"ToolResult: fetch_web_page_content title_len={len(title or '')} text_len={len(text or '')}")
            return json.dumps({"title": title, "text": text}, ensure_ascii=False)
    except asyncio.TimeoutError:
        logger.warning(f"ToolTimeout: fetch_web_page_content timed out after {TOOL_TIMEOUT_SECONDS}s url='{url}'")
        return json.dumps({"title": "", "text": ""}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"ToolError: fetch_web_page_content failed url='{url}' error={type(e).__name__}: {str(e)}")
        return json.dumps({"title": "", "text": ""}, ensure_ascii=False)

tools = [web_search, fetch_web_page_content]
