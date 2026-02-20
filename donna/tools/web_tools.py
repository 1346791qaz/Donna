"""
tools/web_tools.py — Web search and fetch capabilities.

Allows Donna to search for information online and retrieve web page content
for open source projects, documentation, etc.
"""

import logging
import re
from typing import Any

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

logger = logging.getLogger(__name__)

# User-Agent to avoid being blocked by websites
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the web for information and return top results with links and summaries.

    Uses DuckDuckGo's instant answer API which doesn't require authentication.

    Args:
        query: Search query string
        max_results: Maximum results to return (default 5)

    Returns:
        Dict with 'results' list containing title, url, snippet for each result,
        or error dict if search fails.
    """
    if not requests:
        return {"error": "requests library not installed"}

    try:
        # Use DuckDuckGo's search endpoint
        url = "https://duckduckgo.com/html"
        params = {"q": query}
        headers = {"User-Agent": _USER_AGENT}

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        results = []

        # Parse DuckDuckGo results
        for result in soup.select(".result"):
            try:
                title_elem = result.select_one(".result__title")
                url_elem = result.select_one(".result__url")
                snippet_elem = result.select_one(".result__snippet")

                if title_elem and url_elem:
                    title = title_elem.get_text(strip=True)
                    url = url_elem.get("href", "")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet[:200],  # Limit snippet length
                    })

                    if len(results) >= max_results:
                        break
            except Exception:
                continue

        if not results:
            return {"error": f"No results found for '{query}'"}

        logger.info("Web search for %r returned %d results", query, len(results))
        return {"query": query, "results": results}

    except Exception as e:
        logger.exception("Web search failed for query %r", query)
        return {"error": f"Search failed: {str(e)}"}


def fetch_url(url: str, max_length: int = 2000) -> dict[str, Any]:
    """
    Fetch and summarise a web page.

    Extracts main content, removes scripts/styles, and returns a clean text summary.

    Args:
        url: URL to fetch
        max_length: Maximum length of returned text (chars)

    Returns:
        Dict with 'title', 'content', and 'url', or error dict.
    """
    if not requests or not BeautifulSoup:
        return {"error": "requests or beautifulsoup4 library not installed"}

    try:
        headers = {"User-Agent": _USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find main content
        title = soup.title.string if soup.title else "Unknown"

        # Prefer article/main content if available
        content_elem = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"class": re.compile(r"(content|article|main)", re.I)})
            or soup.find("body")
        )

        if not content_elem:
            content_elem = soup

        text = content_elem.get_text(separator="\n", strip=True)

        # Clean up excess whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines[:50])  # First 50 lines

        # Truncate to max_length
        if len(content) > max_length:
            content = content[:max_length] + "…"

        logger.info("Fetched URL %r: %d chars", url, len(content))
        return {
            "url": url,
            "title": title,
            "content": content,
        }

    except Exception as e:
        logger.exception("Failed to fetch URL %r", url)
        return {"error": f"Failed to fetch URL: {str(e)}"}


def search_opensource(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search for open source projects, libraries, and documentation.

    Focuses on GitHub, PyPI, and documentation sites.

    Args:
        query: Search query (e.g., "Python JSON library", "React tutorial")
        max_results: Maximum results to return

    Returns:
        Dict with results focused on open source resources.
    """
    # Enhance query to focus on open source results
    enhanced_query = f"{query} site:github.com OR site:pypi.org OR site:docs.python.org OR site:npmjs.com OR site:crates.io"
    return web_search(enhanced_query, max_results=max_results)
