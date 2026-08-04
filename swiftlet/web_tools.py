"""
Privacy-safe web tools for Swiftlet.

Gives the local LLM awareness of current time and web search results
WITHOUT leaking any user data. The architecture:

  1. User sends a message to Swiftlet proxy
  2. Proxy (this module) detects if the query needs current info
  3. Proxy searches DuckDuckGo (privacy-first, no tracking, no API key)
  4. Search results are injected as context into the system prompt
  5. The LLM processes everything locally — it never touches the internet

Data flow:
  User → Swiftlet Proxy → [DuckDuckGo search] → inject context → local LLM
  
Privacy guarantees:
  - Only the search query goes to DuckDuckGo (not the full conversation)
  - DuckDuckGo does not track users or store searches
  - The LLM model itself has zero network access
  - No user data, API keys, or identifiers are sent
"""

import re
import time
from datetime import datetime, timezone
from html import unescape
from urllib.parse import quote_plus

import httpx

from .logging_config import get_logger

_log = get_logger("web_tools")

# ── Time & Date ──────────────────────────────────────────────────────

def get_current_context() -> str:
    """Return a formatted string with current date, time, and timezone."""
    now = datetime.now()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current time: {now.strftime('%I:%M %p')} (local) / "
        f"{utc_now.strftime('%H:%M')} UTC\n"
        f"Timezone: {time.tzname[0]}"
    )


# ── Query Analysis ───────────────────────────────────────────────────

# Keywords that suggest the user wants current/live information
_CURRENT_KEYWORDS = re.compile(
    r'\b('
    r'today|tonight|yesterday|tomorrow|this week|this month|this year|'
    r'current|currently|latest|recent|recently|right now|'
    r'news|headline|breaking|update|'
    r'weather|forecast|temperature|'
    r'stock|price|market|crypto|bitcoin|'
    r'score|match|game|playing|'
    r'election|poll|vote|'
    r'who is the|who won|what happened|what is happening|'
    r'when is|when does|when will|'
    r'what time|what date|what day|what month|what year|'
    r'how much is|how many|'
    r'release date|launched|announced|'
    r'trending|viral|popular right now'
    r')\b',
    re.IGNORECASE
)

# Queries that are purely about the model itself — no search needed
_SELF_REFERENCE = re.compile(
    r'\b(what are you|who are you|your name|you a|tell me about yourself)\b',
    re.IGNORECASE
)


def needs_web_search(text: str) -> bool:
    """
    Heuristic: does this user message likely need current information?
    Returns True if the text contains keywords suggesting real-time data.
    """
    if _SELF_REFERENCE.search(text):
        return False
    return bool(_CURRENT_KEYWORDS.search(text))


def extract_search_query(text: str) -> str:
    """
    Extract a clean search query from the user message.
    Strips conversational fluff to get a focused search term.
    """
    # Remove common conversational prefixes
    cleaned = re.sub(
        r'^(hey|hi|hello|can you|could you|please|tell me|what is|what are|'
        r'who is|how is|search for|look up|find|google)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()
    
    # Limit to reasonable search length
    if len(cleaned) > 150:
        cleaned = cleaned[:150]
    
    return cleaned or text[:100]


# ── DuckDuckGo Search (Privacy-First) ───────────────────────────────

def search_duckduckgo(query: str, max_results: int = 3) -> list[dict]:
    """
    Search DuckDuckGo's HTML lite version. Zero tracking, no API key.
    
    Returns a list of dicts: [{"title": ..., "snippet": ..., "url": ...}]
    
    Privacy: Only the search query is sent. DuckDuckGo doesn't track users.
    No cookies, no session IDs, no user identifiers.
    """
    results = []
    
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        resp = httpx.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
            },
            timeout=5.0,
            follow_redirects=True,
        )
        
        if resp.status_code != 200:
            _log.warning(f"DuckDuckGo returned {resp.status_code}")
            return results
        
        html = resp.text
        
        # Parse results from DDG HTML lite format
        # Each result is in a <div class="result..."> block
        result_blocks = html.split('class="result ')[1:]
        
        for block in result_blocks[:max_results]:
            title_match = re.search(
                r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            url_match = re.search(
                r'class="result__url"[^>]*href="([^"]*)"', block
            )
            if not url_match:
                url_match = re.search(
                    r'class="result__a"[^>]*href="([^"]*)"', block
                )
            
            title = _strip_html(title_match.group(1)) if title_match else ""
            snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
            link = url_match.group(1) if url_match else ""
            
            if title or snippet:
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": link,
                })
        
        _log.info(f"[web] DuckDuckGo returned {len(results)} results for: {query[:60]}")
        
    except httpx.TimeoutException:
        _log.warning("[web] DuckDuckGo search timed out")
    except Exception as e:
        _log.warning(f"[web] Search error: {e}")
    
    return results


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    return text.strip()


# ── Context Builder ──────────────────────────────────────────────────

def build_web_context(user_message: str, enable_search: bool = True) -> str | None:
    """
    Build a context block to inject into the conversation.
    
    Always includes current date/time.
    Optionally searches the web if the query needs current info.
    
    Returns a formatted context string, or None if nothing to inject.
    """
    parts = []
    
    # Always include current time
    parts.append(get_current_context())
    
    # Conditionally search the web
    if enable_search and needs_web_search(user_message):
        query = extract_search_query(user_message)
        results = search_duckduckgo(query)
        
        if results:
            parts.append("\n--- Web Search Results ---")
            for i, r in enumerate(results, 1):
                parts.append(f"\n[{i}] {r['title']}")
                if r['snippet']:
                    parts.append(f"    {r['snippet']}")
                if r['url']:
                    parts.append(f"    Source: {r['url']}")
    
    if not parts:
        return None
    
    return (
        "The following is real-time context provided by the Swiftlet proxy. "
        "Use this information to answer the user's question accurately.\n\n"
        + "\n".join(parts)
    )
