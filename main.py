"""
Crisalida Search API v2
━━━━━━━━━━━━━━━━━━━━━━
Zero-key · Never-fails · Rich JSON
Engine cascade: SearXNG (12 instances, parallel) → DDG HTML → Wikipedia
Optimised for Render free tier
"""

import asyncio
import hashlib
import logging
import os
import random
import re
import time
import urllib.parse
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("crisalida")

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Crisalida Search API",
    description=(
        "Zero-key web search API. "
        "Engine cascade: SearXNG → DuckDuckGo HTML → Wikipedia. "
        "Never returns an error — always best-effort results."
    ),
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class Result(BaseModel):
    position: int
    title: str
    url: str
    body: str
    source: str  # which engine/instance returned this result


class SearchResponse(BaseModel):
    query: str
    results: List[Result]
    total: int
    engine: str          # engine that succeeded
    cached: bool
    took_ms: int
    timestamp: str


# ── In-memory TTL cache ───────────────────────────────────────────────────────

class _Cache:
    """LRU-ish in-memory cache with TTL. Survives between requests on Render."""

    def __init__(self, maxsize: int = 300, ttl: int = 600):
        self._d: dict = {}
        self.maxsize = maxsize
        self.ttl = ttl  # seconds

    @staticmethod
    def _k(q: str, n: int) -> str:
        return hashlib.md5(f"{q.strip().lower()}|{n}".encode()).hexdigest()

    def get(self, q: str, n: int):
        k = self._k(q, n)
        if k in self._d:
            v, ts = self._d[k]
            if time.monotonic() - ts < self.ttl:
                return v
            del self._d[k]
        return None

    def put(self, q: str, n: int, v):
        if len(self._d) >= self.maxsize:
            # evict oldest
            del self._d[min(self._d, key=lambda x: self._d[x][1])]
        self._d[self._k(q, n)] = (v, time.monotonic())


_cache = _Cache()

# ── Constants ─────────────────────────────────────────────────────────────────

# Realistic browser headers — reduces bot detection on all engines
_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "DNT": "1",
    "Connection": "keep-alive",
}

# Public SearXNG instances — community-run meta-search (DDG+Google+Bing+Brave aggregated)
# No API key required. We try up to 6 in parallel and return the first hit.
SEARXNG_INSTANCES: list[str] = [
    "https://searx.be",
    "https://search.disroot.org",
    "https://searxng.site",
    "https://searx.tiekoetter.com",
    "https://searx.prvcy.eu",
    "https://search.mdosch.de",
    "https://searx.fmac.xyz",
    "https://etsi.me",
    "https://search.inetol.net",
    "https://search.hbubli.cc",
    "https://paulgo.io",
    "https://search.us.projectsegfau.lt",
]

# ── Engine 1: SearXNG — parallel race ─────────────────────────────────────────

async def _one_searxng(
    client: httpx.AsyncClient,
    base: str,
    query: str,
    n: int,
) -> Optional[List[Result]]:
    """Try a single SearXNG instance via its JSON API."""
    try:
        r = await client.get(
            f"{base}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "auto",
                "safesearch": "0",
                "pageno": "1",
            },
            headers={**_HDR, "Accept": "application/json"},
            timeout=10.0,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return None

        data = r.json()
        raw = data.get("results") or []
        if not raw:
            return None

        host = base.split("//")[-1].rstrip("/")
        return [
            Result(
                position=i + 1,
                title=(x.get("title") or "").strip(),
                url=(x.get("url") or ""),
                body=(x.get("content") or x.get("description") or "").strip(),
                source=f"searxng:{host}",
            )
            for i, x in enumerate(raw[:n])
            if x.get("url")
        ]
    except Exception as exc:
        log.debug("searxng %s → %s: %s", base, type(exc).__name__, exc)
        return None


async def search_searxng(query: str, n: int) -> Optional[List[Result]]:
    """
    Race up to 6 SearXNG instances concurrently.
    Returns the first non-empty result; cancels the rest.
    """
    pool = random.sample(SEARXNG_INSTANCES, min(len(SEARXNG_INSTANCES), 6))

    async with httpx.AsyncClient() as client:
        tasks = {
            asyncio.create_task(_one_searxng(client, base, query, n)): base
            for base in pool
        }
        pending = set(tasks)
        result: Optional[List[Result]] = None

        try:
            while pending and result is None:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    try:
                        r = t.result()
                        if r:
                            result = r
                    except Exception:
                        pass
        finally:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    return result


# ── Engine 2: DuckDuckGo HTML ─────────────────────────────────────────────────

class _DDGParser(HTMLParser):
    """
    Minimal parser for DuckDuckGo Lite HTML output.
    html.duckduckgo.com/html/ is far more permissive than the JSON API
    — it's the same endpoint used by curl-friendly DDG wrappers.
    """

    def __init__(self):
        super().__init__()
        self.results: list = []
        self._cur: dict = {}
        self._cap: Optional[str] = None  # "title" | "body"

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._cur = {"url": a.get("href", ""), "title": "", "body": ""}
            self._cap = "title"
        elif tag == "a" and "result__snippet" in cls:
            self._cap = "body"

    def handle_endtag(self, tag):
        if tag == "a":
            if (
                self._cap == "body"
                and self._cur.get("title")
                and self._cur.get("url")
            ):
                self.results.append(dict(self._cur))
                self._cur = {}
            self._cap = None

    def handle_data(self, data):
        data = data.strip()
        if data and self._cap and self._cur is not None:
            prev = self._cur.get(self._cap, "")
            self._cur[self._cap] = (prev + " " + data).strip()


async def search_ddg_html(query: str, n: int) -> Optional[List[Result]]:
    """Scrape DuckDuckGo Lite HTML. No API key, no JS required."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "br-pt"},
                headers={
                    **_HDR,
                    "Accept": "text/html",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        if r.status_code != 200:
            return None

        p = _DDGParser()
        p.feed(r.text)

        if not p.results:
            return None

        return [
            Result(
                position=i + 1,
                title=x["title"],
                url=x["url"],
                body=x["body"],
                source="duckduckgo-html",
            )
            for i, x in enumerate(p.results[:n])
            if x.get("url")
        ]
    except Exception as exc:
        log.warning("ddg-html: %s: %s", type(exc).__name__, exc)
        return None


# ── Engine 3: Wikipedia API ───────────────────────────────────────────────────

async def search_wikipedia(query: str, n: int) -> Optional[List[Result]]:
    """
    Wikipedia MediaWiki API — always available, zero key, zero rate limit
    for reasonable usage. Searches English Wikipedia (broadest index).
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": n,
                    "srprop": "snippet|sectiontitle",
                    "format": "json",
                    "utf8": 1,
                },
                headers=_HDR,
            )
        if r.status_code != 200:
            return None

        items = r.json().get("query", {}).get("search") or []
        if not items:
            return None

        return [
            Result(
                position=i + 1,
                title=x["title"],
                url="https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(x["title"].replace(" ", "_")),
                body=re.sub(r"<[^>]+>", "", x.get("snippet", "")),
                source="wikipedia",
            )
            for i, x in enumerate(items)
        ]
    except Exception as exc:
        log.warning("wikipedia: %s: %s", type(exc).__name__, exc)
        return None


# ── Orchestration ─────────────────────────────────────────────────────────────

async def _do_search(query: str, n: int) -> Tuple[List[Result], str]:
    """
    Waterfall: SearXNG → DDG HTML → Wikipedia.
    Never raises. Always returns (results, engine_name).
    """
    results = await search_searxng(query, n)
    if results:
        return results, "searxng"

    log.info("SearXNG miss for '%s' — trying DDG HTML", query)
    results = await search_ddg_html(query, n)
    if results:
        return results, "duckduckgo-html"

    log.info("DDG HTML miss for '%s' — trying Wikipedia", query)
    results = await search_wikipedia(query, n)
    if results:
        return results, "wikipedia"

    log.warning("All engines failed for '%s'", query)
    return [], "none"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse, summary="Web search")
@app.get("/search/", response_model=SearchResponse, include_in_schema=False)
async def search_endpoint(
    q: str = Query(..., description="Search query (URL-encoded accepted)"),
    max_results: int = Query(5, ge=1, le=20, description="Number of results (1–20)"),
    nocache: bool = Query(False, description="Skip cache and force a fresh search"),
):
    """
    Search the web. Returns rich JSON — never returns a 500 error.

    **Engine waterfall:**
    1. SearXNG public instances (up to 6 raced in parallel)
    2. DuckDuckGo HTML scrape
    3. Wikipedia full-text search

    Results are cached 10 minutes in-process to reduce external load.
    """
    query = urllib.parse.unquote(q).strip()

    if not query:
        return SearchResponse(
            query=query,
            results=[],
            total=0,
            engine="none",
            cached=False,
            took_ms=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    t0 = time.monotonic()

    # ── Cache check ───────────────────────────────────────────────────────────
    if not nocache:
        hit = _cache.get(query, max_results)
        if hit:
            log.info("CACHE HIT '%s'", query)
            return SearchResponse(
                **{
                    **hit,
                    "cached": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    # ── Live search ───────────────────────────────────────────────────────────
    results, engine = await _do_search(query, max_results)
    took_ms = int((time.monotonic() - t0) * 1000)

    log.info(
        "SEARCH '%s' → %d results via %s in %dms",
        query,
        len(results),
        engine,
        took_ms,
    )

    payload = dict(
        query=query,
        results=[r.model_dump() for r in results],
        total=len(results),
        engine=engine,
        cached=False,
        took_ms=took_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if results and not nocache:
        _cache.put(query, max_results, payload)

    return SearchResponse(**payload)


@app.get("/", summary="Health check")
@app.get("/health", summary="Health check", include_in_schema=False)
async def health():
    """Returns API status and usage info."""
    return {
        "status": "online",
        "version": "2.0.0",
        "engines": ["searxng", "duckduckgo-html", "wikipedia"],
        "searxng_pool_size": len(SEARXNG_INSTANCES),
        "cache_entries": len(_cache._d),
        "note": (
            "On Render free tier the service sleeps after 15 min of inactivity. "
            "First request after wake may take ~5 s."
        ),
        "endpoints": {
            "search": "/search?q=meta+ia+latest+model",
            "more_results": "/search?q=...&max_results=10",
            "no_cache": "/search?q=...&nocache=true",
            "docs": "/docs",
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info",
    )
