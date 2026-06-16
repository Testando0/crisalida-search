import asyncio
import random
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
import httpx
from pydantic import BaseModel

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gogoduck")

app = FastAPI(
    title="Gogoduck Web Search API for LLMs",
    docs_url="/docs",
    redoc_url=None
)

# Habilitar CORS para permitir chamadas de qualquer origem (útil para LLMs e integrações)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
]

class SearchResult(BaseModel):
    title: str
    url: str
    body: str
    source: str = "duckduckgo"

class ProxyManager:
    def __init__(self):
        self.proxies = []

    async def refresh_proxies(self):
        """Busca proxies gratuitos de fontes públicas"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get("https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all")
                if response.status_code == 200:
                    self.proxies = [p.strip() for p in response.text.split("\n") if p.strip()]
                    logger.info(f"Sistema: {len(self.proxies)} proxies carregados.")
        except Exception as e:
            logger.error(f"Erro ao carregar proxies: {e}")

    def get_random_proxy(self):
        if not self.proxies:
            return None
        proxy = random.choice(self.proxies)
        return f"http://{proxy}"

proxy_manager = ProxyManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(proxy_manager.refresh_proxies())

async def fetch_ddg_results(query: str, max_results: int = 5) -> List[SearchResult]:
    """Busca no DuckDuckGo com rotação de proxy e User-Agent"""
    proxy_url = proxy_manager.get_random_proxy()
    
    # Tenta com proxy primeiro
    if proxy_url:
        try:
            with DDGS(proxy=proxy_url, timeout=15) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    return [SearchResult(title=r.get("title", ""), url=r.get("href", ""), body=r.get("body", ""), source="duckduckgo") for r in results]
        except Exception as e:
            logger.warning(f"Falha com proxy {proxy_url}. Tentando direto...")

    # Fallback direto (sem proxy)
    try:
        with DDGS(timeout=15) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [SearchResult(title=r.get("title", ""), url=r.get("href", ""), body=r.get("body", ""), source="duckduckgo") for r in results]
    except Exception as e:
        logger.error(f"Erro total no DDG: {e}")
        return []

# Rota de busca com barra opcional para evitar 404
@app.get("/search")
@app.get("/search/")
async def search(
    q: str = Query(..., description="A consulta de busca"),
    max_results: int = Query(5, ge=1, le=10)
):
    logger.info(f"Busca recebida: {q}")
    results = await fetch_ddg_results(q, max_results)
    return results

@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "online", 
        "engine": "gogoduck", 
        "proxies_available": len(proxy_manager.proxies),
        "endpoints": ["/search?q=query", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    # O Render usa a variável de ambiente PORT
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
