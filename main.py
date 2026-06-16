import asyncio
import random
import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
import httpx
from pydantic import BaseModel
import urllib.parse

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gogoduck")

app = FastAPI(
    title="Gogoduck Web Search API for LLMs",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                # Usando múltiplas fontes de proxy para maior variedade
                response = await client.get("https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all")
                if response.status_code == 200:
                    self.proxies = [p.strip() for p in response.text.split("\n") if p.strip()]
                    logger.info(f"Sistema: {len(self.proxies)} proxies carregados.")
        except Exception as e:
            logger.error(f"Erro ao carregar proxies: {e}")

    def get_random_proxy(self):
        if not self.proxies:
            return None
        return f"http://{random.choice(self.proxies)}"

proxy_manager = ProxyManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(proxy_manager.refresh_proxies())

async def fetch_ddg_results(query: str, max_results: int = 5) -> List[SearchResult]:
    """Busca no DuckDuckGo com múltiplas tentativas e estratégias"""
    
    # Limpa a query (remove excesso de espaços e garante decodificação)
    query = urllib.parse.unquote(query).strip()
    
    strategies = [
        {"use_proxy": True, "timeout": 10},
        {"use_proxy": False, "timeout": 15},
        {"use_proxy": True, "timeout": 15}
    ]
    
    for strategy in strategies:
        proxy_url = proxy_manager.get_random_proxy() if strategy["use_proxy"] else None
        try:
            # O DDGS agora recomenda o uso de contextos ou instâncias limpas
            with DDGS(proxy=proxy_url, timeout=strategy["timeout"]) as ddgs:
                # Tentamos usar o backend 'text' que é o mais estável
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            body=r.get("body", ""),
                            source="duckduckgo"
                        ) for r in results
                    ]
        except Exception as e:
            logger.warning(f"Estratégia (proxy={strategy['use_proxy']}) falhou para '{query}': {e}")
            continue
            
    return []

@app.get("/search")
@app.get("/search/")
async def search(
    q: str = Query(..., description="A consulta de busca"),
    max_results: int = Query(5, ge=1, le=10)
):
    # Log para depuração no Render
    logger.info(f"Requisição de busca: {q}")
    
    results = await fetch_ddg_results(q, max_results)
    
    # Se ainda estiver vazio, tentamos uma busca mais genérica (fallback)
    if not results and len(q.split()) > 3:
        logger.info(f"Tentando busca simplificada para: {q}")
        simplified_q = " ".join(q.split()[:5]) # Pega apenas as primeiras 5 palavras
        results = await fetch_ddg_results(simplified_q, max_results)
        
    return results

@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "online", 
        "engine": "gogoduck", 
        "proxies": len(proxy_manager.proxies),
        "usage": "/search?q=seu%20termo%20aqui"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
