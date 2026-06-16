import asyncio
import random
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from duckduckgo_search import DDGS
import httpx
from pydantic import BaseModel

app = FastAPI(title="Gogoduck Web Search API for LLMs")

# Configurações e Headers
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
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all")
                if response.status_code == 200:
                    self.proxies = [p.strip() for p in response.text.split("\n") if p.strip()]
                    print(f"Atualizado: {len(self.proxies)} proxies carregados.")
        except Exception as e:
            print(f"Erro ao carregar proxies: {e}")

    def get_random_proxy(self):
        if not self.proxies:
            return None
        proxy = random.choice(self.proxies)
        return f"http://{proxy}"

proxy_manager = ProxyManager()

@app.on_event("startup")
async def startup_event():
    # Carrega proxies em background
    asyncio.create_task(proxy_manager.refresh_proxies())

async def fetch_ddg_results(query: str, max_results: int = 5) -> List[SearchResult]:
    """Busca no DuckDuckGo com rotação de proxy e User-Agent"""
    proxy_url = proxy_manager.get_random_proxy()
    
    try:
        # Tentativa com proxy
        with DDGS(proxy=proxy_url, timeout=10) as ddgs:
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
        print(f"Erro com proxy {proxy_url}: {e}. Tentando sem proxy...")
    
    # Fallback sem proxy
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    body=r.get("body", ""),
                    source="duckduckgo"
                ) for r in results
            ]
    except Exception as e:
        print(f"Erro total no DDG: {e}")
        return []

@app.get("/search", response_model=List[SearchResult])
async def search(
    q: str = Query(..., description="A consulta de busca"),
    max_results: int = Query(5, ge=1, le=10)
):
    """
    Endpoint principal de busca. Retorna resultados formatados para LLMs.
    Otimizado para ser consumido como ferramenta (Tool Calling) pelo SambaNova.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query não pode ser vazia")
    
    results = await fetch_ddg_results(q, max_results)
    
    if not results:
        # Retorna lista vazia em vez de 500 para não quebrar o LLM
        return []
    
    return results

@app.get("/health")
async def health():
    return {"status": "online", "engine": "gogoduck", "proxies_available": len(proxy_manager.proxies)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
