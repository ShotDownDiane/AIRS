"""WebSearchSkill: search the web using DuckDuckGo (or SerpAPI if key provided)."""

from __future__ import annotations

import logging
import os

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class WebSearchSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Returns top search results with titles, "
            "URLs, and snippets as Markdown."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 8)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = 8) -> str:
        max_results = min(int(max_results), 20)
        logger.info("Web search: %r (max=%d)", query, max_results)

        serpapi_key = os.getenv("SERPAPI_KEY", "")
        if serpapi_key:
            return await self._serpapi_search(query, max_results, serpapi_key)
        else:
            return await self._ddg_search(query, max_results)

    async def _ddg_search(self, query: str, max_results: int) -> str:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        f"### {r.get('title', 'No title')}\n"
                        f"- **URL**: {r.get('href', '')}\n"
                        f"- **Snippet**: {r.get('body', '')}\n"
                    )
            if not results:
                return f"No results found for: {query!r}"
            return f"# Web Search Results for: {query!r}\n\n" + "\n---\n".join(results)
        except Exception as e:
            return f"Web search error: {e}"

    async def _serpapi_search(self, query: str, max_results: int, api_key: str) -> str:
        try:
            import httpx
            params = {
                "q": query,
                "num": max_results,
                "api_key": api_key,
                "engine": "google",
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://serpapi.com/search", params=params, timeout=15)
                data = resp.json()

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(
                    f"### {item.get('title', 'No title')}\n"
                    f"- **URL**: {item.get('link', '')}\n"
                    f"- **Snippet**: {item.get('snippet', '')}\n"
                )
            if not results:
                return f"No results found for: {query!r}"
            return f"# Web Search Results for: {query!r}\n\n" + "\n---\n".join(results)
        except Exception as e:
            return f"SerpAPI search error: {e}"
