"""ArxivSkill: search arXiv for academic papers."""

from __future__ import annotations

import logging
from typing import Any

import arxiv

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ArxivSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return (
            "Search arXiv for academic papers. Returns a Markdown list of papers "
            "with titles, authors, abstracts, and arxiv IDs."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (e.g. 'linear attention transformers')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10, max 30)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = 10) -> str:
        max_results = min(int(max_results), 30)
        logger.info("ArXiv search: %r (max=%d)", query, max_results)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results: list[str] = []
        try:
            for paper in client.results(search):
                authors = ", ".join(str(a) for a in paper.authors[:5])
                if len(paper.authors) > 5:
                    authors += " et al."
                entry = (
                    f"## {paper.title}\n"
                    f"- **ArXiv ID**: {paper.entry_id.split('/')[-1]}\n"
                    f"- **Authors**: {authors}\n"
                    f"- **Published**: {paper.published.date()}\n"
                    f"- **URL**: {paper.entry_id}\n"
                    f"- **Abstract**: {paper.summary[:600].strip()}...\n"
                )
                results.append(entry)
        except Exception as e:
            return f"ArXiv search error: {e}"

        if not results:
            return f"No papers found for query: {query!r}"

        return f"# ArXiv Search Results for: {query!r}\n\n" + "\n---\n".join(results)
