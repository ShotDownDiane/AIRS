"""ArxivDownloadSkill: download paper PDFs from arXiv."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ArxivDownloadSkill(BaseSkill):
    """Downloads arXiv paper PDFs to the workspace."""

    def __init__(self, workspace=None):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "arxiv_download"

    @property
    def description(self) -> str:
        return (
            "Download a paper PDF from arXiv by its ID. "
            "The PDF will be saved to literature/pdfs/{arxiv_id}.pdf in the workspace."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "ArXiv paper ID (e.g. '2104.12518' or '2104.12518v2')",
                },
            },
            "required": ["arxiv_id"],
        }

    async def execute(self, arxiv_id: str) -> str:
        if self._workspace is None:
            return "Error: ArxivDownloadSkill not connected to workspace"

        try:
            import httpx

            # Normalize ID
            arxiv_id = arxiv_id.strip().split("/")[-1]  # Handle full URLs
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # Ensure directory exists
            pdf_dir = self._workspace._safe_path("literature/pdfs")
            pdf_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = pdf_dir / f"{arxiv_id}.pdf"

            if pdf_path.exists():
                return f"Already downloaded: literature/pdfs/{arxiv_id}.pdf"

            logger.info("Downloading arXiv PDF: %s", arxiv_id)
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                resp = await client.get(pdf_url)
                if resp.status_code == 200:
                    pdf_path.write_bytes(resp.content)
                    size_kb = len(resp.content) / 1024
                    return f"Downloaded: literature/pdfs/{arxiv_id}.pdf ({size_kb:.0f} KB)"
                else:
                    return f"Download failed: HTTP {resp.status_code} for {pdf_url}"

        except Exception as e:
            return f"Download error: {e}"
