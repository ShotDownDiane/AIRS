"""LiteratureAgent: surveys academic literature for the research topic."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class LiteratureAgent(BaseAgent):
    """Surveys arXiv and the web to produce a comprehensive literature review."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
