"""WriterAgent: writes the research paper in Markdown and LaTeX."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class WriterAgent(BaseAgent):
    """Writes a complete academic paper with LaTeX compilation."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
