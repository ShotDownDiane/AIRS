"""ReviewerAgent: performs peer review of the paper using GPT-4o."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class ReviewerAgent(BaseAgent):
    """Provides critical peer review of the paper draft."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
