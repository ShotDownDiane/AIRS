"""IdeationAgent: generates novel research ideas from the literature survey."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class IdeationAgent(BaseAgent):
    """Generates and evaluates novel research ideas using OpenAI o3."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
