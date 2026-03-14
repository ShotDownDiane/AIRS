"""AnalystAgent: analyzes experimental results using Gemini."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class AnalystAgent(BaseAgent):
    """Performs statistical analysis and generates visualizations."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
