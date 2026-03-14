"""DesignAgent: creates detailed technical design from the selected idea."""

from __future__ import annotations

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager


class DesignAgent(BaseAgent):
    """Creates implementation-ready technical design documents."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)
