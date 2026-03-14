"""Agent factory: maps stage names to agent classes."""

from __future__ import annotations

from airs.agents.analyst import AnalystAgent
from airs.agents.base import BaseAgent
from airs.agents.coder import CoderAgent
from airs.agents.design import DesignAgent
from airs.agents.experimenter import ExperimenterAgent
from airs.agents.ideation import IdeationAgent
from airs.agents.literature import LiteratureAgent
from airs.agents.reviewer import ReviewerAgent
from airs.agents.writer import WriterAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager

_AGENT_CLASS_MAP: dict[str, type[BaseAgent]] = {
    "literature": LiteratureAgent,
    "ideation": IdeationAgent,
    "design": DesignAgent,
    "coder": CoderAgent,
    "experimenter": ExperimenterAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "reviewer": ReviewerAgent,
}


def build_agent(
    agent_name: str,
    config: AgentConfig,
    workspace: WorkspaceManager,
    ssh_client=None,
    cost_tracker=None,
) -> BaseAgent:
    """Instantiate the right agent class for the given name."""
    cls = _AGENT_CLASS_MAP.get(agent_name, BaseAgent)
    return cls(config=config, workspace=workspace, ssh_client=ssh_client, cost_tracker=cost_tracker)
