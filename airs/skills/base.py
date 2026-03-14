"""BaseSkill: abstract base for all agent tools/skills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from airs.llm.base import ToolDefinition


class BaseSkill(ABC):
    """Abstract base class for all skills that agents can use."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill name (used as tool function name)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for skill parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the skill and return a string result."""
        ...

    def to_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )
