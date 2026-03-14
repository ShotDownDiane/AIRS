"""LLM abstraction: base types shared by all providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool_result"]
    content: str
    tool_call_id: str | None = None  # For tool_result messages
    tool_calls: list["ToolCall"] | None = None  # For assistant messages with tool calls


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON Schema object


class BaseProvider(ABC):
    """Abstract LLM provider. All agents talk through this interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Send messages and return a response (possibly with tool calls)."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


def create_provider(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
) -> BaseProvider:
    """Factory: return the appropriate provider instance."""
    if provider == "claude":
        from airs.llm.claude import ClaudeProvider
        return ClaudeProvider(model=model, api_key=api_key)
    elif provider == "openai":
        from airs.llm.openai import OpenAIProvider
        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        from airs.llm.gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'claude', 'openai', or 'gemini'.")
