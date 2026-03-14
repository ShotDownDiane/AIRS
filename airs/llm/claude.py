"""Claude provider using the Anthropic SDK."""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from airs.llm.base import BaseProvider, LLMResponse, Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str, api_key: str = ""):
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        # Separate system message from the rest
        system = ""
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "tool_result":
                # Must follow an assistant message with tool_use
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        # Build tools list
        api_tools: list[dict] | None = None
        if tools:
            api_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        logger.debug("Claude request: model=%s, msgs=%d", self._model, len(api_messages))
        response = await self._client.messages.create(**kwargs)
        logger.debug("Claude response: stop_reason=%s", response.stop_reason)

        # Parse response
        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        finish = "tool_calls" if tool_calls else response.stop_reason or "stop"

        return LLMResponse(
            content=text_content or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
