"""OpenAI provider."""

from __future__ import annotations

import json
import logging

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from airs.llm.base import BaseProvider, LLMResponse, Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider (also works with o3, gpt-4o, etc.)."""

    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        self._model = model
        kwargs: dict = {"api_key": api_key or None}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
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
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == "tool_result":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls — must use proper format
                api_msg: dict = {"role": "assistant"}
                if msg.content:
                    api_msg["content"] = msg.content
                else:
                    api_msg["content"] = None
                api_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                api_messages.append(api_msg)
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        api_tools = None
        if tools:
            api_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        # o3 and some models don't support temperature
        is_reasoning = self._model.startswith("o")
        kwargs: dict = dict(
            model=self._model,
            messages=api_messages,
        )
        if not is_reasoning:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens

        if api_tools:
            kwargs["tools"] = api_tools

        logger.debug("OpenAI request: model=%s, msgs=%d", self._model, len(api_messages))
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        logger.debug("OpenAI response: finish=%s", choice.finish_reason)

        text_content = choice.message.content or ""
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        finish = "tool_calls" if tool_calls else choice.finish_reason or "stop"
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=text_content or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
        )
