"""Gemini provider using google-generativeai SDK."""

from __future__ import annotations

import json
import logging

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from airs.llm.base import BaseProvider, LLMResponse, Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "quota" in str(exc).lower()


class GeminiProvider(BaseProvider):
    """Google Gemini provider."""

    def __init__(self, model: str, api_key: str = ""):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key or None)
            self._client = genai
        return self._client

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception(_is_rate_limit),
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
        import asyncio
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=self._api_key or None)

        # Build system instruction and history
        system_instruction = ""
        history = []
        pending_user_msg = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                pending_user_msg = msg.content
            elif msg.role == "assistant":
                if pending_user_msg is not None:
                    history.append({"role": "user", "parts": [pending_user_msg]})
                    pending_user_msg = None
                history.append({"role": "model", "parts": [msg.content]})
            elif msg.role == "tool_result":
                # Append tool result as user turn
                history.append({
                    "role": "user",
                    "parts": [f"Tool result (id={msg.tool_call_id}): {msg.content}"],
                })

        # Build tool declarations
        tool_declarations = None
        if tools:
            from google.generativeai.types import content_types
            declarations = []
            for t in tools:
                declarations.append(
                    genai.protos.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=_json_schema_to_gemini(t.parameters),
                    )
                )
            tool_declarations = [genai.protos.Tool(function_declarations=declarations)]

        gen_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system_instruction or None,
            generation_config=gen_config,
            tools=tool_declarations,
        )

        chat = model.start_chat(history=history)

        # Run in thread (SDK is sync)
        user_msg = pending_user_msg or "Continue."
        logger.debug("Gemini request: model=%s", self._model)
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: chat.send_message(user_msg)
        )

        text_content = ""
        tool_calls: list[ToolCall] = []

        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text_content += part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=fc.name,  # Gemini doesn't have IDs; use name
                        name=fc.name,
                        arguments=dict(fc.args),
                    )
                )

        finish = "tool_calls" if tool_calls else "stop"

        return LLMResponse(
            content=text_content or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage={},
        )


def _json_schema_to_gemini(schema: dict) -> dict:
    """Convert JSON Schema to Gemini Schema format."""
    import google.generativeai as genai

    type_map = {
        "string": genai.protos.Type.STRING,
        "number": genai.protos.Type.NUMBER,
        "integer": genai.protos.Type.INTEGER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
        "object": genai.protos.Type.OBJECT,
    }

    props = {}
    for k, v in schema.get("properties", {}).items():
        t = type_map.get(v.get("type", "string"), genai.protos.Type.STRING)
        props[k] = genai.protos.Schema(type=t, description=v.get("description", ""))

    return genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties=props,
        required=schema.get("required", []),
    )
