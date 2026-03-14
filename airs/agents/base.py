"""BaseAgent: universal agentic tool-call loop inherited by all agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from airs.config.loader import AgentConfig
from airs.llm.base import BaseProvider, Message, ToolCall, create_provider
from airs.skills import BaseSkill, build_skill
from airs.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """Emitted during agent execution for live UI updates."""
    type: str           # "tool_call" | "tool_result" | "thinking" | "done" | "error"
    agent_name: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    agent_name: str
    output_files: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    iterations: int = 0
    total_tokens: int = 0


class BaseAgent:
    """
    Universal agentic loop. All 8 research agents inherit from this.

    The loop:
    1. Build initial messages (system prompt + task + input file contents)
    2. Loop until stop or max_iterations:
       a. Call LLM
       b. If tool_calls: execute each skill, append result, continue
       c. If stop: record output, break
    3. Log everything to workspace/logs/agent_runs/
    """

    def __init__(
        self,
        config: AgentConfig,
        workspace: WorkspaceManager,
        llm: BaseProvider | None = None,
        ssh_client=None,
    ):
        self.config = config
        self.workspace = workspace
        self.ssh_client = ssh_client

        # LLM provider
        self.llm = llm or self._build_llm()

        # Build skills
        self.skills: dict[str, BaseSkill] = {}
        for skill_name in config.skills:
            skill = build_skill(
                skill_name,
                workspace=workspace,
                ssh_client=ssh_client,
            )
            self.skills[skill.name] = skill

        logger.info(
            "Agent %s ready: provider=%s model=%s skills=%s",
            config.name,
            config.provider,
            config.model,
            list(self.skills),
        )

    def _build_llm(self) -> BaseProvider:
        from airs.config.loader import load_config
        cfg = load_config()

        # If unified proxy is configured, route everything through OpenAI-compatible API
        if cfg.llm_base_url and cfg.llm_api_key:
            return create_provider(
                "openai",
                self.config.model,
                cfg.llm_api_key,
                base_url=cfg.llm_base_url + "/v1",
            )

        # Otherwise use per-provider API keys
        key_map = {
            "claude": cfg.anthropic_api_key,
            "openai": cfg.openai_api_key,
            "gemini": cfg.google_api_key,
        }
        api_key = key_map.get(self.config.provider, "")
        return create_provider(
            self.config.provider,
            self.config.model,
            api_key,
            base_url=self.config.base_url,
        )

    async def run(
        self,
        task: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AgentResult:
        """Run the agent on the given task string."""
        name = self.config.name
        logger.info("Agent %s starting task (%d chars)", name, len(task))

        # Build tool definitions
        tool_defs = [s.to_tool_definition() for s in self.skills.values()]

        # Build initial messages
        messages = self._build_initial_messages(task)

        # Track for logging
        log_messages: list[dict] = []
        total_tokens = 0
        iterations = 0

        try:
            for i in range(self.config.max_iterations):
                iterations = i + 1

                # Call LLM
                response = await self.llm.chat(
                    messages=messages,
                    tools=tool_defs or None,
                    temperature=self.config.temperature,
                )
                total_tokens += (
                    response.usage.get("input_tokens", 0)
                    + response.usage.get("output_tokens", 0)
                )

                # Emit thinking event if there's text content
                if response.content:
                    if on_event:
                        on_event(AgentEvent(
                            type="thinking",
                            agent_name=name,
                            data={"text": response.content[:500]},
                        ))

                # No tool calls → agent is done
                if not response.tool_calls:
                    # Append final assistant message
                    messages.append(Message(
                        role="assistant",
                        content=response.content or "",
                    ))
                    log_messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                    })
                    if on_event:
                        on_event(AgentEvent(
                            type="done",
                            agent_name=name,
                            data={"iterations": iterations, "total_tokens": total_tokens},
                        ))
                    break

                # Build assistant message with tool calls
                # We serialize tool calls as a structured text block for simpler multi-provider support
                tool_calls_text = json.dumps(
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                     for tc in response.tool_calls],
                    indent=2,
                )
                assistant_content = (response.content or "") + f"\n[TOOL_CALLS]\n{tool_calls_text}"
                messages.append(Message(role="assistant", content=assistant_content))

                # Execute each tool call
                for tc in response.tool_calls:
                    if on_event:
                        on_event(AgentEvent(
                            type="tool_call",
                            agent_name=name,
                            data={"tool": tc.name, "args": tc.arguments},
                        ))

                    result_str = await self._execute_tool(tc)

                    if on_event:
                        on_event(AgentEvent(
                            type="tool_result",
                            agent_name=name,
                            data={"tool": tc.name, "result": result_str[:300]},
                        ))

                    log_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_str[:2000],
                    })

                    messages.append(Message(
                        role="tool_result",
                        content=result_str,
                        tool_call_id=tc.id,
                    ))

            else:
                logger.warning("Agent %s reached max_iterations=%d", name, self.config.max_iterations)

        except Exception as e:
            logger.exception("Agent %s error: %s", name, e)
            self._write_log(name, log_messages, total_tokens, iterations, error=str(e))
            return AgentResult(
                success=False,
                agent_name=name,
                error=str(e),
                iterations=iterations,
                total_tokens=total_tokens,
            )

        # Collect output files
        output_files = self.workspace.list_files(self.config.output_dir)

        # Write final summary to output dir
        final_summary = self._extract_summary(messages)
        if final_summary:
            summary_path = f"{self.config.output_dir}/_summary.md"
            try:
                self.workspace.write(summary_path, final_summary)
            except Exception:
                pass

        # Write agent log
        self._write_log(name, log_messages, total_tokens, iterations)

        logger.info(
            "Agent %s done: %d iterations, %d tokens, %d output files",
            name, iterations, total_tokens, len(output_files),
        )

        return AgentResult(
            success=True,
            agent_name=name,
            output_files=output_files,
            summary=final_summary,
            iterations=iterations,
            total_tokens=total_tokens,
        )

    def _build_initial_messages(self, task: str) -> list[Message]:
        """Build the initial message list with system prompt, context, and task."""
        messages: list[Message] = [
            Message(role="system", content=self.config.system_prompt)
        ]

        # Inject input file contents
        context = ""
        if self.config.input_files:
            context = self.workspace.get_input_files_content(self.config.input_files)
            if context:
                context = f"\n\n# Context from previous stages\n\n{context}"

        user_content = f"# Task\n\n{task}{context}"
        messages.append(Message(role="user", content=user_content))
        return messages

    async def _execute_tool(self, tc: ToolCall) -> str:
        skill = self.skills.get(tc.name)
        if skill is None:
            return f"Unknown tool: {tc.name!r}. Available: {list(self.skills)}"
        try:
            return await skill.execute(**tc.arguments)
        except TypeError as e:
            return f"Tool {tc.name!r} parameter error: {e}"
        except Exception as e:
            logger.exception("Skill %s error", tc.name)
            return f"Tool {tc.name!r} error: {e}"

    def _extract_summary(self, messages: list[Message]) -> str:
        """Extract the last substantial assistant message as summary."""
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content and len(msg.content) > 50:
                # Strip tool call JSON from summary
                content = msg.content
                if "[TOOL_CALLS]" in content:
                    content = content.split("[TOOL_CALLS]")[0].strip()
                if content:
                    return content
        return ""

    def _write_log(
        self,
        name: str,
        messages: list[dict],
        total_tokens: int,
        iterations: int,
        error: str = "",
    ) -> None:
        log = {
            "agent": name,
            "provider": self.config.provider,
            "model": self.config.model,
            "timestamp": datetime.utcnow().isoformat(),
            "iterations": iterations,
            "total_tokens": total_tokens,
            "error": error,
            "messages": messages,
        }
        try:
            self.workspace.write_agent_log(name, log)
        except Exception as e:
            logger.warning("Failed to write agent log: %s", e)
