"""Orchestrator: sequential pipeline state machine."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from airs.agents.base import AgentEvent, AgentResult
from airs.agents.factory import build_agent
from airs.config.loader import AIRSConfig, StageConfig, load_config
from airs.cost import CostTracker
from airs.ssh.client import SSHClient
from airs.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)
console = Console()

# Status badge styles
_STATUS_STYLE = {
    "done": "[green]✓ done[/green]",
    "running": "[yellow]● running[/yellow]",
    "failed": "[red]✗ failed[/red]",
    "pending": "[dim]○ pending[/dim]",
    "skipped": "[blue]⊘ skipped[/blue]",
}


class Orchestrator:
    """Runs the full research pipeline or individual stages."""

    def __init__(
        self,
        project: str,
        workspace_root: Path | None = None,
        config: AIRSConfig | None = None,
    ):
        self.project = project
        self.config = config or load_config()
        self.workspace = WorkspaceManager(project, workspace_root)
        self._ssh: SSHClient | None = None
        self._cost_tracker: CostTracker | None = None

    # ------------------------------------------------------------------
    # SSH
    # ------------------------------------------------------------------

    def _get_ssh(self) -> SSHClient | None:
        if self._ssh is not None:
            return self._ssh
        ssh_cfg = self.config.ssh
        if not ssh_cfg.host:
            return None
        self._ssh = SSHClient(ssh_cfg)
        return self._ssh

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        topic: str,
        from_stage: str | None = None,
        auto_proceed: bool = False,
    ) -> None:
        """Run all pipeline stages in order."""
        self.workspace.init(topic)
        stages = self.config.pipeline.stages

        # Fast-forward to from_stage if specified
        start_idx = 0
        if from_stage:
            for i, s in enumerate(stages):
                if s.name == from_stage:
                    start_idx = i
                    break

        for stage in stages[start_idx:]:
            status = self.workspace.get_stage_status(stage.name)

            # Skip already-done stages
            if status == "done":
                console.print(f"  [green]✓[/green] {stage.display_name} (already done, skipping)")
                continue

            # Build task description
            task = self._build_task(stage, topic)

            # User checkpoint (unless auto_proceed flag set)
            if not stage.auto_proceed and not auto_proceed:
                proceed = self._user_checkpoint(stage, task)
                if proceed == "skip":
                    self.workspace.update_project_state(stage.name, "skipped")
                    console.print(f"  [blue]⊘[/blue] {stage.display_name} skipped")
                    continue
                elif proceed == "edit":
                    task = self._edit_task(task)

            # Run the stage
            result = await self._run_stage(stage, task)

            if not result.success:
                console.print(f"\n[red]Stage {stage.name} failed: {result.error}[/red]")
                retry = Prompt.ask("Retry this stage?", choices=["y", "n"], default="n")
                if retry == "y":
                    result = await self._run_stage(stage, task)
                if not result.success:
                    console.print("[red]Pipeline halted.[/red]")
                    return

        # Final summary
        console.print("\n[bold green]Pipeline complete![/bold green]")
        self._print_status()

    async def _run_stage(self, stage: StageConfig, task: str) -> AgentResult:
        """Run a single stage with Rich UI."""
        agent_config = self.config.agents[stage.agent]

        # Show stage header
        console.print(Panel(
            f"[bold]{stage.display_name}[/bold]\n"
            f"Agent: {agent_config.display_name}  "
            f"Model: {agent_config.provider}/{agent_config.model}",
            title=f"Stage: {stage.name}",
            style="blue",
        ))

        self.workspace.update_project_state(stage.name, "running")

        # Shared cost tracker
        if self._cost_tracker is None:
            self._cost_tracker = CostTracker.load(self.workspace)

        agent = build_agent(
            agent_name=stage.agent,
            config=agent_config,
            workspace=self.workspace,
            ssh_client=self._get_ssh(),
            cost_tracker=self._cost_tracker,
        )

        def on_event(event: AgentEvent) -> None:
            _render_event(event)

        try:
            result = await agent.run(task, on_event=on_event)
        except Exception as e:
            logger.exception("Stage %s error", stage.name)
            self.workspace.update_project_state(stage.name, "failed")
            return AgentResult(success=False, agent_name=stage.agent, error=str(e))

        if result.success:
            self.workspace.update_project_state(
                stage.name, "done", result.output_files
            )
            console.print(
                f"\n[green]✓ {stage.display_name} complete[/green] "
                f"({result.iterations} iterations, {result.total_tokens} tokens)\n"
                f"Output files: {len(result.output_files)}"
            )
        else:
            self.workspace.update_project_state(stage.name, "failed")

        return result

    # ------------------------------------------------------------------
    # Single agent
    # ------------------------------------------------------------------

    async def run_single_agent(
        self,
        stage_name: str,
        task: str | None = None,
        topic: str = "",
    ) -> AgentResult:
        """Run a single stage by name."""
        # Find the stage config
        stage = next(
            (s for s in self.config.pipeline.stages if s.name == stage_name),
            None,
        )
        if stage is None:
            raise ValueError(f"Unknown stage: {stage_name!r}")

        # Ensure workspace exists
        if topic:
            self.workspace.init(topic)

        if task is None:
            task = self._build_task(stage, topic or self.workspace.read_project_state().get("topic", ""))

        return await self._run_stage(stage, task)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return self.workspace.read_project_state()

    def _print_status(self) -> None:
        state = self.workspace.read_project_state()
        stages_state = state.get("stages", {})

        table = Table(title=f"Pipeline Status: {self.project}", show_header=True)
        table.add_column("Stage", style="bold")
        table.add_column("Status")
        table.add_column("Files")

        for stage in self.config.pipeline.stages:
            stage_data = stages_state.get(stage.name, {})
            status = stage_data.get("status", "pending")
            files = len(stage_data.get("output_files", []))
            badge = _STATUS_STYLE.get(status, status)
            table.add_row(stage.display_name, badge, str(files) if files else "-")

        console.print(table)

    # ------------------------------------------------------------------
    # Task builder
    # ------------------------------------------------------------------

    def _build_task(self, stage: StageConfig, topic: str) -> str:
        """Build the task string for a stage."""
        state = self.workspace.read_project_state()
        topic = topic or state.get("topic", "")

        lines = [
            f"Research topic: {topic}",
            f"",
            f"You are the {stage.display_name} agent.",
            f"Your output directory is: {stage.name}/",
            f"",
            f"Complete your stage of the research pipeline as described in your system prompt.",
        ]

        # Add outputs from completed dependency stages
        prior_outputs = []
        for dep in stage.depends_on:
            dep_content = self.workspace.get_stage_outputs(dep)
            if dep_content:
                prior_outputs.append(f"=== Prior stage: {dep} ===\n{dep_content[:3000]}")

        if prior_outputs:
            lines.append("\n# Prior Stage Outputs (Summary)\n")
            lines.extend(prior_outputs)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _user_checkpoint(self, stage: StageConfig, task: str) -> str:
        """Show user checkpoint and return 'proceed' | 'edit' | 'skip'."""
        console.print(Panel(
            Text(task[:1000] + ("..." if len(task) > 1000 else ""), style="dim"),
            title=f"Task preview: {stage.display_name}",
        ))
        choice = Prompt.ask(
            f"Proceed with {stage.display_name}?",
            choices=["y", "edit", "skip"],
            default="y",
        )
        return {"y": "proceed", "edit": "edit", "skip": "skip"}.get(choice, "proceed")

    def _edit_task(self, task: str) -> str:
        """Let user edit the task in a simple multiline input."""
        console.print("[dim]Enter new task (end with '---' on its own line):[/dim]")
        lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            lines.append(line)
        return "\n".join(lines) if lines else task


def _render_event(event: AgentEvent) -> None:
    """Render an agent event to the console."""
    if event.type == "tool_call":
        tool = event.data.get("tool", "?")
        args = event.data.get("args", {})
        # Show first arg as a hint
        hint = next(iter(args.values()), "") if args else ""
        if isinstance(hint, str) and len(hint) > 60:
            hint = hint[:60] + "..."
        console.print(f"  [cyan]→ {tool}[/cyan] [dim]{hint}[/dim]")

    elif event.type == "tool_result":
        tool = event.data.get("tool", "?")
        result = event.data.get("result", "")
        first_line = result.split("\n")[0][:80] if result else ""
        console.print(f"  [green]← {tool}[/green] [dim]{first_line}[/dim]")

    elif event.type == "thinking":
        text = event.data.get("text", "")
        first_line = text.split("\n")[0][:100] if text else ""
        if first_line:
            console.print(f"  [dim]💭 {first_line}[/dim]")

    elif event.type == "done":
        pass  # Handled by caller
