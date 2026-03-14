"""AIRS CLI — Typer-based command interface."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from airs.config.loader import load_config, validate_config

app = typer.Typer(
    name="airs",
    help="AIRS — AI Research System: end-to-end autonomous research pipeline",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# airs run
# ---------------------------------------------------------------------------

@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic or question"),
    project: str = typer.Option(..., "--project", "-p", help="Project name (used as workspace directory)"),
    auto: bool = typer.Option(False, "--auto", help="Skip all user checkpoints"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full research pipeline end-to-end."""
    _setup_logging(verbose)

    console.print(Panel(
        f"[bold]Research Topic:[/bold] {topic}\n"
        f"[bold]Project:[/bold] {project}",
        title="[bold blue]AIRS — AI Research System[/bold blue]",
        subtitle="Starting pipeline...",
    ))

    from airs.orchestrator import Orchestrator
    orch = Orchestrator(project=project)

    asyncio.run(orch.run_pipeline(topic=topic, auto_proceed=auto))


# ---------------------------------------------------------------------------
# airs resume
# ---------------------------------------------------------------------------

@app.command()
def resume(
    project: str = typer.Option(..., "--project", "-p"),
    from_stage: Optional[str] = typer.Option(None, "--from", help="Resume from this stage"),
    auto: bool = typer.Option(False, "--auto"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Resume a pipeline from a checkpoint (or from a specific stage)."""
    _setup_logging(verbose)

    from airs.orchestrator import Orchestrator
    orch = Orchestrator(project=project)

    state = orch.status()
    if not state:
        console.print(f"[red]No workspace found for project: {project}[/red]")
        raise typer.Exit(1)

    topic = state.get("topic", "")
    console.print(f"Resuming project [bold]{project}[/bold] (topic: {topic})")

    asyncio.run(orch.run_pipeline(
        topic=topic,
        from_stage=from_stage,
        auto_proceed=auto,
    ))


# ---------------------------------------------------------------------------
# airs agent
# ---------------------------------------------------------------------------

@app.command()
def agent(
    stage: str = typer.Argument(help="Stage name to run (e.g. 'literature', 'design')"),
    project: str = typer.Option(..., "--project", "-p"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic (required if workspace not yet initialized)"),
    task: Optional[str] = typer.Option(None, "--task", help="Override task description"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a single pipeline stage."""
    _setup_logging(verbose)

    from airs.orchestrator import Orchestrator
    orch = Orchestrator(project=project)

    # If workspace doesn't exist yet, need a topic
    state = orch.status()
    effective_topic = topic or state.get("topic", "")

    if not effective_topic and not state:
        console.print("[red]Provide --topic to initialize the workspace.[/red]")
        raise typer.Exit(1)

    console.print(f"Running stage [bold]{stage}[/bold] for project [bold]{project}[/bold]")

    result = asyncio.run(orch.run_single_agent(
        stage_name=stage,
        task=task,
        topic=effective_topic,
    ))

    if result.success:
        console.print(f"\n[green]✓ Stage {stage} complete[/green]")
        console.print(f"Output files: {len(result.output_files)}")
        for f in result.output_files[:10]:
            console.print(f"  - {f}")
    else:
        console.print(f"\n[red]✗ Stage {stage} failed: {result.error}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# airs status
# ---------------------------------------------------------------------------

@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p"),
):
    """Show pipeline status for a project."""
    from airs.orchestrator import Orchestrator
    orch = Orchestrator(project=project)

    state = orch.status()
    if not state:
        console.print(f"[red]No workspace found for project: {project}[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    stages_state = state.get("stages", {})

    table = Table(
        title=f"Pipeline Status: {project}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Agent")
    table.add_column("Model")
    table.add_column("Files")
    table.add_column("Updated")

    for stage in cfg.pipeline.stages:
        stage_data = stages_state.get(stage.name, {})
        status_val = stage_data.get("status", "pending")
        files = len(stage_data.get("output_files", []))
        updated = stage_data.get("updated_at", "-")
        if updated != "-":
            updated = updated[:16]  # Trim seconds

        agent_cfg = cfg.agents.get(stage.agent)
        model = f"{agent_cfg.provider}/{agent_cfg.model}" if agent_cfg else "-"

        # Style status
        status_styled = {
            "done": "[green]✓ done[/green]",
            "running": "[yellow]● running[/yellow]",
            "failed": "[red]✗ failed[/red]",
            "pending": "[dim]○ pending[/dim]",
            "skipped": "[blue]⊘ skipped[/blue]",
        }.get(status_val, status_val)

        table.add_row(
            stage.display_name,
            status_styled,
            stage.agent,
            model,
            str(files) if files else "-",
            updated,
        )

    console.print(f"\n[bold]Topic:[/bold] {state.get('topic', 'N/A')}")
    console.print(f"[bold]Created:[/bold] {state.get('created_at', 'N/A')[:16]}")
    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# airs serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    project: str = typer.Option(..., "--project", "-p"),
    port: int = typer.Option(8080, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Start the web viewer for a project."""
    import uvicorn
    from airs.web.app import create_app

    web_app = create_app(project=project)
    console.print(
        f"[bold]AIRS Web Viewer[/bold] — project: [cyan]{project}[/cyan]\n"
        f"Open: [link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# airs config validate
# ---------------------------------------------------------------------------

config_app = typer.Typer(help="Config management commands")
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate():
    """Validate all config files and API keys."""
    console.print("Validating AIRS configuration...")

    issues = validate_config()

    if not issues:
        console.print("[green]✓ All configs valid. No issues found.[/green]")
    else:
        console.print(f"[yellow]Found {len(issues)} issue(s):[/yellow]")
        for issue in issues:
            level = "[red]✗[/red]" if "error" in issue.lower() else "[yellow]⚠[/yellow]"
            console.print(f"  {level} {issue}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# airs list-projects
# ---------------------------------------------------------------------------

@app.command("list")
def list_projects():
    """List all research projects in the workspace."""
    from airs.workspace.manager import WorkspaceManager, _DEFAULT_WORKSPACE

    ws_root = _DEFAULT_WORKSPACE
    if not ws_root.exists():
        console.print("[dim]No workspace directory found.[/dim]")
        return

    projects = [d for d in ws_root.iterdir() if d.is_dir()]
    if not projects:
        console.print("[dim]No projects found.[/dim]")
        return

    table = Table(title="Research Projects", show_header=True)
    table.add_column("Project")
    table.add_column("Topic")
    table.add_column("Created")

    for proj_dir in sorted(projects):
        state_file = proj_dir / "project.yaml"
        if state_file.exists():
            import yaml
            with open(state_file) as f:
                state = yaml.safe_load(f) or {}
            topic = state.get("topic", "-")[:60]
            created = state.get("created_at", "-")[:16]
        else:
            topic = "-"
            created = "-"
        table.add_row(proj_dir.name, topic, created)

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
