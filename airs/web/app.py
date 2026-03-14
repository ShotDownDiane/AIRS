"""FastAPI web viewer for AIRS workspace documents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import markdown as md
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from airs.config.loader import load_config
from airs.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(project: str, workspace_root: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI web viewer app."""
    app = FastAPI(title=f"AIRS Viewer — {project}")

    workspace = WorkspaceManager(project, workspace_root)
    cfg = load_config()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # SSE clients: list of asyncio queues
    _sse_clients: list[asyncio.Queue] = []

    # ---------------------------------------------------------------------------
    # File watcher for SSE auto-refresh
    # ---------------------------------------------------------------------------

    class ChangeHandler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent):
            if not event.is_directory:
                msg = json.dumps({"type": "change", "path": str(event.src_path)})
                for q in list(_sse_clients):
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass

    observer = Observer()
    observer.schedule(ChangeHandler(), str(workspace.root), recursive=True)

    @app.on_event("startup")
    async def start_watcher():
        if workspace.root.exists():
            observer.start()

    @app.on_event("shutdown")
    async def stop_watcher():
        observer.stop()
        observer.join()

    # ---------------------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        tree = _build_tree(workspace, cfg, project)
        state = workspace.read_project_state()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "project": project,
            "topic": state.get("topic", ""),
            "tree": tree,
            "content_html": "",
            "current_path": "",
        })

    @app.get("/view/{file_path:path}", response_class=HTMLResponse)
    async def view_file(request: Request, file_path: str):
        tree = _build_tree(workspace, cfg, project)
        state = workspace.read_project_state()

        try:
            raw = workspace.read(file_path)
            content_html = _render_content(file_path, raw)
        except FileNotFoundError:
            content_html = f"<p class='error'>File not found: {file_path}</p>"
        except Exception as e:
            content_html = f"<p class='error'>Error reading file: {e}</p>"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "project": project,
            "topic": state.get("topic", ""),
            "tree": tree,
            "content_html": content_html,
            "current_path": file_path,
        })

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(workspace.read_project_state())

    @app.get("/api/costs")
    async def api_costs():
        from airs.cost import CostTracker
        tracker = CostTracker.load(workspace)
        return JSONResponse(tracker.summary())

    @app.get("/api/tree")
    async def api_tree():
        return JSONResponse({"tree": _build_tree(workspace, cfg, project)})

    @app.get("/events")
    async def sse_events(request: Request):
        """Server-Sent Events endpoint for auto-refresh."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        _sse_clients.append(queue)

        async def generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=30)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield "data: {\"type\": \"ping\"}\n\n"
            finally:
                if queue in _sse_clients:
                    _sse_clients.remove(queue)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tree(workspace: WorkspaceManager, cfg, project: str) -> list[dict]:
    """Build a structured tree for the sidebar."""
    state = workspace.read_project_state()
    stages_state = state.get("stages", {})

    tree = []
    for stage in cfg.pipeline.stages:
        stage_status = stages_state.get(stage.name, {}).get("status", "pending")
        stage_dir = workspace.root / stage.name

        files = []
        if stage_dir.exists():
            for p in sorted(stage_dir.rglob("*")):
                if p.is_file():
                    rel = str(p.relative_to(workspace.root))
                    files.append({
                        "path": rel,
                        "name": p.name,
                        "size": p.stat().st_size,
                    })

        tree.append({
            "name": stage.name,
            "display_name": stage.display_name,
            "status": stage_status,
            "files": files,
        })

    return tree


def _render_content(file_path: str, raw: str) -> str:
    """Render file content to HTML."""
    ext = Path(file_path).suffix.lower()

    if ext in (".md", ".markdown"):
        return md.markdown(
            raw,
            extensions=["tables", "fenced_code", "codehilite", "toc"],
        )
    elif ext in (".py", ".yaml", ".yml", ".json", ".tex", ".txt", ".sh"):
        lang = {
            ".py": "python",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".tex": "latex",
            ".sh": "bash",
        }.get(ext, "")
        escaped = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<pre><code class="language-{lang}">{escaped}</code></pre>'
    else:
        escaped = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"
