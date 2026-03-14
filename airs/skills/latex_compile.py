"""LatexCompileSkill: compile LaTeX files via SSH or local pdflatex."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class LatexCompileSkill(BaseSkill):
    """Compiles LaTeX files. Uses local pdflatex if available, SSH otherwise."""

    def __init__(self, workspace=None, ssh_client=None):
        self._workspace = workspace
        self._ssh = ssh_client

    @property
    def name(self) -> str:
        return "latex_compile"

    @property
    def description(self) -> str:
        return (
            "Compile a LaTeX file to PDF. Provide the workspace-relative path to the .tex file. "
            "Returns compilation log. The PDF will be saved next to the .tex file."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tex_path": {
                    "type": "string",
                    "description": "Workspace-relative path to the .tex file (e.g. 'paper/paper.tex')",
                }
            },
            "required": ["tex_path"],
        }

    async def execute(self, tex_path: str) -> str:
        if self._workspace is None:
            return "Error: LatexCompileSkill not connected to workspace"

        full_path = self._workspace._safe_path(tex_path)

        if not full_path.exists():
            return f"LaTeX file not found: {tex_path}"

        tex_dir = full_path.parent
        tex_name = full_path.name

        # Try local pdflatex first
        if shutil.which("pdflatex"):
            return await self._compile_local(tex_dir, tex_name, tex_path)
        elif self._ssh is not None:
            return self._compile_remote(tex_path)
        else:
            return (
                "pdflatex not found locally and SSH not configured. "
                "Install texlive or configure SSH to a server with LaTeX."
            )

    async def _compile_local(self, tex_dir: Path, tex_name: str, tex_path: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(tex_dir),
                str(tex_dir / tex_name),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(tex_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            log = stdout.decode("utf-8", errors="replace")

            pdf_path = tex_dir / tex_name.replace(".tex", ".pdf")
            if pdf_path.exists():
                return f"Compilation successful. PDF: {tex_path.replace('.tex', '.pdf')}\n\nLog (last 2000 chars):\n{log[-2000:]}"
            else:
                return f"Compilation failed.\n\nLog:\n{log[-3000:]}"
        except asyncio.TimeoutError:
            return "LaTeX compilation timed out after 120s"
        except Exception as e:
            return f"LaTeX compilation error: {e}"

    def _compile_remote(self, tex_path: str) -> str:
        try:
            remote_tex = f"~/airs_workspace/{tex_path}"
            remote_dir = str(Path(remote_tex).parent)
            cmd = f"cd {remote_dir} && pdflatex -interaction=nonstopmode {Path(remote_tex).name}"
            result = self._ssh.exec(cmd, timeout=120)
            return f"Remote compilation result:\n{result.to_string()}"
        except Exception as e:
            return f"Remote LaTeX compilation error: {e}"
