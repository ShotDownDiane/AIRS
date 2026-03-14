"""LocalPythonExecSkill: execute Python code in a subprocess."""

from __future__ import annotations

import asyncio
import logging
import sys
import textwrap

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_MAX_OUTPUT = 8192


class PythonExecSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a subprocess and return stdout + stderr. "
            "Use this for testing code snippets, data analysis, and generating plots. "
            "For matplotlib, use 'import matplotlib; matplotlib.use(\"Agg\")' before plotting "
            "and save figures with 'plt.savefig(path)'."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default 60, max 300)",
                },
            },
            "required": ["code"],
        }

    async def execute(self, code: str, timeout: int = 60) -> str:
        timeout = min(int(timeout), 300)
        logger.info("PythonExec: running code (%d chars)", len(code))

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return f"Execution timed out after {timeout}s"

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            result = ""
            if out:
                result += f"STDOUT:\n{out}"
            if err:
                result += f"\nSTDERR:\n{err}"
            if proc.returncode != 0:
                result += f"\nExit code: {proc.returncode}"

            if not result:
                result = "(No output)"

            # Truncate to avoid overwhelming context
            if len(result) > _MAX_OUTPUT:
                result = result[:_MAX_OUTPUT] + f"\n... (truncated, total {len(result)} chars)"

            return result

        except Exception as e:
            return f"Execution error: {e}"
