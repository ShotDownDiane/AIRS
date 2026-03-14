"""FileWriteSkill: write files to the workspace."""

from __future__ import annotations

import logging

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class FileWriteSkill(BaseSkill):
    """Writes files to the workspace. Injected with workspace manager at runtime."""

    def __init__(self, workspace=None):
        self._workspace = workspace  # WorkspaceManager, set by agent

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file in the research workspace. "
            "Use workspace-relative paths (e.g. 'literature/survey.md'). "
            "Creates parent directories automatically. "
            "Returns a confirmation message."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path (e.g. 'literature/survey.md')",
                },
                "content": {
                    "type": "string",
                    "description": "File content to write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> str:
        if self._workspace is None:
            return "Error: FileWriteSkill not connected to workspace"
        try:
            self._workspace.write(path, content)
            logger.info("FileWrite: %s (%d chars)", path, len(content))
            return f"Successfully wrote {len(content)} characters to {path}"
        except PermissionError as e:
            return f"Permission error: {e}"
        except Exception as e:
            return f"Error writing file {path}: {e}"
