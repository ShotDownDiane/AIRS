"""FileReadSkill: read workspace files."""

from __future__ import annotations

import logging

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class FileReadSkill(BaseSkill):
    """Reads files from the workspace. Injected with workspace manager at runtime."""

    def __init__(self, workspace=None):
        self._workspace = workspace  # WorkspaceManager, set by agent

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file from the research workspace. "
            "Use workspace-relative paths (e.g. 'literature/survey.md'). "
            "Returns the file contents as a string."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path (e.g. 'literature/survey.md')",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> str:
        if self._workspace is None:
            return "Error: FileReadSkill not connected to workspace"
        try:
            content = self._workspace.read(path)
            logger.debug("FileRead: %s (%d chars)", path, len(content))
            return content
        except FileNotFoundError:
            return f"File not found: {path}"
        except PermissionError as e:
            return f"Permission error: {e}"
        except Exception as e:
            return f"Error reading file {path}: {e}"
