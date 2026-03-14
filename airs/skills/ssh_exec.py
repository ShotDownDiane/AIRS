"""SSHExecSkill: execute commands on a remote server via SSH."""

from __future__ import annotations

import logging

from airs.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class SSHExecSkill(BaseSkill):
    """Executes commands on the remote SSH server. Injected with SSHClient at runtime."""

    def __init__(self, ssh_client=None):
        self._ssh = ssh_client  # airs.ssh.client.SSHClient, set by agent

    @property
    def name(self) -> str:
        return "ssh_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command on the remote GPU server via SSH. "
            "Returns stdout and stderr (up to 8000 chars). "
            "For long-running jobs (>5 min), use nohup and check status separately. "
            "The remote workspace is at ~/airs_workspace/."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute on remote server",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 3600)",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, timeout: int = 120) -> str:
        if self._ssh is None:
            return "Error: SSHExecSkill not connected to SSH client. SSH may not be configured."
        timeout = min(int(timeout), 3600)
        try:
            result = self._ssh.exec(command, timeout=timeout)
            return result.to_string()
        except Exception as e:
            return f"SSH execution error: {e}"
