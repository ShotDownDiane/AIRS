"""ExperimenterAgent: runs experiments on the remote GPU server via SSH."""

from __future__ import annotations

from pathlib import Path

from airs.agents.base import BaseAgent
from airs.config.loader import AgentConfig
from airs.workspace.manager import WorkspaceManager

import logging

logger = logging.getLogger(__name__)


class ExperimenterAgent(BaseAgent):
    """Runs experiments on the remote server and collects results."""

    def __init__(self, config: AgentConfig, workspace: WorkspaceManager, **kwargs):
        super().__init__(config, workspace, **kwargs)

    async def run(self, task: str, on_event=None):
        """Override to sync code to remote before running."""
        # Upload code directory to remote if SSH is available
        if self.ssh_client is not None:
            try:
                code_dir = self.workspace.root / "code"
                if code_dir.exists():
                    remote_code = f"{self.ssh_client.config.remote_workspace}/code"
                    logger.info("Uploading code to remote: %s -> %s", code_dir, remote_code)
                    self.ssh_client.connect()
                    self.ssh_client.exec(f"mkdir -p {remote_code}")
                    self.ssh_client.upload_directory(code_dir, remote_code)
                    logger.info("Code upload complete")
            except Exception as e:
                logger.warning("Code upload failed: %s", e)

        return await super().run(task, on_event=on_event)
