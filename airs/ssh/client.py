"""SSHClient: paramiko-based SSH + SFTP client for remote execution."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import paramiko

from airs.config.loader import SSHConfig

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int

    def to_string(self, max_chars: int = 8000) -> str:
        out = ""
        if self.stdout:
            out += f"STDOUT:\n{self.stdout}"
        if self.stderr:
            out += f"\nSTDERR:\n{self.stderr}"
        if self.exit_code != 0:
            out += f"\nExit code: {self.exit_code}"
        if not out:
            out = "(No output)"
        if len(out) > max_chars:
            out = out[:max_chars] + f"\n... (truncated, total {len(out)} chars)"
        return out

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class SSHClient:
    """Paramiko-based SSH client for remote command execution and file transfer."""

    def __init__(self, config: SSHConfig):
        self.config = config
        self._client: paramiko.SSHClient | None = None

    def connect(self) -> None:
        """Establish SSH connection."""
        if self._client is not None:
            try:
                self._client.get_transport().is_active()
                return  # Already connected
            except Exception:
                self._client = None

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = dict(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.username,
            timeout=self.config.connect_timeout,
        )

        # Authentication: prefer key file
        if self.config.key_file:
            key_path = os.path.expanduser(self.config.key_file)
            connect_kwargs["key_filename"] = key_path
        elif self.config.password:
            connect_kwargs["password"] = self.config.password

        logger.info("SSH connecting to %s:%d", self.config.host, self.config.port)
        client.connect(**connect_kwargs)
        self._client = client
        logger.info("SSH connected")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def exec(self, command: str, timeout: int = 300) -> CommandResult:
        """Execute a command and return result."""
        if self._client is None:
            self.connect()

        logger.debug("SSH exec: %s", command[:200])
        _, stdout, stderr = self._client.exec_command(command, timeout=timeout)

        exit_code = stdout.channel.recv_exit_status()
        stdout_str = stdout.read().decode("utf-8", errors="replace")
        stderr_str = stderr.read().decode("utf-8", errors="replace")

        return CommandResult(
            command=command,
            stdout=stdout_str,
            stderr=stderr_str,
            exit_code=exit_code,
        )

    def exec_long(self, command: str, job_name: str) -> str:
        """Run a long job in the background with nohup. Returns PID."""
        remote_ws = self.config.remote_workspace
        log_file = f"{remote_ws}/logs/{job_name}.log"
        pid_file = f"{remote_ws}/logs/{job_name}.pid"

        wrapped = (
            f"mkdir -p {remote_ws}/logs && "
            f"nohup bash -c '{command}' > {log_file} 2>&1 & "
            f"echo $! > {pid_file} && echo $!"
        )
        result = self.exec(wrapped, timeout=30)
        pid = result.stdout.strip()
        return f"Job started with PID {pid}. Log: {log_file}"

    def upload_directory(self, local: str | Path, remote: str) -> None:
        """Upload a local directory to the remote via SFTP."""
        if self._client is None:
            self.connect()

        local = Path(local)
        sftp = self._client.open_sftp()

        try:
            for local_file in sorted(local.rglob("*")):
                if local_file.is_file():
                    rel = local_file.relative_to(local)
                    remote_path = f"{remote}/{rel}".replace("\\", "/")
                    remote_dir = str(Path(remote_path).parent).replace("\\", "/")

                    # Ensure remote directory exists
                    try:
                        sftp.makedirs(remote_dir)
                    except Exception:
                        self.exec(f"mkdir -p {remote_dir}")

                    logger.debug("SFTP upload: %s -> %s", local_file, remote_path)
                    sftp.put(str(local_file), remote_path)
        finally:
            sftp.close()

    def download_file(self, remote: str, local: str | Path) -> None:
        """Download a single file from remote via SFTP."""
        if self._client is None:
            self.connect()

        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)

        sftp = self._client.open_sftp()
        try:
            logger.debug("SFTP download: %s -> %s", remote, local)
            sftp.get(remote, str(local))
        finally:
            sftp.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
