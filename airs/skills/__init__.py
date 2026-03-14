"""Skills package — registry mapping name strings to skill classes."""

from __future__ import annotations

from airs.skills.arxiv import ArxivSkill
from airs.skills.arxiv_download import ArxivDownloadSkill
from airs.skills.base import BaseSkill
from airs.skills.file_read import FileReadSkill
from airs.skills.file_write import FileWriteSkill
from airs.skills.latex_compile import LatexCompileSkill
from airs.skills.python_exec import PythonExecSkill
from airs.skills.ssh_exec import SSHExecSkill
from airs.skills.web_search import WebSearchSkill

# Registry maps skill name → class
SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "arxiv": ArxivSkill,
    "arxiv_download": ArxivDownloadSkill,
    "web_search": WebSearchSkill,
    "file_read": FileReadSkill,
    "file_write": FileWriteSkill,
    "python_exec": PythonExecSkill,
    "ssh_exec": SSHExecSkill,
    "latex_compile": LatexCompileSkill,
}


def build_skill(
    name: str,
    workspace=None,
    ssh_client=None,
) -> BaseSkill:
    """Instantiate a skill by name, injecting workspace/ssh if needed."""
    if name not in SKILL_REGISTRY:
        raise ValueError(f"Unknown skill: {name!r}. Available: {list(SKILL_REGISTRY)}")

    cls = SKILL_REGISTRY[name]

    # Skills that need workspace
    if name in ("file_read", "file_write", "arxiv_download"):
        return cls(workspace=workspace)

    # Skills that need SSH
    if name == "ssh_exec":
        return cls(ssh_client=ssh_client)

    # LaTeX needs both
    if name == "latex_compile":
        return cls(workspace=workspace, ssh_client=ssh_client)

    return cls()


__all__ = [
    "BaseSkill",
    "ArxivSkill",
    "WebSearchSkill",
    "FileReadSkill",
    "FileWriteSkill",
    "PythonExecSkill",
    "SSHExecSkill",
    "LatexCompileSkill",
    "SKILL_REGISTRY",
    "build_skill",
]
