"""Config loader: reads YAML configs + .env, returns validated Pydantic models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Load .env from project root (AIRS/)
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    name: str = ""
    display_name: str
    provider: str           # "claude" | "openai" | "gemini"
    model: str
    fallback_model: str = ""  # Escalation model when primary fails
    base_url: str = ""      # Custom API base URL (for proxy providers)
    temperature: float = 0.7
    max_iterations: int = 30
    skills: list[str] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    output_dir: str
    system_prompt: str


class StageConfig(BaseModel):
    name: str
    display_name: str
    agent: str              # Primary agent name
    agents: list[str] = Field(default_factory=list)  # Multi-agent (e.g. cross-review)
    auto_proceed: bool = True
    depends_on: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    stages: list[StageConfig]


class SSHConfig(BaseModel):
    host: str = ""
    port: int = 22
    username: str = ""
    key_file: str = ""
    password: str = ""
    remote_workspace: str = "~/airs_workspace"
    connect_timeout: int = 30


class AIRSConfig(BaseModel):
    agents: dict[str, AgentConfig]
    pipeline: PipelineConfig
    ssh: SSHConfig

    # API keys from environment
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Unified proxy provider
    llm_base_url: str = ""
    llm_api_key: str = ""

    @model_validator(mode="after")
    def load_api_keys(self) -> "AIRSConfig":
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(configs_dir: Path | None = None) -> AIRSConfig:
    """Load and validate all config files. configs_dir defaults to AIRS/configs/."""
    if configs_dir is None:
        configs_dir = _ROOT / "configs"

    # Agents config
    agents_raw = _load_yaml(configs_dir / "agents.yaml")
    agents = {
        name: AgentConfig(name=name, **cfg)
        for name, cfg in agents_raw["agents"].items()
    }

    # Pipeline config
    pipeline_raw = _load_yaml(configs_dir / "pipeline.yaml")
    pipeline = PipelineConfig(
        stages=[StageConfig(**s) for s in pipeline_raw["pipeline"]["stages"]]
    )

    # SSH config (optional — may not exist yet)
    ssh_path = configs_dir / "ssh.yaml"
    if ssh_path.exists():
        ssh_raw = _load_yaml(ssh_path)
        ssh = SSHConfig(**ssh_raw.get("ssh", {}))
    else:
        ssh = SSHConfig()

    return AIRSConfig(agents=agents, pipeline=pipeline, ssh=ssh)


def validate_config() -> list[str]:
    """Validate config and return list of warnings/errors (empty = OK)."""
    issues: list[str] = []

    try:
        cfg = load_config()
    except Exception as e:
        return [f"Config load error: {e}"]

    # Check API keys
    if cfg.llm_base_url and cfg.llm_api_key:
        pass  # Unified proxy configured, per-provider keys not needed
    else:
        if not cfg.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set in .env")
        if not cfg.openai_api_key:
            issues.append("OPENAI_API_KEY not set in .env (needed for IdeationAgent, ReviewerAgent)")
        if not cfg.google_api_key:
            issues.append("GOOGLE_API_KEY not set in .env (needed for AnalystAgent)")

    # Check agent references in pipeline
    for stage in cfg.pipeline.stages:
        if stage.agent not in cfg.agents:
            issues.append(f"Pipeline stage '{stage.name}' references unknown agent '{stage.agent}'")

    # Check SSH config for experimenter
    if not cfg.ssh.host:
        issues.append("SSH not configured (configs/ssh.yaml missing or empty) — ExperimenterAgent will fail")

    return issues
