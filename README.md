# AIRS — AI Research System

End-to-end autonomous research pipeline: from research idea to peer-reviewed paper.

## What it does

AIRS drives a research idea through 8 sequential stages, each handled by a specialized LLM agent:

| Stage | Agent | Model (Fast) | Fallback | Task |
|-------|-------|-------------|----------|------|
| 1. Literature Review | LiteratureAgent | `gemini-3-flash-preview` | `gemini-3-pro-preview` | Survey arXiv + web |
| 2. Ideation | IdeationAgent | `o3` | — | Generate novel research ideas |
| 3. Technical Design | DesignAgent | `claude-sonnet-4-6` | `claude-opus-4-6` | Implementation-ready design |
| 4. Code Implementation | CoderAgent | `claude-sonnet-4-6-cc` | `claude-opus-4-6-cc` | Write + test code |
| 5. Experiments | ExperimenterAgent | `claude-sonnet-4-6-cc` | `claude-opus-4-6-cc` | Run on remote GPU via SSH |
| 6. Analysis | AnalystAgent | `gemini-3-flash-preview` | `gemini-3-pro-preview` | Statistical analysis + figures |
| 7. Paper Writing | WriterAgent | `gpt-5.3-chat` | — | Method + Experiments sections |
| 8. Cross-Model Review | 3 ReviewerAgents | `claude-sonnet-4-6` / `gpt-5.3-chat` / `gemini-3-pro-preview` | — | Independent reviews from 3 models |

### Design Principles

- **Fast-first execution**: All agents use cost-effective fast models by default. Only escalate to expensive models when the fast model fails.
- **No fabricated data**: Agents are strictly prohibited from inventing experimental results. Missing data uses `[TODO]` placeholders.
- **Cross-model review**: Three independent reviewers (Claude, GPT, Gemini) provide diverse perspectives.
- **Complexity-aware design**: The design agent must provide Big-O analysis for every component.
- **Resume on interrupt**: Pipeline state is persisted in `project.yaml`. Restarting always checks progress first, never resets.

Agents communicate exclusively via files in `workspace/{project}/`.

## Setup

```bash
# Create conda environment (recommended)
conda create -n airs python=3.11 -y
conda activate airs
pip install -e .

# Configure API (unified proxy or per-provider keys)
cp .env.example .env
# Edit .env:
#   LLM_BASE_URL=https://your-proxy.com   # Unified proxy (optional)
#   LLM_API_KEY=sk-...                    # Unified key
#   ANTHROPIC_API_KEY=...                 # Or per-provider keys
#   OPENAI_API_KEY=...
#   GOOGLE_API_KEY=...

# Optional: configure SSH for remote experiments
cp configs/ssh.yaml.example configs/ssh.yaml
# Edit configs/ssh.yaml with your server details
```

## Usage

```bash
# Run full pipeline
airs run --topic "Efficient attention for long-context transformers" --project my_research

# Resume from a stage
airs resume --project my_research --from design

# Run a single stage
airs agent literature --project my_research --topic "linear attention"

# Check status
airs status --project my_research

# Browse results in browser (real-time cost tracking included)
airs serve --project my_research --port 8080

# Validate config
airs config validate

# List all projects
airs list
```

## Model Strategy

### Fast-First with Auto-Escalation

Code and search agents start with **fast, cheap models**. If the fast model fails (errors, max iterations), the orchestrator automatically retries with the **fallback model**:

```
claude-sonnet-4-6-cc (¥7.5/37.5 per M tokens)
        ↓ on failure
claude-opus-4-6-cc (¥12.41/62.05 per M tokens)
```

### Cost Optimization

| Model | Input ¥/M | Output ¥/M | Use Case |
|-------|-----------|------------|----------|
| `gemini-3-flash-preview` | 2.5 | 15.0 | Literature search, analysis |
| `claude-sonnet-4-6-cc` | 7.5 | 37.5 | Code writing (fast) |
| `gpt-5.3-chat` | 8.75 | 70.0 | Paper writing, review |
| `o3` | 10.0 | 40.0 | Deep reasoning (ideation) |
| `gemini-3-pro-preview` | 10.0 | 60.0 | Review, fallback analysis |
| `claude-sonnet-4-6` | 15.0 | 75.0 | Design, review |
| `claude-opus-4-6-cc` | 12.41 | 62.05 | Code fallback |
| `claude-opus-4-6` | 24.82 | 124.1 | Design fallback |

Real-time cost tracking is available in the web viewer (`airs serve`).

## Project Structure

```
AIRS/
├── airs/
│   ├── main.py              # CLI (Typer)
│   ├── orchestrator.py      # Pipeline state machine (fallback + multi-agent)
│   ├── cost.py              # Cost tracking with dmxapi pricing
│   ├── agents/              # 8 agent implementations + BaseAgent
│   ├── llm/                 # LLM providers (OpenAI-compatible proxy)
│   ├── skills/              # Tools: arxiv, web_search, file_read/write, ssh_exec, python_exec
│   ├── ssh/                 # Remote execution via paramiko
│   ├── workspace/           # Filesystem layer
│   ├── config/              # Config loading (Pydantic)
│   └── web/                 # FastAPI web viewer with cost panel
├── configs/
│   ├── pipeline.yaml        # Stage order, multi-agent stages, auto-proceed
│   ├── agents.yaml          # Per-agent: model, fallback_model, skills, prompts
│   └── ssh.yaml.example     # SSH template
├── demos/
│   └── traffic_pems/        # Example pipeline output (PeMS traffic prediction)
└── workspace/               # Runtime data (gitignored)
    └── {project}/
        ├── project.yaml     # Pipeline state
        ├── logs/costs.json  # Detailed cost records
        └── {stage}/         # Stage outputs
```

## Configuration

**`.env`** — API keys (never committed):
```
LLM_BASE_URL=https://your-proxy.com/v1
LLM_API_KEY=sk-...
```

**`configs/agents.yaml`** — Per-agent: model, fallback_model, skills, system prompt.

**`configs/pipeline.yaml`** — Stage order, multi-agent stages (cross-review), auto-proceed.

**`configs/ssh.yaml`** — Remote server for experiments.

## User Checkpoints

The pipeline pauses for user input at:
- After **Ideation**: review and potentially edit the selected idea
- After **Cross-Model Review**: decide whether to revise or accept (3 independent reviews)

Use `--auto` flag to skip all checkpoints (for fully autonomous runs).

## Demo

See `demos/traffic_pems/` for a complete pipeline output on the topic "Traffic flow prediction on PeMS dataset". This demo includes all 8 stages from literature survey through peer review.
