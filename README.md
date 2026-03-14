# AIRS — AI Research System

End-to-end autonomous research pipeline: from research idea to peer-reviewed paper.

## What it does

AIRS drives a research idea through 8 sequential stages, each handled by a specialized LLM agent:

1. **Literature Review** — Claude opus-4.6 surveys arXiv + web, produces `literature/survey.md`
2. **Ideation** — OpenAI o3 generates novel ideas, writes `ideation/ideas.md` + `ideation/selected_idea.md`
3. **Technical Design** — Claude opus-4.6 creates implementation-ready `design/design.md`
4. **Code Implementation** — Claude sonnet-4.6 writes + tests code in `code/experiments/`
5. **Experiments** — Claude sonnet-4.6 runs code on remote GPU via SSH, saves `experiments/results.md`
6. **Analysis** — Gemini 2.5 Pro analyzes results, generates `analysis/analysis.md` + figures
7. **Paper Writing** — Claude opus-4.6 writes full paper + compiles LaTeX to PDF
8. **Peer Review** — GPT-4o reviews the paper, writes `review/review.md`

Agents communicate exclusively via files in `workspace/{project}/`.

## Setup

```bash
pip install -e .
cp .env.example .env
# Fill in API keys in .env

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

# Browse results in browser
airs serve --project my_research --port 8080

# Validate config
airs config validate

# List all projects
airs list
```

## Project Structure

```
AIRS/
├── airs/
│   ├── main.py              # CLI (Typer)
│   ├── orchestrator.py      # Sequential pipeline state machine
│   ├── agents/              # 8 agent implementations
│   ├── llm/                 # LLM providers (Claude, OpenAI, Gemini)
│   ├── skills/              # Tools agents can use
│   ├── ssh/                 # Remote execution via paramiko
│   ├── workspace/           # Filesystem layer
│   ├── config/              # Config loading
│   └── web/                 # FastAPI web viewer
├── configs/
│   ├── pipeline.yaml        # Stage order and settings
│   ├── agents.yaml          # Per-agent config (model, skills, prompts)
│   └── ssh.yaml.example     # SSH template
└── workspace/               # Runtime (gitignored)
    └── {project}/
        ├── project.yaml     # Pipeline state
        ├── literature/
        ├── ideation/
        ├── design/
        ├── code/
        ├── experiments/
        ├── analysis/
        ├── paper/
        └── review/
```

## Configuration

**`.env`** — API keys:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

**`configs/agents.yaml`** — Customize per-agent models, prompts, and skills.

**`configs/pipeline.yaml`** — Change stage order, auto-proceed settings.

**`configs/ssh.yaml`** — Remote server for experiment stage.

## User Checkpoints

The pipeline pauses for user input at:
- After **Ideation**: review and potentially edit the selected idea before proceeding
- After **Review**: decide whether to revise the paper or accept

Use `--auto` flag to skip all checkpoints (for fully autonomous runs).
