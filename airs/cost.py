"""Cost tracking: pricing table and token cost calculator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Pricing per million tokens (RMB) from dmxapi
# Format: model_name -> {input, output}
PRICING_TABLE: dict[str, dict[str, float]] = {
    # Claude Code models (3.4折)
    "claude-opus-4-6-cc":           {"input": 12.41, "output": 62.05},
    "claude-sonnet-4-6-cc":         {"input": 7.5,   "output": 37.5},
    "claude-haiku-4-5-20251001-cc": {"input": 2.5,   "output": 12.5},
    # Claude models (6.8折)
    "claude-opus-4-6":              {"input": 24.82, "output": 124.1},
    "claude-opus-4-6-thinking":     {"input": 24.82, "output": 124.1},
    "claude-sonnet-4-6":            {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6-thinking":   {"input": 15.0,  "output": 75.0},
    "claude-opus-4-5-20251101":     {"input": 25.0,  "output": 125.0},
    "claude-sonnet-4-5-20250929":   {"input": 15.0,  "output": 75.0},
    "claude-haiku-4-5-20251001":    {"input": 5.0,   "output": 25.0},
    # OpenAI models
    "gpt-5.4":                      {"input": 12.5,  "output": 75.0},
    "gpt-5.3-chat":                 {"input": 8.75,  "output": 70.0},
    "gpt-5.3-codex":                {"input": 4.344, "output": 34.748},
    "gpt-5.3-codex-spark":          {"input": 4.344, "output": 34.748},
    "gpt-5.2":                      {"input": 8.75,  "output": 70.0},
    "gpt-5.1":                      {"input": 6.25,  "output": 50.0},
    "gpt-5.1-codex":                {"input": 6.25,  "output": 50.0},
    "gpt-5.1-codex-mini":           {"input": 1.241, "output": 9.928},
    "gpt-5":                        {"input": 6.25,  "output": 50.0},
    "gpt-5-mini":                   {"input": 1.25,  "output": 10.0},
    "gpt-5-nano":                   {"input": 0.25,  "output": 2.0},
    "gpt-5-codex":                  {"input": 3.65,  "output": 29.2},
    "o3":                           {"input": 10.0,  "output": 40.0},
    "o3-mini":                      {"input": 5.5,   "output": 22.0},
    "o4-mini":                      {"input": 5.5,   "output": 22.0},
    "gpt-4.1":                      {"input": 10.0,  "output": 40.0},
    "gpt-4.1-mini":                 {"input": 2.0,   "output": 8.0},
    "gpt-4.1-nano":                 {"input": 0.5,   "output": 2.0},
    "gpt-4o":                       {"input": 12.5,  "output": 50.0},
    "gpt-4o-mini":                  {"input": 0.75,  "output": 3.0},
    # Gemini models
    "gemini-3.1-pro-preview":           {"input": 10.0,  "output": 60.0},
    "gemini-3.1-pro-preview-thinking":  {"input": 10.0,  "output": 60.0},
    "gemini-3.1-flash-lite-preview":    {"input": 1.25,  "output": 7.5},
    "gemini-3-pro-preview":             {"input": 10.0,  "output": 60.0},
    "gemini-3-pro-preview-thinking":    {"input": 10.0,  "output": 60.0},
    "gemini-3-flash-preview":           {"input": 2.5,   "output": 15.0},
    "gemini-3-flash-preview-thinking":  {"input": 2.5,   "output": 15.0},
    "gemini-2.5-pro":                   {"input": 7.5,   "output": 37.5},
    "gemini-2.5-flash":                 {"input": 0.5,   "output": 2.5},
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = {"input": 10.0, "output": 50.0}


def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model. Falls back to default if unknown."""
    return PRICING_TABLE.get(model, _DEFAULT_PRICING)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in RMB for a single LLM call."""
    pricing = get_pricing(model)
    cost = (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
    return round(cost, 6)


@dataclass
class CallRecord:
    """Record of a single LLM API call."""
    timestamp: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_rmb: float
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class CostTracker:
    """Tracks costs across the entire pipeline."""
    records: list[CallRecord] = field(default_factory=list)

    def add(self, agent: str, model: str, input_tokens: int, output_tokens: int,
            tool_calls: list[str] | None = None) -> CallRecord:
        cost = calculate_cost(model, input_tokens, output_tokens)
        record = CallRecord(
            timestamp=datetime.utcnow().isoformat(),
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_rmb=cost,
            tool_calls=tool_calls or [],
        )
        self.records.append(record)
        return record

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost_rmb for r in self.records), 4)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def cost_by_agent(self) -> dict[str, float]:
        by_agent: dict[str, float] = {}
        for r in self.records:
            by_agent[r.agent] = round(by_agent.get(r.agent, 0) + r.cost_rmb, 4)
        return by_agent

    def cost_by_model(self) -> dict[str, float]:
        by_model: dict[str, float] = {}
        for r in self.records:
            by_model[r.model] = round(by_model.get(r.model, 0) + r.cost_rmb, 4)
        return by_model

    def summary(self) -> dict:
        return {
            "total_cost_rmb": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": len(self.records),
            "by_agent": self.cost_by_agent(),
            "by_model": self.cost_by_model(),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "agent": r.agent,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_rmb": r.cost_rmb,
                    "tool_calls": r.tool_calls,
                }
                for r in self.records
            ],
        }

    def save(self, workspace) -> None:
        """Save cost data to workspace."""
        data = self.summary()
        workspace.write("logs/costs.json", json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, workspace) -> "CostTracker":
        """Load cost data from workspace."""
        tracker = cls()
        try:
            raw = workspace.read("logs/costs.json")
            data = json.loads(raw)
            for r in data.get("records", []):
                tracker.records.append(CallRecord(**r))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return tracker
