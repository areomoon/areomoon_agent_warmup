"""
Trajectory schema for ReAct / agent runs.

Single source of truth for:
  - Debug (replay a run from JSONL without spending tokens)
  - Eval (process metrics: n_steps, tool usage, redundancy)
  - Fine-tune data (Week 5: trajectory → SFT messages)

Usage:
    traj = Trajectory(question="...", model="claude-haiku-4-5", temperature=0.0)
    traj.append_step(role="thought_action", content="...", tokens_in=100, tokens_out=50)
    traj.append_step(role="observation", tool_name="convert_units", tool_result="973.15 K")
    save(traj, "trajectories.jsonl")
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


StepRole = Literal["thought_action", "observation", "final", "error"]
RunStatus = Literal["success", "max_steps", "parse_error", "tool_error"]


@dataclass
class Step:
    step_id: int
    role: StepRole
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class Trajectory:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    final_answer: str | None = None
    status: RunStatus = "success"
    steps: list[Step] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: int = 0
    model: str = ""
    temperature: float = 0.0
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def append_step(self, role: StepRole, **kwargs) -> Step:
        step = Step(step_id=len(self.steps), role=role, **kwargs)
        self.steps.append(step)
        if step.tokens_in:
            self.total_tokens += step.tokens_in
        if step.tokens_out:
            self.total_tokens += step.tokens_out
        if step.latency_ms:
            self.total_latency_ms += step.latency_ms
        return step

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def save(traj: Trajectory, path: str | Path = "trajectories.jsonl") -> None:
    """Append a trajectory to a JSONL file. Creates the file if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(traj.to_jsonl_line() + "\n")


def replay(traj: Trajectory) -> None:
    """Pretty-print a trajectory for debugging. Costs zero tokens."""
    print(f"\n=== Run {traj.run_id} | model={traj.model} | t={traj.temperature} ===")
    print(f"Q: {traj.question[:200]}{'...' if len(traj.question) > 200 else ''}")
    for step in traj.steps:
        if step.role == "thought_action":
            print(f"\n[{step.step_id}] AGENT ({step.tokens_out}tok, {step.latency_ms}ms):")
            print(f"  {step.content[:400]}{'...' if len(step.content) > 400 else ''}")
        elif step.role == "observation":
            preview = str(step.tool_result)[:200]
            print(f"\n[{step.step_id}] OBS  ({step.tool_name}): {preview}")
        elif step.role == "final":
            print(f"\n[{step.step_id}] FINAL: {step.content[:300]}")
        elif step.role == "error":
            print(f"\n[{step.step_id}] ERROR: {step.content}")
    print(f"\nStatus: {traj.status} | total={traj.total_tokens}tok, {traj.total_latency_ms}ms")
