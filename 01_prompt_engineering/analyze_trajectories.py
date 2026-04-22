"""
Analyze trajectories.jsonl produced by react_pattern.py.

Loads every trajectory in the JSONL file and reports per-temperature
metrics: success rate, step count, token usage, latency, redundant calls.

Stdlib-only (no pandas). The notebook can show a richer pandas view if you
have it installed.

Usage:
    python analyze_trajectories.py
    python analyze_trajectories.py --path other.jsonl
    python analyze_trajectories.py --filter sweep_trial   # only sweep runs
    python analyze_trajectories.py --show-failures
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}. Run react_pattern.py or run_temperature_sweep first.")
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def step_metrics(traj: dict) -> dict:
    """Per-trajectory derived metrics."""
    obs_steps = [s for s in traj["steps"] if s["role"] == "observation"]
    tool_calls = [s["tool_name"] for s in obs_steps if s.get("tool_name")]
    tool_arg_pairs = [
        (s["tool_name"], json.dumps(s.get("tool_args"), sort_keys=True))
        for s in obs_steps if s.get("tool_name")
    ]
    counts = Counter(tool_arg_pairs)
    redundant = sum(c - 1 for c in counts.values() if c > 1)

    return {
        "run_id": traj["run_id"],
        "temperature": traj["temperature"],
        "status": traj["status"],
        "n_steps": len(traj["steps"]),
        "n_actions": len(obs_steps),
        "tools_used": ",".join(sorted(set(tool_calls))) or "—",
        "redundant_calls": redundant,
        "total_tokens": traj["total_tokens"],
        "total_latency_ms": traj["total_latency_ms"],
        "has_final": traj["final_answer"] is not None,
    }


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def _std(xs):
    return round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate per-temperature summary as list of dicts (no pandas)."""
    by_temp: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_temp[r["temperature"]].append(r)

    out = []
    for t in sorted(by_temp.keys()):
        runs = by_temp[t]
        steps = [r["n_steps"] for r in runs]
        out.append({
            "temperature": t,
            "n_runs": len(runs),
            "success_rate": round(sum(r["status"] == "success" for r in runs) / len(runs), 2),
            "avg_steps": _mean(steps),
            "std_steps": _std(steps),
            "avg_actions": _mean([r["n_actions"] for r in runs]),
            "avg_redundant": _mean([r["redundant_calls"] for r in runs]),
            "avg_tokens": int(_mean([r["total_tokens"] for r in runs])),
            "avg_latency_ms": int(_mean([r["total_latency_ms"] for r in runs])),
        })
    return out


def _print_table(rows: list[dict], cols: list[str] | None = None) -> None:
    if not rows:
        print("(empty)")
        return
    cols = cols or list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="trajectories.jsonl")
    parser.add_argument("--filter", default=None,
                        help="Only include runs whose metadata contains this key (e.g. 'sweep_trial')")
    parser.add_argument("--show-failures", action="store_true",
                        help="Print run_ids of failed trajectories for debugging")
    args = parser.parse_args()

    trajs = load(args.path)
    if args.filter:
        trajs = [t for t in trajs if args.filter in (t.get("metadata") or {})]

    if not trajs:
        print("No trajectories matched.")
        return

    rows = [step_metrics(t) for t in trajs]

    print(f"\n=== {len(rows)} trajectories from {args.path} ===\n")
    print("Per-run table (last 20):")
    _print_table(rows[-20:], cols=["run_id", "temperature", "status", "n_steps",
                                    "n_actions", "redundant_calls", "total_tokens"])

    print("\n=== Summary by temperature ===\n")
    _print_table(summarize(rows))

    print("\n=== Status breakdown ===")
    status_counts = Counter(r["status"] for r in rows)
    for status, n in status_counts.most_common():
        print(f"  {status}: {n}")

    if args.show_failures:
        bad = [r for r in rows if r["status"] != "success"]
        if bad:
            print("\n=== Failed runs (use replay() in trajectory.py to inspect) ===")
            _print_table(bad, cols=["run_id", "temperature", "status", "n_steps"])


if __name__ == "__main__":
    main()
