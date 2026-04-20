"""
ReAct Pattern: Reasoning + Acting for Scientific Information Retrieval
=====================================================================

ReAct interleaves chain-of-thought reasoning with action execution.
The agent thinks about what to do, takes an action (calls a tool),
observes the result, and repeats until it reaches a final answer.

Format:
  Thought: [reasoning about what to do next]
  Action: [tool_name(args)]
  Observation: [result from tool]
  ... (repeat)
  Final Answer: [answer]

For scientific paper agents: the "tools" are search, extraction, and
calculation functions. The reasoning traces help debug failures and
accumulate as examples for the ACE Reflector role.

Note: this is the "prompt-as-protocol" version of ReAct — we parse tool
calls from free-form text using stop sequences. Production agents should
use native tool use (client.messages.create(tools=[...])), which is what
LangGraph/Anthropic Agent SDK builds on top of.

References:
  - ReAct: Synergizing Reasoning and Acting (Yao et al., 2022): https://arxiv.org/abs/2210.03629
  - Anthropic Tool Use: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
"""

from __future__ import annotations  # allow `tuple[...] | None` on Python 3.9

import os
import json
from typing import Callable
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"

# ── Simulated tool functions ──────────────────────────────────────────────────

def search_paper_database(query: str) -> str:
    """Simulate a materials paper database search."""
    # In production: call Materials Project API, CrossRef, or Semantic Scholar
    mock_results = {
        "perovskite conductivity": "Paper: 'Ionic conductivity in La₁₋ₓSrₓMnO₃' — reports σ = 10² S/cm at 800°C",
        "LSMO thin film": "Paper: 'LSMO PLD deposition' — substrate temp range 600–750°C optimal",
        "metal insulator transition": "Paper: 'MIT in manganites' — T_MI depends on Sr doping fraction",
    }
    for key in mock_results:
        if key.lower() in query.lower():
            return mock_results[key]
    return "No relevant papers found for query: " + query


def extract_parameter(text: str, parameter: str) -> str:
    """Simulate parameter extraction from a text snippet."""
    mock_params = {
        "temperature": "700°C (substrate), 800°C (annealing)",
        "pressure": "200 mTorr oxygen partial pressure",
        "thickness": "~50 nm (X-ray reflectometry confirmed)",
    }
    for key in mock_params:
        if key.lower() in parameter.lower():
            return mock_params[key]
    return f"Parameter '{parameter}' not found in provided text."


def calculate_unit_conversion(value: str, from_unit: str, to_unit: str) -> str:
    """Simulate unit conversion."""
    conversions = {
        ("mTorr", "Pa"): lambda v: f"{float(v) * 0.133322:.4f} Pa",
        ("°C", "K"): lambda v: f"{float(v) + 273.15:.2f} K",
        ("nm", "Å"): lambda v: f"{float(v) * 10:.1f} Å",
    }
    key = (from_unit, to_unit)
    if key in conversions:
        try:
            return conversions[key](value)
        except ValueError:
            return f"Could not convert: invalid value '{value}'"
    return f"Conversion from {from_unit} to {to_unit} not implemented"


TOOLS: dict[str, tuple[Callable, str]] = {
    "search_papers": (
        search_paper_database,
        "search_papers(query: str) -> Search materials science papers. Use for background knowledge.",
    ),
    "extract_parameter": (
        extract_parameter,
        "extract_parameter(text: str, parameter: str) -> Extract a specific parameter from text.",
    ),
    "convert_units": (
        calculate_unit_conversion,
        "convert_units(value: str, from_unit: str, to_unit: str) -> Convert between units.",
    ),
}


# ── Simple ReAct loop (without a framework, for clarity) ─────────────────────

REACT_SYSTEM_PROMPT = """You are a scientific data extraction agent. Use the following tools to answer questions:

{tool_descriptions}

Use this format EXACTLY:
Thought: [your reasoning]
Action: tool_name(arg1, arg2, ...)
Observation: [tool result will be inserted here]

Repeat Thought/Action/Observation as needed, then:
Final Answer: [your final answer in JSON format]"""


def parse_action(action_line: str) -> tuple[str, list[str]] | None:
    """Parse 'tool_name(arg1, arg2)' into (tool_name, [args])."""
    action_line = action_line.strip()
    if "(" not in action_line:
        return None
    tool_name = action_line[:action_line.index("(")]
    args_str = action_line[action_line.index("(") + 1:action_line.rindex(")")]
    args = [a.strip().strip("'\"") for a in args_str.split(",") if a.strip()]
    return tool_name.strip(), args


def react_agent(question: str, client, max_steps: int = 6) -> str:
    """
    Simple ReAct loop: Thought → Action → Observation → repeat.
    Returns the final answer string.
    """
    tool_descriptions = "\n".join(f"  {name}: {desc}" for name, (_, desc) in TOOLS.items())
    system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    # Anthropic: system prompt is a top-level parameter, not a message.
    messages = [{"role": "user", "content": question}]

    trajectory = []

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=messages,
            stop_sequences=["Observation:"],  # stop before the model writes its own observation
        )
        agent_text = next(b.text for b in response.content if b.type == "text")
        messages.append({"role": "assistant", "content": agent_text})
        trajectory.append({"step": step, "agent": agent_text})

        # Check if done
        if "Final Answer:" in agent_text:
            final = agent_text.split("Final Answer:")[-1].strip()
            print(f"\n✅ Final Answer after {step + 1} steps:\n{final}")
            print("\n--- Reasoning Trace ---")
            for t in trajectory:
                if "agent" in t:
                    print(f"\nStep {t['step'] + 1} (agent):\n{t['agent']}")
                if "observation" in t:
                    print(f"\nStep {t['step'] + 1} (obs): {t['observation']}")
            return final

        # Extract and execute action
        if "Action:" in agent_text:
            action_line = agent_text.split("Action:")[-1].strip().split("\n")[0]
            parsed = parse_action(action_line)
            if parsed:
                tool_name, args = parsed
                if tool_name in TOOLS:
                    fn, _ = TOOLS[tool_name]
                    try:
                        observation = fn(*args)
                    except Exception as e:
                        observation = f"Tool error: {e}"
                else:
                    observation = f"Unknown tool: {tool_name}"
            else:
                observation = "Could not parse action"

            observation_msg = f"Observation: {observation}"
            messages.append({"role": "user", "content": observation_msg})
            trajectory.append({"step": step, "observation": observation})

    return "Max steps reached without final answer"


def run_example():
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"Anthropic client init failed: {e}\nSet ANTHROPIC_API_KEY in .env")
        return

    question = """From this text, extract all experimental parameters and convert
    oxygen pressure to Pa and substrate temperature to K:

    'LSMO thin films were deposited via PLD at 700°C substrate temperature under
    200 mTorr O₂. Film thickness was ~50 nm.'

    Return as JSON."""

    print(f"Model: {MODEL}")
    print("Question:", question)
    react_agent(question, client)


if __name__ == "__main__":
    run_example()

# TODO: Replace prompt-parsed ReAct with Anthropic native tool use (client.messages.create(tools=[...]))
# TODO: Replace mock tools with real Materials Project API calls
# TODO: Add LangGraph version (Week 3) and compare to this minimal loop
# TODO: Save reasoning traces as training data for fine-tuning
# TODO: Add Reflector step: analyze trace quality after completion
