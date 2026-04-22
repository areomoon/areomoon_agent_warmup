"""
Reflection Agent with LangGraph
================================

A LangGraph-based reflection agent that:
1. Generates an initial extraction
2. Critiques its own output
3. Revises based on the critique
4. Repeats until quality threshold is met or max iterations reached

This is the LangGraph production-ready version of the generator_reflector.py
bare-bones implementation. Uses StateGraph for explicit state management.

References:
  - LangGraph Reflection: https://blog.langchain.com/reflection-agents/
  - Reflexion (arXiv 2303.11366): https://arxiv.org/abs/2303.11366
  - LangGraph docs: https://langchain-ai.github.io/langgraph/
"""

import os
import sys
import json
import pathlib
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "resources"))
from anthropic_helpers import get_client, extract_text, parse_json, DEFAULT_MODEL

load_dotenv()

MODEL = DEFAULT_MODEL


class ExtractionState(TypedDict):
    """State passed between nodes in the LangGraph."""
    text: str                          # Input paper text
    extraction: dict                   # Current extraction result
    critique: str                      # Reflector's critique
    revision_history: Annotated[list, operator.add]  # All revisions
    iteration: int                     # Current iteration count
    quality_score: float               # Self-assessed quality 0-1
    done: bool                         # Whether to stop


def extraction_node(state: ExtractionState, client) -> ExtractionState:
    """Generator node: extract or revise based on prior critique."""
    prior_critique = state.get("critique", "")
    revision_context = ""
    if prior_critique:
        revision_context = f"\n\nPrior critique to address:\n{prior_critique}"

    prompt = f"""Extract experimental parameters from this scientific text as structured JSON.
{revision_context}

Text:
{state['text']}

Return JSON with all numerical parameters, units, and methods you can identify.
Include a "self_quality_score" field (0.0-1.0) for your confidence in the extraction.
Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json(extract_text(response))
    quality = float(result.pop("self_quality_score", 0.7))

    return {
        **state,
        "extraction": result,
        "quality_score": quality,
        "iteration": state.get("iteration", 0) + 1,
        "revision_history": [{"iteration": state.get("iteration", 0) + 1, "extraction": result}],
    }


def reflection_node(state: ExtractionState, client) -> ExtractionState:
    """Reflector node: critique the extraction."""
    prompt = f"""You are a critical reviewer of scientific data extraction.

Original text:
{state['text']}

Current extraction:
{json.dumps(state['extraction'], indent=2)}

Be a tough critic. Check:
1. Are all numerical values correct with proper units?
2. Are there values in the text that were missed?
3. Are there any unit conversion errors (e.g., mTorr vs Pa)?
4. Are chemical formulas correctly transcribed?
5. Is scientific notation handled correctly (e.g., 5×10⁻⁷)?

Return JSON:
{{
  "critique": "specific list of issues found",
  "needs_revision": true/false,
  "quality_score": 0.0-1.0,
  "critical_errors": ["list of must-fix errors"]
}}

Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json(extract_text(response))

    return {
        **state,
        "critique": result.get("critique", ""),
        "quality_score": float(result.get("quality_score", 0.7)),
        "done": not result.get("needs_revision", True),
    }


def should_continue(state: ExtractionState) -> str:
    """Decide whether to revise or stop."""
    if state.get("done", False):
        return "done"
    if state.get("iteration", 0) >= 3:
        return "done"
    if state.get("quality_score", 0) >= 0.9:
        return "done"
    return "reflect"


def build_reflection_graph(client):
    """Build and compile the LangGraph reflection agent."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("Install: pip install langgraph")
        return None

    graph = StateGraph(ExtractionState)

    # Add nodes
    graph.add_node("extract", lambda s: extraction_node(s, client))
    graph.add_node("reflect", lambda s: reflection_node(s, client))

    # Entry point
    graph.set_entry_point("extract")

    # Edges
    graph.add_edge("extract", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"reflect": "extract", "done": END},
    )

    return graph.compile()


def run_example():
    try:
        client = get_client()
    except Exception as e:
        print(f"Anthropic client init failed: {e}\nSet ANTHROPIC_API_KEY in .env")
        return

    app = build_reflection_graph(client)
    if not app:
        return

    sample_text = """
    TiO₂ nanoparticles were synthesized by sol-gel method. Titanium tetraisopropoxide
    (TTIP) was added dropwise to a 0.1 M HNO₃ solution under vigorous stirring at 5°C.
    The resulting sol was aged at 60°C for 24 hours, then dried at 120°C for 8 hours.
    Calcination at 450°C for 2 hours in air yielded anatase-phase TiO₂. BET surface
    area: 85 m²/g. Average crystallite size from XRD (Scherrer equation): 12 nm.
    """

    print("Running LangGraph Reflection Agent...")
    initial_state: ExtractionState = {
        "text": sample_text,
        "extraction": {},
        "critique": "",
        "revision_history": [],
        "iteration": 0,
        "quality_score": 0.0,
        "done": False,
    }

    final_state = app.invoke(initial_state)

    print(f"\n✅ Completed in {final_state['iteration']} iteration(s)")
    print(f"Final quality score: {final_state['quality_score']:.2f}")
    print("\n--- Final Extraction ---")
    print(json.dumps(final_state["extraction"], indent=2))

    if len(final_state["revision_history"]) > 1:
        print(f"\n--- Made {len(final_state['revision_history'])} revisions ---")


if __name__ == "__main__":
    run_example()

# TODO: Add memory: save lessons from reflection as playbook rules
# TODO: Add human-in-the-loop: pause on low quality_score for human review
# TODO: Add parallel reflection: run multiple reflectors and take consensus
