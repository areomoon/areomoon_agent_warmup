"""
Multi-Agent Orchestration: Orchestrator + Specialists
=======================================================

Basic multi-agent pattern where an Orchestrator coordinates
specialist agents:
  - Extractor Agent: pulls raw parameters from text
  - Analyzer Agent:  validates and cross-checks extractions
  - Advisor Agent:   provides domain context and flags issues

This mirrors the MARS system's division of labor and the
Mailbox pattern from Claude Code's agentic harness.

For the materials science extraction job:
  Orchestrator → decides which specialists to call and in what order
  Extractor    → Generator role (raw extraction)
  Analyzer     → Reflector role (validation and cross-checking)
  Advisor      → domain knowledge (known material properties for sanity checks)

References:
  - MARS system (arXiv 2602.00169): https://arxiv.org/abs/2602.00169
  - LLM Powered Autonomous Agents (Lilian Weng): https://lilianweng.github.io/posts/2023-06-23-agent/
"""

import os
import sys
import json
import pathlib
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "resources"))
from anthropic_helpers import get_client, extract_text, parse_json, DEFAULT_MODEL

load_dotenv()

MODEL = DEFAULT_MODEL

SAMPLE_TEXT = """
BiFeO₃ (BFO) multiferroic thin films were grown by chemical vapor deposition (CVD)
on Pt/Ti/SiO₂/Si substrates at 600°C. The Bi:Fe precursor ratio was 1.2:1 (excess Bi
to compensate for Bi volatility). Films of 200 nm thickness were deposited over 4 hours.
Ferroelectric hysteresis loops showed remnant polarization Pr = 60 μC/cm² and coercive
field Ec = 200 kV/cm. Antiferromagnetic Néel temperature TN = 643 K was confirmed by
magnetic susceptibility measurements.
"""

KNOWN_MATERIALS_DB = {
    "BiFeO₃": {
        "typical_Pr_range_uC_cm2": (50, 100),
        "TN_K": 643,
        "typical_deposition_temp_C": (500, 700),
    }
}


@dataclass
class AgentMessage:
    sender: str
    content: str
    data: dict | None = None


def extractor_agent(text: str, client) -> AgentMessage:
    """Specialist: raw parameter extraction."""
    prompt = f"""You are a scientific parameter extraction specialist.
Extract ALL numerical parameters, conditions, and material properties from this text.
Be thorough — get every number, unit, and measurement.

Text: {text}

Return JSON with every extractable parameter.
Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    data = parse_json(extract_text(response))
    return AgentMessage(
        sender="extractor",
        content=f"Extracted {len(data)} parameters",
        data=data,
    )


def analyzer_agent(text: str, extraction: dict, client) -> AgentMessage:
    """Specialist: validation and consistency checking."""
    prompt = f"""You are a scientific data quality analyst.
Validate this extraction against the source text.

Source text: {text}

Extraction: {json.dumps(extraction, indent=2)}

Check:
1. Are all values consistent with the text?
2. Are units correct?
3. Are there any missed parameters?
4. Any physically unreasonable values?

Return JSON:
{{
  "validation_status": "pass|fail|warning",
  "issues": [{{"field": "...", "issue": "...", "severity": "error|warning"}}],
  "missed_parameters": [{{"name": "...", "value": ..., "source_text": "..."}}],
  "corrected_extraction": {{...}}  // full corrected version
}}

Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    data = parse_json(extract_text(response))
    return AgentMessage(
        sender="analyzer",
        content=f"Validation: {data.get('validation_status', 'unknown')} — {len(data.get('issues', []))} issues",
        data=data,
    )


def advisor_agent(material: str, extraction: dict) -> AgentMessage:
    """Specialist: domain knowledge sanity check (no LLM call needed for known materials)."""
    issues = []

    if material in KNOWN_MATERIALS_DB:
        db = KNOWN_MATERIALS_DB[material]

        # Check Pr range
        pr = extraction.get("remnant_polarization_uC_cm2") or extraction.get("Pr_uC_cm2")
        if pr is not None:
            lo, hi = db["typical_Pr_range_uC_cm2"]
            if not (lo <= float(pr) <= hi):
                issues.append(f"Pr={pr} μC/cm² outside typical range [{lo}, {hi}] for {material}")

        # Check TN
        tn = extraction.get("Neel_temperature_K") or extraction.get("TN_K")
        if tn and abs(float(tn) - db["TN_K"]) > 10:
            issues.append(f"TN={tn} K deviates from known value {db['TN_K']} K for {material}")

    status = "FLAGGED" if issues else "OK"
    content = f"Domain check {status}" + (": " + "; ".join(issues) if issues else "")
    return AgentMessage(sender="advisor", content=content, data={"issues": issues})


def orchestrator(text: str, client) -> dict:
    """
    Orchestrator: coordinate specialist agents.
    Implements Plan → Delegate → Collect → Synthesize.
    """
    print("=== Orchestrator: Starting multi-agent extraction ===\n")

    # Step 1: Extract
    print("→ Calling Extractor Agent...")
    extractor_msg = extractor_agent(text, client)
    print(f"  ✅ {extractor_msg.content}")

    # Step 2: Analyze (Reflector role)
    print("\n→ Calling Analyzer Agent...")
    analyzer_msg = analyzer_agent(text, extractor_msg.data, client)
    print(f"  ✅ {analyzer_msg.content}")
    for issue in analyzer_msg.data.get("issues", []):
        print(f"     [{issue['severity']}] {issue['field']}: {issue['issue']}")

    # Use corrected extraction if available
    final_extraction = analyzer_msg.data.get("corrected_extraction") or extractor_msg.data

    # Step 3: Domain sanity check
    material = final_extraction.get("material", "")
    print(f"\n→ Calling Advisor Agent for {material}...")
    advisor_msg = advisor_agent(material, final_extraction)
    print(f"  ✅ {advisor_msg.content}")

    # Step 4: Synthesize final output
    result = {
        "extraction": final_extraction,
        "validation_status": analyzer_msg.data.get("validation_status"),
        "issues": analyzer_msg.data.get("issues", []) + [{"field": "domain", "issue": i, "severity": "warning"} for i in advisor_msg.data.get("issues", [])],
        "agent_trace": [
            {"agent": m.sender, "message": m.content}
            for m in [extractor_msg, analyzer_msg, advisor_msg]
        ],
    }

    print("\n=== Orchestrator: Done ===")
    return result


def run_example():
    try:
        client = get_client()
    except Exception as e:
        print(f"Anthropic client init failed: {e}\nSet ANTHROPIC_API_KEY in .env")
        return

    result = orchestrator(SAMPLE_TEXT, client)

    print("\n--- Final Result ---")
    print(json.dumps({
        "extraction": result["extraction"],
        "validation_status": result["validation_status"],
        "total_issues": len(result["issues"]),
    }, indent=2))


if __name__ == "__main__":
    run_example()

# TODO: Add LangGraph StateGraph for production orchestration
# TODO: Add Mailbox pattern: Extractor sends uncertain values to Orchestrator for review
# TODO: Add parallel fan-out: run Extractor on multiple paper sections simultaneously
# TODO: Connect Advisor to real Materials Project API
