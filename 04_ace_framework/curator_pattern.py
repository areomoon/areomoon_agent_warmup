"""
ACE Curator: Delta Update Playbook Management
=============================================

The Curator is the third role in ACE's Generator-Reflector-Curator (GRC) architecture.
It receives lessons from the Reflector and decides:
  1. Should this be a new rule (append)?
  2. Does it refine an existing rule (merge)?
  3. Does it contradict an existing rule (update/deprecate)?
  4. Is it redundant (skip)?

The Curator is responsible for playbook quality over time.
Without a Curator, the playbook grows unbounded and degrades.

Materials science four-layer playbook structure:
  Layer 1: design_rules        — what materials/compositions to try
  Layer 2: synthesis_protocols — how to make them
  Layer 3: characterization    — how to interpret measurements
  Layer 4: extraction          — how to parse papers accurately

References:
  - ACE Paper (arXiv 2510.04618): https://arxiv.org/abs/2510.04618
  - ACE GitHub: https://github.com/ace-agent/ace
  - ACE community playbook: https://github.com/jmanhype/ace-playbook
"""

import os
import json
from dotenv import load_dotenv
from playbook_evolution import (
    Playbook, PlaybookRule,
    load_playbook, save_playbook,
    append_rule, merge_rule, deduplicate_playbook,
    PLAYBOOK_PATH,
)

load_dotenv()

MATERIAL_LAYERS = ["design_rules", "synthesis_protocols", "characterization", "extraction"]


def curator_agent(
    lessons: list[dict],
    playbook: Playbook,
    client,
) -> tuple[int, int, int]:
    """
    Process lessons from the Reflector and update the playbook.

    Args:
        lessons: List of lesson dicts from Reflector (field, error_type, lesson, confidence_boost)
        playbook: Current playbook to update
        client: OpenAI client for semantic similarity checks

    Returns:
        (appended, merged, skipped) counts
    """
    appended = merged = skipped = 0

    for lesson in lessons:
        rule_text = lesson.get("lesson", "").strip()
        if not rule_text:
            skipped += 1
            continue

        confidence = min(0.5 + lesson.get("confidence_boost", 0.2), 0.95)
        source = f"{lesson.get('error_type', 'observation')} from extraction"

        # Classify into playbook layer
        layer = classify_layer(rule_text, client)

        # Check for semantic duplicate in the same layer
        layer_rules = [r for r in playbook.rules if r.layer == layer and not r.deprecated]
        duplicate = find_semantic_duplicate(rule_text, layer_rules, client)

        if duplicate:
            merge_rule(duplicate, rule_text, confidence)
            merged += 1
            print(f"  MERGE [{layer}]: confidence updated to {duplicate.confidence:.0%}")
        else:
            append_rule(playbook, layer, rule_text, source, confidence)
            appended += 1
            print(f"  APPEND [{layer}] [{confidence:.0%}]: {rule_text[:70]}...")

    # Periodic deduplication (every 20 rules)
    active = len([r for r in playbook.rules if not r.deprecated])
    if active > 0 and active % 20 == 0:
        print("\nRunning periodic deduplication...")
        deduplicate_playbook(playbook, client)

    return appended, merged, skipped


def classify_layer(rule_text: str, client) -> str:
    """Classify a rule into one of the four playbook layers."""
    if not os.getenv("OPENAI_API_KEY"):
        # Fallback: keyword-based classification
        text_lower = rule_text.lower()
        if any(w in text_lower for w in ["extract", "parse", "table", "caption", "unit", "notation"]):
            return "extraction"
        if any(w in text_lower for w in ["raman", "xrd", "tem", "sem", "spectrum", "characteriz"]):
            return "characterization"
        if any(w in text_lower for w in ["synthesis", "sinter", "deposition", "calcin", "anneal", "temperature"]):
            return "synthesis_protocols"
        return "design_rules"

    try:
        from openai import OpenAI
        oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Classify this materials science rule into exactly one category:
- design_rules: about what materials/compositions to use
- synthesis_protocols: about how to make/process materials
- characterization: about how to measure/interpret material properties
- extraction: about how to correctly parse scientific text/papers

Rule: "{rule_text}"

Return JSON: {{"layer": "one of the four categories"}}"""

        response = oai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        layer = result.get("layer", "extraction")
        if layer in MATERIAL_LAYERS:
            return layer
    except Exception:
        pass

    return "extraction"


def find_semantic_duplicate(new_rule: str, existing_rules: list[PlaybookRule], client) -> PlaybookRule | None:
    """Find semantically equivalent rule using LLM comparison."""
    if not existing_rules or not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI
        oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        existing_texts = {i: r.rule for i, r in enumerate(existing_rules)}
        prompt = f"""Is this new rule semantically equivalent (same meaning) to any existing rule?

New rule: "{new_rule}"

Existing rules:
{json.dumps(existing_texts, indent=2)}

Return JSON: {{"is_duplicate": true/false, "duplicate_index": null_or_int}}
Only mark as duplicate if they convey the same actionable instruction."""

        response = oai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("is_duplicate"):
            idx = result.get("duplicate_index")
            if idx is not None and 0 <= idx < len(existing_rules):
                return existing_rules[idx]
    except Exception:
        pass
    return None


def run_full_grc_loop_demo():
    """
    Demonstrate the complete Generator-Reflector-Curator loop
    with playbook persistence.
    """
    print("=== Full GRC Loop Demo ===\n")

    playbook = load_playbook()
    print(f"Loaded playbook with {len(playbook.rules)} existing rules\n")

    # Simulate Reflector output (lessons from 3 extractions)
    simulated_lessons = [
        {
            "field": "base_pressure_torr",
            "error_type": "scientific_notation",
            "lesson": "Parse scientific notation like '5 × 10⁻⁷ Torr' as 5e-7; the Unicode '×' is multiplication",
            "confidence_boost": 0.4,
        },
        {
            "field": "ar_o2_flow_ratio",
            "error_type": "unit_confusion",
            "lesson": "Flow ratios like '15:1 (Ar:O₂)' should be stored as string '15:1', not converted to float",
            "confidence_boost": 0.35,
        },
        {
            "field": "synthesis_protocols",
            "error_type": "domain_knowledge",
            "lesson": "In BiFeO₃ and other Bi-containing perovskites, Bi:B-site precursor ratio > 1 is standard due to Bi volatility",
            "confidence_boost": 0.5,
        },
        {
            "field": "characterization",
            "error_type": "domain_knowledge",
            "lesson": "Remnant polarization (Pr) in BiFeO₃ films is typically 50-100 μC/cm²; values outside this range may indicate measurement error",
            "confidence_boost": 0.6,
        },
    ]

    print("Processing lessons from Reflector...\n")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None

    appended, merged, skipped = curator_agent(simulated_lessons, playbook, client)

    print(f"\nCurator summary: +{appended} appended, ~{merged} merged, {skipped} skipped")

    save_playbook(playbook)

    print("\n--- Updated Playbook ---")
    print(playbook.to_context_string())


if __name__ == "__main__":
    run_full_grc_loop_demo()

# TODO: Add rule expiry: deprecate rules with confidence < 0.3 after N uses
# TODO: Add rule conflict detection: flag rules that contradict each other
# TODO: Track which rules improved extraction accuracy (close the feedback loop)
