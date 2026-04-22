"""
Generator-Reflector Pattern (ACE Core)
=======================================

The Generator-Reflector pattern is the heart of ACE (Agentic Context Engineering)
and the most important pattern for the material science extraction job.

Generator:  Produces an output (extraction, hypothesis, structured data)
Reflector:  Analyzes the output, identifies errors, extracts lessons
Curator:    (see curator_pattern.py) Merges lessons into the playbook

In materials science extraction:
  Generator  → attempts to extract parameters from a paper section
  Reflector  → checks consistency, flags errors, extracts improvement rules
  Loop       → runs N times, each cycle informed by prior lessons

The key difference from simple retry:
  - Reflector creates TRANSFERABLE lessons, not just "try again"
  - Lessons accumulate in a playbook used in future extractions
  - The system improves across papers, not just within one paper

References:
  - ACE: Agentic Context Engineering (arXiv 2510.04618): https://arxiv.org/abs/2510.04618
  - Reflexion (Shinn et al., 2023): https://arxiv.org/abs/2303.11366
  - Andrew Ng Reflection Pattern: https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/
"""

import os
import sys
import json
import pathlib
from dataclasses import dataclass, field
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "resources"))
from anthropic_helpers import get_client, extract_text, parse_json, DEFAULT_MODEL

load_dotenv()

MODEL = DEFAULT_MODEL

SAMPLE_TEXT = """
The synthesis of VO₂ thin films was carried out using reactive DC magnetron sputtering.
A vanadium metal target (99.95% purity) was sputtered in a mixed Ar/O₂ atmosphere with
a flow ratio of 15:1 (Ar:O₂). The base pressure before deposition was 5 × 10⁻⁷ Torr,
and the working pressure during sputtering was 5 mTorr. The substrate temperature was
maintained at 500°C throughout the deposition. Film deposition rate was 2 nm/min,
yielding films of 100 nm thickness after 50 minutes. The VO₂ films underwent a sharp
semiconductor-to-metal transition at 68°C (341 K) with a resistance change of four
orders of magnitude (ΔR/R ~ 10⁴).
"""

EXTRACTION_SCHEMA = {
    "material": None,
    "deposition_method": None,
    "target_purity_percent": None,
    "ar_o2_flow_ratio": None,
    "base_pressure_torr": None,
    "working_pressure_mTorr": None,
    "substrate_temperature_C": None,
    "deposition_rate_nm_per_min": None,
    "film_thickness_nm": None,
    "SMT_temperature_C": None,
    "resistance_change_orders": None,
}


@dataclass
class ExtractionResult:
    data: dict
    confidence_notes: str = ""
    flagged_fields: list[str] = field(default_factory=list)


@dataclass
class ReflectorLesson:
    field: str
    error_type: str
    lesson: str
    confidence_boost: float  # expected confidence improvement if applied


def generator(text: str, schema: dict, playbook_rules: list[str], client) -> ExtractionResult:
    """
    Extract structured parameters from scientific text.
    Uses the current playbook rules to guide extraction.
    """
    rules_section = ""
    if playbook_rules:
        rules_section = "\n\nApply these extraction rules from past experience:\n" + \
                        "\n".join(f"- {r}" for r in playbook_rules)

    prompt = f"""Extract experimental parameters from this scientific text.
Think carefully about units and scientific notation.{rules_section}

Text:
{text}

Return JSON with these fields: {list(schema.keys())}
Also include:
  "confidence_notes": "brief notes on any uncertain extractions"
  "flagged_fields": ["list of fields with low confidence"]

Use null for genuinely missing values.
Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = parse_json(extract_text(response))

    return ExtractionResult(
        data={k: raw.get(k) for k in schema},
        confidence_notes=raw.get("confidence_notes", ""),
        flagged_fields=raw.get("flagged_fields", []),
    )


def reflector(text: str, extraction: ExtractionResult, ground_truth: dict | None, client) -> list[ReflectorLesson]:
    """
    Analyze the extraction result, identify errors, and extract transferable lessons.

    If ground_truth is available: compare directly.
    If not: use self-consistency checks and domain knowledge.
    """
    gt_section = ""
    if ground_truth:
        gt_section = f"\n\nGround truth (for comparison):\n{json.dumps(ground_truth, indent=2)}"

    prompt = f"""You are a scientific data quality reviewer.

Original text:
{text}

Extraction result:
{json.dumps(extraction.data, indent=2)}

Flagged by extractor: {extraction.flagged_fields}
Extractor notes: {extraction.confidence_notes}
{gt_section}

Analyze the extraction and identify:
1. Any errors or inconsistencies with the text
2. Any fields that could have been extracted but weren't
3. Transferable lessons for future extractions of similar papers

Return JSON with:
{{
  "errors_found": [{{"field": "...", "issue": "...", "correct_value": ...}}],
  "lessons": [
    {{
      "field": "field_name_or_general",
      "error_type": "unit_confusion|missed_value|wrong_parsing|scientific_notation",
      "lesson": "actionable rule for future extractions",
      "confidence_boost": 0.0-1.0
    }}
  ],
  "overall_quality": "good|acceptable|poor",
  "summary": "one-sentence summary"
}}

Output valid JSON only, no markdown fences or commentary."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = parse_json(extract_text(response))

    lessons = []
    for l in raw.get("lessons", []):
        lessons.append(ReflectorLesson(
            field=l.get("field", "general"),
            error_type=l.get("error_type", "unknown"),
            lesson=l.get("lesson", ""),
            confidence_boost=l.get("confidence_boost", 0.0),
        ))

    print(f"\nReflector found {len(raw.get('errors_found', []))} errors")
    print(f"Quality: {raw.get('overall_quality')} — {raw.get('summary')}")
    for error in raw.get("errors_found", []):
        print(f"  ⚠️  {error['field']}: {error['issue']}")

    return lessons


def generator_reflector_loop(
    text: str,
    schema: dict,
    client,
    ground_truth: dict | None = None,
    max_cycles: int = 2,
) -> tuple[ExtractionResult, list[str]]:
    """
    Run Generator → Reflector → Generator loop.
    Returns final extraction and accumulated playbook rules.
    """
    playbook_rules: list[str] = []

    for cycle in range(max_cycles):
        print(f"\n{'='*50}")
        print(f"Cycle {cycle + 1}/{max_cycles}")
        print(f"Active playbook rules: {len(playbook_rules)}")

        # Generate
        extraction = generator(text, schema, playbook_rules, client)
        print(f"Generator extracted {sum(1 for v in extraction.data.values() if v is not None)} / {len(schema)} fields")

        # Reflect
        lessons = reflector(text, extraction, ground_truth, client)

        # Update playbook with new lessons
        new_rules = [l.lesson for l in lessons if l.lesson and l.confidence_boost > 0.3]
        playbook_rules.extend(new_rules)
        if new_rules:
            print(f"Added {len(new_rules)} new playbook rules")

    return extraction, playbook_rules


def run_example():
    try:
        client = get_client()
    except Exception as e:
        print(f"Anthropic client init failed: {e}\nSet ANTHROPIC_API_KEY in .env")
        return

    # Ground truth for VO₂ example
    ground_truth = {
        "material": "VO₂",
        "deposition_method": "reactive DC magnetron sputtering",
        "target_purity_percent": 99.95,
        "ar_o2_flow_ratio": "15:1",
        "base_pressure_torr": 5e-7,
        "working_pressure_mTorr": 5.0,
        "substrate_temperature_C": 500,
        "deposition_rate_nm_per_min": 2.0,
        "film_thickness_nm": 100,
        "SMT_temperature_C": 68.0,
        "resistance_change_orders": 4,
    }

    print("Running Generator-Reflector loop on VO₂ thin film paper...")
    final_extraction, learned_rules = generator_reflector_loop(
        SAMPLE_TEXT, EXTRACTION_SCHEMA, client, ground_truth, max_cycles=2
    )

    print("\n--- Final Extraction ---")
    print(json.dumps(final_extraction.data, indent=2))

    print("\n--- Learned Playbook Rules ---")
    for i, rule in enumerate(learned_rules, 1):
        print(f"{i}. {rule}")


if __name__ == "__main__":
    run_example()

# TODO: Persist playbook_rules to a JSON file between sessions
# TODO: Add deduplication: don't add a rule if a semantically similar one exists
# TODO: Track rule performance: did applying rule X actually improve confidence?
# TODO: See curator_pattern.py for the full Curator (delta update) implementation
