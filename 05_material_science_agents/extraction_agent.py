"""
Materials Science Paper Extraction Agent
==========================================

End-to-end extraction agent for scientific papers using the
Generator-Reflector-Curator (GRC) architecture.

This is the "production prototype" that directly maps to the
onboarding job: an agent service that extracts structured
experimental data from scientific literature for materials scientists.

Architecture:
  1. Generator  → extracts parameters from paper section
  2. Reflector  → validates extraction, generates lessons
  3. Curator    → updates playbook with lessons (persisted)
  4. Output     → structured JSON with confidence + provenance

Onboarding action items:
  - Phase 1 (Month 1-2): get Generator + Reflector working well
  - Phase 2 (Month 3-4): fine-tune Generator on accumulated extraction cases

References:
  - MARS system (arXiv 2602.00169): https://arxiv.org/abs/2602.00169
  - LLMatDesign (arXiv 2406.13163): https://arxiv.org/abs/2406.13163
  - ACE (arXiv 2510.04618): https://arxiv.org/abs/2510.04618
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Schema for materials science experimental parameters
# Extend this based on your team's actual requirements
MATERIALS_EXTRACTION_SCHEMA = {
    # Sample identity
    "material_name": None,          # e.g., "La₀.₈Sr₀.₂MnO₃"
    "material_formula": None,        # chemical formula
    "material_class": None,          # perovskite, alloy, 2D material, etc.

    # Synthesis
    "synthesis_method": None,        # PLD, CVD, sol-gel, sputtering, etc.
    "synthesis_temperature_C": None,
    "synthesis_duration_h": None,
    "atmosphere": None,              # air, O₂, Ar, vacuum, etc.
    "pressure_mTorr": None,

    # Sample geometry
    "substrate": None,
    "thickness_nm": None,

    # Post-processing
    "annealing_temperature_C": None,
    "annealing_duration_h": None,

    # Measured properties (extend per material class)
    "electrical_resistivity_ohm_cm": None,
    "transition_temperature_K": None,
    "bandgap_eV": None,
    "conductivity_S_cm": None,

    # Characterization methods used
    "characterization_methods": None,  # list: ["XRD", "SEM", "Raman"]
}


@dataclass
class ExtractionOutput:
    """Full output of the extraction agent."""
    paper_id: str
    section_type: str           # "methods" | "results" | "abstract"
    extracted_data: dict
    confidence_scores: dict     # per-field confidence
    flagged_fields: list[str]   # low confidence or ambiguous
    provenance: dict            # {field: "page X, line Y, section Z"}
    lessons_generated: list[str]
    playbook_rules_applied: list[str]


def load_active_playbook_rules(playbook_path: str = "data/materials_extraction_playbook.json") -> list[str]:
    """Load current playbook rules as a list of strings for context injection."""
    path = Path(playbook_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        rules = [
            r["rule"] for r in data.get("rules", [])
            if not r.get("deprecated", False) and r.get("confidence", 0) > 0.5
        ]
        return rules
    except Exception:
        return []


def generator_extract(text: str, section_type: str, playbook_rules: list[str], client) -> dict:
    """
    Generator role: extract structured parameters from paper section.
    Uses active playbook rules to guide extraction.
    """
    rules_context = ""
    if playbook_rules:
        rules_context = "\n\nApply these extraction rules (from past experience):\n" + \
                       "\n".join(f"  • {r}" for r in playbook_rules[:15])  # limit context size

    prompt = f"""You are a materials science data extraction specialist.
Section type: {section_type}
{rules_context}

Extract ALL experimental parameters from this text. Be thorough and precise with units.

Text:
{text}

Return JSON with:
1. All fields from this schema (null if not mentioned): {list(MATERIALS_EXTRACTION_SCHEMA.keys())}
2. "confidence_scores": {{field: 0.0-1.0}} for each extracted value
3. "flagged_fields": [list of fields with uncertainty]
4. "provenance": {{field: "quote from source text"}} for key values
5. "additional_parameters": {{}} for any domain-specific params not in the schema"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def reflector_validate(text: str, extraction: dict, client) -> dict:
    """
    Reflector role: validate extraction and generate lessons.
    """
    prompt = f"""Validate this materials science parameter extraction.

Source text:
{text}

Extraction:
{json.dumps({k: v for k, v in extraction.items() if k not in ['confidence_scores', 'provenance', 'additional_parameters']}, indent=2)}

Check:
1. Missing values that ARE present in the text
2. Incorrect values or unit errors
3. Scientific notation errors (5×10⁻⁷ → 5e-7)
4. Chemical formula transcription errors

Return JSON:
{{
  "is_valid": true/false,
  "corrections": [{{"field": ..., "wrong": ..., "correct": ..., "reason": ...}}],
  "missed_values": [{{"field": ..., "value": ..., "from_text": "quote"}}],
  "lessons": [
    {{
      "lesson": "transferable extraction rule for future papers",
      "layer": "extraction|characterization|synthesis_protocols|design_rules",
      "confidence_boost": 0.0-1.0
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def run_extraction(
    text: str,
    paper_id: str = "unknown",
    section_type: str = "methods",
    client=None,
) -> ExtractionOutput:
    """
    Run the full extraction pipeline on a paper section.
    Saves generated lessons to the playbook.
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"API client init failed: {e}")
            return None

    # Load current playbook
    playbook_rules = load_active_playbook_rules()
    print(f"[{paper_id}] Using {len(playbook_rules)} active playbook rules")

    # Generator
    print(f"[{paper_id}] Generator extracting from {section_type} section...")
    raw_extraction = generator_extract(text, section_type, playbook_rules, client)

    # Reflector
    print(f"[{paper_id}] Reflector validating...")
    validation = reflector_validate(text, raw_extraction, client)

    if not validation.get("is_valid", True):
        print(f"[{paper_id}] Corrections needed: {len(validation.get('corrections', []))}")
        for corr in validation.get("corrections", []):
            field = corr["field"]
            if field in raw_extraction:
                raw_extraction[field] = corr["correct"]

    # Apply missed values
    for missed in validation.get("missed_values", []):
        field = missed["field"]
        if field in raw_extraction and raw_extraction[field] is None:
            raw_extraction[field] = missed["value"]

    # Save lessons to playbook (Curator step — simplified)
    lessons = validation.get("lessons", [])
    if lessons:
        print(f"[{paper_id}] Saving {len(lessons)} lessons to playbook...")
        _append_lessons_to_playbook(lessons)

    return ExtractionOutput(
        paper_id=paper_id,
        section_type=section_type,
        extracted_data={k: raw_extraction.get(k) for k in MATERIALS_EXTRACTION_SCHEMA},
        confidence_scores=raw_extraction.get("confidence_scores", {}),
        flagged_fields=raw_extraction.get("flagged_fields", []),
        provenance=raw_extraction.get("provenance", {}),
        lessons_generated=[l["lesson"] for l in lessons],
        playbook_rules_applied=playbook_rules[:5],
    )


def _append_lessons_to_playbook(lessons: list[dict]):
    """Quick lesson append to playbook file (simplified Curator)."""
    try:
        from playbook_evolution import load_playbook, save_playbook, append_rule
        playbook = load_playbook()
        for lesson in lessons:
            rule_text = lesson.get("lesson", "").strip()
            if rule_text:
                layer = lesson.get("layer", "extraction")
                confidence = 0.5 + lesson.get("confidence_boost", 0.2)
                append_rule(playbook, layer, rule_text, "reflector", min(confidence, 0.95))
        save_playbook(playbook)
    except Exception as e:
        print(f"Playbook update failed: {e}")


# ── Demo ──────────────────────────────────────────────────────────────────────

SAMPLE_PAPER_SECTION = """
Methods: Thin Film Deposition

VO₂ thin films were deposited on c-cut sapphire (Al₂O₃) substrates using reactive
pulsed DC magnetron sputtering from a metallic vanadium target (99.95% purity, 76 mm
diameter). The deposition chamber was evacuated to a base pressure of 3 × 10⁻⁷ Torr
prior to deposition. Sputtering was performed in a mixed Ar/O₂ atmosphere with total
pressure of 6 mTorr and O₂ partial pressure of 0.5 mTorr (Ar:O₂ = 11:1 flow ratio).
The target-to-substrate distance was 10 cm. Substrate temperature was maintained at
550°C throughout deposition using a resistive heater. Film growth rate was determined
to be 1.5 nm/min by X-ray reflectometry (XRR) on calibration samples, yielding films
of 150 nm after 100 minutes of deposition. No post-deposition annealing was performed.
"""

if __name__ == "__main__":
    result = run_extraction(
        text=SAMPLE_PAPER_SECTION,
        paper_id="vo2_sapphire_2026",
        section_type="methods",
    )
    if result:
        print("\n--- Extraction Output ---")
        # Show non-null fields
        non_null = {k: v for k, v in result.extracted_data.items() if v is not None}
        print(json.dumps(non_null, indent=2))
        print(f"\nFields extracted: {len(non_null)}/{len(MATERIALS_EXTRACTION_SCHEMA)}")
        print(f"Flagged: {result.flagged_fields}")
        print(f"\nLessons generated: {len(result.lessons_generated)}")
        for lesson in result.lessons_generated:
            print(f"  • {lesson}")

# TODO: Add batch processing: run on a folder of PDFs
# TODO: Add cross-paper validation: compare same material across multiple papers
# TODO: Add SFT data collection: save (text, extraction) pairs for fine-tuning
# TODO: Connect to real Materials Project API for property validation
