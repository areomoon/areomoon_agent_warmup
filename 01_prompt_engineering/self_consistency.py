"""
Self-Consistency for Scientific Parameter Extraction
====================================================

Self-consistency samples multiple reasoning paths independently, then
takes the majority vote (or aggregates by most common value).

For numerical extraction from scientific text:
- Run the same extraction prompt N times (temperature > 0)
- Aggregate field-level values by majority vote
- Fields with high agreement → high confidence
- Fields with disagreement → flag for human review

This is especially valuable for:
- Ambiguous numerical values (units unclear)
- Long complex methods sections
- Tables with merged cells

References:
  - Self-Consistency Improves CoT (Wang et al., 2022): https://arxiv.org/abs/2203.11171
"""

import os
import json
from collections import Counter
from typing import Any
from dotenv import load_dotenv

load_dotenv()

SAMPLE_TEXT = """
ZnO nanorods were synthesized hydrothermally. Equimolar solutions of zinc nitrate
hexahydrate (Zn(NO₃)₂·6H₂O) and hexamethylenetetramine (HMT) at 25 mM were mixed
and transferred to a Teflon-lined autoclave. The reaction was carried out at 95°C
for 6 hours. The product was collected by centrifugation at 8000 rpm for 10 minutes,
washed three times with deionized water and once with ethanol, then dried at 60°C
overnight. XRD confirmed wurtzite structure with no secondary phases. The average
nanorod diameter was 200 ± 30 nm and length 2 ± 0.4 μm from SEM analysis.
"""

TARGET_FIELDS = [
    "material",
    "synthesis_method",
    "reaction_temperature_C",
    "reaction_duration_h",
    "concentration_mM",
    "centrifuge_speed_rpm",
    "drying_temperature_C",
    "nanorod_diameter_nm",
    "nanorod_length_um",
    "crystal_structure",
]


def single_extraction(text: str, client, temperature: float = 0.3) -> dict:
    """One extraction attempt with given temperature."""
    prompt = f"""Extract experimental parameters from this scientific text.
Think step by step, then output JSON.

Text: {text}

Fields to extract: {TARGET_FIELDS}
Use null for missing values. Think carefully about units.
Output valid JSON only."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content)


def aggregate_by_majority(samples: list[dict]) -> dict[str, Any]:
    """
    Aggregate N extractions by majority vote per field.
    Returns dict with value and confidence (fraction agreeing).
    """
    all_keys = set()
    for s in samples:
        all_keys.update(s.keys())

    result = {}
    for key in all_keys:
        values = [s.get(key) for s in samples]
        # Normalize: convert floats/ints to strings for counting
        str_values = [str(v) if v is not None else "null" for v in values]
        counter = Counter(str_values)
        most_common_str, count = counter.most_common(1)[0]

        # Convert back to original type
        most_common = next(
            (v for v in values if str(v) == most_common_str),
            None,
        )

        result[key] = {
            "value": most_common,
            "confidence": count / len(samples),
            "all_votes": counter.most_common(),
        }

    return result


def self_consistent_extraction(text: str, client, n_samples: int = 5) -> dict:
    """
    Run extraction N times, aggregate by majority vote.
    Returns values with per-field confidence scores.
    """
    print(f"Running {n_samples} extraction samples...")
    samples = []
    for i in range(n_samples):
        try:
            result = single_extraction(text, client, temperature=0.4)
            samples.append(result)
            print(f"  Sample {i + 1}/{n_samples}: {len(result)} fields extracted")
        except Exception as e:
            print(f"  Sample {i + 1} failed: {e}")

    if not samples:
        return {}

    aggregated = aggregate_by_majority(samples)

    # Print summary
    print("\n--- Self-Consistency Results ---")
    for field, meta in aggregated.items():
        conf = meta["confidence"]
        val = meta["value"]
        flag = "⚠️ " if conf < 0.6 else "✅ "
        print(f"{flag}{field}: {val} (confidence: {conf:.0%})")

    return aggregated


def run_example():
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"OpenAI client init failed: {e}\nSet OPENAI_API_KEY in .env")
        return

    print("Input text:")
    print(SAMPLE_TEXT)

    result = self_consistent_extraction(SAMPLE_TEXT, client, n_samples=3)

    # Separate high-confidence from flagged fields
    high_conf = {k: v["value"] for k, v in result.items() if v["confidence"] >= 0.8}
    low_conf = {k: v for k, v in result.items() if v["confidence"] < 0.8}

    print(f"\n✅ High confidence fields ({len(high_conf)}):")
    print(json.dumps(high_conf, indent=2))

    if low_conf:
        print(f"\n⚠️  Low confidence fields ({len(low_conf)}) — need review:")
        for k, v in low_conf.items():
            print(f"  {k}: votes = {v['all_votes']}")


if __name__ == "__main__":
    run_example()

# TODO: Tune confidence threshold per field type (numerical vs. categorical)
# TODO: Add cross-document consistency check (same material in multiple papers)
# TODO: Save low-confidence cases as annotation tasks for human review
# TODO: Use confidence scores as features for fine-tuning data quality filtering
