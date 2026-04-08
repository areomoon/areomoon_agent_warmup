"""
Chain-of-Thought (CoT) Prompting for Scientific Text Extraction
===============================================================

CoT prompting asks the LLM to reason step-by-step before giving the final answer.
For scientific text, this significantly improves extraction of numerical parameters,
units, and multi-step reasoning about experimental conditions.

Key insight: "Let's think step by step" → model explicitly reasons before answering.
For materials science: reasoning about temperature units, unit conversions, and
which experiment section a value belongs to.

References:
  - Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022): https://arxiv.org/abs/2201.11903
  - Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022): https://arxiv.org/abs/2205.11916
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# Sample scientific text for testing
SAMPLE_TEXT = """
The perovskite oxide La₀.₈Sr₀.₂MnO₃ (LSMO) thin films were deposited via pulsed laser
deposition (PLD) at a substrate temperature of 700°C under an oxygen partial pressure
of 200 mTorr. The laser fluence was maintained at 2 J/cm². Film thickness was controlled
by the number of laser pulses, yielding films of approximately 50 nm as confirmed by
X-ray reflectometry. Post-deposition annealing was performed at 800°C for 2 hours in
1 atm O₂ atmosphere. Electrical resistivity measurements showed a metal-insulator
transition at T_MI = 370 K with a room-temperature resistivity of 1.2 × 10⁻³ Ω·cm.
"""

EXTRACTION_FIELDS = {
    "material": "chemical formula or material name",
    "deposition_method": "synthesis/deposition technique",
    "substrate_temperature_C": "substrate temperature in Celsius",
    "oxygen_pressure_mTorr": "oxygen partial pressure in mTorr",
    "film_thickness_nm": "film thickness in nanometers",
    "annealing_temperature_C": "annealing temperature in Celsius",
    "annealing_duration_h": "annealing duration in hours",
    "MIT_temperature_K": "metal-insulator transition temperature in Kelvin",
    "room_temp_resistivity_ohm_cm": "room temperature resistivity in Ω·cm",
}


def zero_shot_extraction(text: str, client) -> dict:
    """Direct extraction without reasoning prompt."""
    prompt = f"""Extract experimental parameters from the following text as JSON.

Text:
{text}

Return JSON with these fields: {list(EXTRACTION_FIELDS.keys())}
Use null for missing values."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def zero_shot_cot_extraction(text: str, client) -> dict:
    """Zero-shot CoT: add 'think step by step' to trigger reasoning."""
    prompt = f"""Extract experimental parameters from the following scientific text.

Text:
{text}

Let's think step by step:
1. Identify all numerical values and their units
2. Match each value to the correct experimental parameter
3. Convert units if needed (e.g., note original units)
4. Flag any ambiguous values

After reasoning, output a JSON with these fields: {list(EXTRACTION_FIELDS.keys())}
Use null for missing values. Include a "reasoning_notes" field summarizing key decisions."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def few_shot_cot_extraction(text: str, client) -> dict:
    """Few-shot CoT: provide a worked example before the actual query."""
    example_text = """The BaTiO₃ ceramics were sintered at 1300°C for 4 hours in air.
    XRD confirmed single-phase perovskite structure. Dielectric constant at 1 kHz
    reached 2800 at room temperature."""

    example_output = {
        "material": "BaTiO₃",
        "deposition_method": "sintering (ceramic)",
        "substrate_temperature_C": None,
        "annealing_temperature_C": 1300,
        "annealing_duration_h": 4,
        "reasoning_notes": "Sintering temperature treated as equivalent to annealing temperature for ceramics",
    }

    prompt = f"""Extract experimental parameters from scientific text, reasoning step by step.

Example:
Text: {example_text}
Reasoning: Temperature 1300°C is the sintering temperature. Duration 4h is explicit. No film deposition, so substrate_temperature is null.
Output: {json.dumps(example_output, indent=2)}

Now extract from this text:
Text: {text}

Reason step by step, then output JSON with fields: {list(EXTRACTION_FIELDS.keys())}
Include a "reasoning_notes" field."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def run_comparison():
    """Run all three strategies and compare results."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"OpenAI client init failed: {e}")
        print("Set OPENAI_API_KEY in .env to run this example.")
        return

    print("=" * 60)
    print("CoT Comparison: Scientific Parameter Extraction")
    print("=" * 60)
    print(f"\nInput text:\n{SAMPLE_TEXT}\n")

    strategies = [
        ("Zero-shot (no CoT)", zero_shot_extraction),
        ("Zero-shot CoT", zero_shot_cot_extraction),
        ("Few-shot CoT", few_shot_cot_extraction),
    ]

    results = {}
    for name, fn in strategies:
        print(f"\n--- {name} ---")
        try:
            result = fn(SAMPLE_TEXT, client)
            results[name] = result
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    # Compare against ground truth
    ground_truth = {
        "material": "La₀.₈Sr₀.₂MnO₃",
        "deposition_method": "pulsed laser deposition",
        "substrate_temperature_C": 700,
        "oxygen_pressure_mTorr": 200,
        "film_thickness_nm": 50,
        "annealing_temperature_C": 800,
        "annealing_duration_h": 2,
        "MIT_temperature_K": 370,
        "room_temp_resistivity_ohm_cm": 1.2e-3,
    }

    print("\n--- Ground Truth ---")
    print(json.dumps(ground_truth, indent=2))

    print("\n--- Field Accuracy Summary ---")
    for name, result in results.items():
        correct = 0
        total = len(ground_truth)
        for k, v in ground_truth.items():
            if result.get(k) == v:
                correct += 1
        print(f"{name}: {correct}/{total} fields correct")


if __name__ == "__main__":
    run_comparison()

# TODO: Add Gemini API alternative (free tier)
# TODO: Add structured output with JSON schema validation
# TODO: Add confidence scores per extracted field
# TODO: Experiment with domain-specific few-shot examples (LSMO, perovskite, etc.)
