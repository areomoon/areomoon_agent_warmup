"""
LoRA Basics: Low-Rank Adaptation
==================================

LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model weights.
Instead of updating all W (billions of params), it trains two small matrices A and B
such that the weight update ΔW = B @ A (low-rank approximation).

Key parameters:
  r     : rank of the decomposition (typically 4-64; higher = more capacity, more params)
  alpha : scaling factor (usually 2×r; controls how much the LoRA weights affect output)
  target_modules: which layers to adapt (q_proj, v_proj, all-linear, etc.)

For materials science extraction fine-tuning:
  - Recommended: r=16, DoRA=True, target_modules="all-linear", lr=2e-4
  - Training data: (paper_section, extracted_JSON) pairs
  - 200-500 examples minimum for meaningful improvement

References:
  - LoRA paper (arXiv 2106.09685): https://arxiv.org/abs/2106.09685
  - QLoRA paper (arXiv 2305.14314): https://arxiv.org/abs/2305.14314
  - HuggingFace PEFT: https://huggingface.co/docs/peft
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA configuration with recommended defaults for extraction fine-tuning."""
    r: int = 16                              # Rank
    lora_alpha: int = 32                     # Scaling factor (= 2 * r is typical)
    target_modules: str = "all-linear"       # Which layers to adapt
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = True                    # DoRA often outperforms vanilla LoRA


def explain_lora_math():
    """Explain the LoRA math with concrete numbers."""
    print("=== LoRA Mathematics ===\n")
    print("Standard fine-tuning:")
    print("  W_updated = W_original + ΔW")
    print("  ΔW shape: (d_model, d_model) = (4096, 4096) = 16,777,216 params")
    print()
    print("LoRA fine-tuning:")
    print("  ΔW = B @ A, where:")
    print("  A shape: (r, d_model) = (16, 4096) = 65,536 params")
    print("  B shape: (d_model, r) = (4096, 16) = 65,536 params")
    print("  Total LoRA params: 131,072 (< 1% of full fine-tuning!)")
    print()
    print("Scaled output: h = W_original @ x + (alpha/r) * B @ A @ x")
    print("  alpha/r = 32/16 = 2.0 (scaling factor)")
    print()
    print("For Qwen2.5-7B:")
    print("  Full fine-tuning: ~7B params to train")
    print("  LoRA (r=16, all-linear): ~40M params (0.6%)")
    print("  QLoRA (4-bit base + LoRA): runs on T4 GPU (16GB VRAM)")


def generate_training_data_format():
    """Show the expected format for materials extraction SFT training data."""
    print("\n=== Training Data Format ===\n")

    example_pairs = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a materials science data extraction specialist. Extract experimental parameters as structured JSON."
                },
                {
                    "role": "user",
                    "content": "Extract parameters from:\n\nLa₀.₆Sr₀.₄CoO₃ thin films were deposited by PLD at 650°C substrate temperature under 100 mTorr oxygen. Films of 80 nm were grown on LaAlO₃ substrates. Electrical measurements showed metallic behavior with room-temperature resistivity of 5×10⁻⁴ Ω·cm."
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "material_formula": "La₀.₆Sr₀.₄CoO₃",
                        "synthesis_method": "pulsed laser deposition",
                        "substrate_temperature_C": 650,
                        "pressure_mTorr": 100,
                        "atmosphere": "O₂",
                        "substrate": "LaAlO₃",
                        "thickness_nm": 80,
                        "electrical_resistivity_ohm_cm": 5e-4,
                        "material_class": "perovskite oxide",
                    }, indent=2)
                }
            ]
        }
    ]

    print("Format: chat messages with system + user (paper text) + assistant (JSON output)")
    print("\nExample:")
    print(json.dumps(example_pairs[0], indent=2))
    print("\nFor 200-500 such pairs → meaningful fine-tuning improvement")
    print("Collect from your extraction cases during Phase 1 (first 2 months)")


def compare_approaches():
    """Compare fine-tuning approaches for the extraction job."""
    print("\n=== Approach Comparison ===\n")
    approaches = [
        {
            "name": "Base model + prompt engineering",
            "setup_time": "days",
            "data_needed": "0",
            "quality": "good baseline",
            "notes": "Start here. Get this working first."
        },
        {
            "name": "Base model + ACE playbook",
            "setup_time": "weeks",
            "data_needed": "~50 examples to bootstrap rules",
            "quality": "better, improves over time",
            "notes": "Phase 1 target. Playbook accumulates."
        },
        {
            "name": "QLoRA fine-tuned + playbook",
            "setup_time": "1-2 months",
            "data_needed": "200-500 labeled pairs",
            "quality": "best",
            "notes": "Phase 2 target. Combines both."
        },
        {
            "name": "Full fine-tuning",
            "setup_time": "months",
            "data_needed": "1000+",
            "quality": "potentially best",
            "notes": "Usually not needed; QLoRA is sufficient."
        },
    ]

    for a in approaches:
        print(f"{'─'*50}")
        print(f"Method: {a['name']}")
        print(f"  Setup time : {a['setup_time']}")
        print(f"  Data needed: {a['data_needed']}")
        print(f"  Quality    : {a['quality']}")
        print(f"  Notes      : {a['notes']}")


if __name__ == "__main__":
    explain_lora_math()
    generate_training_data_format()
    compare_approaches()

    print("\n=== Recommended LoRA Config for This Job ===")
    config = LoRAConfig()
    print(json.dumps(config.__dict__, indent=2))

# TODO: Run actual LoRA training in Colab — see qlora_training.py
# TODO: After Phase 1, export (text, extraction) pairs → training dataset
# TODO: Experiment with domain-specific base models (if team has access)
