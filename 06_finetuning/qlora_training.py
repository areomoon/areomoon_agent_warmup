"""
QLoRA Training Script for Materials Science Extraction
=======================================================

QLoRA = Quantized LoRA:
  - Base model loaded in 4-bit NF4 quantization (reduces VRAM by ~75%)
  - LoRA adapters trained in full precision on top
  - Enables fine-tuning 7B models on a T4 GPU (16GB VRAM)

This script trains a model to extract structured experimental parameters
from materials science paper sections.

⚠️  Run this on Google Colab (T4 free) or Kaggle (P100 free), NOT on Mac.
    bitsandbytes 4-bit quantization does NOT support Apple Silicon MPS.

Setup on Colab:
  !pip install transformers peft trl datasets bitsandbytes accelerate

References:
  - QLoRA paper (arXiv 2305.14314): https://arxiv.org/abs/2305.14314
  - QLoRA GitHub: https://github.com/artidoro/qlora
  - TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
  - 2026 tutorial: https://oneuptime.com/blog/post/2026-01-30-qlora-fine-tuning/view
"""

import json
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # Change to team's preferred base model
OUTPUT_DIR = "./checkpoints/materials-extractor"
DATASET_PATH = "./data/extraction_training_data.jsonl"

QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    "use_dora": True,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,          # effective batch = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "max_seq_length": 2048,
    "gradient_checkpointing": True,
}

SYSTEM_PROMPT = """You are a materials science data extraction specialist.
Extract ALL experimental parameters from scientific text as structured JSON.
Be precise with values, units, and chemical formulas.
Use null for fields not mentioned in the text."""

# ── Dataset Preparation ────────────────────────────────────────────────────────

def format_sample_for_training(text: str, expected_output: dict) -> dict:
    """
    Format a (paper_text, extraction_dict) pair into chat message format
    for SFTTrainer.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract parameters from:\n\n{text}"},
            {"role": "assistant", "content": json.dumps(expected_output, indent=2)},
        ]
    }


def create_sample_dataset(output_path: str = DATASET_PATH) -> int:
    """
    Create a small sample training dataset.
    In production: replace with your actual extraction cases from Phase 1.
    """
    samples = [
        {
            "text": "La₀.₈Sr₀.₂MnO₃ films deposited by PLD at 700°C, 200 mTorr O₂. Thickness: 50 nm. T_MI = 370 K.",
            "output": {
                "material_formula": "La₀.₈Sr₀.₂MnO₃",
                "synthesis_method": "pulsed laser deposition",
                "substrate_temperature_C": 700,
                "pressure_mTorr": 200,
                "atmosphere": "O₂",
                "thickness_nm": 50,
                "transition_temperature_K": 370,
            }
        },
        {
            "text": "ZnO nanorods via hydrothermal synthesis at 95°C, 6h. Diameter: 200 ± 30 nm, length: 2 μm.",
            "output": {
                "material_name": "ZnO nanorods",
                "synthesis_method": "hydrothermal",
                "synthesis_temperature_C": 95,
                "synthesis_duration_h": 6,
                "particle_diameter_nm": 200,
                "particle_length_um": 2.0,
            }
        },
        {
            "text": "VO₂ films sputtered at 500°C, base pressure 5×10⁻⁷ Torr, working pressure 5 mTorr. Semiconductor-metal transition at 68°C.",
            "output": {
                "material_formula": "VO₂",
                "synthesis_method": "magnetron sputtering",
                "substrate_temperature_C": 500,
                "base_pressure_torr": 5e-7,
                "pressure_mTorr": 5.0,
                "transition_temperature_C": 68,
                "transition_temperature_K": 341,
            }
        },
    ]

    # Convert to JSONL format
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    formatted = [format_sample_for_training(s["text"], s["output"]) for s in samples]

    with open(output_path, "w") as f:
        for sample in formatted:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(formatted)} training samples to {output_path}")
    return len(formatted)


# ── Training Function (run on GPU) ────────────────────────────────────────────

def train_qlora(
    model_id: str = MODEL_ID,
    dataset_path: str = DATASET_PATH,
    output_dir: str = OUTPUT_DIR,
    hf_token: Optional[str] = None,
):
    """
    Full QLoRA training pipeline.
    Run this on Colab/Kaggle with GPU, not on Mac.

    Args:
        model_id: HuggingFace model ID
        dataset_path: Path to JSONL training data
        output_dir: Where to save checkpoints
        hf_token: HuggingFace token for gated models (Llama, etc.)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run on Colab: !pip install transformers peft trl datasets bitsandbytes accelerate")
        return

    print(f"Training QLoRA: {model_id}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU!'}")

    # 1. 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA config
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=QLORA_CONFIG["r"],
        lora_alpha=QLORA_CONFIG["lora_alpha"],
        target_modules=QLORA_CONFIG["target_modules"],
        use_dora=QLORA_CONFIG["use_dora"],
        lora_dropout=QLORA_CONFIG["lora_dropout"],
        bias=QLORA_CONFIG["bias"],
        task_type=QLORA_CONFIG["task_type"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load dataset
    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))
    dataset = Dataset.from_list(samples)

    # 5. Training
    training_args = SFTConfig(
        output_dir=output_dir,
        **TRAINING_CONFIG,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Training complete. Model saved to {output_dir}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_extraction_model(model_path: str, eval_data: list[dict]) -> dict:
    """
    Evaluate extraction quality: Exact Match and Field-level F1.

    Args:
        model_path: Path to fine-tuned model
        eval_data: List of {"text": ..., "expected": {...}} dicts

    Returns:
        {"exact_match_rate": float, "field_f1": float, "total": int}
    """
    exact_matches = 0
    total_fields = 0
    matched_fields = 0

    for sample in eval_data:
        # Placeholder: in production, call the model
        prediction = sample.get("expected", {})  # Replace with actual model call
        expected = sample.get("expected", {})

        for k, v in expected.items():
            if v is None:
                continue
            total_fields += 1
            if prediction.get(k) == v:
                exact_matches += 1
                matched_fields += 1
            elif prediction.get(k) is not None:
                matched_fields += 0.5  # partial credit

    return {
        "exact_match_rate": exact_matches / total_fields if total_fields else 0,
        "field_f1": matched_fields / total_fields if total_fields else 0,
        "total_samples": len(eval_data),
        "total_fields_evaluated": total_fields,
    }


if __name__ == "__main__":
    print("QLoRA Training Script")
    print("=" * 50)
    print(f"Model: {MODEL_ID}")
    print(f"Config: {json.dumps(QLORA_CONFIG, indent=2)}")
    print(f"\nTo train: call train_qlora() on a GPU machine (Colab/Kaggle)")
    print("\nCreating sample dataset...")
    n = create_sample_dataset()
    print(f"\nNext steps:")
    print("1. Accumulate 200-500 extraction cases during Phase 1")
    print("2. Upload to Colab, run train_qlora()")
    print("3. Evaluate with evaluate_extraction_model()")
    print("4. Compare: base model vs. QLoRA vs. base + playbook")

# TODO: Add MLX training script for Apple Silicon (alternative to Colab)
# TODO: Add DPO training: use (good_extraction, bad_extraction) pairs
# TODO: Add model merge: combine QLoRA with base for deployment
