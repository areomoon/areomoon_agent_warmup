"""
ACE Playbook: Grow-and-Refine Evolution
=========================================

Implements the ACE (Agentic Context Engineering) Grow-and-Refine mechanism:
  1. Append  — new lessons add bullets to the playbook
  2. Merge   — semantically similar bullets are merged (not duplicated)
  3. Update  — existing bullets are refined when contradicted by stronger evidence
  4. Prune   — periodic de-duplication removes redundant entries

The playbook is the core persistent artifact in ACE. Unlike fine-tuning
(which bakes knowledge into weights) or RAG (which retrieves at query time),
the playbook keeps knowledge as explicit, human-readable rules in the context.

This file focuses on the storage and evolution mechanics.
See curator_pattern.py for the Curator agent that calls these functions.

References:
  - ACE: Agentic Context Engineering (arXiv 2510.04618): https://arxiv.org/abs/2510.04618
  - ACE GitHub: https://github.com/ace-agent/ace
"""

import os
import sys
import json
import pathlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "resources"))
from anthropic_helpers import get_client, extract_text, parse_json, DEFAULT_MODEL

load_dotenv()

MODEL = DEFAULT_MODEL

PLAYBOOK_PATH = Path("data/materials_extraction_playbook.json")


@dataclass
class PlaybookRule:
    id: str
    layer: str          # "design_rules" | "synthesis_protocols" | "characterization" | "extraction"
    rule: str           # The actual heuristic text
    source: str         # What generated this rule (paper, observation, etc.)
    confidence: float   # 0.0-1.0, updated by feedback
    usage_count: int = 0
    last_updated: str = ""
    deprecated: bool = False


@dataclass
class Playbook:
    version: int = 1
    last_updated: str = ""
    rules: list[PlaybookRule] = field(default_factory=list)

    def get_layer(self, layer: str) -> list[PlaybookRule]:
        return [r for r in self.rules if r.layer == layer and not r.deprecated]

    def to_context_string(self, layers: Optional[list[str]] = None) -> str:
        """Format playbook as a string for injection into LLM context."""
        active_layers = layers or ["design_rules", "synthesis_protocols", "characterization", "extraction"]
        lines = ["# Materials Extraction Playbook\n"]
        for layer in active_layers:
            layer_rules = self.get_layer(layer)
            if layer_rules:
                lines.append(f"## {layer.replace('_', ' ').title()}")
                for r in layer_rules:
                    lines.append(f"- [{r.confidence:.0%}] {r.rule}")
                lines.append("")
        return "\n".join(lines)


def load_playbook(path: Path = PLAYBOOK_PATH) -> Playbook:
    """Load playbook from JSON file, or return empty playbook."""
    if path.exists():
        data = json.loads(path.read_text())
        rules = [PlaybookRule(**r) for r in data.get("rules", [])]
        return Playbook(
            version=data.get("version", 1),
            last_updated=data.get("last_updated", ""),
            rules=rules,
        )
    return Playbook()


def save_playbook(playbook: Playbook, path: Path = PLAYBOOK_PATH):
    """Persist playbook to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": playbook.version,
        "last_updated": datetime.utcnow().isoformat(),
        "rules": [asdict(r) for r in playbook.rules],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Playbook saved: {len(playbook.rules)} rules → {path}")


def append_rule(playbook: Playbook, layer: str, rule: str, source: str, confidence: float = 0.7) -> PlaybookRule:
    """Append a new rule to the playbook."""
    import hashlib
    rule_id = hashlib.md5(f"{layer}:{rule}".encode()).hexdigest()[:8]

    new_rule = PlaybookRule(
        id=rule_id,
        layer=layer,
        rule=rule,
        source=source,
        confidence=confidence,
        last_updated=datetime.utcnow().isoformat(),
    )
    playbook.rules.append(new_rule)
    return new_rule


def is_duplicate(new_rule: str, existing_rules: list[PlaybookRule], client, threshold: float = 0.85) -> Optional[PlaybookRule]:
    """
    Check if a new rule is semantically similar to an existing one.
    Uses LLM for semantic similarity (simple cosine similarity alternative for small playbooks).
    """
    if not existing_rules or not os.getenv("ANTHROPIC_API_KEY"):
        return None  # Skip dedup if no API

    existing_texts = [r.rule for r in existing_rules]
    prompt = f"""Is this new rule semantically equivalent to any existing rule?

New rule: "{new_rule}"

Existing rules:
{json.dumps(existing_texts, indent=2)}

Return JSON: {{"is_duplicate": true/false, "duplicate_index": null_or_int, "similarity": 0.0-1.0}}
Threshold: similarity > {threshold} is considered duplicate.
Output valid JSON only, no markdown fences or commentary."""

    try:
        client_obj = client or get_client()
        response = client_obj.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        result = parse_json(extract_text(response))
        if result.get("is_duplicate") and result.get("duplicate_index") is not None:
            idx = result["duplicate_index"]
            if 0 <= idx < len(existing_rules):
                return existing_rules[idx]
    except Exception:
        pass
    return None


def merge_rule(existing: PlaybookRule, new_text: str, new_confidence: float):
    """Merge new information into an existing rule."""
    if new_confidence > existing.confidence:
        existing.rule = new_text
        existing.confidence = new_confidence
    else:
        # Average the confidence
        existing.confidence = (existing.confidence + new_confidence) / 2
    existing.last_updated = datetime.utcnow().isoformat()
    existing.usage_count += 1


def deduplicate_playbook(playbook: Playbook, client=None) -> int:
    """
    Remove redundant rules within each layer.
    Returns number of rules removed.
    """
    removed = 0
    for layer in ["design_rules", "synthesis_protocols", "characterization", "extraction"]:
        layer_rules = [r for r in playbook.rules if r.layer == layer and not r.deprecated]
        seen_ids = set()

        for i, rule in enumerate(layer_rules):
            if rule.id in seen_ids:
                continue
            # Check against all subsequent rules in same layer
            for j, other in enumerate(layer_rules[i + 1:], start=i + 1):
                if other.id in seen_ids:
                    continue
                # Simple heuristic: if rule texts share >60% words, flag as potential duplicate
                words_a = set(rule.rule.lower().split())
                words_b = set(other.rule.lower().split())
                if len(words_a & words_b) / max(len(words_a | words_b), 1) > 0.6:
                    # Keep higher confidence rule
                    if rule.confidence >= other.confidence:
                        other.deprecated = True
                    else:
                        rule.deprecated = True
                    removed += 1
                    seen_ids.add(other.id)

    print(f"Deduplication: marked {removed} rules as deprecated")
    return removed


def demonstrate_grow_and_refine():
    """Demonstrate the full Grow-and-Refine lifecycle."""
    playbook = Playbook()

    print("=== ACE Grow-and-Refine Demo ===\n")

    # Simulate lessons learned from three extraction episodes
    episodes = [
        {
            "layer": "extraction",
            "rule": "Scientific notation like '5×10⁻⁷' should be parsed as 5e-7 float",
            "source": "VO₂ paper extraction error",
            "confidence": 0.9,
        },
        {
            "layer": "extraction",
            "rule": "When table caption contains 'typical' or 'representative', reduce numerical confidence to 0.6",
            "source": "BaTiO₃ review paper",
            "confidence": 0.85,
        },
        {
            "layer": "synthesis_protocols",
            "rule": "Excess Bi precursor (ratio 1.1-1.3:1) is standard in BiFeO₃ CVD to compensate for Bi volatility",
            "source": "BiFeO₃ CVD paper",
            "confidence": 0.8,
        },
        {
            "layer": "characterization",
            "rule": "Raman D band (~1350 cm⁻¹) + G band (~1580 cm⁻¹) together indicate graphene/carbon contamination",
            "source": "Carbon nanotube paper",
            "confidence": 0.95,
        },
        # Potential duplicate of rule 1
        {
            "layer": "extraction",
            "rule": "Parse '5 × 10⁻⁷ Torr' as 5e-7 in numerical fields; use scientific notation",
            "source": "LSMO paper extraction",
            "confidence": 0.88,
        },
    ]

    print("Adding rules (with duplicate detection):\n")
    for ep in episodes:
        layer_rules = [r for r in playbook.rules if r.layer == ep["layer"] and not r.deprecated]
        dup = is_duplicate(ep["rule"], layer_rules, client=None)  # Skip LLM dedup in demo

        if dup:
            print(f"  MERGE: '{ep['rule'][:50]}...'")
            merge_rule(dup, ep["rule"], ep["confidence"])
        else:
            rule = append_rule(playbook, ep["layer"], ep["rule"], ep["source"], ep["confidence"])
            print(f"  APPEND [{ep['layer']}]: '{rule.rule[:60]}...'")

    print(f"\nTotal rules: {len(playbook.rules)}")

    # Dedup
    print("\nRunning deduplication...")
    deduplicate_playbook(playbook)
    active = [r for r in playbook.rules if not r.deprecated]
    print(f"Active rules after dedup: {len(active)}")

    # Show context string
    print("\n--- Playbook as Context String ---")
    print(playbook.to_context_string())

    # Save
    save_playbook(playbook)


if __name__ == "__main__":
    demonstrate_grow_and_refine()

# TODO: Replace word-overlap dedup with embedding cosine similarity
# TODO: Track rule usage: increment usage_count when rule is applied
# TODO: Decay confidence of rules that haven't been validated recently
# TODO: Partition playbook by material class (oxides, alloys, 2D materials)
