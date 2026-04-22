"""
Shared Anthropic API helpers
============================

Utilities reused across modules 03 and 04 (and 02 where applicable) after
migrating from OpenAI's chat.completions API to Anthropic's messages API.

Anthropic differences to remember:
  - max_tokens is required
  - system prompt goes in a separate `system=` param (not in messages)
  - response body is a list of content blocks; use _extract_text to get text
  - no native response_format={"type": "json_object"}; rely on prompt + _parse_json
"""

from __future__ import annotations

import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 2048


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client; raises if ANTHROPIC_API_KEY is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env")
    return anthropic.Anthropic()


def extract_text(response) -> str:
    """Pull the first text block out of an Anthropic response."""
    return next(b.text for b in response.content if b.type == "text")


def parse_json(text: str) -> dict:
    """Extract JSON object from model output, tolerating optional markdown fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text}")
    return json.loads(match.group(0))


def json_message(
    client,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
) -> dict:
    """Send a single-turn user message and parse the reply as JSON.

    Mirrors the common OpenAI `response_format={"type": "json_object"}` pattern
    used by 03/04: the caller is expected to tell the model to output JSON only.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return parse_json(extract_text(response))
