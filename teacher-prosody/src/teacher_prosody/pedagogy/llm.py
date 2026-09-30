"""LLM beat annotation (Claude API) - one of the annotators compared against expert gold.

Requires `pip install anthropic` and credentials (ANTHROPIC_API_KEY or an `ant auth login`
profile). UNVERIFIED in the build sandbox (no API credentials); `build_request` is unit-tested.
The prompt sees surrounding context because beats are defined by discourse position
(a reveal only exists after a question/build-up).
"""
from __future__ import annotations

import json

from .taxonomy import load_taxonomy

MODEL = "claude-opus-5-5"


def _schema(beats: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "beat": {"type": "string", "enum": beats},
                        "confidence": {"type": "number"},
                        "emphasis_words": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["i", "beat", "confidence", "emphasis_words", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def build_request(sentences: list[str], lesson_topic: str = "", model: str = MODEL) -> dict:
    tax = load_taxonomy()["beats"]
    beats = list(tax)
    defs = "\n".join(f"- {b}: {spec['definition']}" for b, spec in tax.items())
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    system = (
        "You annotate transcripts of an Indian physics teacher's lectures (Hindi, English and Hinglish) "
        "with pedagogical beats, to study how the teacher's voice changes with teaching intent. "
        "Label every sentence with exactly one beat from the taxonomy, judged from its role in the "
        "surrounding discourse, not from keywords alone. Also list the words a skilled teacher would "
        "emphasise in that sentence (0-3 words, copied exactly from the sentence). Use low confidence "
        "when two beats fit.\n\nTaxonomy:\n" + defs
    )
    user = (f"Lesson topic: {lesson_topic or 'unknown'}\n\nSentences:\n{numbered}\n\n"
            "Return one label per sentence index.")
    return {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": _schema(beats)}},
    }


def annotate(sentences: list[str], lesson_topic: str = "", model: str = MODEL, client=None) -> list[dict]:
    import anthropic

    client = client or anthropic.Anthropic()
    req = build_request(sentences, lesson_topic, model)
    # Server-side refusal fallback: a declined request is re-run on Anthropic's recommended model.
    resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **req)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"annotation refused: {getattr(resp, 'stop_details', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("annotation truncated; send fewer sentences per request")
    text = next(b.text for b in resp.content if b.type == "text")
    labels = json.loads(text)["labels"]
    by_i = {d["i"]: d for d in labels}
    return [by_i.get(i, {"i": i, "beat": "explanation", "confidence": 0.0, "emphasis_words": [], "reason": "missing"})
            for i in range(len(sentences))]
