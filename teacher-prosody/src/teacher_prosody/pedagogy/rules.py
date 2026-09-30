"""Rule-based pedagogical beat annotator (transparent baseline for the hybrid annotator).

Uses lexical cues from `data/taxonomy.yaml`, punctuation, formula detection, discourse position
(a statement right after a question is a reveal candidate) and repetition overlap. Outputs a
label, confidence and the evidence that fired, so disagreements with experts can be inspected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..preprocess.lang import is_question_text, phonetic_key
from .taxonomy import load_taxonomy


@dataclass
class BeatLabel:
    beat: str
    conf: float
    evidence: list[str] = field(default_factory=list)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _tokens(t: str) -> set[str]:
    return set(re.findall(r"[\wऀ-ॿ]+", t.lower()))


def annotate(sentences: list[str]) -> list[BeatLabel]:
    tax = load_taxonomy()
    beats, priority = tax["beats"], tax["priority"]
    out: list[BeatLabel] = []
    for i, s in enumerate(sentences):
        low = _norm(s)
        key = phonetic_key(s)  # Devanagari transliterated; romanised spellings unified
        hits: dict[str, list[str]] = {}
        for name, spec in beats.items():
            for cue in spec.get("cues", []):
                c = cue.lower()
                if not c[0].isalnum():
                    found = c in low
                else:
                    ck = phonetic_key(c)
                    found = bool(re.search(r"(?<![\w])" + re.escape(ck) + r"(?![\w])", key))
                if found:
                    hits.setdefault(name, []).append(cue)
        if is_question_text(s):
            hits.setdefault("rhetorical_question", []).append("question form")
        if re.search(r"[=∝]", s):
            hits.setdefault("formula", []).append("equation")
        prev = sentences[i - 1].strip() if i > 0 else ""
        if prev and is_question_text(prev) and not is_question_text(s):
            hits.setdefault("reveal", []).append("follows question")
        for j in range(max(0, i - 3), i):
            a, b = _tokens(s), _tokens(sentences[j])
            if len(a) >= 4 and len(a & b) / max(1, len(a | b)) >= 0.6:
                hits.setdefault("repetition", []).append(f"overlaps sentence {j}")
        if not hits:
            out.append(BeatLabel("explanation", 0.4, ["default"]))
            continue
        order = priority + [b for b in hits if b not in priority]
        # a question that also "follows a question" is still a question
        best = next(b for b in order if b in hits)
        if "repetition" in hits and best in ("explanation", "build_up"):
            best = "repetition"
        n_ev = len(hits[best])
        conf = min(0.9, 0.5 + 0.15 * n_ev) if len(hits) == 1 else min(0.8, 0.45 + 0.1 * n_ev)
        ev = [f"{best}:{e}" for e in hits[best]] + [f"also:{b}" for b in hits if b != best]
        out.append(BeatLabel(best, conf, ev))
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!।])\s+", text.strip())
    return [p for p in parts if p]
