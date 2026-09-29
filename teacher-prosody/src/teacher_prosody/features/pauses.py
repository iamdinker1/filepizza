"""Pause detection and context-conditional pause typing.

Types (rule-based, each with a confidence; revise thresholds from the pilot):
  micro             < 150 ms, inside a phrase
  breath            pause containing an audible unvoiced breath-like region
  phrase_boundary   after a comma / clause connector (toh, aur, lekin, kyunki, so, and, but ...)
  sentence_boundary after sentence-final punctuation
  post_question     after a question (text '?' or rising final), >= 300 ms
  rhetorical        after a rhetorical/Socratic question beat, >= 500 ms, before the teacher answers
  pre_emphasis      >= 150 ms directly before a highly prominent word, not at a sentence boundary
  thinking          next to a filled pause (umm, aaa, hmm) or a long (>600 ms) mid-phrase gap
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

CONNECTORS = {"toh", "to", "aur", "lekin", "par", "kyunki", "kyuki", "isliye", "matlab", "yaani", "jab", "tab",
              "agar", "so", "and", "but", "because", "therefore", "then", "now", "ab", "phir", "fir", "ki", "ke"}
FILLED = {"umm", "um", "uh", "aa", "aaa", "hmm", "mm", "err", "haan", "accha", "achha"}


@dataclass
class Pause:
    start: float
    end: float
    kind: str = "unknown"
    conf: float = 0.5
    prev_word: str = ""
    next_word: str = ""
    has_breath: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def dur(self) -> float:
        return self.end - self.start


def pauses_from_mask(speech: np.ndarray, times: np.ndarray, min_pause: float = 0.05) -> list[Pause]:
    """Interior non-speech runs (leading/trailing silence excluded)."""
    hop = times[1] - times[0]
    sil = ~speech
    d = np.diff(np.concatenate([[0], sil.astype(int), [0]]))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if a == 0 or b >= len(sil):
            continue
        dur = (b - a) * hop
        if dur >= min_pause:
            out.append(Pause(float(times[a]), float(times[a] + dur)))
    return out


def pauses_from_words(words, min_pause: float = 0.05) -> list[Pause]:
    out = []
    for a, b in zip(words[:-1], words[1:]):
        g = b.start - a.end
        if g >= min_pause:
            out.append(Pause(a.end, b.start, prev_word=a.w, next_word=b.w))
    return out


def _clean(w: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]", "", w.lower())


def classify_pauses(pauses: list[Pause], words, prominence: dict[int, float] | None = None,
                    breaths: list[tuple[float, float]] | None = None, question_ends: set[int] | None = None,
                    rhetorical_ends: set[int] | None = None, sentence_ends: set[int] | None = None) -> list[Pause]:
    """Assign a context type to each pause.

    words          aligned words (with punctuation kept in .w if available)
    prominence     word index -> prominence z-score (from features.words)
    question_ends  indices of words ending a question (text '?' or rising final contour)
    rhetorical_ends indices of words ending a rhetorical/Socratic question beat
    sentence_ends  indices of words ending a sentence
    """
    prominence = prominence or {}
    breaths = breaths or []
    question_ends = question_ends or set()
    rhetorical_ends = rhetorical_ends or set()
    sentence_ends = set(sentence_ends or set())
    for i, w in enumerate(words):
        if re.search(r"[.!?।]$", w.w):
            sentence_ends.add(i)
    starts = np.array([w.start for w in words]) if words else np.zeros(0)
    ends = np.array([w.end for w in words]) if words else np.zeros(0)
    for p in pauses:
        # word before = last word ending at/before pause start (+20 ms tolerance)
        prev_i = int(np.searchsorted(ends, p.start + 0.02, side="right") - 1) if len(ends) else -1
        next_i = int(np.searchsorted(starts, p.end - 0.02, side="left")) if len(starts) else -1
        prev_w = words[prev_i] if 0 <= prev_i < len(words) else None
        next_w = words[next_i] if 0 <= next_i < len(words) else None
        p.prev_word = prev_w.w if prev_w else ""
        p.next_word = next_w.w if next_w else ""
        p.has_breath = any(b0 < p.end and b1 > p.start for b0, b1 in breaths)
        pw, nw = _clean(p.prev_word), _clean(p.next_word)
        comma = bool(re.search(r"[,;:—–-]$", p.prev_word))
        kind, conf = "other", 0.4
        if prev_i in rhetorical_ends and p.dur >= 0.5:
            kind, conf = "rhetorical", 0.8
        elif prev_i in question_ends and p.dur >= 0.3:
            kind, conf = "post_question", 0.8
        elif pw in FILLED or nw in FILLED:
            kind, conf = "thinking", 0.7
        elif p.has_breath and p.dur >= 0.15:
            kind, conf = "breath", 0.6
        elif prev_i in sentence_ends:
            kind, conf = "sentence_boundary", 0.8
        elif p.dur < 0.15 and not comma:
            kind, conf = "micro", 0.7
        elif next_i in prominence and prominence[next_i] > 1.0 and p.dur >= 0.15:
            kind, conf = "pre_emphasis", 0.6
        elif comma or pw in CONNECTORS or nw in CONNECTORS:
            kind, conf = "phrase_boundary", 0.6
        elif p.dur < 0.15:
            kind, conf = "micro", 0.7
        elif p.dur > 0.6:
            kind, conf = "thinking", 0.4
        p.kind, p.conf = kind, conf
    return pauses


def pause_distribution(pauses: list[Pause]) -> dict:
    """Duration distribution per pause type (count, median, IQR, p90) - the context-conditional
    summary the brief asks for."""
    out = {}
    kinds = sorted({p.kind for p in pauses})
    for k in kinds:
        d = np.array([p.dur for p in pauses if p.kind == k])
        out[k] = {"n": int(d.size), "median_s": float(np.median(d)), "q25_s": float(np.percentile(d, 25)),
                  "q75_s": float(np.percentile(d, 75)), "p90_s": float(np.percentile(d, 90))}
    return out
