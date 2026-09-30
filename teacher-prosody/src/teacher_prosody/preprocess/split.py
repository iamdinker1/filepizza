"""Session-level train/val/test splits with leakage checks.

The split unit is the teaching *session* (a whole lecture), never the utterance: utterances
from one session share mic, room, mood and topic, so utterance-level splits leak. Held-out test
sessions must also be excluded from retrieval indexes used by style-retrieval architectures.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from ..schema import Recording, Utterance


def _bucket(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(recs: list[Recording], val_frac: float = 0.15, test_frac: float = 0.15,
                  force_test_sessions: set[str] | None = None) -> dict[str, str]:
    """Deterministic, stratified by teacher x recording_condition, at session granularity.

    Every teacher x condition stratum with >= 3 sessions gets at least one val and one test
    session so that recording-condition effects can be separated from teaching style.
    """
    force_test_sessions = force_test_sessions or set()
    strata: dict[tuple, list[str]] = defaultdict(list)
    for r in recs:
        key = (r.teacher_id, r.recording_condition)
        if r.session_id not in strata[key]:
            strata[key].append(r.session_id)
    split: dict[str, str] = {}
    for key, sessions in strata.items():
        sessions = sorted(sessions, key=lambda s: _bucket(f"{key}:{s}"))
        n = len(sessions)
        n_test = max(1, round(test_frac * n)) if n >= 3 else 0
        n_val = max(1, round(val_frac * n)) if n >= 3 else 0
        for i, s in enumerate(sessions):
            split[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for s in force_test_sessions:
        split[s] = "test"
    for r in recs:
        r.split = split.get(r.session_id, "train")
    return split


def check_leakage(utts: list[Utterance], retrieval_index_utt_ids: set[str] | None = None) -> list[str]:
    """Return human-readable problems (empty list = clean)."""
    problems = []
    by_session: dict[str, set[str]] = defaultdict(set)
    for u in utts:
        by_session[u.session_id].add(u.split)
    for s, sp in by_session.items():
        if len(sp) > 1:
            problems.append(f"session {s} spans splits {sorted(sp)}")
    if retrieval_index_utt_ids:
        leaked = [u.utt_id for u in utts if u.split == "test" and u.utt_id in retrieval_index_utt_ids]
        if leaked:
            problems.append(f"{len(leaked)} test utterances are in the retrieval index (e.g. {leaked[:3]})")
    texts = defaultdict(set)
    for u in utts:
        if len(u.text) > 40:
            texts[u.text.strip().lower()].add(u.split)
    dup = [t for t, sp in texts.items() if "test" in sp and len(sp) > 1]
    if dup:
        problems.append(f"{len(dup)} identical long sentences occur in test and another split (repeated lecture?)")
    return problems
