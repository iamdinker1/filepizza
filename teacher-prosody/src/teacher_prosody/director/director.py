"""Rule-based teaching performance director (v0, explainable baseline).

Inputs:  lesson script as beats (text + beat label), a concept map (key terms, the relation
         being taught, variables that change), language spans (computed), teacher profile.
Output:  PerformancePlan with per-beat pace curve / register / pauses / final contour and
         per-word emphasis with channel targets and a written reason for every decision.

Design rules (each is a hypothesis to validate against the teacher's real behaviour):
  * Emphasis budget: <= 2 emphasised words per clause, one strong (3) per beat. Over-emphasis is
    the main way expressive TTS "overacts".
  * Contrastive focus beats lexical stress: negations (NOT / nahi), quantity changes (double,
    half, zyada), and the variable that changes in a conditional get focus.
  * New key terms get emphasis on FIRST mention only (given information is de-accented).
  * Reveals get a preceding pause and a slower rate; build-ups slow down at their end.
  * How emphasis is realised (pitch vs length vs loudness vs pause) comes from the teacher's
    measured `emphasis_channels`, not from a generic TTS default.
"""
from __future__ import annotations

import re

import numpy as np

from ..preprocess.lang import word_langs
from .plan import BeatPlan, PerformancePlan, WordPlan
from .text_norm import normalise

DEFAULT_BEAT = {  # rate_start, rate_end, register_st, energy_db, pause_before, pause_after, final
    "hook": (1.0, 1.0, 1.0, 1.5, 0.3, 0.35, "fall"),
    "introduce_concept": (0.95, 0.9, 0.5, 1.0, 0.5, 0.3, "fall"),
    "definition": (0.88, 0.85, 0.0, 0.5, 0.35, 0.4, "fall"),
    "explanation": (1.0, 1.0, 0.0, 0.0, 0.25, 0.25, "fall"),
    "build_up": (1.0, 0.88, 0.5, 1.0, 0.25, 0.35, "level"),
    "rhetorical_question": (0.95, 0.9, 1.0, 1.0, 0.25, 0.8, "rise"),
    "reveal": (0.85, 0.85, 0.0, 1.5, 0.7, 0.45, "fall"),
    "example": (1.05, 1.05, 0.5, 0.0, 0.3, 0.3, "fall"),
    "formula": (0.8, 0.8, 0.0, 0.5, 0.35, 0.45, "fall"),
    "derivation_step": (0.9, 0.9, 0.0, 0.0, 0.3, 0.35, "level"),
    "numerical_step": (0.9, 0.9, 0.0, 0.0, 0.3, 0.35, "fall"),
    "warning": (0.85, 0.85, -0.5, 2.0, 0.4, 0.4, "fall"),
    "common_mistake": (0.9, 0.9, 0.0, 1.5, 0.35, 0.4, "fall"),
    "recap": (1.05, 1.0, -0.5, -0.5, 0.4, 0.3, "fall"),
    "transition": (1.0, 1.0, 1.0, 1.0, 0.6, 0.25, "fall"),
    "repetition": (0.9, 0.9, 0.0, 1.0, 0.3, 0.35, "fall"),
    "encouragement": (1.0, 1.0, 1.5, 0.5, 0.3, 0.3, "fall"),
    "aside": (1.1, 1.1, -0.5, -1.0, 0.3, 0.3, "fall"),
}
NEGATION = {"not", "no", "never", "nahi", "nahin", "na", "mat", "zero", "kabhi", "nothing", "neither", "none", "नहीं", "मत"}
QUANTITY = {"double", "half", "twice", "triple", "zyada", "jyada", "kam", "more", "less", "increase", "decrease",
            "badh", "ghat", "doguna", "aadha", "same", "constant", "only", "sirf"}
FUNCTION = {"hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "the", "a", "an", "is", "are", "to", "toh", "ab",
            "aur", "and", "of", "in", "on", "par", "yeh", "ye", "woh", "wo", "hota", "hoti", "hote", "hoga", "kar",
            "diya", "de", "do", "gaya", "tha", "thi", "kya", "we", "you", "it", "this", "that", "yahan", "wahan"}


CLAUSE_OPENERS = {"aur", "and", "but", "lekin", "kyunki", "because", "toh", "so", "par", "magar", "jabki", "while"}


def _clean(w: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]", "", w.lower())


def _profile_beat(profile: dict | None, beat: str) -> tuple:
    d = list(DEFAULT_BEAT.get(beat, DEFAULT_BEAT["explanation"]))
    if not profile:
        return tuple(d)
    pb = profile.get("per_beat", {}).get(beat)
    if pb and pb.get("n", 0) >= 5:  # only trust the profile with some evidence
        r = pb.get("rate_rel", {}).get("median")
        if r:
            ratio = d[1] / d[0] if d[0] else 1.0
            d[0], d[1] = r, r * ratio
        f = pb.get("f0_median_rel_st", {}).get("median")
        if f is not None:
            d[2] = f
        e = pb.get("energy_rel_db", {}).get("median")
        if e is not None:
            d[3] = e
        ft = pb.get("final_types", {})
        if ft:
            d[6] = max(ft, key=ft.get) if max(ft, key=ft.get) in ("rise", "fall", "level") else d[6]
    if beat == "reveal":
        p = profile.get("pre_reveal", {}).get("pause_before_s", {}).get("median")
        if p:
            d[4] = p
    if beat == "rhetorical_question":
        p = profile.get("questions", {}).get("post_question_pause_s", {}).get("median")
        if p:
            d[5] = p
    return tuple(d)


def _phrase_pause(profile: dict | None) -> float:
    pz = (profile or {}).get("pauses", {})
    for k in ("phrase_boundary", "sentence_boundary"):
        if pz.get(k, {}).get("n", 0) >= 5:
            return float(np.clip(pz[k]["median_s"], 0.1, 0.6))
    return 0.22


def _channel_weights(profile: dict | None) -> dict:
    ch = (profile or {}).get("emphasis_channels") or {"pitch": 0.45, "duration": 0.3, "energy": 0.15, "pause": 0.1}
    tot = sum(ch.values()) or 1
    return {k: ch.get(k, 0) / tot for k in ("pitch", "duration", "energy", "pause")}


def plan_lesson(lesson_id: str, teacher_id: str, beats: list[dict], concept_map: dict | None = None,
                profile: dict | None = None, lexicon: dict | None = None) -> PerformancePlan:
    """beats: [{"text": ..., "beat": ..., "emphasis": [optional author-marked words]}]

    concept_map: {"key_terms": [...], "changing": [...variables that change...],
                  "relation": "a = F/m", "misconception": "..."}
    """
    concept_map = concept_map or {}
    key_terms = {_clean(t) for t in concept_map.get("key_terms", [])}
    changing = {_clean(t) for t in concept_map.get("changing", [])}
    seen_terms: set[str] = set()
    cw = _channel_weights(profile)
    out: list[BeatPlan] = []
    for bi, b in enumerate(beats):
        beat = b.get("beat", "explanation")
        rs, re_, reg, en, pb, pa, fin = _profile_beat(profile, beat)
        prev_beat = beats[bi - 1]["beat"] if bi > 0 else ""
        rationale = [f"beat={beat}: rate {rs:.2f}->{re_:.2f}x, register {reg:+.1f} st, energy {en:+.1f} dB"]
        if beat == "reveal" and prev_beat in ("rhetorical_question", "build_up"):
            pb = max(pb, 0.6)
            rationale.append(f"pause {pb:.2f}s before reveal: gives the student time to answer the preceding {prev_beat}")
        if beat == "build_up" and bi + 1 < len(beats) and beats[bi + 1].get("beat") == "reveal":
            re_ = min(re_, 0.85)
            rationale.append("build-up slows toward its end because a reveal follows")
        norm = normalise(b["text"], lexicon)
        spoken_words = norm.spoken.split()
        langs = word_langs(spoken_words)
        words = [WordPlan(i=i, w=w) for i, w in enumerate(spoken_words)]
        author = {_clean(x) for x in b.get("emphasis", [])}
        # (level, priority score, reason); priority orders candidates inside the budget
        scores: dict[int, tuple[int, float, str]] = {}
        for wp in words:
            c = _clean(wp.w)
            if not c or c in FUNCTION and c not in NEGATION:
                continue
            if wp.w.isupper() and len(c) > 1 or c in author:
                scores[wp.i] = (3, 3.3, "author-marked emphasis in the script")
            elif c in NEGATION:
                scores[wp.i] = (3, 3.2, "negation: contrastive focus against the student's likely assumption")
            elif c in changing:
                scores[wp.i] = (2, 2.6, "the variable that changes in this relation")
            elif c in QUANTITY:
                scores[wp.i] = (2, 2.4, "quantity change carries the new information")
            elif c in key_terms and c not in seen_terms:
                scores[wp.i] = (2, 2.0, "first mention of a key term (later mentions are given, de-accented)")
            if c in key_terms:
                seen_terms.add(c)  # given from here on, including later in this beat
        # emphasis budget: <= 2 per clause, <= 3 per beat, one strong (3) per beat
        clause_id, cid = [], 0
        for wp in words:
            if _clean(wp.w) in CLAUSE_OPENERS and clause_id:
                cid += 1
            clause_id.append(cid)
            if re.search(r"[,;:?!.।]$", wp.w):
                cid += 1
        kept: dict[int, tuple[int, float, str]] = {}
        for c in set(clause_id):
            cand = sorted([i for i in scores if clause_id[i] == c], key=lambda i: (-scores[i][1], i))
            for i in cand[:2]:
                kept[i] = scores[i]
        for i in sorted(kept, key=lambda i: (-kept[i][1], i))[3:]:
            del kept[i]
        strong = sorted([i for i in kept if kept[i][0] == 3], key=lambda i: (-kept[i][1], i))
        for i in strong[1:]:
            kept[i] = (2, kept[i][1], kept[i][2] + " (downgraded: one strong focus per beat)")
        for i, (lvl, _, why) in kept.items():
            wp = words[i]
            wp.emphasis = lvl
            wp.pitch_st = round(1.2 * lvl * (0.5 + cw["pitch"]), 2)
            wp.dur_scale = round(1 + 0.12 * lvl * (0.5 + cw["duration"]), 3)
            wp.energy_db = round(1.0 * lvl * (0.5 + cw["energy"]), 2)
            # a pre-emphasis pause only for the one strong focus (or the answer word of a reveal)
            if cw["pause"] >= 0.08 and (lvl == 3 or beat == "reveal" and lvl >= 2):
                wp.pre_pause_s = round(0.15 + 0.3 * cw["pause"], 2)
            wp.reason = why
        # explicit clause-boundary pauses (so every backend and the PSOLA editor get the same timing)
        php = _phrase_pause(profile)
        for k in range(1, len(words)):
            if re.search(r"[,;:—–]$", words[k - 1].w) and words[k].pre_pause_s < php:
                words[k].pre_pause_s = round(php, 2)
                if not words[k].reason:
                    words[k].reason = "clause-boundary pause (teacher's median phrase pause)"
        switches = sum(1 for a, c in zip(langs, langs[1:]) if {a, c} == {"hi", "en"})
        if switches:
            rationale.append(f"{switches} Hindi/English switch(es): keep one accent; no register jump at switch points")
        if fin == "rise" and re.search(r"\b(kya|kyun|kaise|kitna|kaun)\b", b["text"].lower()):
            rationale.append("wh-question with kya/kyun: check teacher profile - Hindi wh-questions often FALL; rise kept only if profile agrees")
        out.append(BeatPlan(
            beat_id=f"{lesson_id}-b{bi:03d}", beat=beat, text=b["text"], spoken=norm.spoken,
            prev_text=beats[bi - 1]["text"] if bi > 0 else "", next_text=beats[bi + 1]["text"] if bi + 1 < len(beats) else "",
            rate_start=round(rs, 3), rate_end=round(re_, 3), register_st=round(reg, 2), energy_db=round(en, 2),
            pause_before_s=round(pb, 2), pause_after_s=round(pa, 2), final_contour=fin, words=words,
            formulas=[{"written": f.written, "spoken": f.spoken} for f in norm.formulas],
            must_review=bool(norm.formulas) or switches >= 3, rationale=rationale))
    return PerformancePlan(lesson_id, teacher_id, out, profile_version=(profile or {}).get("version", ""))
