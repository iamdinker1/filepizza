"""Rank N cached-lecture candidates per beat; gate, regenerate or send to human review.

Sub-scores (each 0..1, higher is better; NaN = not measured):
  text        1 - CER of ASR transcript vs planned spoken text           (needs asr_fn)
  formula     every planned formula reading found in the transcript       (needs asr_fn)
  speaker     cosine similarity to the teacher centroid, rescaled         (needs embed_fn)
  emphasis    planned emphasis words are the most prominent ones          (needs word timings)
  pauses      planned pause positions realised (F1 with tolerance)        (needs word timings)
  pace        articulation rate close to plan target (teacher x beat)
  profile     utterance features inside the teacher's per-beat IQR (NOT rewarding variety)
  quality     clipping, SNR, internal dead air, click-like spikes
  continuity  F0 / loudness match with the previous chosen beat
Hard gates reject candidates regardless of score; formula and code-switch-heavy beats and
low-confidence alignments are routed to the human review queue even when they pass.
Weights are a prior - calibrate them on blinded human pairwise preferences (`calibrate_weights`).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio import Audio
from ..features.energy import extract_energy, speech_mask
from ..features.f0 import extract_f0
from ..features.rate import syllable_count, words_rate
from ..features.words import emphasis_alignment, word_prosody

DEFAULT_WEIGHTS = {"text": 3.0, "formula": 2.0, "speaker": 2.0, "emphasis": 1.5, "pauses": 1.0, "pace": 1.0,
                   "profile": 1.0, "quality": 1.5, "continuity": 1.0}
GATES = {"cer_max": 0.08, "speaker_min": 0.5, "clipping_max": 0.002, "dead_air_max_s": 1.8}


@dataclass
class Scored:
    beat_id: str
    k: int
    sub: dict
    total: float
    decision: str  # accept | reject | review
    reasons: list[str] = field(default_factory=list)


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\wऀ-ॿ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = _norm_text(ref), _norm_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.cer(r, h))


def pause_f1(planned_gaps: list[int], words, min_gap: float = 0.12) -> float:
    """planned_gaps: indices i meaning 'pause before word i'. Realised if gap before word i >= min_gap."""
    realised = {i for i in range(1, len(words)) if words[i].start - words[i - 1].end >= min_gap}
    planned = set(planned_gaps)
    if not planned and not realised:
        return 1.0
    tp = len(planned & realised)
    p = tp / len(realised) if realised else 0.0
    r = tp / len(planned) if planned else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _quality(audio: Audio, f0, e, sp) -> tuple[float, dict]:
    clip = float(np.mean(np.abs(audio.y) >= 0.999))
    hop = e.times[1] - e.times[0]
    sil = ~sp
    d = np.diff(np.concatenate([[0], sil.astype(int), [0]]))
    runs = [(b - a) * hop for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if a > 0 and b < len(sil)]
    dead = max(runs) if runs else 0.0
    # click-like: sample-to-sample jumps far above the local scale
    dy = np.abs(np.diff(audio.y))
    thr = 12 * (np.median(dy) + 1e-6)
    clicks = int(np.sum(dy > max(thr, 0.3)))
    snr = e.snr_db
    q = 1.0
    q -= min(1.0, clip * 200)
    q -= 0.3 if dead > GATES["dead_air_max_s"] else 0.0
    q -= min(0.3, clicks * 0.05)
    q -= 0.2 if snr < 20 else 0.0
    return max(0.0, q), {"clipping": clip, "dead_air_s": dead, "clicks": clicks, "snr_db": snr}


def score_candidate(cand, beat_plan, teacher: dict, prev_tail: dict | None = None,
                    asr_fn: Callable[[Audio], str] | None = None, embed_fn: Callable[[Audio], np.ndarray] | None = None,
                    teacher_centroid: np.ndarray | None = None, weights: dict | None = None) -> Scored:
    """teacher: {"ref_hz": float, "artic_rate_sps": float, "per_beat": profile per_beat dict (optional)}"""
    weights = weights or DEFAULT_WEIGHTS
    a = cand.audio
    f0 = extract_f0(a, floor=60, ceiling=500, two_pass=False)
    e = extract_energy(a)
    sp = speech_mask(e, f0.voiced)
    sub, reasons, info = {}, [], {}

    hyp = asr_fn(a) if asr_fn else None
    if hyp is not None:
        c = cer(beat_plan.spoken, hyp)
        sub["text"] = max(0.0, 1 - c)
        info["cer"] = c
        if c > GATES["cer_max"]:
            reasons.append(f"GATE text CER {c:.2f} > {GATES['cer_max']}")
        if beat_plan.formulas:
            ok = [_norm_text(f["spoken"]) in _norm_text(hyp) for f in beat_plan.formulas]
            sub["formula"] = float(np.mean(ok))
            if not all(ok):
                reasons.append("REVIEW formula reading not found verbatim in transcript")
    else:
        sub["text"] = np.nan
        reasons.append("REVIEW text not verified (no ASR backend)")

    if embed_fn is not None and teacher_centroid is not None:
        emb = embed_fn(a)
        cs = float(np.dot(emb, teacher_centroid) / (np.linalg.norm(emb) * np.linalg.norm(teacher_centroid) + 1e-9))
        sub["speaker"] = float(np.clip((cs - 0.3) / 0.6, 0, 1))
        info["speaker_cos"] = cs
        if cs < GATES["speaker_min"]:
            reasons.append(f"GATE speaker similarity {cs:.2f} < {GATES['speaker_min']}")
    else:
        sub["speaker"] = np.nan

    words = cand.words
    if words and len(words) == len(beat_plan.words):
        rows = word_prosody(words, f0, e, teacher.get("ref_hz") or f0.median())
        tgt = [w.i for w in beat_plan.words if w.emphasis >= 2]
        if tgt:
            al = emphasis_alignment(rows, tgt)
            sub["emphasis"] = float(0.5 * al["rank_pct"] + 0.5 * al["precision"])
            info["emphasis"] = al
        planned_gaps = [w.i for w in beat_plan.words if w.pre_pause_s > 0] + \
                       [i + 1 for i, w in enumerate(beat_plan.words[:-1]) if re.search(r"[,;:?!.।]$", w.w)]
        sub["pauses"] = pause_f1(sorted(set(planned_gaps)), words)
        rate = words_rate(words)["artic_rate_sps"]
    else:
        if words:
            reasons.append("REVIEW word timings do not match plan words (alignment needed)")
        syl = sum(syllable_count(w.w) for w in beat_plan.words)
        phon = float(sp.sum() * (e.times[1] - e.times[0]))
        rate = syl / phon if phon > 0.3 else np.nan
    target = (teacher.get("artic_rate_sps") or 5.0) * (beat_plan.rate_start + beat_plan.rate_end) / 2
    if np.isfinite(rate):
        sub["pace"] = float(np.exp(-abs(np.log(rate / target)) / 0.15))
        info["artic_rate"] = rate
        info["target_rate"] = target

    pb = (teacher.get("per_beat") or {}).get(beat_plan.beat, {})
    if pb.get("f0_range_st", {}).get("n", 0) >= 5:
        v = f0.hz[f0.voiced]
        if v.size > 10:
            rng = float(np.percentile(12 * np.log2(v / np.median(v)), 95) - np.percentile(12 * np.log2(v / np.median(v)), 5))
            q25, q75 = pb["f0_range_st"]["q25"], pb["f0_range_st"]["q75"]
            iqr = max(0.5, q75 - q25)
            dist = max(0.0, q25 - rng, rng - q75) / iqr
            sub["profile"] = float(np.exp(-dist))
            info["f0_range_st"] = rng

    qv, qinfo = _quality(a, f0, e, sp)
    sub["quality"] = qv
    info.update(qinfo)
    if qinfo["clipping"] > GATES["clipping_max"]:
        reasons.append("GATE clipping")
    if qinfo["dead_air_s"] > GATES["dead_air_max_s"]:
        reasons.append(f"GATE dead air {qinfo['dead_air_s']:.1f}s")

    if prev_tail:
        head = a.slice(0, min(0.8, a.duration))
        fh = extract_f0(head, floor=60, ceiling=500, two_pass=False).median()
        if np.isfinite(fh) and np.isfinite(prev_tail.get("f0_hz", np.nan)):
            jump = abs(12 * np.log2(fh / prev_tail["f0_hz"]))
            sub["continuity"] = float(np.exp(-max(0.0, jump - 1.5) / 2.0))
            info["join_f0_jump_st"] = jump

    num = sum(weights.get(k, 0) * v for k, v in sub.items() if v is not None and np.isfinite(v))
    den = sum(weights.get(k, 0) for k, v in sub.items() if v is not None and np.isfinite(v))
    total = num / den if den else 0.0
    decision = "reject" if any(r.startswith("GATE") for r in reasons) else "review" if any(r.startswith("REVIEW") for r in reasons) else "accept"
    if beat_plan.must_review and decision == "accept":
        decision, reasons = "review", reasons + ["REVIEW beat flagged must_review (formula / heavy code-switching)"]
    s = Scored(cand.beat_id, cand.k, {**sub, "_info": info}, float(total), decision, reasons)
    return s


def rank(cands, beat_plan, teacher: dict, **kw) -> list[Scored]:
    scored = [score_candidate(c, beat_plan, teacher, **kw) for c in cands]
    order = {"accept": 0, "review": 1, "reject": 2}
    return sorted(scored, key=lambda s: (order[s.decision], -s.total))


def regeneration_action(ranked: list[Scored], round_i: int, max_rounds: int = 3) -> dict:
    """What to do next for a beat after ranking."""
    best = ranked[0]
    if best.decision == "accept" and best.total >= 0.7:
        return {"action": "use", "k": best.k}
    if best.decision == "review" and best.total >= 0.7:
        return {"action": "use_after_review", "k": best.k,
                "why": "; ".join(r for r in best.reasons if r.startswith("REVIEW"))}
    if round_i + 1 >= max_rounds:
        return {"action": "human_review", "k": best.k, "why": "max regeneration rounds reached"}
    subs = [r.sub for r in ranked if r.decision != "reject"]
    if not subs:
        return {"action": "regenerate", "why": "all candidates failed hard gates", "change": "new seeds; shorten segment if text errors"}
    mean = lambda k: np.nanmean([s.get(k, np.nan) for s in subs]) if any(np.isfinite(s.get(k, np.nan)) for s in subs) else np.nan
    if np.isfinite(mean("emphasis")) and mean("emphasis") < 0.5:
        return {"action": "regenerate", "why": "planned emphasis not realised",
                "change": "add prosody reference clip for this beat, or PSOLA post-edit best candidate"}
    if np.isfinite(mean("pace")) and mean("pace") < 0.5:
        return {"action": "regenerate", "why": "pace off target", "change": "adjust speed / duration control"}
    return {"action": "regenerate", "why": f"best score {best.total:.2f} below 0.7", "change": "new seeds"}


def write_review_queue(path: str | Path, items: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, default=float) + "\n")
    return path


def calibrate_weights(pairs: list[tuple[dict, dict, int]], keys: list[str] | None = None) -> dict:
    """Bradley-Terry style fit: pairs = (sub_a, sub_b, 1 if humans preferred a else 0).
    Missing sub-scores are imputed with 0.5. Returns non-negative weights."""
    from sklearn.linear_model import LogisticRegression

    keys = keys or [k for k in DEFAULT_WEIGHTS]
    X, y = [], []
    for a, b, pref in pairs:
        va = np.array([a.get(k, np.nan) for k in keys], float)
        vb = np.array([b.get(k, np.nan) for k in keys], float)
        va[~np.isfinite(va)] = 0.5
        vb[~np.isfinite(vb)] = 0.5
        X.append(va - vb)
        y.append(pref)
        X.append(vb - va)
        y.append(1 - pref)
    clf = LogisticRegression(fit_intercept=False, C=1.0).fit(np.array(X), np.array(y))
    w = np.clip(clf.coef_[0], 0, None)
    return dict(zip(keys, map(float, w)))


def false_positives(auto: list[Scored], human_ok: dict[tuple[str, int], bool]) -> list[dict]:
    """Auto-accepted candidates that human raters rejected - inspect these before trusting the judge."""
    return [{"beat_id": s.beat_id, "k": s.k, "total": s.total, "sub": {k: v for k, v in s.sub.items() if k != "_info"}}
            for s in auto if s.decision == "accept" and human_ok.get((s.beat_id, s.k)) is False]


def scored_to_dict(s: Scored) -> dict:
    d = asdict(s)
    return json.loads(json.dumps(d, default=float))
