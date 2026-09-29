"""Teacher profile: dynamic, context-conditional patterns rather than global averages.

What it captures (each with n, median and IQR so thin evidence is visible):
  per_beat        rate / F0 / energy / final contour / emphasis density by pedagogical beat,
                  all relative to the teacher's own session baseline
  pre_reveal      how pace and pausing change in the ~2 s before a reveal or conclusion
  questions       final contour types and post-question pause lengths
  transitions     energy / F0 reset at transition beats vs the previous utterance's end
  emphasis        which channels (pitch / duration / energy / pause) realise prominent words
  code_switch     pitch, pace and pausing at Hindi<->English switch points vs other word boundaries
  confounds       variance decomposition by beat / session / condition / topic
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..analyze import RecordingFeatures, UtteranceAnalysis
from ..features.pauses import pause_distribution
from ..features.f0 import to_semitones
from ..features.rate import rate_change_before, words_rate
from ..preprocess.lang import word_langs
from ..schema import Utterance
from .confounds import variance_decomposition


@dataclass
class ProfileItem:
    utt: Utterance
    ana: UtteranceAnalysis
    rf: RecordingFeatures


def _dist(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75))}


def _session_baselines(items: list[ProfileItem]) -> dict[str, dict]:
    by = defaultdict(list)
    for it in items:
        by[it.utt.session_id].append(it.ana.stats)
    base = {}
    for s, stats in by.items():
        base[s] = {k: float(np.nanmedian([st.get(k, np.nan) for st in stats]))
                   for k in ("artic_rate_sps", "f0_median_st", "f0_range_st", "energy_mean_db")}
    return base


LONG_SILENCE_S = 3.0  # longer gaps in lecture recordings are usually board work / demos, not prosody


def build_profile(items: list[ProfileItem], teacher_id: str) -> dict:
    items = sorted(items, key=lambda it: (it.utt.recording_id, it.utt.start))
    base = _session_baselines(items)
    rel_rows = []
    for it in items:
        st, b = it.ana.stats, base[it.utt.session_id]
        rel_rows.append({
            "beat": it.utt.beat or "unlabelled", "session": it.utt.session_id, "condition": it.utt.recording_condition,
            "topic": it.utt.topic,
            "rate_rel": st.get("artic_rate_sps", np.nan) / b["artic_rate_sps"] if b["artic_rate_sps"] else np.nan,
            "f0_median_rel_st": st.get("f0_median_st", np.nan) - b["f0_median_st"],
            "f0_range_st": st.get("f0_range_st", np.nan),
            "energy_rel_db": st.get("energy_mean_db", np.nan) - b["energy_mean_db"],
            "final_delta_st": st.get("final_delta_st", np.nan), "final_type": st.get("final_type", "unknown"),
            "emph_density": (np.mean([w.prominence > 1.0 for w in it.ana.words]) if it.ana.words else np.nan),
            # raw (un-normalised) values kept to test which features are recording-driven
            "raw_rate": st.get("artic_rate_sps", np.nan), "raw_f0_median_st": st.get("f0_median_st", np.nan),
            "raw_energy_dbfs": st.get("energy_abs_db", np.nan),
        })

    per_beat = {}
    for beat in sorted({r["beat"] for r in rel_rows}):
        rs = [r for r in rel_rows if r["beat"] == beat]
        per_beat[beat] = {
            "n": len(rs),
            "rate_rel": _dist([r["rate_rel"] for r in rs]),
            "f0_median_rel_st": _dist([r["f0_median_rel_st"] for r in rs]),
            "f0_range_st": _dist([r["f0_range_st"] for r in rs]),
            "energy_rel_db": _dist([r["energy_rel_db"] for r in rs]),
            "final_types": dict(Counter(r["final_type"] for r in rs)),
            "emph_density": _dist([r["emph_density"] for r in rs]),
        }

    # pre-reveal dynamics + pause before reveals
    pre = {"lead_in_slowdown": [], "lead_in_rate_slope": [], "pause_before_s": []}
    trans = {"energy_reset_db": [], "f0_reset_st": []}
    q = {"final_types": Counter(), "post_question_pause_s": []}
    for i, it in enumerate(items):
        prev = items[i - 1] if i > 0 and items[i - 1].utt.recording_id == it.utt.recording_id else None
        gap = it.utt.start - prev.utt.end if prev else np.nan
        if it.utt.beat == "reveal" and prev is not None:
            # lead-in = the utterance before the reveal (build-up / question): does it slow toward its end?
            pw = prev.utt.words
            if len(pw) >= 4 and np.mean([w.conf for w in pw]) >= 0.5:
                cut = len(pw) - max(1, len(pw) // 3)
                r1, r2 = words_rate(pw[:cut])["artic_rate_sps"], words_rate(pw[cut:])["artic_rate_sps"]
                pre["lead_in_slowdown"].append(r2 / r1 if r1 else np.nan)
            else:
                rc = rate_change_before(prev.utt.end, it.rf.rate_t, it.rf.rate, lead=2.0)
                pre["lead_in_rate_slope"].append(rc["slope"])
            if np.isfinite(gap) and gap <= LONG_SILENCE_S:
                pre["pause_before_s"].append(gap)
        if it.utt.beat == "transition" and prev is not None:
            e = it.rf.energy
            trans["energy_reset_db"].append(e.mean_norm(it.utt.start, it.utt.start + 1.0) - e.mean_norm(prev.utt.end - 1.0, prev.utt.end))
            f = it.rf.f0
            a = f.at(it.utt.start, it.utt.start + 1.0)
            b = f.at(prev.utt.end - 1.0, prev.utt.end)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size and b.size:
                trans["f0_reset_st"].append(float(12 * np.log2(np.median(a) / np.median(b))))
        if it.ana.stats.get("is_question"):
            q["final_types"][it.ana.stats.get("final_type", "unknown")] += 1
            nxt = items[i + 1] if i + 1 < len(items) and items[i + 1].utt.recording_id == it.utt.recording_id else None
            if nxt and nxt.utt.start - it.utt.end <= LONG_SILENCE_S:
                q["post_question_pause_s"].append(nxt.utt.start - it.utt.end)

    # emphasis channels: share of clearly prominent words that use each channel (multi-label:
    # a teacher may raise pitch AND lengthen AND pause before the same word)
    prom = [w for it in items for w in it.ana.words if w.prominence > 1.0]
    chan = {ch: float(np.mean([getattr(w, f"z_{ch}") > 0.8 for w in prom])) if prom else 0.0
            for ch in ("pitch", "duration", "energy", "pause")}

    return {
        "teacher_id": teacher_id,
        "n_utterances": len(items),
        "hours": float(sum(it.utt.duration for it in items) / 3600),
        "sessions": sorted(base),
        "session_baselines": base,
        "per_beat": per_beat,
        "pre_reveal": {k: _dist(v) for k, v in pre.items()},
        "questions": {"final_types": dict(q["final_types"]), "post_question_pause_s": _dist(q["post_question_pause_s"])},
        "transitions": {k: _dist(v) for k, v in trans.items()},
        "emphasis_channels": chan,
        "n_prominent_words": len(prom),
        "pauses": pause_distribution([p for it in items for p in it.ana.pauses]),
        "code_switch": code_switch_behaviour(items),
        "confounds": variance_decomposition(
            rel_rows, ["rate_rel", "f0_median_rel_st", "f0_range_st", "energy_rel_db", "final_delta_st",
                       "raw_rate", "raw_f0_median_st", "raw_energy_dbfs"]),
    }


def code_switch_behaviour(items: list[ProfileItem]) -> dict:
    """Compare word boundaries where the language switches with those where it does not."""
    rows = {"switch": defaultdict(list), "same": defaultdict(list)}
    for it in items:
        ws = it.utt.words
        if len(ws) < 3:
            continue
        langs = word_langs([w.w for w in ws])
        ref = it.rf.ref_hz
        for k in range(1, len(ws)):
            a, b = langs[k - 1], langs[k]
            if a not in ("hi", "en") or b not in ("hi", "en"):
                continue
            kind = "switch" if a != b else "same"
            fa = it.rf.f0.at(ws[k - 1].start, ws[k - 1].end)
            fb = it.rf.f0.at(ws[k].start, ws[k].end)
            fa, fb = fa[np.isfinite(fa)], fb[np.isfinite(fb)]
            if fa.size and fb.size:
                rows[kind]["f0_jump_st"].append(float(to_semitones(np.median(fb), ref) - to_semitones(np.median(fa), ref)))
            rows[kind]["gap_s"].append(ws[k].start - ws[k - 1].end)
            rows[kind]["next_word_dur_s"].append(ws[k].end - ws[k].start)
    return {kind: {k: _dist(v) for k, v in d.items()} for kind, d in rows.items()}
