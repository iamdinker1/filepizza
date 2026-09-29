"""End-to-end analysis of one or more lecture recordings (manifest-driven).

manifest.yaml
  teacher_id: rajwant
  recordings:
    - path: data/raw/lecture01.mp4
      session_id: s01
      condition: studio            # studio | classroom | online_live | phone | unknown
      topic: newtons_laws
      lesson_type: conceptual
      transcript: data/raw/lecture01.vtt   # optional: .vtt/.srt/.json captions or MFA .TextGrid
      start: 120                   # optional crop (s)
      duration: 600
      consent_ref: CONSENT-2026-014
      consent_scope: [analysis]
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .analyze import analyze_recording, analyze_utterance
from .audio import Audio
from .dashboard import report
from .pedagogy.rules import annotate
from .preprocess.backends import proportional_align, read_captions, read_textgrid
from .preprocess.ingest import ingest, segment_at_pauses
from .preprocess.lang import lang_spans, word_langs
from .preprocess.quality import assess
from .preprocess.split import assign_splits, check_leakage
from .profile.teacher_profile import ProfileItem, build_profile
from .schema import Utterance, Word, write_json, write_jsonl


def _speech_runs(rf, t0: float, t1: float) -> list[tuple[float, float]]:
    e = rf.energy
    m = (e.times >= t0) & (e.times < t1)
    sp = rf.speech[m]
    tt = e.times[m]
    if not sp.any():
        return [(t0, t1)]
    d = np.diff(np.concatenate([[0], sp.astype(int), [0]]))
    hop = e.times[1] - e.times[0]
    return [(float(tt[a]), float(tt[b - 1] + hop)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _group_words(words: list[Word], min_pause: float = 0.45, max_len: float = 15.0) -> list[list[Word]]:
    groups: list[list[Word]] = []
    for w in words:
        if groups and (w.start - groups[-1][-1].end < min_pause) and (w.end - groups[-1][0].start < max_len):
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def apply_gold_beats(utts: list[Utterance], gold: list[dict], annotator: str = "gold") -> dict[str, tuple[str, str]]:
    """Overwrite beat labels with gold spans (max time overlap). Returns utt_id -> (rule_label, gold_label)."""
    pairs = {}
    for u in utts:
        best, ov = None, 0.0
        for g in gold:
            o = min(u.end, g["end"]) - max(u.start, g["start"])
            if o > ov:
                best, ov = g, o
        if best is not None and ov > 0.5 * u.duration:
            pairs[u.utt_id] = (u.beat, best["beat"])
            u.beat, u.beat_source = best["beat"], f"gold:{best.get('annotator', annotator)}"
    return pairs


def utterances_for_recording(rec, audio: Audio, rf, transcript: str | None) -> list[Utterance]:
    utts: list[Utterance] = []
    base = dict(recording_id=rec.recording_id, teacher_id=rec.teacher_id, session_id=rec.session_id,
                topic=rec.topic, recording_condition=rec.recording_condition)
    if transcript and transcript.endswith(".TextGrid"):
        words, _ = read_textgrid(transcript)
        for gi, g in enumerate(_group_words(words)):
            text = " ".join(w.w for w in g)
            utts.append(Utterance(utt_id=f"{rec.recording_id}_{gi:04d}", start=g[0].start, end=g[-1].end, text=text,
                                  words=g, **base))
    elif transcript:
        segs = read_captions(transcript)
        for gi, s in enumerate(segs):
            toks = s.text.split()
            words = proportional_align(toks, audio, _speech_runs(rf, s.start, s.end))
            utts.append(Utterance(utt_id=f"{rec.recording_id}_{gi:04d}", start=s.start, end=s.end, text_asr=s.text,
                                  text=s.text, words=words, **base))
    else:
        for gi, s in enumerate(segment_at_pauses(audio)):
            utts.append(Utterance(utt_id=f"{rec.recording_id}_{gi:04d}", start=s.start, end=s.end, **base))
    for u in utts:
        if u.text:
            u.lang_spans = lang_spans(u.text)
            for w, l in zip(u.words, word_langs([w.w for w in u.words])):
                w.lang = l
    texts = [u.text for u in utts]
    if any(texts):
        for u, lab in zip(utts, annotate(texts)):
            if u.text:
                u.beat, u.beat_source = lab.beat, "rule"
    return utts


def run_manifest(manifest_path: str | Path, out_dir: str | Path, dashboard: bool = True) -> dict:
    man = yaml.safe_load(Path(manifest_path).read_text())
    out = Path(out_dir)
    teacher = man["teacher_id"]
    recs, items, all_utts, trace_html, label_pairs = [], [], [], [], {}
    for r in man["recordings"]:
        rec, audio = ingest(r["path"], out, teacher, r["session_id"], r.get("recording_id"), r.get("source_uri", ""),
                            r.get("condition", "unknown"), r.get("topic", ""), r.get("lesson_type", ""),
                            r.get("consent_ref", ""), r.get("consent_scope", []), start=r.get("start"),
                            duration=r.get("duration"))
        if not rec.cleared_for("analysis"):
            print(f"WARNING {rec.recording_id}: no consent record for 'analysis' - results must stay internal to the pilot")
        rf = analyze_recording(audio)
        utts = utterances_for_recording(rec, audio, rf, r.get("transcript"))
        if r.get("gold_beats"):
            gold = r["gold_beats"] if isinstance(r["gold_beats"], list) else json.loads(Path(r["gold_beats"]).read_text())
            label_pairs.update(apply_gold_beats(utts, gold))
        good = [w for u in utts for w in u.words if w.conf >= 0.5]
        if good:
            from .features.rate import syllable_count

            rf.syl_dur = sum(w.end - w.start for w in good) / max(1, sum(syllable_count(w.w) for w in good))
        for u in utts:
            ana = analyze_utterance(u, rf)
            m = rf.f0.slice_mask(u.start, u.end)
            em = (rf.energy.times >= u.start) & (rf.energy.times < u.end)
            align_conf = float(np.mean([w.conf for w in u.words])) if u.words else None
            u.quality = assess(audio.slice(u.start, u.end), _sub_f0(rf.f0, m), rf.energy, rf.speech[em], align_conf=align_conf)
            items.append(ProfileItem(u, ana, rf))
        recs.append((rec, rf))
        all_utts.extend(utts)
        write_json(out / "recordings" / f"{rec.recording_id}.json", asdict(rec))
        write_json(out / "recordings" / f"{rec.recording_id}.summary.json", rf.summary)
        # emphasis traces for the 3 utterances with the clearest single focus
        cands = [it for it in items if it.utt.recording_id == rec.recording_id and len(it.ana.words) >= 3
                 and "align_low" not in it.utt.quality.flags]
        cands.sort(key=lambda it: -max(w.prominence for w in it.ana.words))
        for it in cands[:3]:
            trace_html.append(report.img(report.emphasis_trace(rf.audio, rf.f0, rf.energy, it.utt.words, it.ana.words, rf.ref_hz,
                                                               f"{it.utt.utt_id} [{it.utt.beat}] {it.utt.text[:90]}")))
    assign_splits([r for r, _ in recs])
    split_of = {r.session_id: r.split for r, _ in recs}
    for u in all_utts:
        u.split = split_of.get(u.session_id, "train")
    write_jsonl(out / "utterances.jsonl", all_utts)
    usable = [it for it in items if it.utt.quality.usable_for_style]
    profile = build_profile(usable, teacher)
    profile["version"] = "v0"
    profile["n_excluded_quality"] = len(items) - len(usable)
    write_json(out / "profile.json", json.loads(json.dumps(profile, default=float)))
    problems = check_leakage(all_utts)
    agreement = None
    if label_pairs:
        from .pedagogy.agreement import cohen_kappa, per_class_f1

        rule, gold = zip(*label_pairs.values())
        agreement = {"n": len(rule), "rule_vs_gold_kappa": cohen_kappa(list(rule), list(gold)),
                     "accuracy": float(np.mean([a == b for a, b in label_pairs.values()])),
                     "per_class": per_class_f1(list(gold), list(rule))}
    result = {"n_recordings": len(recs), "n_utterances": len(all_utts), "n_usable": len(usable),
              "leakage_problems": problems, "profile": str(out / "profile.json"), "annotator_agreement": agreement}
    if dashboard:
        pauses_by_kind: dict[str, list[float]] = {}
        for it in usable:
            for p in it.ana.pauses:
                pauses_by_kind.setdefault(p.kind, []).append(p.dur)
        sections = [("Recordings", "".join(report.img(report.overview_plots(rf)) + f"<pre>{json.dumps(rf.summary, indent=1, default=float)}</pre>"
                                           for _, rf in recs)),
                    ("Emphasis traces (clearest focus per recording)", "".join(trace_html) or "<p>No aligned words.</p>"),
                    ("Pauses by context", report.img(report.pause_plot(pauses_by_kind))),
                    ("Teacher profile", report.profile_tables(profile))]
        report.build(out / "dashboard.html", f"Teacher prosody – {teacher}", sections,
                     note="Word timings from a proportional fallback aligner are flagged align_low and excluded from traces.")
        result["dashboard"] = str(out / "dashboard.html")
    write_json(out / "run_summary.json", result)
    return result


def _sub_f0(f0, mask):
    from .features.f0 import F0Track

    return F0Track(f0.times[mask], f0.hz[mask], f0.conf[mask], f0.floor, f0.ceiling, f0.method)
