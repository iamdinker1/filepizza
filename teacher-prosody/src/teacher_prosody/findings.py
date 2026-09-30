"""Teacher findings report from one real recording + an ASR/caption transcript.

Produces measured, teacher-specific patterns with sample sizes and bootstrap intervals, the
listening clips that back each claim, and a gold-set annotation sheet for expert correction.
Word timings come from the approximate nuclei aligner unless an MFA TextGrid is given, so
word-level numbers are labelled APPROXIMATE; utterance-level and pause numbers are acoustic.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analyze import analyze_recording, analyze_utterance  # noqa: E402
from .audio import load, save  # noqa: E402
from .dashboard import report  # noqa: E402
from .eval.longform import cadence_repetition, drift, phrase_final_shapes  # noqa: E402
from .features.f0 import to_semitones  # noqa: E402
from .features.pauses import classify_pauses, pauses_from_mask  # noqa: E402
from .features.rate import words_rate  # noqa: E402
from .pedagogy.rules import annotate  # noqa: E402
from .pipeline import _sub_f0, utterances_for_recording  # noqa: E402
from .preprocess.lang import is_question_text, phonetic_key, word_langs  # noqa: E402
from .preprocess.quality import assess  # noqa: E402
from .profile.teacher_profile import ProfileItem, build_profile  # noqa: E402
from .schema import Recording, write_json, write_jsonl  # noqa: E402

HI_FUNCTION = {"hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "men", "mein", "to", "toh", "ab", "aur", "ye", "yeh", "vo",
               "voh", "kya", "na", "ho", "hi", "bhi", "tha", "thi", "the", "ek", "is", "us", "par", "pe", "kar", "jo",
               "hum", "ham", "ap", "aap", "apko", "hamare", "hamara", "koi", "kuch", "jab", "tab", "agar", "lekin"}


def _boot_median_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or b.size < 3:
        return {"n_a": int(a.size), "n_b": int(b.size)}
    d = [np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size)) for _ in range(n)]
    return {"median_a": float(np.median(a)), "median_b": float(np.median(b)), "diff": float(np.median(a) - np.median(b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n_a": int(a.size), "n_b": int(b.size)}


def _dist(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90))}


def _key(w: str) -> str:
    return phonetic_key(re.sub(r"[^\wऀ-ॿ]", "", w)).strip()


def _png(fig):
    return report.img(report._png(fig))


def accent_peaks(rf, utts, min_st: float = 3.0, min_sep_s: float = 0.25) -> list[dict]:
    """Transcript-free pitch accents: F0 peaks >= min_st above each utterance's declination line.
    For each: size (st), whether energy also peaks (+3 dB over the utterance median within 100 ms),
    and whether a >= 150 ms pause ends within 400 ms before it (pre-accent pause)."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    st_all = to_semitones(rf.f0.hz, rf.ref_hz)
    hop = rf.f0.times[1] - rf.f0.times[0]
    e = rf.energy
    out = []
    for u in utts:
        m = rf.f0.slice_mask(u.start, u.end)
        t, st = rf.f0.times[m], st_all[m]
        ok = np.isfinite(st)
        if ok.sum() < 30:
            continue
        coef = np.polyfit(t[ok], st[ok], 1)
        resid = st - np.polyval(coef, t)
        keep = ok & (np.abs(resid) <= np.nanpercentile(np.abs(resid[ok]), 80))
        if keep.sum() >= 10:
            coef = np.polyfit(t[keep], st[keep], 1)
        r = np.where(ok, st - np.polyval(coef, t), -99.0)
        r = median_filter(r, size=5)
        idx, _ = find_peaks(r, height=min_st, distance=max(1, int(min_sep_s / hop)))
        em = (e.times >= u.start) & (e.times < u.end)
        e_med = float(np.median(e.db_norm[em & rf.speech[: len(e.db)]])) if (em & rf.speech[: len(e.db)]).any() else 0.0
        for i in idx:
            tp = float(t[i])
            win = (e.times >= tp - 0.1) & (e.times <= tp + 0.1)
            e_peak = bool(win.any() and e.db_norm[win].max() >= e_med + 3)
            pre = (e.times >= tp - 0.55) & (e.times < tp - 0.05)
            sil = ~rf.speech[: len(e.db)][pre]
            run, best = 0, 0
            for x in sil:
                run = run + 1 if x else 0
                best = max(best, run)
            out.append({"t": tp, "size_st": float(r[i]), "energy_copeak": e_peak,
                        "pre_pause": best * hop >= 0.15, "utt": u.utt_id})
    return out


def teacher_findings(audio_path: str, transcript: str, out_dir: str, teacher_id: str, session_id: str = "s01",
                     consent_ref: str = "", n_clips: int = 4) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio = load(audio_path)
    rf = analyze_recording(audio)
    rec = Recording(recording_id=f"{teacher_id}_{session_id}", teacher_id=teacher_id, session_id=session_id,
                    source_uri=str(audio_path), original_path=str(audio_path), sha256="", duration_s=audio.duration,
                    sample_rate=audio.sr, consent_ref=consent_ref, consent_scope=["analysis"])
    utts = utterances_for_recording(rec, audio, rf, transcript)
    good = [w for u in utts for w in u.words]
    from .features.rate import syllable_count

    rf.syl_dur = sum(w.end - w.start for w in good) / max(1, sum(syllable_count(w.w) for w in good))
    items, anas = [], []
    for u in utts:
        ana = analyze_utterance(u, rf)
        m = rf.f0.slice_mask(u.start, u.end)
        em = (rf.energy.times >= u.start) & (rf.energy.times < u.end)
        u.quality = assess(audio.slice(u.start, u.end), _sub_f0(rf.f0, m), rf.energy, rf.speech[em],
                           align_conf=float(np.mean([w.conf for w in u.words])) if u.words else None)
        anas.append(ana)
        items.append(ProfileItem(u, ana, rf))
    write_jsonl(out / "utterances.jsonl", utts)
    usable = [it for it in items if it.utt.quality.usable_for_style or it.utt.quality.flags == ["align_low"]]
    profile = build_profile(usable, teacher_id)
    profile["version"] = "v0-asr-approx-align"
    write_json(out / "profile.json", json.loads(json.dumps(profile, default=float)))

    minutes = audio.duration / 60
    F = {"teacher_id": teacher_id, "minutes": minutes, "n_utterances": len(utts), "n_words_asr": len(good),
         "alignment": "approximate (syllable-nuclei anchored); word-level numbers are APPROXIMATE"}
    v = rf.f0.hz[rf.f0.voiced]
    F["voice"] = {"median_f0_hz": rf.ref_hz, "f0_p5_p95_hz": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
                  "f0_span_p5_p95_st": float(12 * np.log2(np.percentile(v, 95) / np.percentile(v, 5))),
                  "utt_f0_range_st": _dist([a.stats.get("f0_range_st") for a in anas]),
                  "speech_fraction": float(rf.speech.mean()), "snr_db": rf.energy.snr_db,
                  "artic_rate_sps_recording": float(np.nanmedian(rf.rate)),
                  "utt_artic_rate_sps": _dist([a.stats.get("artic_rate_sps") for a in anas]),
                  "words_per_speech_minute": len(good) / max(1e-6, rf.speech.mean() * minutes)}

    # ---- pauses (acoustic, from the speech mask)
    gaps = [b.start - a.end for a, b in zip(utts, utts[1:])]
    intra = []
    for u in utts:
        em = (rf.energy.times >= u.start) & (rf.energy.times < u.end)
        ps = pauses_from_mask(rf.speech[em], rf.energy.times[em], min_pause=0.08) if em.sum() > 3 else []
        q_end = {len(u.words) - 1} if is_question_text(u.text) else set()
        intra += classify_pauses(ps, u.words, question_ends=q_end, breaths=rf.breaths)
    by_kind: dict[str, list[float]] = {}
    for p in intra:
        by_kind.setdefault(p.kind, []).append(p.dur)
    F["pauses"] = {"between_utterances_s": _dist(gaps), "n_gaps_over_1_5s": int(sum(g > 1.5 for g in gaps)),
                   "pauses_per_minute": (len(intra) + len(gaps)) / minutes,
                   "within_utterance_by_type": {k: _dist(v) for k, v in by_kind.items()},
                   "breath_candidates_per_minute": len(rf.breaths) / minutes}

    # ---- questions
    q_idx = [i for i, u in enumerate(utts) if is_question_text(u.text)]
    nq_idx = [i for i in range(len(utts)) if i not in set(q_idx)]
    fin_q = Counter(anas[i].stats.get("final_type") for i in q_idx)
    fin_nq = Counter(anas[i].stats.get("final_type") for i in nq_idx)
    gap_after = lambda idx: [utts[i + 1].start - utts[i].end for i in idx if i + 1 < len(utts)]
    tails = Counter(_key(utts[i].words[-1].w) for i in q_idx if utts[i].words)
    F["questions"] = {"n_question_utterances": len(q_idx), "share_of_utterances": len(q_idx) / max(1, len(utts)),
                      "per_minute": len(q_idx) / minutes, "final_types_question": dict(fin_q),
                      "final_types_other": dict(fin_nq), "question_tail_words": dict(tails.most_common(8)),
                      "gap_after_question_vs_other_s": _boot_median_diff(gap_after(q_idx), gap_after(nq_idx)),
                      "final_register_st_question": _dist([anas[i].stats.get("final_register_st") for i in q_idx]),
                      "final_register_st_other": _dist([anas[i].stats.get("final_register_st") for i in nq_idx])}

    # ---- immediate repetition device ("mass mass")
    reps = []
    for u, a in zip(utts, anas):
        for k in range(len(u.words) - 1):
            w1, w2 = _key(u.words[k].w), _key(u.words[k + 1].w)
            if w1 and w1 == w2 and len(w1) >= 2 and w1 not in HI_FUNCTION:
                r1, r2 = a.words[k], a.words[k + 1]
                reps.append({"utt": u.utt_id, "word": u.words[k].w, "t": u.words[k].start,
                             "dur_ratio": r2.dur / max(1e-3, r1.dur), "f0_max_diff_st": r2.f0_max_st - r1.f0_max_st,
                             "energy_diff_db": r2.energy_mean_db - r1.energy_mean_db})
    F["repetition"] = {"n": len(reps), "per_minute": len(reps) / minutes,
                       "top_words": Counter(r["word"] for r in reps).most_common(10),
                       "second_vs_first": {k: _dist([r[k] for r in reps]) for k in ("dur_ratio", "f0_max_diff_st", "energy_diff_db")},
                       "note": "APPROXIMATE word timing; verify on clips"}

    # ---- emphasis: which words carry the main focus
    tops, all_l = [], Counter()
    for u, a in zip(utts, anas):
        if len(a.words) < 4:
            continue
        langs = word_langs([w.w for w in u.words])
        all_l.update(langs)
        best = max(a.words, key=lambda r: r.prominence)
        tops.append((best, langs[best.idx], _key(best.w)))
    top_l = Counter(l for _, l, _ in tops)
    tot_all, tot_top = sum(all_l.values()) or 1, sum(top_l.values()) or 1
    F["emphasis"] = {"n_utterances": len(tops),
                     "main_focus_language_share": {l: top_l[l] / tot_top for l in top_l},
                     "all_words_language_share": {l: all_l[l] / tot_all for l in all_l},
                     "lift_english_terms": (top_l["en"] / tot_top) / max(1e-6, all_l["en"] / tot_all),
                     "main_focus_on_function_word_share": float(np.mean([k in HI_FUNCTION for *_, k in tops])) if tops else None,
                     "top_focus_words": Counter(r.w for r, _, _ in tops).most_common(15),
                     "channels_on_prominent_words": profile.get("emphasis_channels"),
                     "note": "APPROXIMATE word timing"}
    acc = accent_peaks(rf, utts)
    F["pitch_accents"] = {"n": len(acc), "per_minute_of_speech": len(acc) / max(1e-6, rf.speech.mean() * minutes),
                          "size_st": _dist([a["size_st"] for a in acc]),
                          "share_with_energy_peak": float(np.mean([a["energy_copeak"] for a in acc])) if acc else None,
                          "share_with_pre_pause": float(np.mean([a["pre_pause"] for a in acc])) if acc else None,
                          "per_utterance": _dist([sum(a["utt"] == u.utt_id for a in acc) for u in utts]),
                          "method": "transcript-free: F0 peaks >= 3 st above the utterance declination line"}
    # measured accent sizes feed the director's emphasis targets
    profile["accents"] = {"size_st": F["pitch_accents"]["size_st"], "share_with_energy_peak": F["pitch_accents"]["share_with_energy_peak"],
                          "share_with_pre_pause": F["pitch_accents"]["share_with_pre_pause"]}
    gaps_ok = [g for g in gaps if g <= 3.0]
    profile["utterance_gaps_s"] = _dist(gaps_ok)
    profile["n_long_silences_over_3s"] = int(sum(g > 3.0 for g in gaps))
    write_json(out / "profile.json", json.loads(json.dumps(profile, default=float)))
    F["code_switch"] = profile.get("code_switch")
    F["beats_rule_based"] = {b: {"n": d["n"], "rate_rel": d["rate_rel"].get("median"),
                                 "f0_rel_st": d["f0_median_rel_st"].get("median"), "energy_rel_db": d["energy_rel_db"].get("median"),
                                 "finals": d["final_types"]} for b, d in profile["per_beat"].items()}
    F["transitions"] = profile.get("transitions")
    F["pre_reveal"] = profile.get("pre_reveal")
    ends = [u.end for u in utts]
    F["long_form"] = {"drift": drift(rf),
                      "cadence_repeat_first_10min": cadence_repetition(phrase_final_shapes(rf, [e for e in ends if e <= 600])),
                      "cadence_repeat_all": cadence_repetition(phrase_final_shapes(rf, ends))}

    # ---- clips for listening / gold-set work
    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)
    clip_index = []

    def add_clip(tag, u, why):
        a0, a1 = max(0, u.start - 0.3), min(audio.duration, u.end + 0.6)
        if a1 - a0 > 14:
            a1 = a0 + 14
        p = save(clips_dir / f"{tag}_{u.utt_id}.wav", audio.slice(a0, a1))
        clip_index.append({"file": p.name, "start": round(a0, 2), "end": round(a1, 2), "text": u.text, "why": why})

    by_prom = sorted([(max(r.prominence for r in a.words), u) for u, a in zip(utts, anas) if len(a.words) >= 4],
                     key=lambda x: -x[0])
    for _, u in by_prom[:n_clips]:
        add_clip("emphasis", u, "strongest single focus in the lecture")
    for i in q_idx[:n_clips]:
        add_clip("question", utts[i], f"question form, final={anas[i].stats.get('final_type')}")
    seen = set()
    for r in reps:
        if r["utt"] not in seen and len(seen) < n_clips:
            seen.add(r["utt"])
            add_clip("repeat", next(u for u in utts if u.utt_id == r["utt"]), f"immediate repetition of '{r['word']}'")
    (clips_dir / "index.json").write_text(json.dumps(clip_index, ensure_ascii=False, indent=1))
    F["clips"] = clip_index

    # ---- gold-set sheet (expert fills the blank columns)
    labs = annotate([u.text for u in utts])
    with open(out / "gold_set_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "start", "end", "asr_text", "rule_beat", "rule_conf", "question_form", "final_type",
                    "auto_main_focus", "corrected_text", "expert_beat", "expert_emphasis_words", "expert_final", "notes"])
        for u, a, lab in zip(utts, anas, labs):
            best = max(a.words, key=lambda r: r.prominence).w if a.words else ""
            w.writerow([u.utt_id, f"{u.start:.2f}", f"{u.end:.2f}", u.text, lab.beat, f"{lab.conf:.2f}",
                        is_question_text(u.text), a.stats.get("final_type"), best, "", "", "", "", ""])

    # ---- figures + html
    figs = []
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].hist(to_semitones(v, rf.ref_hz), bins=60, color="#2a6fdb")
    ax[0].set_xlabel("F0, semitones re median")
    ax[0].set_title(f"median {rf.ref_hz:.0f} Hz", fontsize=9)
    ax[1].hist([g for g in gaps if g < 4], bins=40, color="#8a5cc2")
    ax[1].set_xlabel("pause between utterances (s)")
    ax[2].plot(rf.rate_t / 60, rf.rate, lw=0.5, color="#e08a1e")
    ax[2].set_xlabel("minute")
    ax[2].set_ylabel("syll/s")
    fig.tight_layout()
    figs.append(("Voice, pauses and tempo", _png(fig)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ga, gb = gap_after(q_idx), gap_after(nq_idx)
    ax[0].hist([gb, ga], bins=np.linspace(0, 3, 31), label=["after statements", "after questions"], color=["#aaa", "#2a6fdb"])
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("pause after utterance (s)")
    kinds = ["rise", "level", "fall"]
    x = np.arange(3)
    nq, nn = max(1, sum(fin_q.values())), max(1, sum(fin_nq.values()))
    ax[1].bar(x - 0.2, [fin_q[k] / nq for k in kinds], 0.4, label="questions", color="#2a6fdb")
    ax[1].bar(x + 0.2, [fin_nq[k] / nn for k in kinds], 0.4, label="other", color="#aaa")
    ax[1].set_xticks(x, kinds)
    ax[1].set_ylabel("share")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(("Questions: how they end and how long he waits", _png(fig)))
    traces = []
    for _, u in by_prom[:3]:
        a = anas[utts.index(u)]
        traces.append(report.img(report.emphasis_trace(audio, rf.f0, rf.energy, u.words, a.words, rf.ref_hz,
                                                       f"{u.utt_id}: {u.text[:80]} (APPROX word timing)")))
    sections = [(h, img) for h, img in figs] + [
        ("Emphasis traces (approximate word timing)", "".join(traces)),
        ("Measured findings (JSON)", "<pre>" + json.dumps({k: v for k, v in F.items() if k != "clips"}, indent=1,
                                                          ensure_ascii=False, default=float) + "</pre>"),
        ("Teacher profile", report.profile_tables(profile))]
    report.build(out / "findings.html", f"Prosody findings – {teacher_id}", sections,
                 note="ASR: Whisper-turbo (uncorrected). Word timing: approximate syllable-nuclei alignment. "
                      "Beats: rule-based on ASR text. Acoustic pause/pitch/tempo numbers do not depend on the transcript.")
    write_json(out / "findings.json", json.loads(json.dumps(F, default=float)))
    return F
