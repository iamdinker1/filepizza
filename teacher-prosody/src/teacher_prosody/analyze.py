"""Run the acoustic analysis once per recording, then summarise each utterance.

Recording-level tracks (F0, energy, VAD, syllable nuclei, local rate, breath candidates) are
computed once so that per-speaker/per-recording normalisation uses the whole session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import Audio
from .features.energy import EnergyTrack, extract_energy, sound_regions_without_voicing, speech_mask
from .features.f0 import F0Track, extract_f0, f0_stats, final_contour
from .features.pauses import Pause, classify_pauses, pause_distribution, pauses_from_mask, pauses_from_words
from .features.rate import Nuclei, local_rate_curve, syllable_nuclei, words_rate
from .features.words import WordProsody, word_prosody
from .preprocess.lang import cmi, word_langs
from .schema import Utterance


@dataclass
class RecordingFeatures:
    audio: Audio
    f0: F0Track
    energy: EnergyTrack
    speech: np.ndarray
    nuclei: Nuclei
    rate_t: np.ndarray
    rate: np.ndarray
    breaths: list[tuple[float, float]]
    ref_hz: float
    syl_dur: float | None = None  # speaker mean syllable duration (from aligned words), set by caller

    @property
    def summary(self) -> dict:
        v = self.f0.hz[self.f0.voiced]
        return {
            "duration_s": self.audio.duration, "ref_hz": self.ref_hz, "f0_floor": self.f0.floor,
            "f0_ceiling": self.f0.ceiling, "voiced_frac": float(self.f0.voiced.mean()),
            "octave_fixed_frac": float(self.f0.octave_fixed.mean()) if self.f0.octave_fixed.size else 0.0,
            "f0_p5_p95_hz": [float(np.percentile(v, 5)), float(np.percentile(v, 95))] if v.size else None,
            "snr_db": self.energy.snr_db, "speech_level_db": self.energy.speech_level_db,
            "speech_frac": float(self.speech.mean()), "n_syllable_nuclei": int(len(self.nuclei.times)),
            "median_artic_rate_sps": float(np.nanmedian(self.rate)) if np.isfinite(self.rate).any() else None,
            "n_breath_candidates": len(self.breaths),
        }


def analyze_recording(audio: Audio, f0_method: str = "praat") -> RecordingFeatures:
    f0 = extract_f0(audio, method=f0_method)
    e = extract_energy(audio)
    sp = speech_mask(e, f0.voiced)
    nuc = syllable_nuclei(e, f0)
    rt, r = local_rate_curve(nuc.times, sp, e.times)
    breaths = sound_regions_without_voicing(e, f0.voiced)
    return RecordingFeatures(audio, f0, e, sp, nuc, rt, r, breaths, f0.median())


@dataclass
class UtteranceAnalysis:
    utt_id: str
    stats: dict
    words: list[WordProsody] = field(default_factory=list)
    pauses: list[Pause] = field(default_factory=list)


def analyze_utterance(u: Utterance, rf: RecordingFeatures, question_rise_st: float = 1.5) -> UtteranceAnalysis:
    f0, e = rf.f0, rf.energy
    m = f0.slice_mask(u.start, u.end)
    st = f0_stats(f0.hz[m], f0.times[m], rf.ref_hz)
    fin = final_contour(f0.hz, f0.times, u.end, rf.ref_hz, start=u.start)
    em = (e.times >= u.start) & (e.times < u.end)
    sp = rf.speech[em] if em.any() else np.zeros(0, bool)
    nuc_in = rf.nuclei.times[(rf.nuclei.times >= u.start) & (rf.nuclei.times < u.end)]
    phon = float(sp.sum() * (e.times[1] - e.times[0])) if sp.size else 0.0
    stats = {
        **{f"f0_{k}": v for k, v in st.items()},
        **fin,
        "energy_mean_db": e.mean_norm(u.start, u.end),
        "energy_abs_db": float(np.mean(e.db[em & rf.speech[: len(e.db)]])) if (em & rf.speech[: len(e.db)]).any() else np.nan,
        "energy_range_db": float(np.percentile(e.db_norm[em], 95) - np.percentile(e.db_norm[em], 5)) if em.sum() > 5 else np.nan,
        "nuclei_artic_rate_sps": len(nuc_in) / phon if phon > 0.3 else np.nan,
        "speech_frac": float(sp.mean()) if sp.size else 0.0,
        "duration_s": u.duration,
        "beat": u.beat,
        "text": u.text,
        "cmi": cmi(u.text) if u.text else 0.0,
        "is_question": u.text.strip().endswith("?") or fin.get("final_type") == "rise" and fin.get("final_delta_st", 0) > question_rise_st,
    }
    rows: list[WordProsody] = []
    pauses: list[Pause] = []
    if u.words:
        stats.update({f"words_{k}": v for k, v in words_rate(u.words).items()})
        if np.mean([w.conf for w in u.words]) >= 0.5 and np.isfinite(stats.get("words_artic_rate_sps", np.nan)):
            stats["artic_rate_sps"] = stats["words_artic_rate_sps"]
            stats["rate_source"] = "alignment"
        rows = word_prosody(u.words, f0, e, rf.ref_hz, syl_dur=rf.syl_dur)
        prom = {r.idx: r.prominence for r in rows}
        q_end = {len(u.words) - 1} if stats["is_question"] else set()
        rhet = q_end if u.beat == "rhetorical_question" else set()
        pauses = classify_pauses(pauses_from_words(u.words), u.words, prominence=prom, breaths=rf.breaths,
                                 question_ends=q_end, rhetorical_ends=rhet)
        langs = word_langs([w.w for w in u.words])
        stats["n_switches"] = sum(1 for a, b in zip(langs, langs[1:]) if {a, b} == {"hi", "en"})
    else:
        pm = rf.speech[em]
        pauses = pauses_from_mask(pm, e.times[em]) if pm.size > 2 else []
    stats.setdefault("artic_rate_sps", stats["nuclei_artic_rate_sps"])
    stats.setdefault("rate_source", "nuclei")
    stats["n_pauses"] = len(pauses)
    stats["pause_time_s"] = float(sum(p.dur for p in pauses))
    return UtteranceAnalysis(u.utt_id, stats, rows, pauses)


def pause_summary(analyses: list[UtteranceAnalysis]) -> dict:
    return pause_distribution([p for a in analyses for p in a.pauses])
