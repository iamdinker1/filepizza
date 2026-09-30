"""Impose a performance plan on existing speech with PSOLA (Praat Manipulation via parselmouth).

This is the CPU-feasible "DSP post-edit" prototype: take a TTS candidate (or a teacher recording),
its word timings, and per-word targets from `director.render.to_word_targets`, then
  * stretch/compress each word (DurationTier),
  * add focal F0 excursions / register offsets (PitchTier, shaped as a rise-fall over the word),
  * apply energy offsets (gain envelope),
  * insert planned pre-word pauses.
PSOLA stays natural for modest edits (roughly +/-4 st, 0.7-1.4x duration); beyond that it
sounds processed, which the QC artifact checks should catch. It is a prosody prototyping tool
and an A/B instrument, not the final production vocoder.
"""
from __future__ import annotations

import numpy as np

from ..audio import Audio, concat, silence
from ..schema import Word


def _gain_curve(n: int, sr: int, spans: list[tuple[float, float, float]], ramp: float = 0.02) -> np.ndarray:
    g_db = np.zeros(n)
    t = np.arange(n) / sr
    for a, b, db in spans:
        if db == 0:
            continue
        w = np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0, 1)
        g_db += w * db
    return 10 ** (g_db / 20)


def apply_word_targets(audio: Audio, words: list[Word], targets: list[dict], max_st: float = 5.0,
                       dur_range=(0.6, 1.6), f0_floor: float = 60.0, f0_ceiling: float = 500.0) -> tuple[Audio, list[Word]]:
    """Returns edited audio and the new word timings."""
    import parselmouth
    from parselmouth.praat import call

    if len(words) != len(targets):
        raise ValueError(f"{len(words)} words vs {len(targets)} targets - align plan words to audio words first")
    snd = parselmouth.Sound(audio.y.astype(np.float64), sampling_frequency=audio.sr)
    manip = call(snd, "To Manipulation", 0.01, f0_floor, f0_ceiling)
    pitch_tier = call(manip, "Extract pitch tier")
    dur_tier = call(manip, "Extract duration tier")

    # pitch: multiply existing points inside each word by a smooth rise-fall bump
    n_pts = call(pitch_tier, "Get number of points")
    pts = [(call(pitch_tier, "Get time from index", i), call(pitch_tier, "Get value at index", i)) for i in range(1, n_pts + 1)]
    call(pitch_tier, "Remove points between", 0, audio.duration)
    for t, f in pts:
        st = 0.0
        for w, tg in zip(words, targets):
            if w.start <= t <= w.end and tg.get("pitch_st"):
                x = (t - w.start) / max(1e-3, w.end - w.start)
                bump = np.sin(np.pi * np.clip(x * 1.15, 0, 1)) if tg.get("emphasis", 0) else 1.0  # peak slightly early
                st += float(np.clip(tg["pitch_st"], -max_st, max_st)) * bump
        call(pitch_tier, "Add point", t, f * 2 ** (st / 12))

    # duration: piecewise-constant scale per word, with 5 ms transitions
    call(dur_tier, "Add point", 0.0, 1.0)
    for w, tg in zip(words, targets):
        s = float(np.clip(tg.get("dur_scale", 1.0), *dur_range))
        if abs(s - 1) < 1e-3:
            continue
        call(dur_tier, "Add point", max(0.0, w.start - 0.005), 1.0)
        call(dur_tier, "Add point", w.start, s)
        call(dur_tier, "Add point", w.end, s)
        call(dur_tier, "Add point", w.end + 0.005, 1.0)
    call([manip, pitch_tier], "Replace pitch tier")
    call([manip, dur_tier], "Replace duration tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(out.values[0], dtype=np.float32)

    # new word times: integrate the duration scale
    def warp(t: float) -> float:
        acc, prev = 0.0, 0.0
        for w, tg in zip(words, targets):
            s = float(np.clip(tg.get("dur_scale", 1.0), *dur_range))
            if t <= w.start:
                break
            acc += w.start - prev
            seg_end = min(t, w.end)
            acc += (seg_end - w.start) * s
            prev = w.end
            if t <= w.end:
                return acc
        return acc + (t - prev)

    new_words = [Word(w=w.w, start=warp(w.start), end=warp(w.end), lang=w.lang, conf=w.conf, norm=w.norm) for w in words]
    edited = Audio(y, audio.sr)

    # energy
    spans = [(nw.start, nw.end, float(tg.get("energy_db", 0.0))) for nw, tg in zip(new_words, targets)]
    edited = Audio((edited.y * _gain_curve(len(edited.y), edited.sr, spans)).astype(np.float32), edited.sr)

    # pre-word pauses (cut at the word start and insert silence)
    pieces, cursor, shift, final_words = [], 0.0, 0.0, []
    for nw, tg in zip(new_words, targets):
        p = float(tg.get("pre_pause_s", 0.0))
        if p > 0 and nw.start > cursor:
            pieces.append(edited.slice(cursor, nw.start))
            pieces.append(silence(p, edited.sr))
            cursor = nw.start
            shift += p
        final_words.append(Word(w=nw.w, start=nw.start + shift, end=nw.end + shift, lang=nw.lang, conf=nw.conf, norm=nw.norm))
    pieces.append(edited.slice(cursor, edited.duration))
    peak = max(1e-9, max(float(np.max(np.abs(p.y))) if len(p.y) else 0 for p in pieces))
    result = concat(pieces)
    if peak > 0.99:
        result = Audio((result.y / peak * 0.98).astype(np.float32), result.sr)
    return result, final_words
