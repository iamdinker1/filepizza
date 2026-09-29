"""Ingest raw lecture recordings: decode, hash, keep the original, write a cleaned 16 kHz copy,
and cut the recording into utterance-sized segments at pauses.

Cleaning is deliberately mild (DC/high-pass + loudness normalisation). Prosody features are
measured on this lightly-cleaned audio, never on a generative enhancer's output, because
enhancers can move F0 and energy. Heavy separation (Demucs/MossFormer) may be used only to
*decide* segment boundaries or reject noisy spans.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..audio import Audio, load, save, sha256_file
from ..features.energy import extract_energy, speech_mask
from ..features.f0 import extract_f0
from ..schema import Recording


def highpass(audio: Audio, cutoff: float = 60.0) -> Audio:
    sos = butter(4, cutoff, btype="highpass", fs=audio.sr, output="sos")
    return Audio(sosfiltfilt(sos, audio.y).astype(np.float32), audio.sr)


def loudness_normalise(audio: Audio, target_lufs: float = -23.0) -> Audio:
    import pyloudnorm as pyln

    meter = pyln.Meter(audio.sr)
    if audio.duration < 0.5:
        return audio
    lufs = meter.integrated_loudness(audio.y.astype(np.float64))
    if not np.isfinite(lufs):
        return audio
    gain = 10 ** ((target_lufs - lufs) / 20)
    y = audio.y * gain
    peak = np.max(np.abs(y))
    if peak > 0.98:  # never clip; accept being quieter than target
        y = y / peak * 0.98
    return Audio(y.astype(np.float32), audio.sr)


def clean(audio: Audio) -> Audio:
    return loudness_normalise(highpass(audio))


def ingest(path: str | Path, out_dir: str | Path, teacher_id: str, session_id: str, recording_id: str | None = None,
           source_uri: str = "", condition: str = "unknown", topic: str = "", lesson_type: str = "",
           consent_ref: str = "", consent_scope: list[str] | None = None, sr: int = 16000,
           start: float | None = None, duration: float | None = None) -> tuple[Recording, Audio]:
    path = Path(path)
    out_dir = Path(out_dir)
    recording_id = recording_id or path.stem
    audio = load(path, sr=sr, start=start, duration=duration)
    cleaned = clean(audio)
    cleaned_path = save(out_dir / "clean" / f"{recording_id}.wav", cleaned)
    rec = Recording(recording_id=recording_id, teacher_id=teacher_id, session_id=session_id,
                    source_uri=source_uri or str(path), original_path=str(path), sha256=sha256_file(path),
                    duration_s=cleaned.duration, sample_rate=sr, recording_condition=condition, topic=topic,
                    lesson_type=lesson_type, consent_ref=consent_ref, consent_scope=list(consent_scope or []),
                    cleaned_path=str(cleaned_path))
    return rec, cleaned


@dataclass
class Segment:
    start: float
    end: float


def segment_at_pauses(audio: Audio, min_pause: float = 0.45, max_len: float = 20.0, min_len: float = 1.0,
                      pad: float = 0.1) -> list[Segment]:
    """Cut at pauses >= min_pause; split over-long stretches at their longest internal pause."""
    f0 = extract_f0(audio)
    e = extract_energy(audio)
    sp = speech_mask(e, f0.voiced)
    t = e.times
    hop = t[1] - t[0]
    d = np.diff(np.concatenate([[0], sp.astype(int), [0]]))
    runs = [(t[a], t[b - 1] + hop) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    segs: list[list[tuple[float, float]]] = []
    for r in runs:
        if segs and r[0] - segs[-1][-1][1] < min_pause:
            segs[-1].append(r)
        else:
            segs.append([r])

    def split(group):
        s, e_ = group[0][0], group[-1][1]
        if e_ - s <= max_len or len(group) == 1:
            return [group]
        gaps = [group[i + 1][0] - group[i][1] for i in range(len(group) - 1)]
        # prefer the longest gap that leaves both sides >= min_len
        order = np.argsort(gaps)[::-1]
        for k in order:
            left, right = group[:k + 1], group[k + 1:]
            if left[-1][1] - left[0][0] >= min_len and right[-1][1] - right[0][0] >= min_len:
                return split(left) + split(right)
        return [group]

    out = []
    for g in segs:
        for sub in split(g):
            a, b = sub[0][0], sub[-1][1]
            if b - a >= min_len:
                out.append(Segment(max(0.0, a - pad), min(audio.duration, b + pad)))
    return out
