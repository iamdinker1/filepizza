"""Quality flags that keep non-teacher audio out of the style statistics.

Checks here are cheap DSP heuristics that run on CPU with no model downloads. They are a
first filter, not a replacement for diarization + speaker verification (pyannote community-1 +
a WavLM/ECAPA teacher-enrolment embedding; see `preprocess/backends.py`), which must run on
classroom recordings where students speak.
"""
from __future__ import annotations

import numpy as np

from ..audio import Audio
from ..features.energy import EnergyTrack
from ..features.f0 import F0Track
from ..schema import Quality


def clipping_fraction(audio: Audio, thr: float = 0.999) -> float:
    return float(np.mean(np.abs(audio.y) >= thr))


def spectral_flatness_frames(audio: Audio, hop: float = 0.02, n_fft: int = 1024) -> np.ndarray:
    import librosa

    return librosa.feature.spectral_flatness(y=audio.y, n_fft=n_fft, hop_length=int(hop * audio.sr))[0]


def music_like_fraction(audio: Audio, f0: F0Track, speech: np.ndarray, hop: float = 0.01) -> float:
    """Fraction of the segment that looks like sustained tonal non-speech (music / jingles).

    Speech alternates voiced/unvoiced every ~100-300 ms; music keeps long stable voicing with
    stable pitch. Heuristic: voiced runs > 1.5 s whose F0 stays within +/-0.5 st for most frames.
    """
    v = f0.voiced
    if v.size == 0:
        return 0.0
    d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
    long_stable = 0
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) * hop < 1.5:
            continue
        seg = 12 * np.log2(f0.hz[a:b] / np.nanmedian(f0.hz[a:b]))
        if np.mean(np.abs(np.diff(seg)) < 0.05) > 0.6:
            long_stable += b - a
    return float(long_stable / len(v))


def mfcc_embedding(audio: Audio) -> np.ndarray:
    """Weak speaker fingerprint (MFCC mean+std). WEAK - only for a first-pass filter / tests;
    replace with WavLM-SV or ECAPA embeddings for real decisions."""
    import librosa

    m = librosa.feature.mfcc(y=audio.y, sr=audio.sr, n_mfcc=20)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def assess(audio: Audio, f0: F0Track, energy: EnergyTrack, speech: np.ndarray, teacher_sim: float | None = None,
           align_conf: float | None = None, min_snr: float = 15.0, sim_thr: float | None = None) -> Quality:
    q = Quality()
    q.snr_db = float(energy.snr_db)
    q.clipping_frac = clipping_fraction(audio)
    q.speech_frac = float(np.mean(speech)) if speech.size else 0.0
    q.music_like_frac = music_like_fraction(audio, f0, speech)
    q.teacher_sim = teacher_sim
    q.align_conf = align_conf
    if q.snr_db < min_snr:
        q.flags.append("low_snr")
    if q.clipping_frac > 0.001:
        q.flags.append("clipping")
    if q.music_like_frac > 0.2:
        q.flags.append("music")
    if q.speech_frac < 0.3:
        q.flags.append("mostly_silence")
    if teacher_sim is not None and sim_thr is not None and teacher_sim < sim_thr:
        q.flags.append("not_teacher")
    if align_conf is not None and align_conf < 0.5:
        q.flags.append("align_low")
    return q
