"""RMS energy envelope, recording-gain normalisation, peaks/slopes and a voicing-aware VAD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..audio import Audio

HOP_S = 0.01
WIN_S = 0.025


@dataclass
class EnergyTrack:
    times: np.ndarray
    db: np.ndarray  # absolute RMS dBFS
    noise_floor_db: float
    speech_level_db: float  # median level of active speech frames (recording gain proxy)

    @property
    def db_norm(self) -> np.ndarray:
        """dB relative to this recording's active-speech level (removes microphone/gain offsets)."""
        return self.db - self.speech_level_db

    @property
    def snr_db(self) -> float:
        return self.speech_level_db - self.noise_floor_db

    def mean_norm(self, t0: float, t1: float) -> float:
        m = (self.times >= t0) & (self.times < t1)
        return float(np.mean(self.db_norm[m])) if m.any() else float("nan")

    def max_norm(self, t0: float, t1: float) -> float:
        m = (self.times >= t0) & (self.times < t1)
        return float(np.max(self.db_norm[m])) if m.any() else float("nan")


def rms_db(audio: Audio, hop: float = HOP_S, win: float = WIN_S):
    hop_n, win_n = int(round(hop * audio.sr)), int(round(win * audio.sr))
    y = np.pad(audio.y.astype(np.float64), (win_n // 2, win_n // 2))
    n = 1 + max(0, (len(y) - win_n) // hop_n)
    idx = np.arange(win_n)[None, :] + hop_n * np.arange(n)[:, None]
    frames = y[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    times = np.arange(n) * hop
    return times, 20 * np.log10(rms + 1e-10)


def extract_energy(audio: Audio, hop: float = HOP_S) -> EnergyTrack:
    t, db = rms_db(audio, hop)
    noise = float(np.percentile(db, 10))
    active = db > noise + 10
    level = float(np.median(db[active])) if active.sum() > 10 else float(np.percentile(db, 90))
    return EnergyTrack(t, db, noise, level)


def smooth(x: np.ndarray, frames: int) -> np.ndarray:
    return uniform_filter1d(x, size=max(1, frames), mode="nearest")


def energy_peaks(e: EnergyTrack, prominence_db: float = 3.0, smooth_frames: int = 5):
    s = smooth(e.db_norm, smooth_frames)
    idx, props = find_peaks(s, prominence=prominence_db)
    return e.times[idx], s[idx], props["prominences"]


def energy_slope(e: EnergyTrack, smooth_frames: int = 7) -> np.ndarray:
    """dB per second, smoothed."""
    s = smooth(e.db_norm, smooth_frames)
    return np.gradient(s, e.times)


def speech_mask(e: EnergyTrack, voiced: np.ndarray | None = None, rel_thr: float = 0.35,
                min_speech: float = 0.03, min_gap: float = 0.05, voiced_reach: float = 0.15) -> np.ndarray:
    """Frame-level speech/non-speech decision.

    A frame is speech if it is above an adaptive threshold between noise floor and speech level
    AND lies within `voiced_reach` seconds of a voiced frame (so breaths, clicks and room noise
    that never voice are not counted as speech). Gaps shorter than `min_gap` are filled and
    bursts shorter than `min_speech` removed.
    """
    thr = e.noise_floor_db + max(6.0, rel_thr * (e.speech_level_db - e.noise_floor_db))
    m = e.db > thr
    hop = e.times[1] - e.times[0] if len(e.times) > 1 else HOP_S
    if voiced is not None:
        v = np.zeros(len(e.times), dtype=bool)
        n = min(len(voiced), len(v))
        v[:n] = voiced[:n]
        reach = int(round(voiced_reach / hop))
        near_voiced = uniform_filter1d(v.astype(float), size=2 * reach + 1, mode="constant") > 0
        m &= near_voiced
    m = _fill_short(m, int(round(min_gap / hop)), value=False)
    m = _fill_short(m, int(round(min_speech / hop)), value=True)
    return m


def _fill_short(mask: np.ndarray, max_len: int, value: bool) -> np.ndarray:
    """Flip runs equal to `value` shorter than max_len (interior runs only for gaps)."""
    m = mask.copy()
    d = np.diff(np.concatenate([[not value], m == value, [not value]]).astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(starts, ends):
        if b - a < max_len:
            if value is False and (a == 0 or b == len(m)):
                continue  # keep leading/trailing silence
            m[a:b] = not value
    return m


def sound_regions_without_voicing(e: EnergyTrack, voiced: np.ndarray, min_dur: float = 0.12, max_dur: float = 0.9):
    """Candidate breaths: audible (> noise + 6 dB), never voiced, clearly below speech level."""
    hop = e.times[1] - e.times[0]
    audible = e.db > e.noise_floor_db + 6
    n = min(len(voiced), len(audible))
    v = np.zeros(len(audible), dtype=bool)
    v[:n] = voiced[:n]
    reach = int(round(0.1 / hop))
    near_voiced = uniform_filter1d(v.astype(float), size=2 * reach + 1, mode="constant") > 0
    cand = audible & ~near_voiced
    out = []
    d = np.diff(np.concatenate([[0], cand.astype(int), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        dur = (b - a) * hop
        if min_dur <= dur <= max_dur and np.max(e.db[a:b]) < e.speech_level_db - 8:
            out.append((float(e.times[a]), float(e.times[b - 1] + hop)))
    return out
