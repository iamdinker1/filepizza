"""Voiced F0 with confidence, octave-error handling and speaker/recording normalisation.

Default extractor is Praat's autocorrelation pitch (via parselmouth) with the common
two-pass floor/ceiling heuristic (first pass 60-700 Hz, then floor = 0.75*Q25 and
ceiling = 1.5*Q75; Hirst 2011, De Looze & Hirst 2008). pYIN (librosa) is available as a
cross-check. Neural trackers (RMVPE, torchcrepe, PENN) plug in through `extract_f0(..., method=)`
once GPU/network access exists; they are not bundled.

Normalisations offered (compare them with `profile.confounds.compare_normalisations`):
  st_median   12*log2(f0 / median_f0)          semitones re speaker x recording median (default)
  z_logf0     (ln f0 - mean) / std             z-scored log F0
  erb         ERB-rate relative to median       psychoacoustic scale
  st_baseline semitones re fitted phrase baseline (removes declination)
Absolute Hz is always kept for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import Audio

HOP_S = 0.01


@dataclass
class F0Track:
    times: np.ndarray  # frame centres, seconds
    hz: np.ndarray  # NaN where unvoiced / masked
    conf: np.ndarray  # 0..1 voicing strength
    floor: float
    ceiling: float
    method: str
    octave_fixed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def voiced(self) -> np.ndarray:
        return np.isfinite(self.hz)

    def median(self) -> float:
        v = self.hz[self.voiced]
        return float(np.median(v)) if v.size else float("nan")

    def at(self, t0: float, t1: float) -> np.ndarray:
        m = (self.times >= t0) & (self.times < t1)
        return self.hz[m]

    def slice_mask(self, t0: float, t1: float) -> np.ndarray:
        return (self.times >= t0) & (self.times < t1)


def _praat_pitch(audio: Audio, floor: float, ceiling: float, hop: float):
    import parselmouth

    snd = parselmouth.Sound(audio.y.astype(np.float64), sampling_frequency=audio.sr)
    pitch = snd.to_pitch_ac(time_step=hop, pitch_floor=floor, pitch_ceiling=ceiling, very_accurate=False,
                            voicing_threshold=0.45, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14)
    arr = pitch.selected_array
    hz = arr["frequency"].astype(float)
    conf = np.clip(arr["strength"].astype(float), 0, 1)
    hz[hz <= 0] = np.nan
    return np.asarray(pitch.xs()), hz, conf


def _pyin(audio: Audio, floor: float, ceiling: float, hop: float):
    import librosa

    hop_len = int(round(hop * audio.sr))
    frame = 2048 if audio.sr > 22050 else 1024
    f0, vflag, vprob = librosa.pyin(audio.y, fmin=floor, fmax=ceiling, sr=audio.sr, frame_length=frame,
                                    hop_length=hop_len, center=True)
    t = librosa.frames_to_time(np.arange(len(f0)), sr=audio.sr, hop_length=hop_len)
    f0 = np.where(vflag, f0, np.nan)
    return t, f0.astype(float), np.nan_to_num(vprob).astype(float)


def extract_f0(audio: Audio, floor: float | None = None, ceiling: float | None = None, method: str = "praat",
               hop: float = HOP_S, two_pass: bool = True, min_conf: float = 0.0, fix_octaves: bool = True) -> F0Track:
    """Extract F0. If floor/ceiling are not given, estimate them per speaker with two passes."""
    fn = {"praat": _praat_pitch, "pyin": _pyin}[method]
    if floor is None or ceiling is None:
        if two_pass:
            _, hz0, _ = fn(audio, 60.0, 700.0, hop)
            v = hz0[np.isfinite(hz0)]
            if v.size > 20:
                q25, q75 = np.percentile(v, [25, 75])
                floor = floor or max(50.0, 0.75 * q25)
                ceiling = ceiling or min(900.0, 1.5 * q75)
        floor = floor or 60.0
        ceiling = ceiling or 500.0
    t, hz, conf = fn(audio, floor, ceiling, hop)
    if min_conf > 0:
        hz = np.where(conf >= min_conf, hz, np.nan)
    fixed = np.zeros_like(hz, dtype=bool)
    if fix_octaves:
        hz, fixed = correct_octave_errors(hz)
    return F0Track(t, hz, conf, float(floor), float(ceiling), method, fixed)


def _running_median_voiced(hz: np.ndarray, win: int = 31) -> np.ndarray:
    """Median over the surrounding `win` *voiced* frames (unvoiced gaps skipped)."""
    from scipy.ndimage import median_filter

    idx = np.flatnonzero(np.isfinite(hz))
    ref = np.full_like(hz, np.nan)
    if idx.size == 0:
        return ref
    ref[idx] = median_filter(hz[idx], size=min(win, idx.size), mode="nearest")
    return ref


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def correct_octave_errors(hz: np.ndarray, win: int = 51, tol_st: float = 3.0, jump_st: float = 7.0):
    """Fold back halving/doubling errors without flattening real pitch accents.

    A run of frames ~+/-12 st away from the local voiced median is treated as an octave error
    only if it is entered or left through an abrupt frame-to-frame jump (> `jump_st`), which a
    genuine accent (a smooth rise) does not produce. Very short (< 4 frame) spikes that jump in
    and out but are not near an octave are masked as unreliable. Returns (hz, fixed_mask).
    """
    hz = hz.copy()
    ref = _running_median_voiced(hz, win)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 12 * np.log2(hz / ref)
    fixed = np.zeros(hz.shape, dtype=bool)
    vidx = np.flatnonzero(np.isfinite(hz))
    # frame-to-frame jump (st) w.r.t. previous voiced frame
    jump = np.zeros_like(hz)
    if vidx.size > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            jump[vidx[1:]] = 12 * np.log2(hz[vidx[1:]] / hz[vidx[:-1]])

    def abrupt_edges(a: int, b: int) -> tuple[bool, bool]:
        enter = abs(jump[a]) > jump_st
        nxt = vidx[vidx >= b]
        leave = bool(nxt.size) and abs(jump[nxt[0]]) > jump_st
        return enter, leave

    for sign, factor in ((+1, 0.5), (-1, 2.0)):
        with np.errstate(invalid="ignore"):
            cand = np.abs(d - sign * 12) < tol_st
        for a, b in _runs(cand):
            enter, leave = abrupt_edges(a, b)
            if enter or leave:
                hz[a:b] *= factor
                fixed[a:b] = True
    with np.errstate(invalid="ignore"):
        spike = np.isfinite(d) & ~fixed & (np.abs(d) > jump_st)
    for a, b in _runs(spike):
        if b - a < 4 and all(abrupt_edges(a, b)):
            hz[a:b] = np.nan
    return hz, fixed


# ---------------------------------------------------------------- normalisation

def to_semitones(hz: np.ndarray, ref_hz: float) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 12.0 * np.log2(hz / ref_hz)


def hz_to_erb(hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1 + 0.00437 * hz)


def normalise(hz: np.ndarray, method: str = "st_median", stats: dict | None = None,
              times: np.ndarray | None = None, phrase_bounds: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Normalise F0 (NaN-preserving). `stats` carries speaker x recording reference values
    (median_hz, mean_logf0, std_logf0); if absent they are computed from `hz` itself."""
    v = hz[np.isfinite(hz)]
    stats = dict(stats or {})
    stats.setdefault("median_hz", float(np.median(v)) if v.size else np.nan)
    if method == "st_median":
        return to_semitones(hz, stats["median_hz"])
    if method == "z_logf0":
        lv = np.log(v) if v.size else np.array([np.nan])
        mu = stats.get("mean_logf0", float(np.mean(lv)))
        sd = stats.get("std_logf0", float(np.std(lv)) or 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.log(hz) - mu) / sd
    if method == "erb":
        return hz_to_erb(hz) - hz_to_erb(np.array(stats["median_hz"]))
    if method == "st_baseline":
        if times is None:
            raise ValueError("st_baseline needs frame times")
        st = to_semitones(hz, stats["median_hz"])
        out = np.full_like(st, np.nan)
        bounds = phrase_bounds or [(float(times[0]), float(times[-1]) + 1e-6)]
        for a, b in bounds:
            m = (times >= a) & (times < b) & np.isfinite(st)
            if m.sum() < 5:
                continue
            # baseline = linear fit through the lower part of the contour (declination line)
            tt, ss = times[m], st[m]
            lo = ss <= np.percentile(ss, 40)
            coef = np.polyfit(tt[lo], ss[lo], 1) if lo.sum() >= 3 else np.polyfit(tt, ss, 1)
            out[m] = ss - np.polyval(coef, tt)
        return out
    raise ValueError(f"unknown normalisation {method}")


def f0_stats(hz: np.ndarray, times: np.ndarray, ref_hz: float) -> dict:
    """Descriptive statistics of a voiced F0 stretch (semitones re ref_hz)."""
    m = np.isfinite(hz)
    if m.sum() < 3:
        return {"n_voiced": int(m.sum())}
    st = to_semitones(hz[m], ref_hz)
    t = times[m]
    slope = float(np.polyfit(t - t[0], st, 1)[0]) if t[-1] > t[0] else 0.0
    d = np.diff(st)
    return {
        "n_voiced": int(m.sum()),
        "mean_hz": float(np.mean(hz[m])),
        "median_hz": float(np.median(hz[m])),
        "mean_st": float(np.mean(st)),
        "median_st": float(np.median(st)),
        "range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "std_st": float(np.std(st)),
        "max_st": float(np.max(st)),
        "slope_st_per_s": slope,
        "resets": int(np.sum(d > 3.0)),  # upward jumps > 3 st between adjacent voiced frames
    }


def final_contour(hz: np.ndarray, times: np.ndarray, end: float, ref_hz: float, window: float = 0.35,
                  start: float | None = None) -> dict:
    """Shape of the utterance ending.

    final_delta_st     movement inside the last `window` s of voicing (a within-word rise/fall)
    final_register_st  median of that window relative to the utterance body before it (a step up
                       onto the last word is how many speakers - and TTS - realise a question)
    Rising if delta > +1.5 st, or the final register is > +2.5 st above the body without falling;
    falling if delta < -1.5 st or register < -2.5 st; otherwise level.
    """
    m = np.isfinite(hz) & (times <= end) & (times >= (start if start is not None else -np.inf))
    nan = {"final_slope": np.nan, "final_delta_st": np.nan, "final_register_st": np.nan, "final_type": "unknown"}
    if m.sum() < 4:
        return nan
    last_t = times[m][-1]
    w = m & (times >= last_t - window)
    st = to_semitones(hz[w], ref_hz)
    tt = times[w]
    if len(st) < 4:
        return nan
    slope = float(np.polyfit(tt, st, 1)[0])
    k = max(1, len(st) // 4)
    delta = float(np.median(st[-k:]) - np.median(st[:k]))
    body = m & (times < last_t - window - 0.15)
    reg = float(np.median(st) - np.median(to_semitones(hz[body], ref_hz))) if body.sum() >= 5 else np.nan
    if delta > 1.5 or (np.isfinite(reg) and reg > 2.5 and delta > -1.0):
        kind = "rise"
    elif delta < -1.5 or (np.isfinite(reg) and reg < -2.5):
        kind = "fall"
    else:
        kind = "level"
    return {"final_slope": slope, "final_delta_st": delta, "final_register_st": reg, "final_type": kind}
