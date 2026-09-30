"""Long-form checks at 5 / 15 / 30 / 60 minutes: drift, repetitive cadence, overacting, stitching.

  drift          slope of rolling median F0 (st), articulation rate and loudness over time
  cadence_repeat share of phrase-final contours that are near-duplicates of each other
                 (a "sing-song" TTS reuses one template; a real teacher varies with content)
  overacting     share of words with prominence > 1.5 vs the teacher reference
  dead_air       share of pauses > 1.5 s
Compare every number against the same measurement on real teacher lectures, not an absolute bar.
"""
from __future__ import annotations

import numpy as np

from ..analyze import RecordingFeatures
from ..features.f0 import to_semitones

CHECKPOINTS_MIN = (5, 15, 30, 60)


def _rolling(times: np.ndarray, x: np.ndarray, win_s: float = 60.0, step_s: float = 30.0):
    c = np.arange(times[0], times[-1], step_s)
    out = np.full(len(c), np.nan)
    for i, t in enumerate(c):
        m = (times >= t - win_s / 2) & (times < t + win_s / 2) & np.isfinite(x)
        if m.sum() > 20:
            out[i] = np.median(x[m])
    return c, out


def drift(rf: RecordingFeatures) -> dict:
    st = to_semitones(rf.f0.hz, rf.ref_hz)
    c1, f = _rolling(rf.f0.times, st)
    c2, r = _rolling(rf.rate_t, rf.rate)
    e = np.where(rf.speech[: len(rf.energy.db_norm)], rf.energy.db_norm, np.nan)
    c3, l = _rolling(rf.energy.times, e)

    def slope_per_10min(c, y):
        m = np.isfinite(y)
        return float(np.polyfit(c[m] / 600.0, y[m], 1)[0]) if m.sum() >= 3 else np.nan

    return {"f0_st_per_10min": slope_per_10min(c1, f), "rate_sps_per_10min": slope_per_10min(c2, r),
            "loudness_db_per_10min": slope_per_10min(c3, l)}


def phrase_final_shapes(rf: RecordingFeatures, phrase_ends: list[float], window: float = 0.5, n_pts: int = 8) -> np.ndarray:
    """Resampled final-contour shape (st, mean-removed) for each phrase end."""
    st = to_semitones(rf.f0.hz, rf.ref_hz)
    shapes = []
    for t in phrase_ends:
        m = (rf.f0.times >= t - window) & (rf.f0.times <= t) & np.isfinite(st)
        if m.sum() < n_pts:
            continue
        y = st[m]
        x = np.linspace(0, 1, len(y))
        s = np.interp(np.linspace(0, 1, n_pts), x, y)
        shapes.append(s - s.mean())
    return np.array(shapes)


def cadence_repetition(shapes: np.ndarray, tol_st: float = 0.6) -> float:
    """Fraction of phrase-final contours that have a near-identical twin (RMS diff < tol)."""
    if len(shapes) < 4:
        return float("nan")
    d = np.sqrt(((shapes[:, None, :] - shapes[None, :, :]) ** 2).mean(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.mean(d.min(1) < tol_st))


def overacting(prominence: list[float], ref_prominence: list[float], thr: float = 1.5) -> dict:
    a = float(np.mean(np.asarray(prominence) > thr)) if prominence else np.nan
    b = float(np.mean(np.asarray(ref_prominence) > thr)) if ref_prominence else np.nan
    return {"strong_emphasis_rate": a, "teacher_rate": b, "ratio": a / b if b else np.nan}


def checkpoint_report(rf: RecordingFeatures, phrase_ends: list[float], pauses_s: list[float]) -> dict:
    out = {}
    for m in CHECKPOINTS_MIN:
        t = m * 60.0
        if rf.audio.duration < t * 0.9:
            continue
        ends = [x for x in phrase_ends if x <= t]
        ps = np.asarray([p for p in pauses_s])
        out[f"{m}min"] = {
            "cadence_repeat": cadence_repetition(phrase_final_shapes(rf, ends)),
            "dead_air_frac": float(np.mean(ps > 1.5)) if ps.size else np.nan,
        }
    out["drift"] = drift(rf)
    return out
