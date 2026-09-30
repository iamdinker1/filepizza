"""Objective diagnostics comparing a system's lecture audio with the real teacher.

Contour-level comparison (DTW over F0 / energy) is only meaningful when the two renditions say
the same text in the same context, so `contour_distance` refuses unless texts match (CER <= 0.05).
Distribution-level comparisons (pause durations by type, articulation rate, F0 range, emphasis
density) work on unmatched scripts and are the main generalisation measure.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from ..features.f0 import F0Track, to_semitones


def dtw(a: np.ndarray, b: np.ndarray, band: float = 0.2) -> tuple[float, list[tuple[int, int]]]:
    """Sakoe-Chiba banded DTW on 1-D sequences (NaNs removed beforehand). Returns (mean cost, path)."""
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return float(D[n, m] / len(path)), path[::-1]


def contour_distance(ref: F0Track, hyp: F0Track, ref_text: str, hyp_text: str, ref_hz_ref: float, ref_hz_hyp: float,
                     max_cer: float = 0.05, decimate: int = 2) -> dict:
    """Speaker-normalised F0 contour distance (st) after DTW; each voice relative to its own median."""
    from ..qc.rank import cer

    c = cer(ref_text, hyp_text)
    if c > max_cer:
        return {"refused": f"texts differ (CER {c:.2f}); use distribution metrics instead"}
    a = to_semitones(ref.hz[ref.voiced], ref_hz_ref)[::decimate]
    b = to_semitones(hyp.hz[hyp.voiced], ref_hz_hyp)[::decimate]
    if len(a) < 5 or len(b) < 5:
        return {"refused": "too little voicing"}
    cost, path = dtw(a, b)
    ia, ib = zip(*path)
    corr = float(np.corrcoef(a[list(ia)], b[list(ib)])[0, 1])
    return {"dtw_st": cost, "corr": corr, "range_ref_st": float(np.ptp(a)), "range_hyp_st": float(np.ptp(b))}


def distribution_distances(ref: dict[str, list[float]], hyp: dict[str, list[float]]) -> dict:
    """Per-feature Wasserstein-1 and KS statistic, e.g. keys 'pause_post_question_s', 'artic_rate', 'f0_range_st'."""
    out = {}
    for k in sorted(set(ref) & set(hyp)):
        a = np.asarray([x for x in ref[k] if np.isfinite(x)])
        b = np.asarray([x for x in hyp[k] if np.isfinite(x)])
        if a.size < 3 or b.size < 3:
            out[k] = {"n_ref": int(a.size), "n_hyp": int(b.size)}
            continue
        ks = ks_2samp(a, b)
        out[k] = {"w1": float(wasserstein_distance(a, b)), "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
                  "median_ref": float(np.median(a)), "median_hyp": float(np.median(b)), "n_ref": int(a.size), "n_hyp": int(b.size)}
    return out


def teacher_likeness(dd: dict, scales: dict | None = None) -> float:
    """Single summary (0..1) of distribution closeness - a DIAGNOSTIC, never a substitute for listeners."""
    scales = scales or {}
    vals = []
    for k, v in dd.items():
        if "w1" in v:
            s = scales.get(k) or max(1e-6, abs(v["median_ref"]) * 0.25 or 1.0)
            vals.append(np.exp(-v["w1"] / s))
    return float(np.mean(vals)) if vals else float("nan")
