"""Separate teaching behaviour from session / recording-condition / topic effects.

For each utterance-level feature we report eta-squared (share of variance explained) by beat,
session, recording condition and topic. A feature whose variance is mostly session/condition
is a recording artefact for style-learning purposes and should be normalised away or dropped;
a feature with high beat eta^2 and low session eta^2 is teaching behaviour worth modelling.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..features.f0 import normalise


def eta_squared(values: np.ndarray, groups: list) -> float:
    v = np.asarray(values, float)
    m = np.isfinite(v)
    v = v[m]
    g = [x for x, keep in zip(groups, m) if keep]
    if v.size < 3 or len(set(g)) < 2:
        return float("nan")
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    by = defaultdict(list)
    for x, gi in zip(v, g):
        by[gi].append(x)
    ss_between = sum(len(xs) * (np.mean(xs) - grand) ** 2 for xs in by.values())
    return float(ss_between / ss_tot)


def variance_decomposition(rows: list[dict], features: list[str], factors=("beat", "session", "condition", "topic")) -> dict:
    out = {}
    for f in features:
        vals = np.array([r.get(f, np.nan) if r.get(f) is not None else np.nan for r in rows], float)
        out[f] = {fac: eta_squared(vals, [r.get(fac, "") for r in rows]) for fac in factors}
        s, b = out[f].get("session", np.nan), out[f].get("beat", np.nan)
        out[f]["verdict"] = ("recording/session-driven" if np.isfinite(s) and np.isfinite(b) and s > 2 * b and s > 0.2
                             else "teaching-driven" if np.isfinite(b) and np.isfinite(s) and b > s else "mixed/unclear")
    return out


def compare_normalisations(utt_f0: list[tuple[np.ndarray, np.ndarray]], sessions: list[str], beats: list[str],
                           session_stats: dict[str, dict], methods=("hz", "st_median", "z_logf0", "erb", "st_baseline")) -> dict:
    """Which F0 normalisation removes recording/session offsets while keeping beat contrasts?

    utt_f0: per utterance (hz, times). session_stats[session] = {"median_hz", "mean_logf0", "std_logf0"}.
    Score = eta^2(beat) - eta^2(session) on the utterance median (higher is better); also reports both.
    """
    out = {}
    for meth in methods:
        med = []
        for (hz, t), s in zip(utt_f0, sessions):
            if meth == "hz":
                x = hz
            else:
                x = normalise(hz, meth, stats=session_stats[s], times=t)
            v = x[np.isfinite(x)]
            med.append(float(np.median(v)) if v.size else np.nan)
        e_s, e_b = eta_squared(np.array(med), sessions), eta_squared(np.array(med), beats)
        out[meth] = {"eta2_session": e_s, "eta2_beat": e_b, "score": (e_b - e_s) if np.isfinite(e_b) and np.isfinite(e_s) else np.nan}
    return out
