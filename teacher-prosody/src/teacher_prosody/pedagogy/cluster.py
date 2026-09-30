"""Unsupervised discovery of recurring prosodic states, compared with pedagogical labels.

Features are z-scored WITHIN session so clusters reflect delivery, not recording condition.
Model selection by BIC over Gaussian mixtures. The comparison with beat labels (ARI, NMI,
contingency) says whether the taxonomy matches audible behaviour - and which beats to merge.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

FEATURES = ["f0_median_st", "f0_range_st", "f0_slope_st_per_s", "final_delta_st", "energy_mean_db",
            "energy_range_db", "artic_rate_sps", "pause_time_s", "duration_s"]


def feature_matrix(stats: list[dict], sessions: list[str], features: list[str] = FEATURES) -> np.ndarray:
    X = np.array([[float(s.get(f, np.nan)) if s.get(f) is not None else np.nan for f in features] for s in stats], float)
    by = defaultdict(list)
    for i, s in enumerate(sessions):
        by[s].append(i)
    for idx in by.values():
        sub = X[idx]
        mu = np.nanmean(sub, 0)
        sd = np.nanstd(sub, 0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
        X[idx] = (sub - mu) / sd
    return np.nan_to_num(X)


def discover_states(X: np.ndarray, k_range=range(2, 9), seed: int = 0) -> tuple[np.ndarray, int, list[float]]:
    from sklearn.mixture import GaussianMixture

    best, best_k, bics = None, None, []
    for k in k_range:
        if k >= len(X):
            break
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(X)
        bic = gm.bic(X)
        bics.append(float(bic))
        if best is None or bic < best[0]:
            best, best_k = (bic, gm), k
    return best[1].predict(X), best_k, bics


def compare_with_beats(states: np.ndarray, beats: list[str]) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    table: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for s, b in zip(states, beats):
        table[b][int(s)] += 1
    return {"ari": float(adjusted_rand_score(beats, states)), "nmi": float(normalized_mutual_info_score(beats, states)),
            "contingency": {b: dict(v) for b, v in table.items()}}
