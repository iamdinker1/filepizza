"""Inter-annotator agreement for the gold set (nominal labels)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """ratings[item] = labels from each rater (same number of raters per item)."""
    labels = sorted({l for r in ratings for l in r})
    idx = {l: i for i, l in enumerate(labels)}
    n_items, n_raters = len(ratings), len(ratings[0])
    M = np.zeros((n_items, len(labels)))
    for i, r in enumerate(ratings):
        assert len(r) == n_raters, "Fleiss' kappa needs a fixed number of raters per item"
        for l in r:
            M[i, idx[l]] += 1
    p_j = M.sum(0) / (n_items * n_raters)
    P_i = ((M * M).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    Pbar, Pe = P_i.mean(), (p_j**2).sum()
    return 1.0 if Pe == 1 else float((Pbar - Pe) / (1 - Pe))


def krippendorff_alpha_nominal(ratings: list[list[str | None]]) -> float:
    """ratings[item] = labels per rater, None for missing. Handles unequal coverage."""
    values = sorted({v for r in ratings for v in r if v is not None})
    idx = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for r in ratings:
        vals = [v for v in r if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    o[idx[a], idx[b]] += 1 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (n_c**2).sum()) / (n * (n - 1))
    return 1.0 if d_e == 0 else float(1 - d_o / d_e)


def confusion(a: list[str], b: list[str]) -> dict:
    labels = sorted(set(a) | set(b))
    m = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b):
        m[x][y] += 1
    return m


def per_class_f1(gold: list[str], pred: list[str]) -> dict:
    out = {}
    for l in sorted(set(gold) | set(pred)):
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[l] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "support": sum(g == l for g in gold)}
    return out
