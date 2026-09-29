"""Link word (and phone) spans to F0, energy, duration and pauses; score prominence.

Prominence is a transparent weighted sum of within-utterance z-scores of four channels:
  pitch    focal F0 excursion (st above local voiced baseline)
  energy   mean normalised energy
  duration lengthening vs expected (syllable count x speaker mean syllable duration)
  pause    silence immediately before the word (pre-emphasis pause)
Per-word channel contributions are kept so we can see HOW a teacher emphasises (pitch vs
length vs loudness vs pausing), which matters more for control than a single score.
The weights are a starting point; fit them to gold emphasis labels (`fit_prominence_weights`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .energy import EnergyTrack
from .f0 import F0Track, to_semitones
from .rate import syllable_count

DEFAULT_WEIGHTS = {"pitch": 0.4, "energy": 0.25, "duration": 0.25, "pause": 0.10}


@dataclass
class WordProsody:
    idx: int
    w: str
    start: float
    end: float
    lang: str
    dur: float
    n_syll: int
    dur_per_syll: float
    f0_mean_st: float
    f0_max_st: float
    f0_excursion_st: float  # max st minus local baseline (median st in +/-1.5 s)
    f0_slope_st_s: float
    energy_mean_db: float
    energy_max_db: float
    pause_before: float
    pause_after: float
    dur_z: float = 0.0
    z_pitch: float = 0.0
    z_energy: float = 0.0
    z_duration: float = 0.0
    z_pause: float = 0.0
    prominence: float = 0.0
    main_channel: str = ""

    def to_dict(self):
        return asdict(self)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    out = np.zeros_like(x)
    if m.sum() >= 2:
        sd = np.std(x[m])
        out[m] = (x[m] - np.mean(x[m])) / (sd if sd > 1e-9 else 1.0)
    return out


def word_prosody(words, f0: F0Track, energy: EnergyTrack, ref_hz: float, syl_dur: float | None = None,
                 weights: dict | None = None, baseline_win: float = 1.5) -> list[WordProsody]:
    """Per-word acoustic measurements + prominence for one utterance (or a longer stretch)."""
    weights = weights or DEFAULT_WEIGHTS
    st = to_semitones(f0.hz, ref_hz)
    rows: list[WordProsody] = []
    n_syll = [syllable_count(w.norm or w.w) for w in words]
    if syl_dur is None:
        tot = sum(w.end - w.start for w in words)
        syl_dur = tot / max(1, sum(n_syll))
    for i, w in enumerate(words):
        m = f0.slice_mask(w.start, w.end)
        seg = st[m]
        v = seg[np.isfinite(seg)]
        tt = f0.times[m][np.isfinite(seg)]
        bm = f0.slice_mask(w.start - baseline_win, w.end + baseline_win)
        base_v = st[bm]
        base_v = base_v[np.isfinite(base_v)]
        base = float(np.median(base_v)) if base_v.size else np.nan
        f0_mean = float(np.mean(v)) if v.size else np.nan
        f0_max = float(np.percentile(v, 95)) if v.size else np.nan
        slope = float(np.polyfit(tt, v, 1)[0]) if v.size >= 4 and tt[-1] > tt[0] else np.nan
        pb = w.start - words[i - 1].end if i > 0 else 0.0
        pa = words[i + 1].start - w.end if i + 1 < len(words) else 0.0
        dur = w.end - w.start
        rows.append(WordProsody(
            idx=i, w=w.w, start=w.start, end=w.end, lang=getattr(w, "lang", "other"), dur=dur, n_syll=n_syll[i],
            dur_per_syll=dur / max(1, n_syll[i]), f0_mean_st=f0_mean, f0_max_st=f0_max,
            f0_excursion_st=(f0_max - base) if np.isfinite(f0_max) and np.isfinite(base) else np.nan,
            f0_slope_st_s=slope, energy_mean_db=energy.mean_norm(w.start, w.end),
            energy_max_db=energy.max_norm(w.start, w.end), pause_before=max(0.0, pb), pause_after=max(0.0, pa)))
    if not rows:
        return rows
    expected = np.array([r.n_syll * syl_dur for r in rows])
    ratio = np.log(np.array([r.dur for r in rows]) / np.maximum(expected, 1e-3))
    zp = _z(np.array([r.f0_excursion_st for r in rows]))
    ze = _z(np.array([r.energy_mean_db for r in rows]))
    zd = _z(ratio)
    zq = _z(np.minimum(np.array([r.pause_before for r in rows]), 1.5))
    for k, r in enumerate(rows):
        r.dur_z, r.z_pitch, r.z_energy, r.z_duration, r.z_pause = float(zd[k]), float(zp[k]), float(ze[k]), float(zd[k]), float(zq[k])
        contrib = {"pitch": weights["pitch"] * zp[k], "energy": weights["energy"] * ze[k],
                   "duration": weights["duration"] * zd[k], "pause": weights["pause"] * zq[k]}
        r.prominence = float(sum(contrib.values()))
        r.main_channel = max(contrib, key=contrib.get)
    return rows


def top_prominent(rows: list[WordProsody], k: int = 3) -> list[WordProsody]:
    return sorted(rows, key=lambda r: -r.prominence)[:k]


def emphasis_alignment(rows: list[WordProsody], target_idx: list[int], k: int | None = None) -> dict:
    """How well realised prominence matches the planned emphasis words.

    precision@k / recall@k of the top-k prominent words vs targets (k = len(targets) by default),
    plus the mean prominence rank percentile of the targets (1.0 = targets are the most prominent).
    """
    if not rows or not target_idx:
        return {"precision": np.nan, "recall": np.nan, "rank_pct": np.nan}
    k = k or len(target_idx)
    top = {r.idx for r in top_prominent(rows, k)}
    tgt = set(target_idx)
    order = np.argsort([-r.prominence for r in rows])
    rank = {rows[j].idx: pos for pos, j in enumerate(order)}
    pct = np.mean([1 - rank[t] / max(1, len(rows) - 1) for t in tgt if t in rank])
    return {"precision": len(top & tgt) / k, "recall": len(top & tgt) / len(tgt), "rank_pct": float(pct)}


def fit_prominence_weights(feature_rows: list[list[WordProsody]], gold: list[list[int]]) -> dict:
    """Fit channel weights to gold emphasis labels with logistic regression (non-negative weights
    are enforced by clipping; refit if many clip)."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for rows, g in zip(feature_rows, gold):
        gs = set(g)
        for r in rows:
            X.append([r.z_pitch, r.z_energy, r.z_duration, r.z_pause])
            y.append(int(r.idx in gs))
    if len(set(y)) < 2:
        return dict(DEFAULT_WEIGHTS)
    clf = LogisticRegression(class_weight="balanced").fit(np.array(X), np.array(y))
    w = np.clip(clf.coef_[0], 0, None)
    w = w / w.sum() if w.sum() > 0 else np.array(list(DEFAULT_WEIGHTS.values()))
    return dict(zip(["pitch", "energy", "duration", "pause"], map(float, w)))


def phrase_final_lengthening(rows: list[WordProsody], final_idx: set[int]) -> dict:
    """Ratio of per-syllable duration of phrase-final vs non-final words."""
    fin = [r.dur_per_syll for r in rows if r.idx in final_idx]
    non = [r.dur_per_syll for r in rows if r.idx not in final_idx]
    if not fin or not non:
        return {"ratio": np.nan, "n_final": len(fin)}
    return {"ratio": float(np.median(fin) / np.median(non)), "n_final": len(fin), "n_nonfinal": len(non)}
