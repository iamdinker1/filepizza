"""Restyle existing TTS audio toward a teacher's measured delivery (transcript-free, PSOLA).

What is matched (measured identically on the teacher and on the TTS audio):
  pace        articulation rate (syllable nuclei per second of phonation)
  pauses      every silence 0.12-3 s is quantile-mapped onto the teacher's pause distribution
  accents     F0 excursions above each unit's declination line are scaled so the accent-size
              distribution matches the teacher's; dips are scaled half as much
  endings     flattest non-rising unit endings get a final rise until the rise share matches
              the teacher's (the last unit of a clip is left falling)
What is NOT changed: the words, the voice timbre and, by default, the voice's own pitch register.
This is a prosody experiment (Prototype A, stage 1). PSOLA degrades with large edits, so every
edit is clipped, and the before/after report says how far each dimension actually moved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analyze import analyze_recording
from ..audio import Audio
from ..features.f0 import final_contour, to_semitones
from ..preprocess.ingest import Segment, segment_at_pauses


@dataclass
class StyleStats:
    duration_s: float
    median_f0_hz: float
    artic_rate_sps: float
    pauses_s: list[float]
    accent_sizes_st: list[float]
    accents_per_speech_min: float
    rise_share: float
    unit_f0_range_st: float
    n_units: int
    rise_delta_st: float = float("nan")
    units: list[Segment] = field(default_factory=list, repr=False)
    finals: list[dict] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {"summary": self.summary(), "pauses_s": list(map(float, self.pauses_s)),
                "accent_sizes_st": list(map(float, self.accent_sizes_st))}

    @staticmethod
    def from_json(d: dict) -> "StyleStats":
        sm = d["summary"]
        return StyleStats(duration_s=sm["duration_s"], median_f0_hz=sm["median_f0_hz"], artic_rate_sps=sm["artic_rate_sps"],
                          pauses_s=d["pauses_s"], accent_sizes_st=d["accent_sizes_st"],
                          accents_per_speech_min=sm["accents_per_speech_min"], rise_share=sm["rise_share"],
                          unit_f0_range_st=sm["unit_f0_range_st"], n_units=sm["n_units"], rise_delta_st=sm["rise_delta_st"])

    def summary(self) -> dict:
        p = np.asarray(self.pauses_s)
        a = np.asarray(self.accent_sizes_st)
        return {
            "duration_s": round(self.duration_s, 1), "median_f0_hz": round(self.median_f0_hz, 1),
            "artic_rate_sps": round(self.artic_rate_sps, 2),
            "pause_median_s": round(float(np.median(p)), 3) if p.size else None,
            "pause_p75_s": round(float(np.percentile(p, 75)), 3) if p.size else None,
            "pauses_per_min": round(p.size / (self.duration_s / 60), 1),
            "accent_size_median_st": round(float(np.median(a)), 2) if a.size else None,
            "accent_size_p90_st": round(float(np.percentile(a, 90)), 2) if a.size else None,
            "accents_per_speech_min": round(self.accents_per_speech_min, 1),
            "rise_share": round(self.rise_share, 3), "rise_delta_st": round(self.rise_delta_st, 2),
            "unit_f0_range_st": round(self.unit_f0_range_st, 2), "n_units": self.n_units,
        }


def _silences(rf, min_s=0.12, max_s=3.0):
    hop = rf.energy.times[1] - rf.energy.times[0]
    sil = ~rf.speech
    d = np.diff(np.concatenate([[0], sil.astype(int), [0]]))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if a == 0 or b >= len(sil):
            continue
        dur = (b - a) * hop
        if min_s <= dur <= max_s:
            out.append((float(rf.energy.times[a]), float(rf.energy.times[a] + dur)))
    return out


def _decl_resid(t, st):
    ok = np.isfinite(st)
    if ok.sum() < 10:
        return None, None
    coef = np.polyfit(t[ok], st[ok], 1)
    r = st - np.polyval(coef, t)
    keep = ok & (np.abs(r) <= np.nanpercentile(np.abs(r[ok]), 80))
    if keep.sum() >= 8:
        coef = np.polyfit(t[keep], st[keep], 1)
    return coef, st - np.polyval(coef, t)


def measure_style(audio: Audio, rf=None, min_pause_unit: float = 0.45) -> StyleStats:
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    rf = rf or analyze_recording(audio)
    units = segment_at_pauses(audio, min_pause=min_pause_unit)
    st_all = to_semitones(rf.f0.hz, rf.ref_hz)
    hop = rf.f0.times[1] - rf.f0.times[0]
    acc, finals, ranges = [], [], []
    for u in units:
        m = rf.f0.slice_mask(u.start, u.end)
        t, st = rf.f0.times[m], st_all[m]
        coef, r = _decl_resid(t, st)
        if coef is None:
            finals.append({"final_type": "unknown"})
            continue
        rr = median_filter(np.where(np.isfinite(r), r, -99.0), size=5)
        idx, _ = find_peaks(rr, height=3.0, distance=max(1, int(0.25 / hop)))
        acc += [float(rr[i]) for i in idx]
        v = st[np.isfinite(st)]
        ranges.append(float(np.percentile(v, 95) - np.percentile(v, 5)))
        finals.append(final_contour(rf.f0.hz, rf.f0.times, u.end, rf.ref_hz, start=u.start))
    known = [f for f in finals if f.get("final_type") in ("rise", "fall", "level")]
    rises = [f["final_delta_st"] for f in known if f["final_type"] == "rise" and np.isfinite(f.get("final_delta_st", np.nan))]
    sil = _silences(rf)
    speech_min = float(rf.speech.mean() * audio.duration / 60)
    return StyleStats(
        duration_s=audio.duration, median_f0_hz=rf.ref_hz, artic_rate_sps=float(np.nanmedian(rf.rate)),
        pauses_s=[b - a for a, b in sil], accent_sizes_st=acc, accents_per_speech_min=len(acc) / max(1e-6, speech_min),
        rise_share=(sum(f["final_type"] == "rise" for f in known) / len(known)) if known else float("nan"),
        unit_f0_range_st=float(np.median(ranges)) if ranges else float("nan"), n_units=len(units),
        rise_delta_st=float(np.median(rises)) if rises else float("nan"), units=units, finals=finals)


def _qmap(x: float, src: np.ndarray, tgt: np.ndarray) -> float:
    """Quantile-map x from the source distribution onto the target distribution."""
    q = float(np.searchsorted(np.sort(src), x, side="right")) / max(1, len(src))
    return float(np.quantile(tgt, min(max(q, 0.0), 1.0)))


def restyle(audio: Audio, target: StyleStats, strength: float = 1.0, match_register: bool = False,
            rate_limits=(0.8, 1.25), pause_limits=(0.35, 3.0), accent_limits=(0.7, 1.8), rise_st: float | None = None,
            rise_window_s: float = 0.3, keep_fall_before_gap_s: float = 2.0) -> tuple[Audio, dict]:
    import parselmouth
    from parselmouth.praat import call

    rf = analyze_recording(audio)
    src = measure_style(audio, rf)
    s = float(np.clip(strength, 0, 1.5))
    # --- factors
    rate_k = float(np.clip((src.artic_rate_sps / target.artic_rate_sps) ** s, *rate_limits))  # duration multiplier for speech
    sa, ta = np.median(src.accent_sizes_st) if src.accent_sizes_st else 4.0, np.median(target.accent_sizes_st)
    acc_k = float(np.clip((ta / sa) ** s, *accent_limits))
    reg_shift = to_semitones(np.array([target.median_f0_hz]), src.median_f0_hz)[0] * s if match_register else 0.0
    rise_st = rise_st if rise_st is not None else (target.rise_delta_st if np.isfinite(target.rise_delta_st) else 3.0)
    # --- which unit endings to turn into rises (flattest first, spread over the clip, last unit kept falling)
    idx_known = [i for i, f in enumerate(src.finals) if f.get("final_type") in ("rise", "fall", "level")]
    n_rise_now = sum(src.finals[i]["final_type"] == "rise" for i in idx_known)
    want = int(round(min(1.0, target.rise_share * s + src.rise_share * (1 - s)) * len(idx_known))) if idx_known else 0
    def gap_after(i):
        return src.units[i + 1].start - src.units[i].end if i + 1 < len(src.units) else float("inf")

    # endings before a long gap (clip / section end) stay falling
    cand = [i for i in idx_known if src.finals[i]["final_type"] != "rise" and gap_after(i) < keep_fall_before_gap_s]
    cand.sort(key=lambda i: -(src.finals[i].get("final_delta_st") or -9))  # level / gentle falls first
    to_rise = set(cand[: max(0, want - n_rise_now)])

    snd = parselmouth.Sound(audio.y.astype(np.float64), sampling_frequency=audio.sr)
    floor = max(50.0, 0.7 * np.nanpercentile(rf.f0.hz[rf.f0.voiced], 5))
    ceil = min(700.0, 1.6 * np.nanpercentile(rf.f0.hz[rf.f0.voiced], 95))
    manip = call(snd, "To Manipulation", 0.01, floor, ceil)
    pt = call(manip, "Extract pitch tier")
    n = call(pt, "Get number of points")
    pts = np.array([(call(pt, "Get time from index", i), call(pt, "Get value at index", i)) for i in range(1, n + 1)])
    call(pt, "Remove points between", 0, audio.duration)
    unit_of = np.full(len(pts), -1)
    for ui, u in enumerate(src.units):
        unit_of[(pts[:, 0] >= u.start) & (pts[:, 0] <= u.end)] = ui
    new_st = to_semitones(pts[:, 1], src.median_f0_hz)
    for ui, u in enumerate(src.units):
        mk = unit_of == ui
        if mk.sum() < 10:
            continue
        t, st = pts[mk, 0], new_st[mk]
        coef, r = _decl_resid(t, st)
        if coef is None:
            continue
        r2 = np.where(r > 0, r * acc_k, r * (1 + (acc_k - 1) * 0.5))
        out = np.polyval(coef, t) + r2
        if ui in to_rise:
            t_end = t[-1]
            ramp = np.clip((t - (t_end - rise_window_s)) / rise_window_s, 0, 1)
            out = out + rise_st * ramp ** 1.5
        new_st[mk] = out
    new_st = new_st + reg_shift
    hz = src.median_f0_hz * 2 ** (new_st / 12)
    for (t, _), f in zip(pts, np.clip(hz, 50, 700)):
        call(pt, "Add point", float(t), float(f))

    # --- durations: speech rate everywhere, quantile-mapped pauses
    dt = call(manip, "Extract duration tier")
    sil = _silences(rf)
    tgt_p = np.asarray(target.pauses_s)
    src_p = np.asarray([b - a for a, b in sil])
    call(dt, "Add point", 0.0, rate_k)
    pause_log = []
    for a, b in sil:
        want_d = _qmap(b - a, src_p, tgt_p) if tgt_p.size and src_p.size else (b - a)
        want_d = (b - a) + s * (want_d - (b - a))
        k = float(np.clip(want_d / (b - a), *pause_limits))
        eps = 0.004
        call(dt, "Add point", a - eps, rate_k)
        call(dt, "Add point", a, k)
        call(dt, "Add point", b, k)
        call(dt, "Add point", b + eps, rate_k)
        pause_log.append({"t": round(a, 2), "from_s": round(b - a, 3), "to_s": round((b - a) * k, 3)})
    call([manip, pt], "Replace pitch tier")
    call([manip, dt], "Replace duration tier")
    res = call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(res.values[0], dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0.98:
        y = y / peak * 0.98
    out = Audio(y, audio.sr)
    # time warp old -> new (piecewise-constant duration factors), for evaluating the same units after editing
    grid = np.arange(0.0, audio.duration + 0.001, 0.001)
    fac = np.full(grid.shape, rate_k)
    for (a, b), pl in zip(sil, pause_log):
        fac[(grid >= a) & (grid < b)] = pl["to_s"] / max(1e-6, pl["from_s"])
    cum = np.concatenate([[0.0], np.cumsum(fac[:-1] * 0.001)])
    scale = out.duration / cum[-1] if cum[-1] > 0 else 1.0
    warp = lambda t: float(np.interp(t, grid, cum * scale))
    edits = {"rate_duration_factor": round(rate_k, 3), "accent_scale": round(acc_k, 3), "register_shift_st": round(reg_shift, 2),
             "units": len(src.units), "endings_turned_rising": len(to_rise), "rise_added_st": round(rise_st, 2),
             "pauses_remapped": len(pause_log), "pause_examples": pause_log[:8],
             "_warp": warp, "_units": src.units, "_to_rise": sorted(to_rise), "_source": src}
    return out, edits


def evaluate_restyle(out: Audio, edits: dict, target: StyleStats) -> dict:
    """Measure the edited audio on the SAME units as the source (mapped through the time warp)."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    rf = analyze_recording(out)
    warp, units, src = edits["_warp"], edits["_units"], edits["_source"]
    st_all = to_semitones(rf.f0.hz, src.median_f0_hz)
    hop = rf.f0.times[1] - rf.f0.times[0]
    finals, acc = [], []
    for u in units:
        a, b = warp(u.start), warp(u.end)
        m = rf.f0.slice_mask(a, b)
        coef, r = _decl_resid(rf.f0.times[m], st_all[m])
        if coef is not None:
            rr = median_filter(np.where(np.isfinite(r), r, -99.0), size=5)
            idx, _ = find_peaks(rr, height=3.0, distance=max(1, int(0.25 / hop)))
            acc += [float(rr[i]) for i in idx]
        finals.append(final_contour(rf.f0.hz, rf.f0.times, b, src.median_f0_hz, start=a))
    known = [f for f in finals if f.get("final_type") in ("rise", "fall", "level")]
    sil = _silences(rf)
    p = np.asarray([y - x for x, y in sil])
    before, tgt = src.summary(), target.summary()
    after = {"artic_rate_sps": round(float(np.nanmedian(rf.rate)), 2),
             "pause_median_s": round(float(np.median(p)), 3) if p.size else None,
             "pause_p75_s": round(float(np.percentile(p, 75)), 3) if p.size else None,
             "accent_size_median_st": round(float(np.median(acc)), 2) if acc else None,
             "accent_size_p90_st": round(float(np.percentile(acc, 90)), 2) if acc else None,
             "rise_share": round(sum(f["final_type"] == "rise" for f in known) / max(1, len(known)), 3),
             "median_f0_hz": round(rf.ref_hz, 1), "duration_s": round(out.duration, 1)}
    keys = ["artic_rate_sps", "pause_median_s", "pause_p75_s", "accent_size_median_st", "accent_size_p90_st", "rise_share", "median_f0_hz", "duration_s"]
    return {k: {"target_rajwant": tgt.get(k), "before": before.get(k), "after": after.get(k)} for k in keys}


def split_clips(audio: Audio, min_gap_s: float = 3.0, pad_s: float = 0.25) -> list[tuple[float, float]]:
    """Split a concatenation of TTS clips at silences >= min_gap_s."""
    rf = analyze_recording(audio)
    hop = rf.energy.times[1] - rf.energy.times[0]
    sil = ~rf.speech
    d = np.diff(np.concatenate([[0], sil.astype(int), [0]]))
    cuts = [(rf.energy.times[a], rf.energy.times[a] + (b - a) * hop) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if (b - a) * hop >= min_gap_s and a > 0 and b < len(sil)]
    bounds, start = [], 0.0
    for a, b in cuts:
        bounds.append((max(0.0, start), min(audio.duration, a + pad_s)))
        start = b - pad_s
    bounds.append((max(0.0, start), audio.duration))
    return [(a, b) for a, b in bounds if b - a > 1.0]
