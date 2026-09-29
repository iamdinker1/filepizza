"""Stitch per-beat audio into a lecture without clipped breaths, tonal jumps or dead air.

Rules
  * trim only true silence at beat edges (keep breath-like unvoiced sound: breaths are part of
    a teacher's rhythm and cutting them makes joins audible)
  * insert the PLANNED inter-beat pause, measured from the last/first voiced frame
  * equal-power crossfade (default 15 ms) instead of hard cuts
  * match each beat's loudness to the lecture target with a smooth gain (no per-word pumping)
  * report boundary diagnostics: F0 jump (st) and loudness jump (dB) across each join, silence
    ratio, and any pause longer than the allowed maximum
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import Audio
from ..features.energy import extract_energy
from ..features.f0 import extract_f0


@dataclass
class StitchReport:
    joins: list[dict] = field(default_factory=list)
    silence_ratio: float = 0.0
    long_pauses: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _edge_trim(a: Audio, thr_db: float = -50.0, keep_s: float = 0.02) -> Audio:
    """Trim leading/trailing silence below thr_db relative to peak (breaths are usually above it)."""
    if len(a.y) == 0:
        return a
    win = int(0.01 * a.sr)
    frames = np.lib.stride_tricks.sliding_window_view(a.y, win)[::win]
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    on = np.flatnonzero(db > db.max() + thr_db)
    if on.size == 0:
        return a
    s = max(0, on[0] * win - int(keep_s * a.sr))
    e = min(len(a.y), (on[-1] + 1) * win + int(keep_s * a.sr))
    return Audio(a.y[s:e], a.sr)


def _lufs(a: Audio) -> float:
    import pyloudnorm as pyln

    if a.duration < 0.4:
        return float(20 * np.log10(np.sqrt(np.mean(a.y**2)) + 1e-9))
    return float(pyln.Meter(a.sr).integrated_loudness(a.y.astype(np.float64)))


def _edge_f0(a: Audio, which: str, span: float = 0.8) -> float:
    seg = a.slice(max(0, a.duration - span), a.duration) if which == "end" else a.slice(0, min(span, a.duration))
    if seg.duration < 0.1:
        return float("nan")
    f = extract_f0(seg, floor=60, ceiling=500, two_pass=False)
    v = f.hz[f.voiced]
    return float(np.median(v)) if v.size else float("nan")


def stitch(beats: list[Audio], pauses_s: list[float], target_lufs: float = -20.0, xfade_s: float = 0.015,
           max_pause_s: float = 1.6, f0_jump_flag_st: float = 4.0, loud_jump_flag_db: float = 4.0) -> tuple[Audio, StitchReport]:
    """pauses_s[i] = silence between beat i and i+1 (len = len(beats) - 1)."""
    assert len(pauses_s) == max(0, len(beats) - 1)
    rep = StitchReport()
    sr = beats[0].sr
    trimmed = [_edge_trim(b) for b in beats]
    leveled = []
    for b in trimmed:
        l = _lufs(b)
        g = 10 ** ((target_lufs - l) / 20) if np.isfinite(l) else 1.0
        leveled.append(Audio((b.y * g).astype(np.float32), sr))
    xf = int(xfade_s * sr)
    out = leveled[0].y.copy()
    for i, b in enumerate(leveled[1:]):
        prev = leveled[i]
        p = float(pauses_s[i])
        if p > max_pause_s:
            rep.long_pauses.append(p)
        f_end, f_start = _edge_f0(prev, "end"), _edge_f0(b, "start")
        jump = 12 * np.log2(f_start / f_end) if np.isfinite(f_end) and np.isfinite(f_start) else float("nan")
        l_end = _lufs(prev.slice(max(0, prev.duration - 1.0), prev.duration))
        l_start = _lufs(b.slice(0, min(1.0, b.duration)))
        rep.joins.append({"after_beat": i, "pause_s": p, "f0_jump_st": float(jump), "loudness_jump_db": float(l_start - l_end)})
        if np.isfinite(jump) and abs(jump) > f0_jump_flag_st:
            rep.flags.append(f"join {i}: F0 jump {jump:+.1f} st")
        if abs(l_start - l_end) > loud_jump_flag_db:
            rep.flags.append(f"join {i}: loudness jump {l_start - l_end:+.1f} dB")
        gap = np.zeros(max(0, int(p * sr)), np.float32)
        # equal-power fades into/out of the gap
        if xf > 0 and len(out) > xf and len(b.y) > xf:
            fade = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2
            out[-xf:] *= fade[::-1]
            head = b.y.copy()
            head[:xf] *= fade
        else:
            head = b.y
        out = np.concatenate([out, gap, head])
    lecture = Audio(out.astype(np.float32), sr)
    e = extract_energy(lecture)
    rep.silence_ratio = float(np.mean(e.db < e.noise_floor_db + 6))
    if rep.silence_ratio > 0.35:
        rep.flags.append(f"silence ratio {rep.silence_ratio:.0%} (too much dead air?)")
    return lecture, rep
