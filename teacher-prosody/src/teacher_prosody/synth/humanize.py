"""Humanize layer: give monotonous TTS the within-sentence dynamics of a real teacher, keeping the voice.

Measured on Rajwant Sir (30 min) vs Bunty ElevenLabs TTS (9.4 min), what separates a human lecture
from a monotonous one is not the averages but the variation inside sentences:
  tempo swing (local-rate CV)       0.175 vs 0.143
  syllable-timing variety (CV)      0.49  vs 0.42
  phrase-final slowing              1.13x vs 0.88x   (he slows into phrase ends; TTS hurries)
  pitch range in a sentence         12.0  vs 10.4 st
  pause variety (CV)                0.85  vs 0.77, with longer breath groups
The layer is transcript-free. It uses the TTS audio's own syllable nuclei, pauses and pitch accents
(where the TTS already put stress) and exaggerates their contrast toward the teacher's numbers:
  1. stretch stressed syllables and tighten unstressed ones        (syllable-timing variety)
  2. slow the last syllables before every pause                     (phrase-final slowing)
  3. slow around each phrase's key point, speed through the rest    (tempo swing)
  4. pauses: quantile-map onto the teacher's; merge very short breaks; occasional pre-key pause
     (speech is then rescaled so its average pace is the teacher's articulation rate)
  5. widen pitch excursions around the declination line; lift the key accent a little; +dB on it
One PSOLA pass (Praat) at the input's own sample rate. `calibrate` searches a single strength
multiplier so the measured dynamics land closest to the teacher's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyze import analyze_recording
from ..audio import Audio
from ..features.f0 import to_semitones
from ..preprocess.ingest import segment_at_pauses

TARGET_KEYS = ("tempo_swing_cv", "tempo_span", "syllable_timing_cv", "final_slowing", "pitch_range_st", "pause_cv")


def measure_dynamics(audio: Audio, rf=None) -> dict:
    """Within-sentence dynamics (the numbers in the module docstring), transcript-free."""
    rf = rf or analyze_recording(audio if audio.sr == 16000 else _to16k(audio))
    a16 = audio if audio.sr == 16000 else _to16k(audio)
    units = segment_at_pauses(a16, min_pause=0.45)
    st = to_semitones(rf.f0.hz, rf.ref_hz)
    rate_cv, rate_span, f_range, syl_cv, final_len = [], [], [], [], []
    for u in units:
        if u.end - u.start < 2.0:
            continue
        m = (rf.rate_t >= u.start) & (rf.rate_t < u.end) & np.isfinite(rf.rate)
        if m.sum() > 10:
            r = rf.rate[m]
            rate_cv.append(np.std(r) / np.mean(r))
            rate_span.append(np.percentile(r, 90) / max(1e-6, np.percentile(r, 10)))
        v = st[rf.f0.slice_mask(u.start, u.end)]
        v = v[np.isfinite(v)]
        if v.size > 20:
            f_range.append(np.percentile(v, 95) - np.percentile(v, 5))
        n = rf.nuclei.times[(rf.nuclei.times >= u.start) & (rf.nuclei.times < u.end)]
        if len(n) >= 6:
            d = np.diff(n)
            d = d[d < 0.6]
            if d.size >= 5:
                syl_cv.append(np.std(d) / np.mean(d))
                final_len.append(d[-1] / np.median(d[:-1]))
    sil = [b - a for a, b in _silences(rf)]
    med = lambda x: float(np.median(x)) if len(x) else float("nan")
    return {"tempo_swing_cv": med(rate_cv), "tempo_span": med(rate_span), "syllable_timing_cv": med(syl_cv),
            "final_slowing": med(final_len), "pitch_range_st": med(f_range),
            "pause_cv": float(np.std(sil) / np.mean(sil)) if sil else float("nan"),
            "pause_p90_s": float(np.percentile(sil, 90)) if sil else float("nan"),
            "units_per_min": len(units) / (a16.duration / 60), "artic_rate_sps": float(np.nanmedian(rf.rate))}


def _to16k(a: Audio) -> Audio:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(a.sr, 16000)
    return Audio(resample_poly(a.y, 16000 // g, a.sr // g).astype(np.float32), 16000)


def _silences(rf, min_s=0.12, max_s=3.0):
    hop = rf.energy.times[1] - rf.energy.times[0]
    sil = ~rf.speech
    d = np.diff(np.concatenate([[0], sil.astype(int), [0]]))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        dur = (b - a) * hop
        if a > 0 and b < len(sil) and min_s <= dur <= max_s:
            out.append((float(rf.energy.times[a]), float(rf.energy.times[a] + dur)))
    return out


@dataclass
class HumanizeParams:
    strength: float = 1.0
    stress_stretch: float = 0.28      # max extra duration on the most stressed syllables
    unstress_squeeze: float = 0.12    # max duration cut on the least stressed syllables
    final_stretch: float = 0.30       # extra duration on the last syllable(s) before a pause
    keypoint_slow: float = 0.15       # slow-down around each phrase's key accent (+-0.6 s)
    connect_fast: float = 0.10        # speed-up through the rest of the phrase (not the ending)
    pitch_expand: float = 0.15        # widen excursions around the declination line
    key_accent_lift_st: float = 1.5   # extra lift on each phrase's key accent
    key_accent_db: float = 1.5        # loudness bump on the key accent
    prekey_pause_share: float = 0.17  # share of phrases that get a short pause before the key accent
    prekey_pause_s: float = 0.2
    merge_breaks_below_s: float = 0.28  # shorten very short inter-phrase breaks (longer breath groups)


def humanize(audio: Audio, teacher_pauses: list[float] | None = None, params: HumanizeParams | None = None,
             teacher_rate: float | None = None) -> tuple[Audio, dict]:
    import parselmouth
    from parselmouth.praat import call
    from scipy.ndimage import median_filter, uniform_filter1d

    p = params or HumanizeParams()
    s = float(p.strength)
    a16 = audio if audio.sr == 16000 else _to16k(audio)
    rf = analyze_recording(a16)
    units = segment_at_pauses(a16, min_pause=0.3)
    st_all = to_semitones(rf.f0.hz, rf.ref_hz)
    hop = 0.005
    T = audio.duration
    grid = np.arange(0, T + hop, hop)
    fac = np.ones_like(grid)
    nuc = rf.nuclei.times
    log = {"phrases": 0, "prekey_pauses": 0, "stressed": 0, "unstressed": 0}
    key_accents: list[tuple[float, float]] = []  # (time, span half-width)
    rng = np.random.default_rng(7)
    for ui, u in enumerate(units):
        m = rf.f0.slice_mask(u.start, u.end)
        t, st = rf.f0.times[m], st_all[m]
        ok = np.isfinite(st)
        ns = nuc[(nuc >= u.start) & (nuc < u.end)]
        if ok.sum() < 10 or len(ns) < 3:
            continue
        log["phrases"] += 1
        coef = np.polyfit(t[ok], st[ok], 1)
        resid = np.where(ok, st - np.polyval(coef, t), np.nan)
        # syllable spans = midpoints between nuclei
        edges = np.concatenate([[u.start], (ns[:-1] + ns[1:]) / 2, [u.end]])
        # stress score per syllable: pitch residual at nucleus (+ loudness), z-scored within the phrase
        r_at = np.array([np.nanmax(resid[(t >= a) & (t < b)]) if np.any(np.isfinite(resid[(t >= a) & (t < b)])) else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
        e_at = np.interp(ns, rf.energy.times, rf.energy.db_norm)
        z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
        score = np.nan_to_num(0.65 * z(r_at) + 0.35 * z(e_at))
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (grid >= a) & (grid < b)
            if score[k] > 0.5:
                fac[sel] *= 1 + s * p.stress_stretch * min(1.0, (score[k] - 0.5) / 1.5)
                log["stressed"] += 1
            elif score[k] < -0.3:
                fac[sel] *= 1 - s * p.unstress_squeeze * min(1.0, (-score[k] - 0.3) / 1.2)
                log["unstressed"] += 1
        # phrase-final slowing: lean into the end - from the second-to-last nucleus through the tail fully,
        # the syllable before it by half
        fin_a = ns[-2] if len(ns) >= 2 else edges[-2]
        fac[(grid >= fin_a) & (grid < u.end)] *= 1 + s * p.final_stretch
        if len(ns) >= 3:
            fac[(grid >= edges[-3]) & (grid < fin_a)] *= 1 + 0.5 * s * p.final_stretch
        # tempo swing at the ~1 s scale: slow around the phrase's key point, move faster through the rest
        kk = int(np.argmax(score))
        tk = ns[kk]
        key_accents.append((tk, (edges[kk + 1] - edges[kk]) / 2))
        key_win = (grid >= max(u.start, tk - 0.6)) & (grid < min(u.end, tk + 0.6))
        fac[key_win] *= 1 + s * p.keypoint_slow
        if u.end - u.start > 1.6:
            rest = (grid >= u.start) & (grid < (edges[-3] if len(ns) >= 3 else u.end)) & ~key_win
            fac[rest] *= 1 - s * p.connect_fast
        # occasional short pause before the key point, only where the signal already dips between words
        if kk > 0 and rng.random() < p.prekey_pause_share * min(1.0, s):
            win = (rf.energy.times >= edges[kk] - 0.15) & (rf.energy.times < ns[kk] - 0.03)
            if win.any():
                j = np.flatnonzero(win)[np.argmin(rf.energy.db_norm[win])]
                if rf.energy.db_norm[j] < -12:
                    t0 = rf.energy.times[j]
                    sel = (grid >= t0 - 0.02) & (grid < t0 + 0.02)
                    fac[sel] = max(1.0, s * p.prekey_pause_s / 0.04)
                    log["prekey_pauses"] += 1
    # overall speaking pace: rescale speech (not pauses) so its mean duration factor gives the teacher's
    # articulation rate (or keeps the TTS's own when no rate is given); contrasts stay, the average moves
    sil = _silences(rf)
    speech = np.zeros(len(grid), bool)
    for u in units:
        speech |= (grid >= u.start) & (grid < u.end)
    for a, b in sil:
        speech &= ~((grid >= a) & (grid < b))
    want = 1.0
    if teacher_rate:
        want = float(np.clip((float(np.nanmedian(rf.rate)) / teacher_rate) ** min(1.0, s), 0.85, 1.15))
    if speech.any():
        fac[speech] *= want / float(np.mean(fac[speech]))
    log["speech_duration_factor"] = round(want, 3)
    # pauses: quantile-map to the teacher's distribution; very short inter-phrase breaks get shorter
    if teacher_pauses is not None and sil:
        src_p = np.sort([b - a for a, b in sil])
        tgt_p = np.asarray(teacher_pauses)
        for a, b in sil:
            L = b - a
            q = np.searchsorted(src_p, L, side="right") / len(src_p)
            want = float(np.quantile(tgt_p, min(1.0, q)))
            want = L + s * (want - L)
            if L < p.merge_breaks_below_s:
                want = min(want, L * (1 - 0.4 * min(1.0, s)))
            sel = (grid >= a) & (grid < b)
            fac[sel] = np.clip(want / L, 0.35, 3.0)
    fac = np.clip(uniform_filter1d(fac, size=7), 0.55, 6.0)  # ~35 ms smoothing, no hard steps

    # ---- PSOLA pass at the input's own rate
    snd = parselmouth.Sound(audio.y.astype(np.float64), sampling_frequency=audio.sr)
    v = rf.f0.hz[rf.f0.voiced]
    manip = call(snd, "To Manipulation", 0.01, max(50.0, 0.7 * np.percentile(v, 5)), min(700.0, 1.6 * np.percentile(v, 95)))
    pt = call(manip, "Extract pitch tier")
    npts = call(pt, "Get number of points")
    pts = np.array([(call(pt, "Get time from index", i), call(pt, "Get value at index", i)) for i in range(1, npts + 1)])
    call(pt, "Remove points between", 0, T)
    if len(pts):
        st_p = to_semitones(pts[:, 1], rf.ref_hz)
        new = st_p.copy()
        for u in units:
            mk = (pts[:, 0] >= u.start) & (pts[:, 0] <= u.end)
            if mk.sum() < 8:
                continue
            c = np.median(st_p[mk])  # widen the whole phrase's movement, declination included
            new[mk] = c + (st_p[mk] - c) * (1 + s * p.pitch_expand)
        for tk, hw in key_accents:
            bump = np.exp(-0.5 * ((pts[:, 0] - tk) / max(0.04, hw)) ** 2)
            new += s * p.key_accent_lift_st * bump
        new = median_filter(new, size=3)
        for (tt, _), stv in zip(pts, new):
            call(pt, "Add point", float(tt), float(np.clip(rf.ref_hz * 2 ** (stv / 12), 50, 700)))
    dt = call(manip, "Extract duration tier")
    step = 2  # a point every 10 ms
    for i in range(0, len(grid), step):
        call(dt, "Add point", float(grid[i]), float(fac[i]))
    call([manip, pt], "Replace pitch tier")
    call([manip, dt], "Replace duration tier")
    res = call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(res.values[0], dtype=np.float32)
    # loudness bump on key accents, placed through the time warp
    cum = np.concatenate([[0.0], np.cumsum(fac[:-1] * hop)])
    cum *= (len(y) / audio.sr) / max(cum[-1], 1e-9)
    warp = lambda t0: float(np.interp(t0, grid, cum))
    if p.key_accent_db and key_accents:
        tt = np.arange(len(y)) / audio.sr
        g = np.zeros(len(y))
        for tk, hw in key_accents:
            c = warp(tk)
            g += s * p.key_accent_db * np.exp(-0.5 * ((tt - c) / max(0.05, hw * 1.2)) ** 2)
        y = y * (10 ** (np.minimum(g, 4.0) / 20)).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0.98:
        y = y / peak * 0.98
    log.update({"strength": s, "duration_in_s": round(T, 2), "duration_out_s": round(len(y) / audio.sr, 2),
                "mean_duration_factor": round(float(np.mean(fac)), 3)})
    return Audio(y.astype(np.float32), audio.sr), log


def teacher_profile(teacher_audio: Audio) -> dict:
    """Everything `humanize` needs from the teacher, measured once (a few minutes of clean lecture is enough):
    his pause lengths, articulation rate and the within-sentence dynamics to aim for."""
    a16 = teacher_audio if teacher_audio.sr == 16000 else _to16k(teacher_audio)
    rf = analyze_recording(a16)
    d = measure_dynamics(a16, rf)
    return {"pauses_s": [round(b - a, 3) for a, b in _silences(rf)], "artic_rate_sps": d["artic_rate_sps"],
            "dynamics": {k: d[k] for k in TARGET_KEYS}}


def distance_to_target(d: dict, target: dict, keys=TARGET_KEYS) -> float:
    """Mean relative gap to the teacher on the dynamics that matter."""
    gaps = [abs(d[k] - target[k]) / max(1e-6, abs(target[k])) for k in keys if np.isfinite(d.get(k, np.nan))]
    return float(np.mean(gaps)) if gaps else float("inf")


def calibrate(sample: Audio, target: dict, teacher_pauses: list[float] | None, teacher_rate: float | None = None,
              strengths=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)) -> tuple[float, list[dict]]:
    """Pick the strength whose output dynamics are closest to the teacher's (on a sample clip)."""
    trials = []
    for s in strengths:
        out, _ = humanize(sample, teacher_pauses, HumanizeParams(strength=s), teacher_rate)
        d = measure_dynamics(out)
        trials.append({"strength": s, "distance": distance_to_target(d, target), **{k: round(d[k], 3) for k in d}})
    best = min(trials, key=lambda r: r["distance"])
    return best["strength"], trials
