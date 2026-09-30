"""Local speaking rate inside sentences, not just words per minute.

Two independent estimators so they can cross-check each other:
  * transcript-free syllable nuclei (intensity peaks that are voiced, >= 2 dB dips between them;
    after de Jong & Wempe 2009, Behav. Res. Methods 41:385-390);
  * alignment-based syllables/phones per second of phonation time.
Articulation rate excludes pauses; speech rate includes them. Both are reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .energy import EnergyTrack, smooth
from .f0 import F0Track

_DEVA_VOWEL_SIGNS = set("ािीुूृॄेैोौॅॉ")
_DEVA_INDEP_VOWELS = set("अआइईउऊऋएऐओऔऍऑ")
_DEVA_CONS = set(chr(c) for c in range(0x0915, 0x093A)) | set("क़ख़ग़ज़ड़ढ़फ़य़")
_VIRAMA = "्"


def syllable_count(word: str) -> int:
    """Approximate syllable count for English, romanised Hindi or Devanagari words."""
    w = word.strip().lower()
    if not w:
        return 0
    if any("ऀ" <= ch <= "ॿ" for ch in w):
        n = 0
        for i, ch in enumerate(w):
            if ch in _DEVA_INDEP_VOWELS:
                n += 1
            elif ch in _DEVA_CONS:
                nxt = w[i + 1] if i + 1 < len(w) else ""
                if nxt != _VIRAMA:
                    n += 1  # consonant carries inherent or explicit vowel
        # word-final schwa deletion in Hindi: final bare consonant is not a syllable
        if len(w) >= 2 and w[-1] in _DEVA_CONS and n > 1:
            n -= 1
        return max(1, n)
    if re.fullmatch(r"[\d.,]+", w):
        return max(1, len(re.sub(r"\D", "", w)))  # rough: one per digit; normalise numbers first
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1  # silent final e (English)
    return max(1, n)


@dataclass
class Nuclei:
    times: np.ndarray
    heights_db: np.ndarray


def syllable_nuclei(energy: EnergyTrack, f0: F0Track, min_dip_db: float = 2.0, rel_floor_db: float = 25.0,
                    smooth_frames: int = 3, min_sep_s: float = 0.07) -> Nuclei:
    s = smooth(energy.db, smooth_frames)
    floor = np.percentile(s, 99) - rel_floor_db
    hop = energy.times[1] - energy.times[0]
    idx, _ = find_peaks(s, height=floor, prominence=min_dip_db, distance=max(1, int(min_sep_s / hop)))
    # keep voiced peaks only
    voiced = np.interp(energy.times[idx], f0.times, f0.voiced.astype(float), left=0, right=0) > 0.5
    idx = idx[voiced]
    return Nuclei(energy.times[idx], s[idx])


def local_rate_curve(nuclei_t: np.ndarray, speech: np.ndarray, times: np.ndarray, win: float = 1.5,
                     step: float = 0.05, min_phonation: float = 0.3):
    """Articulation rate (syllables per second of phonation) in a sliding window."""
    hop = times[1] - times[0]
    centres = np.arange(times[0], times[-1], step)
    rate = np.full(len(centres), np.nan)
    cum_speech = np.concatenate([[0], np.cumsum(speech.astype(float))])
    for i, c in enumerate(centres):
        a, b = c - win / 2, c + win / 2
        ia, ib = np.searchsorted(times, [a, b])
        phon = (cum_speech[ib] - cum_speech[ia]) * hop
        if phon >= min_phonation:
            n = np.sum((nuclei_t >= a) & (nuclei_t < b))
            rate[i] = n / phon
    return centres, rate


def words_rate(words, t0: float | None = None, t1: float | None = None, pause_min: float = 0.15) -> dict:
    """Alignment-based rate over a word list (objects with .w .start .end)."""
    ws = [w for w in words if (t0 is None or w.start >= t0) and (t1 is None or w.end <= t1)]
    if not ws:
        return {"n_words": 0}
    syl = sum(syllable_count(w.norm or w.w) for w in ws)
    span = ws[-1].end - ws[0].start
    gaps = [b.start - a.end for a, b in zip(ws[:-1], ws[1:])]
    pause_time = sum(g for g in gaps if g >= pause_min)
    phon = max(1e-3, span - pause_time)
    return {
        "n_words": len(ws),
        "n_syll": syl,
        "speech_rate_sps": syl / max(span, 1e-3),
        "artic_rate_sps": syl / phon,
        "wpm": 60 * len(ws) / max(span, 1e-3),
        "pause_time_s": pause_time,
    }


def rate_change_before(anchor_t: float, centres: np.ndarray, rate: np.ndarray, lead: float = 2.0,
                       baseline: tuple[float, float] | None = None) -> dict:
    """Did the teacher slow down / speed up in the `lead` seconds before `anchor_t` (e.g. a reveal)?

    Returns the slope (syll/s per s) over the lead window and the ratio of the final 0.75 s rate
    to a baseline window (default: the 3 s preceding the lead window)."""
    m = (centres >= anchor_t - lead) & (centres < anchor_t) & np.isfinite(rate)
    if m.sum() < 3:
        return {"slope": np.nan, "ratio_to_baseline": np.nan}
    slope = float(np.polyfit(centres[m], rate[m], 1)[0])
    b0, b1 = baseline or (anchor_t - lead - 3.0, anchor_t - lead)
    mb = (centres >= b0) & (centres < b1) & np.isfinite(rate)
    mf = (centres >= anchor_t - 0.75) & (centres < anchor_t) & np.isfinite(rate)
    ratio = float(np.nanmean(rate[mf]) / np.nanmean(rate[mb])) if mb.any() and mf.any() else np.nan
    return {"slope": slope, "ratio_to_baseline": ratio}
