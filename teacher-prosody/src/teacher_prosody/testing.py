"""Synthetic test material with known ground truth.

MOCK DATA - NOT TEACHER SPEECH. Used by unit tests and the offline demo when no real
lecture audio is available:
  * `synth_voice`  source-filter "vowel" with an exact F0 contour (for F0 / octave tests)
  * `espeak_utterance`  word-by-word espeak-ng synthesis concatenated with known pauses, giving
    exact word timings and controllable per-word pitch / speed / loudness (for emphasis, pause and
    rate tests, and as a stand-in "TTS candidate" generator).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

from .audio import Audio, DEFAULT_SR
from .schema import Word


def synth_voice(f0_hz: np.ndarray, sr: int = DEFAULT_SR, hop: float = 0.01,
                formants=((700, 80), (1220, 90), (2600, 120)), amp: np.ndarray | None = None,
                noise_db: float = -60.0, seed: int = 0) -> Audio:
    """Impulse-train source through a formant cascade. f0_hz: per-hop F0 (0/NaN = silence)."""
    rng = np.random.default_rng(seed)
    n = int(len(f0_hz) * hop * sr)
    f0_s = np.interp(np.arange(n) / sr, np.arange(len(f0_hz)) * hop, np.nan_to_num(f0_hz))
    amp_s = np.ones(n) if amp is None else np.interp(np.arange(n) / sr, np.arange(len(amp)) * hop, amp)
    phase = np.cumsum(f0_s / sr)
    src = np.zeros(n)
    src[1:][np.diff(np.floor(phase)) > 0] = 1.0
    src[f0_s <= 0] = 0
    y = src
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    y = lfilter([1], [1, -0.97], y) * amp_s  # glottal tilt-ish
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
    y += rng.normal(0, 10 ** (noise_db / 20), n)
    return Audio(y.astype(np.float32), sr)


def have_espeak() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def espeak_word(text: str, voice: str = "en-us", pitch: int = 50, speed: int = 160, amp: int = 100,
                sr: int = DEFAULT_SR) -> Audio:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak-ng not installed (apt-get install espeak-ng)")
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "w.wav"
        cmd = [exe, "-v", voice, "-p", str(pitch), "-s", str(speed), "-a", str(amp), "-w", str(out)]
        subprocess.run(cmd + [text], check=True, capture_output=True)
        y, file_sr = sf.read(str(out), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        from math import gcd

        g = gcd(file_sr, sr)
        y = resample_poly(y, sr // g, file_sr // g).astype(np.float32)
    return trim(Audio(y, sr))


def trim(a: Audio, thr_db: float = -40.0, pad_s: float = 0.005) -> Audio:
    win = int(0.01 * a.sr)
    if len(a.y) < win:
        return a
    frames = np.lib.stride_tricks.sliding_window_view(a.y, win)[::win // 2]
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    on = np.flatnonzero(db > db.max() + thr_db)
    if on.size == 0:
        return a
    s = max(0, on[0] * (win // 2) - int(pad_s * a.sr))
    e = min(len(a.y), on[-1] * (win // 2) + win + int(pad_s * a.sr))
    return Audio(a.y[s:e], a.sr)


@dataclass
class WordSpec:
    text: str
    voice: str = "en-us"
    pitch: int = 50
    speed: int = 160
    amp: int = 100
    pause_after: float = 0.06
    lang: str = "en"


def espeak_utterance(spec: list[WordSpec], sr: int = DEFAULT_SR, lead: float = 0.3, tail: float = 0.3,
                     noise_db: float = -65.0, seed: int = 0) -> tuple[Audio, list[Word]]:
    """Concatenate per-word synthesis with exact pauses. Returns audio + ground-truth word timings."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * sr), np.float32)]
    words, t = [], lead
    for s in spec:
        a = espeak_word(s.text, s.voice, s.pitch, s.speed, 100, sr)  # loudness applied below, linearly
        words.append(Word(w=s.text, start=t, end=t + a.duration, lang=s.lang))
        parts.append(a.y * (s.amp / 100.0))
        t += a.duration
        parts.append(np.zeros(int(round(s.pause_after * sr)), np.float32))
        t += int(round(s.pause_after * sr)) / sr
    parts.append(np.zeros(int(tail * sr), np.float32))
    y = np.concatenate(parts)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.7
    y = y + rng.normal(0, 10 ** (noise_db / 20), len(y)).astype(np.float32)
    return Audio(y.astype(np.float32), sr), words


def write_textgrid(path, words: list[Word], duration: float) -> None:
    """Long-format TextGrid with a 'words' tier (gaps filled with empty intervals)."""
    ivs, t = [], 0.0
    for w in words:
        if w.start > t + 1e-4:
            ivs.append((t, w.start, ""))
        ivs.append((w.start, w.end, w.w))
        t = w.end
    if duration > t + 1e-4:
        ivs.append((t, duration, ""))
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "words"', "        xmin = 0", f"        xmax = {duration}", f"        intervals: size = {len(ivs)}"]
    for i, (a, b, txt) in enumerate(ivs, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{txt}"']
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
