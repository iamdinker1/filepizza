"""Performance-first voice conversion (Prototype B): re-render a source performance (TTS take or a
human reader) in the teacher's voice with kNN-VC (Baas et al., Interspeech 2023; MIT licence).

kNN-VC replaces every 20 ms WavLM-Large layer-6 frame of the source with the mean of its k nearest
frames from the teacher's own recordings, then vocodes with a HiFi-GAN trained on WavLM features.
No training: the teacher's audio *is* the model, so more (clean) reference audio helps up to ~5-8 min
and beyond. Timing follows the source; pitch and pronunciation move toward the teacher because the
output frames are literally his.

USE ONLY WITH THE TEACHER'S SIGNED VOICE-CLONING / CONVERSION CONSENT. Label outputs as synthetic.

Models (GitHub releases of bshall/knn-vc, v0.1): WavLM-Large.pt, prematch_g_02500000.pt, plus the
repo's python files, in one directory (see `load_knnvc`). Output is 16 kHz. The vocoder was trained on
English (LibriSpeech); Hindi works in practice but expect some muffling on retroflexes/aspirates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..audio import Audio


def load_knnvc(model_dir: str | Path, device: str = "cpu", prematched: bool = True):
    import json

    import torch

    d = Path(model_dir)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from hifigan.models import Generator as HiFiGAN  # type: ignore
    from hifigan.utils import AttrDict  # type: ignore
    from matcher import KNeighborsVC  # type: ignore
    from wavlm.WavLM import WavLM, WavLMConfig  # type: ignore

    h = AttrDict(json.loads((d / "hifigan" / "config_v1_wavlm.json").read_text()))
    gen = HiFiGAN(h).to(device)
    sd = torch.load(d / ("prematch_g_02500000.pt" if prematched else "g_02500000.pt"), map_location=device, weights_only=False)
    gen.load_state_dict(sd["generator"])
    gen.eval()
    gen.remove_weight_norm()
    ck = torch.load(d / "WavLM-Large.pt", map_location=device, weights_only=False)
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    wavlm = wavlm.to(device).eval()
    return KNeighborsVC(wavlm, gen, h, device)


def _features(knn, audio: Audio, chunk_s: float = 20.0):
    import torch

    assert audio.sr == 16000, "kNN-VC works at 16 kHz"
    feats = []
    step = int(chunk_s * 16000)
    y = audio.y
    for i in range(0, len(y), step):
        seg = y[i:i + step]
        if len(seg) < 1600:  # < 0.1 s tail
            continue
        x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))[None]
        feats.append(knn.get_features(x, vad_trigger_level=0))
    return torch.cat(feats, dim=0).cpu()


def build_matching_set(knn, teacher_audio: Audio, cache: str | Path | None = None):
    import torch

    if cache and Path(cache).exists():
        return torch.load(cache)
    m = _features(knn, teacher_audio)
    if cache:
        torch.save(m, cache)
    return m


def convert(knn, source: Audio, matching_set, topk: int = 4, chunk_s: float = 20.0, loudness_db: float = -20.0) -> Audio:
    """Convert `source` (16 kHz) into the matching-set speaker. Chunked so memory stays bounded;
    chunk joins are cross-faded over 20 ms."""
    import torch

    q = _features(knn, source, chunk_s)
    outs = []
    per = int(chunk_s * 50)  # 50 feature frames per second
    for i in range(0, q.shape[0], per):
        with torch.inference_mode():
            wav = knn.match(q[i:i + per], matching_set, topk=topk, tgt_loudness_db=None)
        outs.append(wav.numpy().astype(np.float32))
    fade = int(0.02 * 16000)
    y = outs[0]
    for o in outs[1:]:
        if len(y) > fade and len(o) > fade:
            r = np.linspace(0, 1, fade, dtype=np.float32)
            y = np.concatenate([y[:-fade], y[-fade:] * (1 - r) + o[:fade] * r, o[fade:]])
        else:
            y = np.concatenate([y, o])
    import pyloudnorm as pyln

    meter = pyln.Meter(16000)
    lufs = meter.integrated_loudness(y.astype(np.float64))
    if np.isfinite(lufs):
        y = y * 10 ** ((loudness_db - lufs) / 20)
    peak = float(np.max(np.abs(y)))
    if peak > 0.98:
        y = y / peak * 0.98
    return Audio(y.astype(np.float32), 16000)


def transplant_intonation(converted: Audio, source: Audio, target_median_hz: float, strength: float = 1.0,
                          range_scale: float = 1.0, max_move_st: float = 8.0) -> Audio:
    """Give the converted audio the SOURCE's intonation shape (semitones relative to the source median,
    optionally widened by `range_scale` to the teacher's pitch span) placed at the teacher's register,
    via PSOLA. Timing is identical between source and kNN-VC output, so contours line up frame by frame.
    kNN-VC alone scrambles intonation (each output frame brings its own pitch), so this is usually needed."""
    import parselmouth
    from parselmouth.praat import call

    from ..features.f0 import extract_f0

    from scipy.ndimage import median_filter

    sf0 = extract_f0(source)
    s_med = sf0.median()
    # continuous, smoothed source contour in st (interpolated across unvoiced gaps) so that every voiced
    # frame of the converted audio gets a target, whatever its own voicing decision was
    st = 12 * np.log2(sf0.hz / s_med)
    ok = np.isfinite(st)
    st_c = np.interp(sf0.times, sf0.times[ok], st[ok]) if ok.sum() > 2 else np.zeros_like(sf0.times)
    st_c = median_filter(st_c, size=5)
    snd = parselmouth.Sound(converted.y.astype(np.float64), sampling_frequency=converted.sr)
    manip = call(snd, "To Manipulation", 0.01, 70, 500)
    pt = call(manip, "Extract pitch tier")
    n = call(pt, "Get number of points")
    pts = [(call(pt, "Get time from index", i), call(pt, "Get value at index", i)) for i in range(1, n + 1)]
    call(pt, "Remove points between", 0, converted.duration)
    tgt_st = np.interp([t for t, _ in pts], sf0.times, st_c)
    for (t, f), st_t in zip(pts, tgt_st):
        tgt = target_median_hz * 2 ** (st_t * range_scale / 12)
        move = float(np.clip(12 * np.log2(tgt / f), -max_move_st, max_move_st)) * strength
        call(pt, "Add point", t, float(np.clip(f * 2 ** (move / 12), 60, 500)))
    call([manip, pt], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(out.values[0], dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    return Audio((y / peak * 0.98 if peak > 0.98 else y).astype(np.float32), converted.sr)
