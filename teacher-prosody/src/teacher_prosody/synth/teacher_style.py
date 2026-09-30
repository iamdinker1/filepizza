"""Teacher-style converter: a TTS take (e.g. Bunty on ElevenLabs) in, the same words out in the same voice
with a teacher's delivery (pace, pitch movement, emphasis, flow).

Editing finished audio cannot add delivery (PSOLA restyling was inaudible, stronger time-warping sounded
unnatural, and Vevo-style re-speaking produced gibberish on Hindi), so the converter RE-SPEAKS the words
with a VoxCPM2 (Apache-2.0) LoRA trained on "teacher delivery in the target voice": the teacher's real
lectures re-voiced into the target voice with kNN-VC (`voice_convert.convert`), transcribed, and used as
the fine-tuning set. Rajwant Sir's pilot: 22 min, 152 sentences.

Per sentence the output is checked against the script with Whisper (phonetic CER) and regenerated up to
`max_tries` times, because VoxCPM2 varies run to run and its published Hindi WER is the weakest of the
open models we compared. The result is synthetic speech: label it and keep the provenance file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..audio import Audio

SENT_END = re.compile(r"(?<=[।.?!])\s+")


def split_script(text: str, max_chars: int = 160) -> list[str]:
    """Sentences for one generation each; over-long ones are split at commas (then hard-wrapped at spaces)."""
    out: list[str] = []
    for sent in SENT_END.split(" ".join(text.split())):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        cur = ""
        for part in re.split(r"(?<=,)\s+", sent):
            while len(part) > max_chars:  # no commas left: wrap at the last space before the limit
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(part[:cut].strip())
                part = part[cut:].strip()
            if cur and len(cur) + 1 + len(part) > max_chars:
                out.append(cur)
                cur = part
            else:
                cur = f"{cur} {part}".strip()
        if cur:
            out.append(cur)
    return out


def pause_after(sentence: str) -> float:
    """Gap after a sentence, in seconds. Questions get the longest wait (the teacher lets it land),
    clause breaks the shortest; values sit inside Rajwant Sir's measured pause range (p90 ~1.0 s)."""
    s = sentence.rstrip()
    if s.endswith("?"):
        return 0.8
    if s.endswith(("।", ".", "!")):
        return 0.55
    return 0.3


def join(pieces: list[np.ndarray], sentences: list[str], sr: int, fade_s: float = 0.01) -> np.ndarray:
    f = int(fade_s * sr)
    ramp = np.linspace(0, 1, f, dtype=np.float32) if f else np.zeros(0, np.float32)
    parts = []
    for y, sent in zip(pieces, sentences):
        y = np.asarray(y, dtype=np.float32).copy()
        if f and len(y) > 2 * f:
            y[:f] *= ramp
            y[-f:] *= ramp[::-1]
        parts += [y, np.zeros(int(pause_after(sent) * sr), np.float32)]
    return np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)


@dataclass
class SentenceResult:
    text: str
    tries: int
    cer: float | None
    heard: str = ""


@dataclass
class ConvertReport:
    sentences: list[SentenceResult] = field(default_factory=list)

    def summary(self) -> dict:
        c = [s.cer for s in self.sentences if s.cer is not None]
        return {"sentences": len(self.sentences), "retried": sum(s.tries > 1 for s in self.sentences),
                "median_cer": float(np.median(c)) if c else None, "max_cer": float(max(c)) if c else None,
                "over_threshold": [s.text for s in self.sentences if s.cer is not None and s.cer > 0.15]}


class TeacherStyleConverter:
    def __init__(self, base_model: str, lora_path: str | None, device: str = "cpu", asr=None):
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(base_model, load_denoiser=False, optimize=False, device=device,
                                            lora_weights_path=lora_path)
        self.sr = self.model.tts_model.sample_rate
        self.asr = asr

    def _heard(self, y: np.ndarray) -> str:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(self.sr, 16000)
        y16 = resample_poly(y, 16000 // g, self.sr // g).astype(np.float32)
        return " ".join(s.text for s in self.asr.transcribe(Audio(y16, 16000)))

    def render(self, sentences: list[str], reference_wav: str | None = None, max_tries: int = 3,
               cer_ok: float = 0.12, cfg_value: float = 2.0, timesteps: int = 10) -> tuple[list[np.ndarray], ConvertReport]:
        from ..preprocess.lang import phonetic_key
        from ..qc.rank import cer

        rep, out = ConvertReport(), []
        for sent in sentences:
            best = None
            for k in range(1, max_tries + 1):
                y = self.model.generate(text=sent, reference_wav_path=reference_wav, cfg_value=cfg_value,
                                        inference_timesteps=timesteps)
                if self.asr is None:
                    best = (y, None, "", k)
                    break
                heard = self._heard(y)
                c = cer(phonetic_key(sent), phonetic_key(heard))
                if best is None or c < best[1]:
                    best = (y, c, heard, k)
                if c <= cer_ok:
                    break
            out.append(best[0])
            rep.sentences.append(SentenceResult(sent, best[3], best[1], best[2]))
        return out, rep

    def convert(self, script: str, reference_wav: str | None = None, **kw) -> tuple[Audio, ConvertReport]:
        sents = split_script(script)
        pieces, rep = self.render(sents, reference_wav, **kw)
        y = join(pieces, sents, self.sr)
        peak = float(np.max(np.abs(y))) if y.size else 1.0
        return Audio((y / peak * 0.95 if peak > 0.95 else y).astype(np.float32), self.sr), rep


def build_training_set(teacher: Audio, target_voice: Audio, knn, asr, out_dir: str, min_s: float = 3.0,
                       max_s: float = 12.0, val_share: float = 0.05, seed: int = 0) -> dict:
    """Fine-tuning set of "teacher delivery in the target voice": re-voice the whole lecture with kNN-VC
    (sample-aligned, so cuts line up), cut 3-12 s sentences at pauses, transcribe the ORIGINAL audio
    (clearer than the converted one) and drop implausible transcripts (Whisper loops, empty windows).
    Writes wav/*.wav, train.jsonl and val.jsonl (VoxCPM manifest format) into `out_dir`."""
    import json
    import random
    from pathlib import Path

    import soundfile as sf

    from ..preprocess.ingest import segment_at_pauses
    from .voice_convert import build_matching_set, convert

    assert teacher.sr == 16000 and target_voice.sr == 16000
    out = Path(out_dir)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    conv = convert(knn, teacher, build_matching_set(knn, target_voice), topk=4)
    utts, cur = [], []
    for u in segment_at_pauses(teacher, min_pause=0.3):
        if cur and (u.end - cur[0].start > max_s or u.start - cur[-1].end > 1.2):
            utts.append((cur[0].start, cur[-1].end))
            cur = []
        cur.append(u)
    if cur:
        utts.append((cur[0].start, cur[-1].end))
    utts = [(max(0.0, a - 0.12), b + 0.15) for a, b in utts if min_s <= b - a <= max_s + 0.5]
    rows = []
    for i, (a, b) in enumerate(utts):
        i0, i1 = int(a * 16000), int(b * 16000)
        text = " ".join(s.text.strip() for s in asr.transcribe(Audio(teacher.y[i0:i1], 16000))).strip()
        cps = len(text.replace(" ", "")) / (b - a)
        if not text or not 4.0 <= cps <= 22.0:
            continue
        p = out / "wav" / f"u{i:05d}.wav"
        sf.write(p, conv.y[i0:i1], 16000)
        rows.append({"audio": str(p.resolve()), "text": text, "start": round(float(a), 2), "end": round(float(b), 2)})
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_share))
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"audio": r["audio"], "text": r["text"]}, ensure_ascii=False) + "\n")
    return {"utterances": len(rows), "minutes": round(sum(r["end"] - r["start"] for r in rows) / 60, 1),
            "candidates": len(utts), "val": n_val}
