"""TTS backend adapters: one interface, many engines, N candidates per segment.

  EspeakMock        MOCK: local espeak-ng, used in tests / offline demo. Not a teacher voice.
  ElevenLabsTTS     UNVERIFIED here (no key/network): REST /v1/text-to-speech/{voice}/with-timestamps
                    with previous_text / next_text stitching; returns char-level timings.
  CommandTTS        wraps any open model's CLI (VoxCPM2, MOSS-TTS, Chatterbox, FastPitch...) via a
                    command template, so GPU inference can live in its own environment.

Every backend returns Candidate objects with audio + (if available) word timings, so the QC
ranker can score them identically.
"""
from __future__ import annotations

import base64
import io
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio import Audio, load
from ..schema import Word


@dataclass
class Candidate:
    beat_id: str
    k: int
    audio: Audio
    words: list[Word] = field(default_factory=list)
    backend: str = ""
    meta: dict = field(default_factory=dict)


class EspeakMock:
    """MOCK TTS: espeak-ng with per-candidate random pitch/speed jitter and optional plan-following.
    Generates word-by-word so exact word timings are known."""

    name = "espeak-mock"

    def __init__(self, voice_en: str = "en-us", voice_hi: str = "hi", base_pitch: int = 45, base_speed: int = 165):
        self.voice_en, self.voice_hi = voice_en, voice_hi
        self.base_pitch, self.base_speed = base_pitch, base_speed

    def generate(self, beat_plan, n: int = 4, follow_plan: bool | list[bool] = True, seed: int = 0) -> list[Candidate]:
        from ..preprocess.lang import word_langs
        from ..testing import WordSpec, espeak_utterance

        rng = np.random.default_rng(seed)
        langs = word_langs([w.w for w in beat_plan.words])
        out = []
        for k in range(n):
            follow = follow_plan[k] if isinstance(follow_plan, list) else follow_plan
            jitter_p = int(rng.integers(-8, 9))
            jitter_s = int(rng.integers(-15, 16))
            rate = (beat_plan.rate_start + beat_plan.rate_end) / 2
            spec = []
            for w, l in zip(beat_plan.words, langs):
                emph = w.emphasis if follow else 0
                spec.append(WordSpec(
                    text=w.w, voice=self.voice_hi if l == "hi" and any("ऀ" <= c <= "ॿ" for c in w.w) else self.voice_en,
                    pitch=int(np.clip(self.base_pitch + jitter_p + (w.pitch_st * 4 if emph else 0), 0, 99)),
                    speed=int(np.clip((self.base_speed + jitter_s) * (rate if follow else 1.0) / (w.dur_scale if emph else 1.0), 80, 400)),
                    amp=int(np.clip(100 * 10 ** ((w.energy_db if emph else 0) / 20), 20, 200)),
                    pause_after=0.05, lang=l))
                if follow and w.pre_pause_s > 0 and spec[:-1]:
                    spec[-2].pause_after += w.pre_pause_s
            audio, words = espeak_utterance(spec, lead=0.05, tail=0.05, seed=seed + k)
            out.append(Candidate(beat_plan.beat_id, k, audio, words, self.name,
                                 {"follow_plan": follow, "jitter_pitch": jitter_p, "jitter_speed": jitter_s}))
        return out


class ElevenLabsTTS:
    """UNVERIFIED in the build sandbox. Uses the documented with-timestamps endpoint."""

    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps?output_format=pcm_16000"

    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ["ELEVENLABS_API_KEY"]

    def generate_request(self, req: dict, n: int = 4, seed0: int = 0) -> list[Candidate]:
        import requests

        out = []
        for k in range(n):
            body = dict(req["body"], seed=seed0 + k)
            r = requests.post(self.URL.format(voice_id=req["voice_id"]), json=body,
                              headers={"xi-api-key": self.key}, timeout=120)
            r.raise_for_status()
            d = r.json()
            pcm = np.frombuffer(base64.b64decode(d["audio_base64"]), dtype="<i2").astype(np.float32) / 32768
            words = _chars_to_words(d.get("alignment") or {})
            out.append(Candidate(req["beat_id"], k, Audio(pcm, 16000), words, self.name,
                                 {"request_id": r.headers.get("request-id"), "seed": seed0 + k}))
        return out


def _chars_to_words(al: dict) -> list[Word]:
    chars = al.get("characters", [])
    st, en = al.get("character_start_times_seconds", []), al.get("character_end_times_seconds", [])
    words, cur, t0, t1 = [], "", None, None
    for c, a, b in zip(chars, st, en):
        if c.isspace():
            if cur:
                words.append(Word(w=cur, start=t0, end=t1))
            cur, t0 = "", None
            continue
        if t0 is None:
            t0 = a
        cur += c
        t1 = b
    if cur:
        words.append(Word(w=cur, start=t0, end=t1))
    return words


class CommandTTS:
    """Run an external TTS CLI. Template fields: {text_file} {out_wav} {ref_wav} {seed}.
    Example: 'python -m voxcpm_infer --text-file {text_file} --ref {ref_wav} --seed {seed} --out {out_wav}'"""

    name = "command"

    def __init__(self, template: str, ref_wav: str = "", name: str = "command"):
        self.template, self.ref_wav, self.name = template, ref_wav, name

    def generate_text(self, beat_id: str, text: str, n: int = 4, seed0: int = 0) -> list[Candidate]:
        out = []
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d) / "in.txt"
            tf.write_text(text, encoding="utf-8")
            for k in range(n):
                wav = Path(d) / f"c{k}.wav"
                cmd = self.template.format(text_file=tf, out_wav=wav, ref_wav=self.ref_wav, seed=seed0 + k)
                subprocess.run(shlex.split(cmd), check=True)
                out.append(Candidate(beat_id, k, load(wav), [], self.name, {"cmd": cmd}))
        return out
