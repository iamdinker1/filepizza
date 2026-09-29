"""Pluggable ASR / alignment / diarization / speaker-embedding backends.

Real backends import heavy optional dependencies lazily and need model downloads
(huggingface.co, download.pytorch.org). They are UNVERIFIED in the build sandbox, whose
network policy blocked those hosts; they follow the libraries' documented APIs and should be
smoke-tested on the first GPU box. CPU-only fallbacks that always work are marked FALLBACK and
set low confidence so their output is flagged, never silently trusted.

Recommended pilot stack (see docs/research_matrix.md):
  ASR        faster-whisper large-v3 (hi) + human correction of 30-60 min
  alignment  Montreal Forced Aligner, merged Hindi+English lexicon, adapted on the teacher
  diarize    pyannote community-1 + teacher-enrolment embedding threshold
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import Audio
from ..features.rate import syllable_count
from ..schema import Phone, Word


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[Word]
    conf: float = 1.0


# ------------------------------------------------------------------ ASR

class FasterWhisperASR:
    """UNVERIFIED here (needs faster-whisper + model download)."""

    def __init__(self, model: str = "large-v3", language: str | None = "hi", device: str = "auto",
                 compute_type: str = "int8", initial_prompt: str | None = None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        # A Hinglish prompt nudges script choice (Devanagari Hindi + Latin English) and physics terms.
        self.initial_prompt = initial_prompt or "Ab dekho, velocity constant nahi hai. Force = mass × acceleration."

    def transcribe(self, audio: Audio) -> list[AsrSegment]:
        segs, _ = self.model.transcribe(audio.y, language=self.language, word_timestamps=True, vad_filter=True,
                                        initial_prompt=self.initial_prompt, condition_on_previous_text=False)
        out = []
        for s in segs:
            words = [Word(w=w.word.strip(), start=w.start, end=w.end, conf=w.probability) for w in (s.words or [])]
            out.append(AsrSegment(s.start, s.end, s.text.strip(), words, float(np.exp(s.avg_logprob))))
        return out


def read_captions(path: str | Path) -> list[AsrSegment]:
    """SRT / WebVTT / JSON ([{start,end,text}]) transcripts, e.g. existing lecture captions."""
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return [AsrSegment(float(d["start"]), float(d["end"]), d["text"], []) for d in json.loads(txt)]

    def ts(s):
        s = s.replace(",", ".")
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    out = []
    for block in re.split(r"\n\s*\n", txt):
        m = re.search(r"([\d:.,]+)\s*-->\s*([\d:.,]+)[^\n]*\n(.+)", block, re.S)
        if m:
            text = re.sub(r"<[^>]+>", "", m.group(3)).replace("\n", " ").strip()
            if text:
                out.append(AsrSegment(ts(m.group(1)), ts(m.group(2)), text, []))
    return out


# ------------------------------------------------------------------ alignment

def read_textgrid(path: str | Path, word_tier: str = "words", phone_tier: str = "phones"):
    """Minimal long-format Praat TextGrid reader (MFA output). Returns (words, phones)."""
    txt = Path(path).read_text(encoding="utf-8")
    tiers = {}
    for m in re.finditer(r'name = "([^"]+)"(.*?)(?=item \[\d+\]:|\Z)', txt, re.S):
        ivs = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', m.group(2))
        tiers[m.group(1)] = [(float(a), float(b), t) for a, b, t in ivs]
    words = [Word(w=t, start=a, end=b, conf=1.0) for a, b, t in tiers.get(word_tier, []) if t.strip()]
    phones = []
    for a, b, t in tiers.get(phone_tier, []):
        if not t.strip() or t in ("sil", "sp", "spn"):
            continue
        wi = next((i for i, w in enumerate(words) if w.start - 1e-3 <= a and b <= w.end + 1e-3), -1)
        phones.append(Phone(p=t, start=a, end=b, word_idx=wi))
    return words, phones


def run_mfa(corpus_dir: str | Path, dictionary: str, acoustic_model: str, out_dir: str | Path, jobs: int = 4) -> Path:
    """Run `mfa align` (Montreal Forced Aligner CLI must be installed, e.g. via conda). UNVERIFIED here.
    corpus_dir holds <utt>.wav + <utt>.lab (normalised spoken text) pairs."""
    subprocess.run(["mfa", "align", str(corpus_dir), dictionary, acoustic_model, str(out_dir), "-j", str(jobs),
                    "--clean", "--output_format", "long_textgrid"], check=True)
    return Path(out_dir)


def proportional_align(text_words: list[str], audio: Audio, speech_runs: list[tuple[float, float]]) -> list[Word]:
    """FALLBACK aligner: spread words over detected speech runs in proportion to syllable counts.

    Confidence is fixed at 0.3 so `quality.assess` flags `align_low`. Good enough for rough
    utterance-level rate or for demos; NOT for word-level prosody claims.
    """
    if not text_words or not speech_runs:
        return []
    syl = np.array([max(1, syllable_count(w)) for w in text_words], float)
    total_speech = sum(b - a for a, b in speech_runs)
    per_syl = total_speech / syl.sum()
    words, run_i, t = [], 0, speech_runs[0][0]
    for w, s in zip(text_words, syl):
        dur = s * per_syl
        while run_i < len(speech_runs) and t + dur > speech_runs[run_i][1] + 1e-6:
            left = speech_runs[run_i][1] - t
            if left > 0.5 * dur or run_i == len(speech_runs) - 1:
                break
            run_i += 1
            t = speech_runs[run_i][0]
        words.append(Word(w=w, start=t, end=t + dur, conf=0.3))
        t += dur
    return words


# ------------------------------------------------------------------ diarization / speaker similarity

class PyannoteDiarizer:
    """UNVERIFIED here: pyannote community-1 needs an HF token and accepting the model conditions."""

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1", token: str | None = None):
        from pyannote.audio import Pipeline

        self.pipe = Pipeline.from_pretrained(model, token=token)

    def __call__(self, wav_path: str):
        out = self.pipe(wav_path)
        ann = getattr(out, "speaker_diarization", out)
        return [(turn.start, turn.end, spk) for turn, _, spk in ann.itertracks(yield_label=True)]


class SpeakerEmbedder:
    """Speaker embedding for teacher-vs-other decisions and for speaker-similarity scoring.

    method="ecapa"  speechbrain spkrec-ecapa-voxceleb (UNVERIFIED here)
    method="mfcc"   FALLBACK MFCC statistics (weak; tests / smoke runs only)
    """

    def __init__(self, method: str = "mfcc"):
        self.method = method
        if method == "ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def __call__(self, audio: Audio) -> np.ndarray:
        if self.method == "ecapa":
            import torch

            with torch.no_grad():
                return self.model.encode_batch(torch.tensor(audio.y)[None]).squeeze().numpy()
        from .quality import mfcc_embedding

        return mfcc_embedding(audio)
