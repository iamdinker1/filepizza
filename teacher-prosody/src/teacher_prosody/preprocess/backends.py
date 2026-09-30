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


def nuclei_align(text_words: list[str], speech_runs: list[tuple[float, float]], nuclei_t: np.ndarray) -> list[Word]:
    """APPROXIMATE aligner when no forced aligner is available: map each word's syllables onto
    detected syllable nuclei (intensity peaks) in order, so local tempo and pauses shape the timing.

    Word i spans from the midpoint before its first syllable's nucleus to the midpoint after its
    last. Falls back to `proportional_align` if nuclei and syllable counts disagree by > 40%.
    Confidence 0.45 (vs 0.3 for proportional): usable for utterance/phrase-level analysis and
    rough word prominence, not for phone-level claims. Replace with MFA when available.
    """
    if not text_words or not speech_runs:
        return []
    syl = [max(1, syllable_count(w)) for w in text_words]
    total = sum(syl)
    t0, t1 = speech_runs[0][0], speech_runs[-1][1]
    nuc = np.sort(nuclei_t[(nuclei_t >= t0) & (nuclei_t <= t1)])
    if len(nuc) < 2 or not (0.6 * total <= len(nuc) <= 1.4 * total):
        return proportional_align(text_words, None, speech_runs)
    # syllable index -> nucleus time via linear resampling of the nuclei sequence
    pos = np.linspace(0, len(nuc) - 1, total)
    syl_t = np.interp(pos, np.arange(len(nuc)), nuc)
    edges = np.concatenate([[t0], (syl_t[:-1] + syl_t[1:]) / 2, [t1]])
    # never let a boundary sit inside a speech run's silence gap: snap to the gap edge nearest in time
    gaps = [(a[1], b[0]) for a, b in zip(speech_runs[:-1], speech_runs[1:]) if b[0] - a[1] > 0.12]
    words, k = [], 0
    for w, n in zip(text_words, syl):
        a, b = edges[k], edges[k + n]
        for g0, g1 in gaps:
            if g0 < a < g1:
                a = g1
            if g0 < b < g1:
                b = g0
        words.append(Word(w=w, start=float(a), end=float(max(b, a + 0.03)), conf=0.45))
        k += n
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


class SherpaWhisperASR:
    """Whisper (ONNX, int8) via sherpa-onnx with Silero VAD chunking. Runs on CPU with models from
    GitHub releases (k2-fsa/sherpa-onnx `asr-models`), so it works where huggingface.co is blocked.

    Runs on 4 CPUs (whisper-turbo int8). Each <= 8 s window is padded to 30 s because sherpa-onnx caps
    output tokens by input length, which truncated Hindi before (see `decode`). Hindi mode keeps English physics terms in Latin script and Hindi in
    Devanagari. sherpa-onnx decodes each BPE token to text on its own and drops partial UTF-8
    bytes (broken Devanagari), so `hexify_tokens` rewrites the token table to emit hex bytes that
    are re-assembled and decoded here. No word timestamps: pair with an aligner.
    """

    PAD_TO = int(29.9 * 16000)

    def __init__(self, model_dir: str, vad_model: str, language: str = "hi", num_threads: int = 4,
                 prefix: str = "turbo", max_chunk_s: float = 8.0, min_silence_s: float = 0.35):
        import sherpa_onnx

        d = Path(model_dir)
        hex_tokens = d / f"{prefix}-tokens-hex.txt"
        if not hex_tokens.exists():
            hexify_tokens(d / f"{prefix}-tokens.txt", hex_tokens)
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / f"{prefix}-encoder.int8.onnx"), decoder=str(d / f"{prefix}-decoder.int8.onnx"),
            tokens=str(hex_tokens), language=language, task="transcribe", num_threads=num_threads)
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = vad_model
        cfg.silero_vad.min_silence_duration = min_silence_s
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = max_chunk_s
        cfg.sample_rate = 16000
        self.vad_cfg = cfg

    def chunks(self, audio: Audio) -> list[tuple[float, np.ndarray]]:
        import sherpa_onnx

        assert audio.sr == 16000, "resample to 16 kHz first"
        vad = sherpa_onnx.VoiceActivityDetector(self.vad_cfg, buffer_size_in_seconds=60)
        win = self.vad_cfg.silero_vad.window_size
        out = []
        y = audio.y
        for i in range(0, len(y), win):
            vad.accept_waveform(y[i:i + win])
            while not vad.empty():
                out.append((vad.front.start / 16000, np.asarray(vad.front.samples, dtype=np.float32)))
                vad.pop()
        vad.flush()
        while not vad.empty():
            out.append((vad.front.start / 16000, np.asarray(vad.front.samples, dtype=np.float32)))
            vad.pop()
        return out

    def decode(self, samples: np.ndarray) -> str:
        # sherpa-onnx caps Whisper output at ~6 tokens per second of input audio, and Devanagari costs
        # ~1-2 tokens per character, so unpadded Hindi is cut off mid-sentence. Padding with silence to
        # Whisper's native 30 s window lifts the cap to ~180 tokens (enough for <= ~8 s of speech).
        if len(samples) < self.PAD_TO:
            samples = np.concatenate([samples, np.zeros(self.PAD_TO - len(samples), np.float32)])
        s = self.rec.create_stream()
        s.accept_waveform(16000, samples)
        self.rec.decode_stream(s)
        raw = bytes.fromhex("".join(re.findall(r"\{([0-9a-f]*)\}", s.result.text)))
        return raw.decode("utf-8", errors="ignore").strip()

    def merged_chunks(self, audio: Audio, max_window_s: float = 8.0) -> list[tuple[float, np.ndarray]]:
        """Merge neighbouring VAD chunks into windows <= max_window_s. Whisper's encoder always pays
        for a 30 s window, so many short calls are slow, and very short chunks decode poorly; windows
        longer than ~8 s of Hindi overflow the output cap (see `decode`)."""
        spans = [(t0, t0 + len(x) / 16000) for t0, x in self.chunks(audio)]
        merged: list[list[float]] = []
        for a, b in spans:
            if merged and b - merged[-1][0] <= max_window_s:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return [(a, audio.y[int(a * 16000):int(b * 16000)]) for a, b in merged]

    def transcribe(self, audio: Audio, progress: bool = False) -> list[AsrSegment]:
        out = []
        chunks = self.merged_chunks(audio)
        for k, (t0, samples) in enumerate(chunks):
            text = self.decode(samples)
            out.append(AsrSegment(t0, t0 + len(samples) / 16000, text, []))
            if progress and k % 5 == 0:
                print(f"  asr chunk {k + 1}/{len(chunks)} @ {t0 / 60:.1f} min: {text[:60]}", flush=True)
        return out


def hexify_tokens(src: str | Path, dst: str | Path) -> None:
    import base64

    lines = []
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        parts = line.split(" ")
        if len(parts) == 2:
            try:
                raw = base64.b64decode(parts[0])
                lines.append(base64.b64encode(("{" + raw.hex() + "}").encode()).decode() + " " + parts[1])
                continue
            except Exception:
                pass
        lines.append(line)
    Path(dst).write_text("\n".join(lines) + "\n", encoding="utf-8")
