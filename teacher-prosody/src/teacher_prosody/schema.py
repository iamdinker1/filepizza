"""Dataset schema. One JSONL row per utterance, one JSON per recording.

Everything needed to trace a measured prosodic pattern back to the exact audio and the
exact text is kept: original audio, cleaned audio, raw ASR text, corrected text,
language spans, word/phone timestamps, teacher, topic, session, recording condition,
consent reference and quality flags. The JSON Schema in `schema/utterance.schema.json`
mirrors these dataclasses and is checked in tests.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

Lang = Literal["hi", "en", "math", "other"]
Script = Literal["deva", "latn", "sym", "mixed"]
Split = Literal["train", "val", "test", "unassigned"]
RecordingCondition = Literal["studio", "classroom", "online_live", "phone", "unknown"]


@dataclass
class Recording:
    recording_id: str
    teacher_id: str
    session_id: str  # one teaching session; the unit for train/val/test splits
    source_uri: str  # where the original came from (YouTube URL, drive id, file path)
    original_path: str
    sha256: str
    duration_s: float
    sample_rate: int
    recording_condition: RecordingCondition = "unknown"
    subject: str = "physics"
    topic: str = ""
    lesson_type: str = ""  # conceptual / derivation / numericals / recap / mixed
    consent_ref: str = ""  # id of the signed consent + scope record; empty means NOT cleared
    consent_scope: list[str] = field(default_factory=list)  # e.g. ["analysis", "tts_training", "voice_clone"]
    cleaned_path: str = ""
    split: Split = "unassigned"
    notes: str = ""

    def cleared_for(self, use: str) -> bool:
        return bool(self.consent_ref) and use in self.consent_scope


@dataclass
class Word:
    w: str  # surface form as in corrected text
    start: float
    end: float
    lang: Lang = "other"
    conf: float = 1.0  # alignment confidence 0..1
    norm: str = ""  # normalised/spoken form (e.g. "m/s" -> "metre per second")


@dataclass
class Phone:
    p: str
    start: float
    end: float
    word_idx: int


@dataclass
class LangSpan:
    start_char: int
    end_char: int
    lang: Lang
    script: Script


@dataclass
class Quality:
    snr_db: Optional[float] = None
    clipping_frac: Optional[float] = None
    speech_frac: Optional[float] = None
    music_like_frac: Optional[float] = None
    teacher_sim: Optional[float] = None  # speaker similarity to teacher centroid
    align_conf: Optional[float] = None
    asr_conf: Optional[float] = None
    flags: list[str] = field(default_factory=list)

    @property
    def usable_for_style(self) -> bool:
        blocking = {"not_teacher", "overlap", "music", "clipping", "low_snr", "align_fail", "student_speech"}
        return not (set(self.flags) & blocking)


@dataclass
class Utterance:
    utt_id: str
    recording_id: str
    teacher_id: str
    session_id: str
    start: float
    end: float
    text_asr: str = ""
    text: str = ""  # human-corrected text (falls back to ASR text when uncorrected)
    text_corrected_by: str = ""  # "" = not corrected, else annotator id
    lang_spans: list[LangSpan] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    phones: list[Phone] = field(default_factory=list)
    speaker: str = "teacher"
    topic: str = ""
    recording_condition: RecordingCondition = "unknown"
    beat: str = ""  # pedagogical beat label (see pedagogy/taxonomy.yaml)
    beat_source: str = ""  # gold:<annotator> | rule | llm:<model> | cluster
    quality: Quality = field(default_factory=Quality)
    split: Split = "unassigned"

    @property
    def duration(self) -> float:
        return self.end - self.start

    @staticmethod
    def from_dict(d: dict) -> "Utterance":
        d = dict(d)
        d["lang_spans"] = [LangSpan(**x) for x in d.get("lang_spans", [])]
        d["words"] = [Word(**x) for x in d.get("words", [])]
        d["phones"] = [Phone(**x) for x in d.get("phones", [])]
        d["quality"] = Quality(**d.get("quality", {}))
        return Utterance(**d)


def to_dict(obj) -> dict:
    return asdict(obj)


def write_jsonl(path: str | Path, rows: Iterable) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r if isinstance(r, dict) else asdict(r), ensure_ascii=False) + "\n")
    return path


def read_utterances(path: str | Path) -> list[Utterance]:
    with open(path, encoding="utf-8") as f:
        return [Utterance.from_dict(json.loads(line)) for line in f if line.strip()]


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj if isinstance(obj, (dict, list)) else asdict(obj), f, ensure_ascii=False, indent=2)
    return path
