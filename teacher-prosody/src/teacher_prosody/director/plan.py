"""Performance plan: the contract between the teaching director and any synthesis backend.

All prosodic targets are RELATIVE to the teacher's own baselines (rate multipliers, semitone
offsets from the teacher's median F0, dB offsets from their speech level), so one plan can be
rendered by an SSML engine, a tag-driven LLM-TTS, an explicit-control model (FastPitch) or the
PSOLA post-editor, and compared on equal terms.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class WordPlan:
    i: int  # index in the beat's spoken word list
    w: str
    emphasis: int = 0  # 0 none, 1 light, 2 clear, 3 strong (at most one per beat)
    pitch_st: float = 0.0  # peak F0 offset on the word (st)
    dur_scale: float = 1.0  # duration multiplier for the word
    energy_db: float = 0.0
    pre_pause_s: float = 0.0  # silence inserted right before the word
    reason: str = ""


@dataclass
class BeatPlan:
    beat_id: str
    beat: str
    text: str  # as written in the script
    spoken: str  # normalised spoken form (what the TTS reads)
    prev_text: str = ""
    next_text: str = ""
    rate_start: float = 1.0  # x teacher median articulation rate at beat start
    rate_end: float = 1.0  # ... and at beat end (linear ramp) - e.g. slow into a reveal
    register_st: float = 0.0  # F0 register offset
    range_scale: float = 1.0  # F0 range multiplier
    energy_db: float = 0.0
    pause_before_s: float = 0.25
    pause_after_s: float = 0.25
    final_contour: str = "fall"  # fall | rise | level
    words: list[WordPlan] = field(default_factory=list)
    reference_clips: list[str] = field(default_factory=list)  # retrieved teacher moments (ids)
    formulas: list[dict] = field(default_factory=list)  # written/spoken pairs for pronunciation QC
    must_review: bool = False
    rationale: list[str] = field(default_factory=list)

    @property
    def emphasised(self) -> list[WordPlan]:
        return [w for w in self.words if w.emphasis > 0]


@dataclass
class PerformancePlan:
    lesson_id: str
    teacher_id: str
    beats: list[BeatPlan]
    profile_version: str = ""
    director: str = "rules-v0"

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @staticmethod
    def from_dict(d: dict) -> "PerformancePlan":
        beats = []
        for b in d["beats"]:
            b = dict(b)
            b["words"] = [WordPlan(**w) for w in b.get("words", [])]
            beats.append(BeatPlan(**b))
        return PerformancePlan(d["lesson_id"], d["teacher_id"], beats, d.get("profile_version", ""), d.get("director", ""))
