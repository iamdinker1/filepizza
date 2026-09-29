"""Render a PerformancePlan into backend-specific controls.

  to_ssml          Azure-style SSML (prosody rate/pitch/volume, break). Note Azure docs: <lang> cannot
                   be combined with prosody/break, emphasis works on few voices - so emphasis is
                   rendered as nested <prosody>.
  to_elevenlabs    request bodies with previous_text/next_text stitching context; pauses as
                   <break> (Multilingual v2) or [pause]-style tags (v3/v4). Caps-for-emphasis is an
                   EXPERIMENT flag, off by default.
  to_inline_tags   generic inline tags for tag-driven LLM-TTS (e.g. MOSS-TTS `[pause X.Ys]`).
  to_word_targets  numeric per-word targets for explicit-control models / PSOLA post-editing.
Do not assume a tag is obeyed: the QC step measures whether each planned emphasis/pause happened.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from .plan import BeatPlan, PerformancePlan


def _pct(x: float) -> str:
    return f"{(x - 1) * 100:+.0f}%"


def to_ssml(plan: PerformancePlan, voice: str = "hi-IN-AartiNeural", lang: str = "hi-IN") -> str:
    parts = [f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang}">', f'<voice name="{voice}">']
    for b in plan.beats:
        if b.pause_before_s > 0:
            parts.append(f'<break time="{int(b.pause_before_s * 1000)}ms"/>')
        rate = (b.rate_start + b.rate_end) / 2
        parts.append(f'<prosody rate="{_pct(rate)}" pitch="{b.register_st:+.1f}st" volume="{b.energy_db:+.1f}dB">')
        for w in b.words:
            if w.pre_pause_s > 0:
                parts.append(f'<break time="{int(w.pre_pause_s * 1000)}ms"/>')
            if w.emphasis:
                parts.append(f'<prosody pitch="{w.pitch_st:+.1f}st" rate="{_pct(1 / w.dur_scale)}" volume="{w.energy_db:+.1f}dB">'
                             f"{escape(w.w)}</prosody>")
            else:
                parts.append(escape(w.w))
        parts.append("</prosody>")
        if b.pause_after_s > 0:
            parts.append(f'<break time="{int(b.pause_after_s * 1000)}ms"/>')
    parts += ["</voice>", "</speak>"]
    return " ".join(parts)


def _eleven_pause(seconds: float, model_id: str) -> str:
    if seconds < 0.15:
        return ""
    if "v2" in model_id or "flash" in model_id or "turbo" in model_id:
        return f'<break time="{min(seconds, 3.0):.1f}s" />'
    return "[long pause]" if seconds >= 0.8 else "[pause]"


def to_elevenlabs(plan: PerformancePlan, voice_id: str, model_id: str = "eleven_multilingual_v2",
                  caps_emphasis: bool = False, stability: float = 0.5, similarity: float = 0.85,
                  seed: int | None = None) -> list[dict]:
    """One request per beat. Pauses between beats are NOT baked in; the stitcher inserts the
    planned silence so it can be controlled exactly."""
    reqs = []
    for b in plan.beats:
        toks = []
        for w in b.words:
            if w.pre_pause_s > 0:
                toks.append(_eleven_pause(w.pre_pause_s, model_id))
            toks.append(w.w.upper() if caps_emphasis and w.emphasis >= 2 else w.w)
        body = {"text": " ".join(t for t in toks if t), "model_id": model_id,
                "voice_settings": {"stability": stability, "similarity_boost": similarity},
                "previous_text": b.prev_text or None, "next_text": b.next_text or None}
        if "v2" in model_id:
            body["voice_settings"]["speed"] = round(max(0.7, min(1.2, (b.rate_start + b.rate_end) / 2)), 2)
        if seed is not None:
            body["seed"] = seed
        reqs.append({"beat_id": b.beat_id, "voice_id": voice_id, "body": body})
    return reqs


def to_inline_tags(plan: PerformancePlan, pause_fmt: str = "[pause {s:.1f}s]", emph_fmt: str = "{w}") -> list[str]:
    out = []
    for b in plan.beats:
        toks = []
        for w in b.words:
            if w.pre_pause_s > 0:
                toks.append(pause_fmt.format(s=w.pre_pause_s))
            toks.append(emph_fmt.format(w=w.w) if w.emphasis >= 2 else w.w)
        out.append(" ".join(toks))
    return out


def to_word_targets(b: BeatPlan) -> list[dict]:
    """Per-word numeric targets with the beat pace ramp folded in (for explicit control / PSOLA)."""
    n = max(1, len(b.words) - 1)
    out = []
    for w in b.words:
        frac = w.i / n
        rate = b.rate_start + (b.rate_end - b.rate_start) * frac
        out.append({"i": w.i, "w": w.w, "dur_scale": w.dur_scale / rate, "pitch_st": w.pitch_st + b.register_st,
                    "energy_db": w.energy_db, "pre_pause_s": w.pre_pause_s, "emphasis": w.emphasis})
    return out
