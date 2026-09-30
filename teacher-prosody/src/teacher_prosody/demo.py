"""Offline end-to-end demo on MOCK data (espeak-ng voices, not a teacher).

It exercises every stage with known ground truth so the code paths are verified before real
lectures arrive:
  1. build a mock 3-session "teacher" corpus with scripted teaching behaviours (pre-reveal
     slowdown, post-question pause, NOT emphasis, transition energy reset), sessions recorded at
     different gains/noise to test confound handling; write wav + TextGrid + manifest
  2. run the analysis pipeline -> utterances.jsonl, profile.json, dashboard.html
  3. plan a Newton's-second-law micro-lesson with the director (+ SSML / ElevenLabs renders)
  4. generate 4 mock TTS candidates per beat (2 plan-following, 2 flat), rank with QC
  5. PSOLA-edit a flat candidate to impose the plan and re-score it
  6. stitch the chosen beats, report join diagnostics
  7. package a blinded listening test (no ratings are simulated)
Nothing here is evidence about real TTS quality; it proves the machinery works.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .audio import Audio, save
from .testing import WordSpec, espeak_utterance, write_textgrid

MOCK_LESSONS = {
    "s01": [
        ("explanation", "Force aur acceleration ka relation samajhte hain."),
        ("build_up", "Ab dekho, mass constant hai aur force ko double kar diya."),
        ("rhetorical_question", "Toh acceleration ka kya hoga?"),
        ("reveal", "Acceleration bhi double ho jaayega."),
        ("common_mistake", "Velocity is NOT constant here, yeh galti mat karna."),
        ("transition", "Chalo, ab next example dekhte hain."),
        ("example", "Maan lo ek ball ko dheere se push kiya."),
        ("recap", "Toh summary yeh hai ki force se acceleration badhta hai."),
    ],
    "s02": [
        ("explanation", "Momentum ko samajhna bahut zaroori hai."),
        ("build_up", "Ab socho, velocity same hai aur mass ko double kar diya."),
        ("rhetorical_question", "Toh momentum ka kya hoga?"),
        ("reveal", "Momentum bhi double ho jaayega."),
        ("common_mistake", "Momentum is NOT energy, dono ko confuse mat karna."),
        ("transition", "Chalo, aage badhte hain."),
        ("example", "Jaise ek truck aur ek cycle same speed par chal rahe hain."),
        ("recap", "Toh total yeh hua ki momentum mass aur velocity dono par depend karta hai."),
    ],
    "s03": [
        ("explanation", "Friction ek force hai jo motion ko oppose karti hai."),
        ("build_up", "Ab dekho, surface rough kar diya aur weight same rakha."),
        ("rhetorical_question", "Toh friction ka kya hoga?"),
        ("reveal", "Friction zyada ho jaayega."),
        ("common_mistake", "Friction is NOT always bad, yeh yaad rakhna."),
        ("transition", "Chalo, ab numerical karte hain."),
        ("example", "Maan lo ek block table par pada hai."),
        ("recap", "Toh summary yeh hai ki friction surface par depend karta hai."),
    ],
}
SESSION_COND = {"s01": ("studio", 1.0, -65.0), "s02": ("classroom", 0.4, -50.0), "s03": ("online_live", 0.7, -58.0)}
KEYS = {"double", "not", "zyada", "acceleration", "momentum", "friction"}


def _teacher_specs(beat: str, text: str) -> list[WordSpec]:
    """Scripted 'teacher' behaviour per beat (the ground truth the analysis should recover)."""
    words = text.split()
    n = len(words)
    specs = []
    for i, w in enumerate(words):
        c = w.strip(".,?!").lower()
        s = WordSpec(text=w, pitch=48, speed=175, amp=100, pause_after=0.07)
        if beat == "build_up":
            s.speed = int(180 - 45 * i / max(1, n - 1))  # slows toward the end
            s.pitch = 52
        elif beat == "rhetorical_question":
            s.pitch = 55 if i < n - 1 else 80  # rising final
            s.speed = 160
        elif beat == "reveal":
            s.speed = 135
            if c in KEYS and c not in ("acceleration", "momentum", "friction"):
                s.pitch, s.speed, s.amp = 82, 100, 160
        elif beat == "common_mistake":
            if c == "not":
                s.pitch, s.speed, s.amp = 85, 105, 170
            if i + 1 < n and words[i + 1].lower() == "not":
                s.pause_after = 0.22  # pre-emphasis pause
        elif beat == "transition":
            s.pitch, s.amp = (68, 140) if i == 0 else (55, 115)
        elif beat == "example":
            s.speed, s.pitch = 190, 52
        elif beat == "recap":
            s.speed, s.pitch, s.amp = 195, 44, 85
        if w.endswith(",") and s.pause_after < 0.2:
            s.pause_after = 0.2
        specs.append(s)
    return specs


INTER_UTT = {"rhetorical_question": 0.95, "transition": 0.55}


def build_mock_corpus(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rec_entries = []
    for sess, lesson in MOCK_LESSONS.items():
        cond, gain, noise = SESSION_COND[sess]
        chunks, words_all, gold, t = [], [], [], 0.0
        for bi, (beat, text) in enumerate(lesson):
            a, words = espeak_utterance(_teacher_specs(beat, text), lead=0.0, tail=0.0, noise_db=-90, seed=bi)
            for w in words:
                words_all.append(type(w)(w=w.w, start=w.start + t, end=w.end + t, lang=w.lang))
            chunks.append(a.y)
            gold.append({"start": t, "end": t + a.duration, "beat": beat, "annotator": "mock_script"})
            t += a.duration
            gap = INTER_UTT.get(beat, 0.6)
            if bi + 1 < len(lesson) and lesson[bi + 1][0] == "reveal":
                gap = 0.95
            chunks.append(np.zeros(int(gap * a.sr), np.float32))
            t += gap
        y = np.concatenate(chunks) * gain
        y = y + np.random.default_rng(7).normal(0, 10 ** (noise / 20), len(y)).astype(np.float32)
        audio = Audio(y.astype(np.float32), 16000)
        wav = save(out / f"mock_{sess}.wav", audio)
        write_textgrid(out / f"mock_{sess}.TextGrid", words_all, audio.duration)
        (out / f"mock_{sess}.beats.json").write_text(json.dumps(gold, indent=1))
        rec_entries.append({"path": str(wav), "session_id": sess, "recording_id": f"mock_{sess}", "condition": cond,
                            "topic": lesson[0][1].split()[0].lower(), "lesson_type": "conceptual",
                            "transcript": str(out / f"mock_{sess}.TextGrid"), "consent_ref": "MOCK-SYNTHETIC",
                            "consent_scope": ["analysis"], "gold_beats": str(out / f"mock_{sess}.beats.json")})
    man = out / "manifest.yaml"
    man.write_text(yaml.safe_dump({"teacher_id": "mock_teacher", "recordings": rec_entries}, sort_keys=False))
    return man


LESSON = [
    {"beat": "introduce_concept", "text": "Aaj hum Newton ka second law samjhenge."},
    {"beat": "build_up", "text": "Ab dekho, mass constant rakha hai aur force ko double kar diya."},
    {"beat": "rhetorical_question", "text": "Toh acceleration ka kya hoga?"},
    {"beat": "reveal", "text": "Acceleration bhi double ho jaayega, kyunki a = F/m."},
    {"beat": "common_mistake", "text": "Velocity is NOT constant here, yeh galti mat karna."},
]
CONCEPT_MAP = {"key_terms": ["force", "acceleration", "mass", "velocity"], "changing": ["force"], "relation": "a = F/m",
               "misconception": "doubling force doubles velocity"}


def run(out_dir: str | Path) -> dict:
    from .director.director import plan_lesson
    from .director.render import to_elevenlabs, to_ssml, to_word_targets
    from .eval.listening_test import package
    from .pipeline import run_manifest
    from .qc.rank import rank, regeneration_action, scored_to_dict
    from .synth.backends import Candidate, EspeakMock
    from .synth.prosody_edit import apply_word_targets
    from .synth.stitch import stitch
    from .dashboard import report
    from .features.energy import extract_energy
    from .features.f0 import extract_f0
    from .features.words import word_prosody

    out = Path(out_dir)
    man = build_mock_corpus(out / "corpus")
    res = run_manifest(man, out / "analysis")
    profile = json.loads(Path(res["profile"]).read_text())

    plan = plan_lesson("nlm2-demo", "mock_teacher", LESSON, CONCEPT_MAP, profile)
    plan.to_json(out / "plan.json")
    (out / "plan.ssml").write_text(to_ssml(plan))
    (out / "elevenlabs_requests.json").write_text(json.dumps(to_elevenlabs(plan, "<VOICE_ID>"), indent=2, ensure_ascii=False))

    base = profile.get("session_baselines", {})
    teacher = {"ref_hz": None, "artic_rate_sps": float(np.nanmedian([b["artic_rate_sps"] for b in base.values()])) if base else 5.0,
               "per_beat": profile.get("per_beat", {})}
    tts = EspeakMock()
    chosen, rankings, edits, stim = [], {}, [], []
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    for bi, bp in enumerate(plan.beats):
        cands = tts.generate(bp, n=4, follow_plan=[True, True, False, False], seed=100 + bi)
        ranked = rank(cands, bp, teacher)
        rankings[bp.beat_id] = {"ranked": [scored_to_dict(s) for s in ranked], "next": regeneration_action(ranked, 0)}
        # PSOLA: impose the plan on the best FLAT candidate and re-score
        flat = [c for c in cands if not c.meta["follow_plan"]]
        fb = max(flat, key=lambda c: next(s.total for s in ranked if s.k == c.k))
        edited_audio, edited_words = apply_word_targets(fb.audio, fb.words, to_word_targets(bp))
        ec = Candidate(bp.beat_id, 99, edited_audio, edited_words, "espeak-mock+psola", {"follow_plan": "psola"})
        es = rank([ec], bp, teacher)[0]
        before = next(s for s in ranked if s.k == fb.k)
        edits.append({"beat_id": bp.beat_id, "flat_k": fb.k, "emphasis_before": before.sub.get("emphasis"),
                      "emphasis_after": es.sub.get("emphasis"), "pauses_before": before.sub.get("pauses"),
                      "pauses_after": es.sub.get("pauses"), "total_before": before.total, "total_after": es.total})
        best = next(c for c in cands if c.k == ranked[0].k)
        chosen.append(best)
        for c, tag in ((best, "ranked_best"), (fb, "flat"), (ec, "flat_plus_psola")):
            p = save(out / "candidates" / f"{bp.beat_id}_{tag}.wav", c.audio)
            stim.append({"item": bp.beat_id, "system": tag, "path": str(p)})
    (out / "ranking.json").write_text(json.dumps(rankings, indent=1, default=float, ensure_ascii=False))
    (out / "psola_edits.json").write_text(json.dumps(edits, indent=1, default=float))

    lecture, rep = stitch([c.audio for c in chosen], [b.pause_before_s for b in plan.beats[1:]])
    save(out / "lecture_mock.wav", lecture)
    (out / "stitch_report.json").write_text(json.dumps(rep.__dict__, indent=1, default=float))
    lt = package(stim, out / "listening_test")

    # emphasis trace of the reference "teacher" NOT utterance and of the chosen candidate
    a, words = espeak_utterance(_teacher_specs("common_mistake", "Velocity is NOT constant here."), seed=3)
    f0, e = extract_f0(a), extract_energy(a)
    rows = word_prosody(words, f0, e, f0.median())
    trace = report.emphasis_trace(a, f0, e, words, rows, f0.median(), "MOCK teacher: 'Velocity is NOT constant here.'")
    save(out / "velocity_not_constant_mock.wav", a)
    cm = chosen[-1]
    f0c, ec2 = extract_f0(cm.audio), extract_energy(cm.audio)
    rows_c = word_prosody(cm.words, f0c, ec2, f0c.median())
    trace_c = report.emphasis_trace(cm.audio, f0c, ec2, cm.words, rows_c, f0c.median(), "Chosen candidate for the common_mistake beat")
    sections = [
        ("Emphasis trace: 'Velocity is NOT constant'", report.img(trace) + "<pre>" + json.dumps(
            [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.to_dict().items()
              if k in ("w", "f0_excursion_st", "energy_mean_db", "dur_z", "pause_before", "prominence", "main_channel")} for r in rows],
            indent=1) + "</pre>"),
        ("Chosen candidate, same beat", report.img(trace_c)),
        ("Performance plan (director v0)", "<pre>" + json.dumps(json.loads(plan.to_json()), indent=1, ensure_ascii=False)[:6000] + "</pre>"),
        ("PSOLA post-edit effect on flat candidates", "<pre>" + json.dumps(edits, indent=1, default=float) + "</pre>"),
        ("Stitch report", "<pre>" + json.dumps(rep.__dict__, indent=1, default=float) + "</pre>"),
    ]
    report.build(out / "demo_report.html", "Teacher prosody – offline demo (MOCK voices)", sections,
                 note="All audio here is espeak-ng synthesis standing in for a teacher and a TTS; it verifies the pipeline, not quality.")
    summary = {"analysis": res, "plan": str(out / "plan.json"), "psola_edits": edits, "stitch_flags": rep.flags,
               "listening_test": lt, "demo_report": str(out / "demo_report.html")}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
