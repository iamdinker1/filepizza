"""Command line: `tp <command>` (or `python -m teacher_prosody.cli <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tp", description="Teacher prosody research toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="run the analysis pipeline on a manifest of recordings")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)

    n = sub.add_parser("normalize", help="show the spoken form of a Hinglish physics sentence")
    n.add_argument("text")

    p = sub.add_parser("plan", help="make a performance plan for a lesson script (YAML)")
    p.add_argument("--script", required=True, help="YAML: {lesson_id, teacher_id, concept_map, beats:[{beat,text}]}")
    p.add_argument("--profile", help="profile.json from `tp analyze`")
    p.add_argument("--out", required=True)
    p.add_argument("--ssml-voice", default="hi-IN-AartiNeural")
    p.add_argument("--elevenlabs-voice", default="<VOICE_ID>")

    lb = sub.add_parser("annotate", help="rule-based beat labels for sentences in a text file")
    lb.add_argument("text_file")

    fd = sub.add_parser("findings", help="teacher findings report from one recording + transcript")
    fd.add_argument("--audio", required=True)
    fd.add_argument("--transcript", required=True, help="ASR/caption JSON [{start,end,text}], SRT/VTT, or MFA TextGrid")
    fd.add_argument("--teacher", required=True)
    fd.add_argument("--out", required=True)
    fd.add_argument("--consent-ref", default="")

    tr = sub.add_parser("transcribe", help="Whisper (sherpa-onnx, CPU) transcription to JSON windows")
    tr.add_argument("--audio", required=True)
    tr.add_argument("--model-dir", required=True, help="dir with <prefix>-encoder/decoder.int8.onnx + tokens")
    tr.add_argument("--vad", required=True, help="silero_vad.onnx")
    tr.add_argument("--language", default="hi")
    tr.add_argument("--out", required=True)

    rs = sub.add_parser("restyle", help="move TTS audio toward a teacher's measured delivery (PSOLA, transcript-free)")
    rs.add_argument("--audio", required=True, help="TTS audio (any format ffmpeg reads)")
    rs.add_argument("--target", required=True, help="teacher style JSON from --save-target, or teacher audio to measure")
    rs.add_argument("--save-target", help="write the measured teacher style JSON here")
    rs.add_argument("--out", required=True, help="output .wav")
    rs.add_argument("--sr", type=int, default=44100)
    rs.add_argument("--strength", type=float, default=1.0)
    rs.add_argument("--match-register", action="store_true", help="also move the voice's pitch register (usually sounds processed)")

    vc = sub.add_parser("convert", help="re-render a performance in the teacher's voice (kNN-VC; needs voice-cloning consent)")
    vc.add_argument("--audio", required=True, help="source performance (TTS take or human read)")
    vc.add_argument("--teacher-audio", required=True, help="teacher reference speech (5-30 min, clean)")
    vc.add_argument("--model-dir", required=True, help="dir with kNN-VC code + WavLM-Large.pt + prematch_g_02500000.pt")
    vc.add_argument("--out", required=True)
    vc.add_argument("--consent-ref", required=True, help="id of the signed voice-cloning consent record")
    vc.add_argument("--cache", help="where to cache the teacher matching set (.pt)")
    vc.add_argument("--topk", type=int, default=4)
    vc.add_argument("--transplant-intonation", action="store_true", help="impose the source's intonation at the teacher's register")

    d = sub.add_parser("demo", help="offline end-to-end demo on mock (espeak) audio")
    d.add_argument("--out", default="out/demo")

    lp = sub.add_parser("listening-package", help="blind stimuli for a listening test")
    lp.add_argument("--stimuli", required=True, help="CSV with columns item,system,path")
    lp.add_argument("--out", required=True)

    la = sub.add_parser("listening-analyze", help="summarise ratings with cluster-bootstrap CIs")
    la.add_argument("--ratings", required=True)
    la.add_argument("--key", required=True)
    la.add_argument("--question", default="teacher_explaining")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        from .pipeline import run_manifest

        print(json.dumps(run_manifest(args.manifest, args.out), indent=1))
    elif args.cmd == "normalize":
        from .director.text_norm import normalise
        from .preprocess.lang import cmi, tag_tokens

        r = normalise(args.text)
        print(json.dumps({"spoken": r.spoken, "formulas": [f.__dict__ for f in r.formulas],
                          "tokens": [(t, l) for t, _, _, l in tag_tokens(args.text)], "cmi": cmi(args.text)},
                         ensure_ascii=False, indent=1))
    elif args.cmd == "plan":
        import yaml

        from .director.director import plan_lesson
        from .director.render import to_elevenlabs, to_ssml

        sc = yaml.safe_load(Path(args.script).read_text())
        prof = json.loads(Path(args.profile).read_text()) if args.profile else None
        plan = plan_lesson(sc["lesson_id"], sc.get("teacher_id", ""), sc["beats"], sc.get("concept_map"), prof, sc.get("lexicon"))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        plan.to_json(out / "plan.json")
        (out / "plan.ssml").write_text(to_ssml(plan, voice=args.ssml_voice))
        (out / "elevenlabs_requests.json").write_text(json.dumps(to_elevenlabs(plan, args.elevenlabs_voice), indent=1, ensure_ascii=False))
        print(f"wrote {out}/plan.json, plan.ssml, elevenlabs_requests.json")
    elif args.cmd == "annotate":
        from .pedagogy.rules import annotate, split_sentences

        sents = split_sentences(Path(args.text_file).read_text(encoding="utf-8"))
        for s, lab in zip(sents, annotate(sents)):
            print(f"{lab.beat:20s} {lab.conf:.2f}  {s}")
    elif args.cmd == "findings":
        from .findings import teacher_findings

        F = teacher_findings(args.audio, args.transcript, args.out, args.teacher, consent_ref=args.consent_ref)
        print(json.dumps({k: F[k] for k in ("minutes", "n_utterances", "voice", "questions")}, indent=1, default=float, ensure_ascii=False))
    elif args.cmd == "transcribe":
        from .audio import load
        from .preprocess.backends import SherpaWhisperASR

        asr = SherpaWhisperASR(args.model_dir, args.vad, language=args.language)
        segs = asr.transcribe(load(args.audio), progress=True)
        Path(args.out).write_text(json.dumps([{"start": s.start, "end": s.end, "text": s.text} for s in segs],
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    elif args.cmd == "restyle":
        from .audio import load, save
        from .synth.restyle import StyleStats, evaluate_restyle, measure_style, restyle

        if args.target.endswith(".json"):
            target = StyleStats.from_json(json.loads(Path(args.target).read_text()))
        else:
            target = measure_style(load(args.target))
            if args.save_target:
                Path(args.save_target).write_text(json.dumps(target.to_json()))
        src = load(args.audio, sr=args.sr)
        out, edits = restyle(src, target, strength=args.strength, match_register=args.match_register)
        save(args.out, out)
        ev = evaluate_restyle(out, edits, target)
        rep = {"evaluation": ev, "edits": {k: v for k, v in edits.items() if not k.startswith("_")}}
        Path(args.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1, default=float))
        print(json.dumps(ev, indent=1, default=float))
    elif args.cmd == "convert":
        from .audio import load, save
        from .synth.voice_convert import build_matching_set, convert, load_knnvc, transplant_intonation

        knn = load_knnvc(args.model_dir)
        teacher = load(args.teacher_audio)
        m = build_matching_set(knn, teacher, cache=args.cache)
        src = load(args.audio)
        out = convert(knn, src, m, topk=args.topk)
        if args.transplant_intonation:
            from .features.f0 import extract_f0

            out = transplant_intonation(out, src, extract_f0(teacher).median())
        save(args.out, out)
        Path(args.out).with_suffix(".provenance.json").write_text(json.dumps(
            {"synthetic": True, "method": "kNN-VC (WavLM-Large layer 6, prematched HiFi-GAN)", "source": args.audio,
             "teacher_reference": args.teacher_audio, "consent_ref": args.consent_ref, "topk": args.topk,
             "intonation_transplant": args.transplant_intonation}, indent=1))
        print(f"wrote {args.out} (SYNTHETIC voice; provenance saved next to it)")
    elif args.cmd == "demo":
        from .demo import run

        print(json.dumps(run(args.out), indent=1, default=float))
    elif args.cmd == "listening-package":
        import csv

        from .eval.listening_test import package

        with open(args.stimuli) as f:
            stim = list(csv.DictReader(f))
        print(json.dumps(package(stim, args.out), indent=1))
    elif args.cmd == "listening-analyze":
        from .eval.listening_test import load_ratings, summarise

        print(json.dumps(summarise(load_ratings(args.ratings, args.key), args.question), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
