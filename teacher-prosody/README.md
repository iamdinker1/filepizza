# teacher-prosody

Measure how a teacher speaks while teaching, turn that into an explainable performance plan, and
check whether generated audio for cached one-to-one AI lectures actually delivers it. Covered:
pitch, local pace, pauses, energy, emphasis, question delivery, build-up/reveal, and Hindi–English–Hinglish switching.

> **Status: pilot tooling. Nothing here is evidence about real TTS quality yet.** Every stage
> runs end to end on mock espeak-ng audio with scripted "teacher" behaviour (`tp demo`).
> Real-lecture analysis is waiting on consented recordings. Components marked **UNVERIFIED**
> follow documented APIs but could not be run in the build sandbox. That sandbox had no network
> access to huggingface.co, YouTube or TTS APIs, and no GPU.

## What is where

| Stage | Module | Runs on CPU here | Notes |
|---|---|---|---|
| Ingest, clean, segment | `preprocess/ingest.py` | yes | Keeps the original audio; features are measured on lightly cleaned audio, never on enhancer output |
| ASR / alignment / diarization | `preprocess/backends.py` | captions + TextGrid readers, proportional FALLBACK aligner | faster-whisper, MFA, pyannote, ECAPA are **UNVERIFIED** here |
| Quality flags | `preprocess/quality.py` | yes | Low SNR, clipping, music-like, not-teacher, low alignment confidence |
| Session-level splits, leakage | `preprocess/split.py` | yes | The split unit is the session, never the utterance |
| Language spans, code-mixing index | `preprocess/lang.py` | yes | Heuristic Devanagari / romanised Hindi / English / maths tagger |
| F0 (Praat, pYIN), octave repair, normalisations | `features/f0.py` | yes | Semitones re median, z-log-F0, ERB, phrase-baseline; compared in `profile/confounds.py` |
| Energy, VAD, breaths | `features/energy.py` | yes | dB relative to the recording's speech level removes gain differences |
| Local rate | `features/rate.py` | yes | Syllable nuclei (no transcript) and alignment-based |
| Pauses by context | `features/pauses.py` | yes | micro, breath, phrase, sentence, post-question, rhetorical, pre-emphasis, thinking |
| Word prosody and prominence | `features/words.py` | yes | Per-channel z-scores (pitch / duration / energy / pause) |
| Pedagogical beats | `pedagogy/` | rules, agreement, clustering | LLM annotator (Claude API) **UNVERIFIED**: no key in the sandbox |
| Teacher profile | `profile/teacher_profile.py` | yes | Per-beat dynamics, pre-reveal, questions, transitions, emphasis channels, code-switch, confounds |
| Text normalisation | `director/text_norm.py` | yes | `F = ma` → "F equals M A", `9.8 m/s^2` → "nine point eight metre per second square" |
| Performance director | `director/director.py`, `plan.py` | yes | Rule-based v0 with a written reason for every emphasis and pause |
| Renderers | `director/render.py` | yes | Azure SSML, ElevenLabs requests with stitching context, inline tags, numeric word targets |
| TTS adapters | `synth/backends.py` | EspeakMock only | ElevenLabs and CommandTTS (open models) **UNVERIFIED** |
| PSOLA prosody post-edit | `synth/prosody_edit.py` | yes | Imposes plan duration, F0, energy and pauses on any candidate |
| Beat stitching | `synth/stitch.py` | yes | Planned pauses, equal-power fades, loudness levelling, join diagnostics |
| Candidate ranking and QC | `qc/rank.py` | yes (text and speaker checks need backends) | Hard gates, regeneration rules, review queue, weight calibration |
| Objective eval | `eval/objective.py` | yes | DTW only when texts match; distribution distances otherwise |
| Listening tests | `eval/listening_test.py` | yes | Blinded packaging; two-way cluster-bootstrap CIs; ICC; disagreements |
| Long-form checks | `eval/longform.py` | yes | Drift, cadence repetition, over-acting, dead air at 5/15/30/60 min |
| Dashboard | `dashboard/report.py` | yes | Self-contained HTML |

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock.txt && pip install -e .
# optional: apt-get install espeak-ng   (mock voices for tests/demo)
# optional GPU/network extras: pip install -e ".[asr,align,diarize,tts]"
```

## Commands

```bash
tp demo --out out/demo                    # full offline pipeline on MOCK audio (~20 s on 4 CPUs)
tp analyze --manifest manifest.yaml --out out/rajwant   # real recordings (see pipeline.py for the manifest format)
tp normalize "Force double kar diya toh a = F/m = 5 m/s^2 hoga?"
tp plan --script lesson.yaml --profile out/rajwant/profile.json --out out/plan
tp listening-package --stimuli stimuli.csv --out out/lt
tp listening-analyze --ratings ratings.csv --key out/lt/KEY_DO_NOT_SHARE.json
```

`tp demo` writes `analysis/dashboard.html`, `analysis/profile.json`, `plan.json`, `plan.ssml`,
`elevenlabs_requests.json`, `ranking.json`, `psola_edits.json`, `lecture_mock.wav`,
`stitch_report.json`, `listening_test/` and `demo_report.html`.

## Consent

`Recording.consent_ref` / `consent_scope` must name a signed record before a recording is used,
with scope `analysis`, `tts_training` or `voice_clone`. The pipeline warns when a recording has
no `analysis` consent. Voice-identity uses such as cloning, fine-tuning or conversion need their
own explicit scope.

See `docs/research_matrix.md` for the sourced model and tool comparison.
