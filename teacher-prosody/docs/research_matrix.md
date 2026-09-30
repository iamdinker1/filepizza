# Research comparison and decision matrix (checked 2026-09-29)

**Evidence labels:** [D] official doc/README/LICENSE/model card · [D*] official page seen only via a search
snippet (the cloud sandbox's proxy blocked huggingface.co, arxiv.org and most vendor sites) · [P] paper or
vendor-reported benchmark · [C] community/third-party report · [U] unverified / hypothesis.
**Re-verify every price, licence and access rule on the live page before a purchase or legal decision.**
Nothing here was measured on PW audio yet; the pilot (see `README.md`) is what turns these into evidence.

## 1. Decision matrix (what to prototype)

| Role | Candidate | Why it is in | What would knock it out | Status |
|---|---|---|---|---|
| **Fastest credible baseline** | ElevenLabs PVC (Eleven v4 or Multilingual v2) of the consented teacher, careful text normalisation, request stitching (`previous_text`/`next_text`/`previous_request_ids`) | Most mature cloning + Hindi listed + stitching + timestamps endpoint + a live-latency tier (v4 Turbo / Flash) [D] | Accent flips at Hinglish switches (v4 speaks target-language native accent [D]); tag over-acting; v4 released 2026-09-28, no track record | To test |
| **Prototype A: explicit prosody plan → controllable TTS** | FastSpeech2/FastPitch Hindi+English (IITM `Fastspeech2_HS`, CC-BY-4.0; AI4Bharat `en+hi` FastPitch) fine-tuned on the teacher, driven by this repo's performance plan (per-word duration / F0 / energy / pause) | Only family with *documented* per-phoneme F0/duration/energy control, Hindi checkpoints and permissive licences [D]; emphasis-by-lengthening shown to work (+40% emphasis ID) [P] | Naturalness ceiling below LLM-TTS; Hinglish G2P; vocoder fine-tune needed | To test |
| **Prototype B: performance first → voice conversion** | Source performance (ElevenLabs/Sarvam/human guide read, or Prototype A) → any-to-one RVC/Applio (MIT) trained on 30–60 min of the teacher, optionally with the plan's F0 curve; MeanVC2 (Apache) as zero-shot baseline | Separates prosody from timbre; frame-level F0 control; swap in the best Hindi generator later [D/U] | No open VC trained on Hindi (accent leakage, retroflex loss) [U]; artifacts on long stretches; timing must survive conversion | To test |
| Open LLM-TTS challenger | VoxCPM2 (Apache-2.0, Hindi official, official LoRA 5–10 min) + LoRA on teacher; MOSS-TTS v1.5 (Apache, Hindi, `[pause X.Ys]`, duration tokens) | Permissive + Hindi + fine-tuning + some inline control [D] | Hindi WER 19.7 on MiniMax set for VoxCPM2 [P]; MOSS Hindi unbenchmarked [U] | Stage-2 |
| India-specialist API | Sarvam Bulbul v3/v4 (Hinglish-native, cloning 30–60 s + consent) | Best-documented Hinglish/numeric/STEM CER (vendor study) [P] | 2,500 chars/request; limited prosody controls; v4 gated [D] | To test in baseline bake-off |
| Max SSML control | Azure Professional Voice (HD) | Only API with pitch **contour** + rate + break SSML [D] | Limited-access approval, 300+ studio utterances, `lang` can't combine with prosody/break (bad for code-switch), emphasis only 3 en-US voices [D] | Only if A fails |

**Excluded for commercial use** (non-commercial or no-Hindi, keep as design references): XTTS-v2 (CPML, no
commercial licence obtainable [D]), F5-TTS base weights (CC-BY-NC, Emilia [D]), Fish S2 Pro / OpenAudio
(research licence [D]), Higgs Audio v3 (non-commercial [D]), Spark-TTS, Llasa, MaskGCT (NC [D*]),
VibeVoice (TTS code withdrawn [D]), CSM, Dia (no Hindi [D]), IndexTTS-2.5 (no Hindi; bilibili licence
revenue clause [D]), ProsodyFM/ProsodyLM/Vevo (NC, no Hindi [D]), Seed-VC (GPL-3.0, archived [D]).
**Licence needing legal review:** IndicF5 (HF card MIT but no LICENSE in repo and F5 base was NC [D*/U]),
Veena / svara-TTS (Llama-derived weights [U]), Gemini voice replication (not available in India via AI Studio [D*]).

## 2. Commercial APIs

| Provider / model (release) | Cloning & data | Hindi / Hinglish | Prosody controls | Live tier | ≈ cost / lecture-hour (55k chars) |
|---|---|---|---|---|---|
| ElevenLabs Eleven v4 (2026-09-28) | IVC + PVC (PVC ≥30 min, 2–3 h ideal) [D] | Hindi in 90+ langs [D] | audio tags, `[pause]`, stability/similarity; no SSML / speed [D] | v4 Turbo ~150 ms first speech [vendor] | $4.40 at $0.08/1k chars [D]; launch discount to 12 Oct [C] |
| ElevenLabs Multilingual v2 / v3 | IVC + PVC (PVC not optimised for v3) [D] | Hindi [D] | v2: `<break>` ≤3 s, speed 0.7–1.2, style; v3: tags, no break/speed [D] | Flash v2.5 ~75 ms [vendor] | $4.40 |
| OpenAI gpt-4o-mini-tts (2025-12-15) | custom voices gated, consent recording [D] | Hindi listed; quality unverified | free-text `instructions` only | gpt-realtime-2 / GPT-Live-1 | ~$0.90 [D] |
| Gemini 3.8 Flash TTS (GA 2026-09-23) | voice replication 30 s + consent [D] — **not available in India via AI Studio** [D*] | 130 langs [D] | style prompts + inline tags; pause lengths model-chosen [D] | Gemini Live | ~$0.81 (doubles 2027-01-01) [C] |
| Google Chirp 3 HD / Instant Custom Voice | ICV allow-listed [D] | hi-IN voices [D] | pace, `[pause]`, IPA [D] | streaming | $1.65 / $3.30 [C] |
| Azure Professional Voice / Personal Voice | 300+ utterances + talent consent, Limited Access [D] | hi-IN, bilingual Aarti/Arjun [vendor] | full SSML prosody pitch/contour/range/rate/volume, break [D] | Voice Live | ~$1.32–2.64 + hosting/training [C] |
| Sarvam Bulbul v3 / v4 (v4 2026-07-30) | 30–60 s clone + consent checkbox [D] | India-first, native Hinglish [vendor] | v3 pace 0.5–2.0; v4 pitch/loudness/emotion (gated) [D/C] | streaming <250 ms [C] | ~₹165 (~$1.9) [D] |
| Cartesia Sonic-3.6 (2026-08-18) | IVC ~10 s, PVC 30 min–3 h [D] | Hinglish in Devanagari/Latin/mixed [D] | speed, volume, `<break>` (splits generation), emotion EN only [D]; PVC ignores speed/volume [D] | <90 ms [vendor] | ~$2.06 [D] |

## 3. Open-weight TTS (commercial-relevant subset)

| Model (latest) | Licence | Hindi | Explicit controls | Fine-tuning | Notes / failure modes |
|---|---|---|---|---|---|
| VoxCPM2 (2026-04) | Apache-2.0 code+weights [D] | official (30 langs) [D] | "(style)" prefix on cloned voice | official full + LoRA, 5–10 min data [D] | Hindi SIM 85.6 best of 5 but WER 19.7 worst [P]; run-to-run variance, 1–3 retries advised [D] |
| MOSS-TTS v1.5 (2026-05/06) | Apache-2.0 [D] | added in v1.5, code-switching claimed; no Hindi metrics [D/U] | duration tokens, inline `[pause X.Ys]`, IPA [D] | official full fine-tune [D] | 8B / 4B; claims 1-hour stable generation [D] |
| Chatterbox Multilingual V3 (2026-06) | MIT, PerTh watermark [D] | Hindi + Hindi fine-tune pack [D] | `exaggeration`, `cfg_weight` [D] | no official scripts | ~40 s/call; reference-accent carry-over (set `cfg_weight=0`) [D] |
| Qwen3-TTS 1.7B (2026-01) | Apache-2.0 [D*] | not listed [D] | instructions only on non-clone models [D] | official single-speaker FT [D] | Hindi only via fine-tune; accent issues reported [C] |
| Fun-CosyVoice3-0.5B (2025-12) | Apache-2.0 [D] | not listed | instruction speed/volume/emotion [D] | scripts [D] | would need continued pre-training for Hindi [U] |
| Zonos v0.1 / ZONOS2 (2026-06) | Apache / MIT [D] | ZONOS2 Hindi tier 3 [D] | rate buckets, pitch variation, emotion sliders [D] | – | clones from ECAPA embedding → timbre not prosody [U] |
| IITM Fastspeech2_HS | CC-BY-4.0 [D] | Hindi (16 langs) [D] | `<alpha>` local rate, `<sil=500ms>`; FS2 pitch/energy/duration predictors [D] | ESPnet recipe [D] | best permissive explicit-control Hindi base found |
| AI4Bharat Indic-TTS FastPitch | code MIT; ckpt licence not stated [D] | `hi`, `en+hi` [D] | per-symbol F0 & duration (code work to override) [D/U] | Coqui recipes | 2023-era quality |
| Indic Parler-TTS | Apache-2.0 [D*] | 20 Indic + EN [D*] | utterance-level description (pitch/pace/expressivity) [D*] | parler-tts + dataspeech | no zero-shot cloning; emotion not tested for Hindi [D*] |
| IndicF5 | **unclear** [D*/U] | Hindi + 10 Indic [D] | implicit (reference) only | F5 recipes | numerals → gibberish, roman script truncated [C] |

## 4. Tooling for the analysis pipeline

| Stage | Pilot choice (to validate) | Licence | Notes |
|---|---|---|---|
| ASR | faster-whisper large-v3/turbo + IndicConformer (Devanagari); optional Sarvam Saaras `codemix` | MIT / MIT [D/C] | Whisper Hindi WER claims vary 17–37% [C]; hand-correct 30–60 min |
| Forced alignment | **MFA** with merged Hindi+English lexicon, acoustic model adapted on the teacher's code-mixed speech | MIT [D] | code-mixed MFA models: 4.15 ms mean boundary error vs ~38 ms monolingual [P, arXiv 2607.25581]; MMS/ctc-forced-aligner only as NC cross-check [D] |
| Diarization / student rejection | pyannote community-1 + teacher-enrolment embedding cosine threshold | MIT code, CC-BY-4.0 gated pipeline [D] | measure F0/energy on un-enhanced audio [U] |
| Music / applause | inaSpeechSegmenter, YAMNet / PANNs | MIT / Apache [D] | |
| F0 | SwiftF0 or RMVPE primary, Praat cross-check (this repo) | MIT / Apache / GPL [D] | benchmark by SwiftF0's author [D]; report octave-error rate |
| VC | RVC/Applio any-to-one (MIT), MeanVC2 (Apache) | [D] | Seed-VC GPL + archived; Vevo2 NC |
| Eval | DNSMOS, WavLM-SV SIM, CER after script normalisation, UTMOSv2 (EN/JA-trained), P.808 ACR/CCR with Hindi raters | MIT / CC-BY [D] | EmphAssess/AutoPCP are English or NC; no standard "teacher-likeness" metric exists [U] |
| Prosody tokens | derive own tokens from quantised F0/energy/duration (commercial-safe) | – | FACodec / Vevo2 tokenizers for research only |

## Sources
Commercial: elevenlabs.io/docs/overview/models · elevenlabs.io/blog/eleven-v4 · elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4 ·
elevenlabs.io/pricing/api · elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/request-stitching ·
elevenlabs.io/docs/eleven-creative/voices/voice-cloning/professional-voice-cloning ·
elevenlabs.io/docs/help-center/product/voices/voice-cloning/can-i-create-a-professional-voice-clone-of-someone-elses-voice ·
elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps · elevenlabs.io/docs/overview/capabilities/voice-changer ·
elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/pronunciation-dictionaries ·
developers.openai.com/api/docs/guides/text-to-speech · developers.openai.com/api/docs/guides/custom-voices · developers.openai.com/api/docs/pricing ·
ai.google.dev/gemini-api/docs/voice-replication · ai.google.dev/gemini-api/docs/speech-generation · docs.cloud.google.com/text-to-speech/docs/chirp3-hd ·
learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice · learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-create-consent ·
www.sarvam.ai/blogs/bulbul-v3 · docs.sarvam.ai/api/getting-started/models/bulbul · www.sarvam.ai/api-pricing · docs.sarvam.ai/creative-voice-cloning ·
docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags · docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices-pro · www.cartesia.ai/pricing ·
artificialanalysis.ai/text-to-speech/leaderboard
Open models: github.com/OpenBMB/VoxCPM · github.com/OpenMOSS/MOSS-TTS · github.com/resemble-ai/chatterbox · github.com/QwenLM/Qwen3-TTS ·
github.com/FunAudioLLM/CosyVoice · github.com/fishaudio/fish-speech · github.com/index-tts/index-tts · github.com/boson-ai/higgs-audio ·
github.com/microsoft/VibeVoice · github.com/Zyphra/Zonos · github.com/Zyphra/ZONOS2 · github.com/k2-fsa/OmniVoice · github.com/SWivid/F5-TTS ·
github.com/AI4Bharat/IndicF5 · huggingface.co/ai4bharat/indic-parler-tts · github.com/AI4Bharat/Indic-TTS · github.com/smtiitm/Fastspeech2_HS ·
github.com/idiap/coqui-ai-TTS · github.com/coqui-ai/TTS/discussions/4304 · github.com/yl4579/StyleTTS2 · github.com/hexgrad/kokoro
Prosody control research: github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch · arxiv.org/abs/2408.06827 (PRESENT) ·
github.com/ubisoft/ubisoft-laforge-daft-exprt · github.com/XianghengHee/ProsodyFM · github.com/auspicious3000/ProsodyLM · github.com/jishengpeng/ControlSpeech ·
github.com/open-mmlab/Amphion (FACodec, Vevo2) · arxiv.org/abs/2307.07062 (zero-data word emphasis) · github.com/facebookresearch/emphassess ·
github.com/hi-paris/Prosody-Control-French-TTS (LLM→SSML planner) · arxiv.org/abs/2609.01016 (code-switched accent guidance)
Tooling: github.com/MontrealCorpusTools/Montreal-Forced-Aligner · arxiv.org/abs/2607.25581 · github.com/MahmoudAshraf97/ctc-forced-aligner · github.com/m-bain/whisperX ·
github.com/SYSTRAN/faster-whisper · github.com/AI4Bharat/vistaar · github.com/pyannote/pyannote-audio · github.com/ina-foss/inaSpeechSegmenter ·
github.com/lars76/swift-f0 · github.com/lars76/pitch-benchmark · github.com/Dream-High/RMVPE · github.com/YannickJadoul/Parselmouth ·
github.com/IAHispano/Applio · github.com/ASLP-lab/MeanVC2 · github.com/Plachtaa/seed-vc · github.com/BytedanceSpeech/seed-tts-eval ·
github.com/sarulab-speech/UTMOSv2 · github.com/microsoft/DNS-Challenge · github.com/microsoft/P.808 · github.com/facebookresearch/stopes (AutoPCP)
