# Stage C: listener-preferred voice conversion (29 Sep 2026)

Code: tag `stage-c-voice-conversion` (commit a9da714). Outputs are **synthetic** speech in the
teacher's voice. They were made under the teacher's confirmed consent for internal R&D voice conversion, and
are kept outside this repository together with a provenance file (`STAGE_C_PROVENANCE.json`, with checksums).

## Recipe
1. Source: ElevenLabs lecture clips (44.1 kHz MP3).
2. `tp restyle --audio <clips>.mp3 --target <teacher_style>.json --out restyled_44k.wav --sr 44100`
   (moves pace and pauses toward the teacher's measured delivery; strength 1.0).
3. Resample the restyled file to 16 kHz.
4. `tp convert --audio restyled_16k.wav --teacher-audio <teacher 30-min lecture>.wav --model-dir models/knnvc --consent-ref <ID> --out C.wav`
   (kNN-VC: WavLM-Large layer 6, prematched HiFi-GAN, k=4; 20 s chunks with 40 ms look-ahead, sample-aligned).
5. **No** `--transplant-intonation`: version D (with it) matched the teacher's register on paper
   but sounded worse to the listener.

## Measured (8.5 min)
| | Teacher | Source | Stage C |
|---|---|---|---|
| Median pitch | 213 Hz | 131 Hz | 183 Hz |
| Pitch span (5th–95th pct) | 12.6 st | 10.5 st | 11.3 st |
| Source melody kept (contour correlation) | – | 1.00 | 0.87 |
| Transcript difference vs source, median (noise floor 5.5%) | – | – | 9.2% |

## Listener feedback
- "C is much better" than D (the PSOLA intonation transplant).
- "Clarity in C still needs to be improved."
- "It is very good": preserve this stage.

Lesson: an objective register match did not predict preference. Calibrate every automatic
score against listeners before optimising it.
