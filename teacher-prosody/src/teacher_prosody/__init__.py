"""teacher_prosody: learn and reproduce how a teacher speaks while teaching.

Sub-packages
------------
preprocess  ingest, cleaning, VAD, ASR/alignment backends, language spans, quality flags, splits
features    F0, energy, local rate, pauses, durations, word-level prosody / prominence
pedagogy    beat taxonomy, rule/LLM annotators, agreement, unsupervised clustering
profile     dynamic teacher profile with session / recording-condition confound checks
director    text normalisation, performance plans, renderers (SSML, ElevenLabs, control curves)
synth       TTS backend interface, PSOLA prosody editing, beat stitching
qc          candidate ranking, hard gates, regeneration rules, human review queue
eval        objective diagnostics, blinded listening tests, long-form drift checks
dashboard   static HTML acoustic dashboard
"""

__version__ = "0.1.0"
