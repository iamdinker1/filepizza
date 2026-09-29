import xml.etree.ElementTree as ET

import numpy as np

from teacher_prosody.director.director import plan_lesson
from teacher_prosody.director.render import to_elevenlabs, to_ssml, to_word_targets
from teacher_prosody.features.f0 import extract_f0
from teacher_prosody.qc.rank import calibrate_weights, cer, pause_f1, rank, regeneration_action
from teacher_prosody.schema import Word
from teacher_prosody.synth.stitch import stitch
from teacher_prosody.testing import synth_voice
from tests.conftest import needs_espeak

LESSON = [{"beat": "build_up", "text": "Ab dekho, mass constant rakha hai aur force ko double kar diya."},
          {"beat": "rhetorical_question", "text": "Toh acceleration ka kya hoga?"},
          {"beat": "reveal", "text": "Acceleration bhi double ho jaayega, kyunki a = F/m."},
          {"beat": "common_mistake", "text": "Velocity is NOT constant here, yeh galti mat karna."}]
CM = {"key_terms": ["force", "acceleration", "mass", "velocity"], "changing": ["force"]}


def test_director_emphasis_budget_and_reasons():
    p = plan_lesson("t", "teacher", LESSON, CM)
    for b in p.beats:
        assert sum(w.emphasis >= 3 for w in b.words) <= 1
        assert len(b.emphasised) <= 3
        assert all(w.reason for w in b.emphasised)
    build = [w.w.strip(",.").lower() for w in p.beats[0].emphasised]
    assert "force" in build and "double" in build
    mistake = {w.w: w.emphasis for w in p.beats[3].words}
    assert mistake["NOT"] == 3
    assert p.beats[2].pause_before_s >= 0.6  # reveal after a question
    assert p.beats[2].formulas and p.beats[2].must_review
    # given information: acceleration already introduced in beat 1 is not re-emphasised in the reveal
    assert "Acceleration" not in [w.w for w in p.beats[2].emphasised]


def test_renderers():
    p = plan_lesson("t", "teacher", LESSON, CM)
    ET.fromstring(to_ssml(p))  # well-formed XML
    reqs = to_elevenlabs(p, "VOICE")
    assert reqs[1]["body"]["previous_text"] == LESSON[0]["text"]
    assert reqs[1]["body"]["next_text"] == LESSON[2]["text"]
    tg = to_word_targets(p.beats[0])
    assert len(tg) == len(p.beats[0].words) and all(t["dur_scale"] > 0 for t in tg)


def test_cer_and_pause_f1():
    assert cer("Force double kar diya", "force double kar diya") == 0
    ws = [Word("a", 0, 0.3), Word("b", 0.3, 0.6), Word("c", 0.9, 1.2)]
    assert pause_f1([2], ws) == 1.0
    assert pause_f1([1], ws) == 0.0


def test_calibrate_weights_recovers_preference():
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(200):
        a = {"emphasis": rng.random(), "pace": rng.random()}
        b = {"emphasis": rng.random(), "pace": rng.random()}
        pairs.append((a, b, int(a["emphasis"] > b["emphasis"])))  # humans only care about emphasis
    w = calibrate_weights(pairs, ["emphasis", "pace"])
    assert w["emphasis"] > 5 * max(w["pace"], 1e-6)


def test_stitch_inserts_planned_pauses():
    a = synth_voice(np.full(100, 140.0))
    b = synth_voice(np.full(100, 150.0))
    lec, rep = stitch([a, b], [0.5])
    assert abs(lec.duration - (a.duration + b.duration + 0.5)) < 0.1
    assert rep.joins[0]["pause_s"] == 0.5 and abs(rep.joins[0]["f0_jump_st"] - 12 * np.log2(150 / 140)) < 0.5


@needs_espeak
def test_rank_prefers_plan_following_and_psola_helps():
    from teacher_prosody.synth.backends import EspeakMock
    from teacher_prosody.synth.prosody_edit import apply_word_targets

    p = plan_lesson("t", "teacher", LESSON, CM)
    bp = p.beats[3]
    cands = EspeakMock().generate(bp, n=4, follow_plan=[True, True, False, False], seed=3)
    teacher = {"ref_hz": None, "artic_rate_sps": 5.0}
    ranked = rank(cands, bp, teacher)
    follow = {c.k for c in cands if c.meta["follow_plan"]}
    assert ranked[0].k in follow
    assert all(r.decision == "review" for r in ranked)  # no ASR backend -> text unverified -> human review
    flat = next(c for c in cands if not c.meta["follow_plan"])
    before = next(r for r in ranked if r.k == flat.k).sub["emphasis"]
    ea, ew = apply_word_targets(flat.audio, flat.words, to_word_targets(bp))
    flat.audio, flat.words = ea, ew
    after = rank([flat], bp, teacher)[0].sub["emphasis"]
    assert after > before
    assert regeneration_action(ranked, 0)["action"] in ("use_after_review", "regenerate")


@needs_espeak
def test_psola_changes_duration_and_pitch():
    from teacher_prosody.synth.prosody_edit import apply_word_targets
    from teacher_prosody.testing import WordSpec, espeak_utterance

    a, ws = espeak_utterance([WordSpec("one"), WordSpec("two"), WordSpec("three")], seed=0)
    tg = [{"dur_scale": 1.0}, {"dur_scale": 1.3, "pitch_st": 4.0, "emphasis": 2}, {"dur_scale": 1.0}]
    ea, ew = apply_word_targets(a, ws, tg)
    assert abs((ew[1].end - ew[1].start) / (ws[1].end - ws[1].start) - 1.3) < 0.05
    f_before = extract_f0(a).at(ws[1].start, ws[1].end)
    f_after = extract_f0(ea).at(ew[1].start, ew[1].end)
    gain = 12 * np.log2(np.nanpercentile(f_after, 90) / np.nanpercentile(f_before, 90))
    assert gain > 2.0


def test_reveal_answer_and_measured_accent_sizes():
    lesson = [{"beat": "rhetorical_question", "text": "Resultant kitna hoga?"},
              {"beat": "reveal", "text": "20 N. Seedha addition."}]
    prof = {"accents": {"size_st": {"n": 500, "q25": 3.9, "median": 4.8, "p90": 7.4}, "share_with_pre_pause": 0.17,
                        "share_with_energy_peak": 0.9}}
    p = plan_lesson("t", "teacher", lesson, {"key_terms": ["resultant"]}, prof)
    ans = p.beats[1].emphasised
    assert ans and ans[0].w == "twenty" and ans[0].pitch_st == 4.8


@needs_espeak
def test_restyle_moves_pace_and_pauses_toward_target():
    from teacher_prosody.synth.restyle import StyleStats, evaluate_restyle, measure_style, restyle
    from teacher_prosody.testing import WordSpec, espeak_utterance

    spec = []
    for k in range(6):  # slow speaker with long pauses (MOCK)
        spec += [WordSpec("force", speed=130), WordSpec("double", speed=130),
                 WordSpec("hoga", speed=130, pitch=45, pause_after=0.9 if k % 2 else 0.6)]
    src, _ = espeak_utterance(spec, seed=2)
    s = measure_style(src)
    target = StyleStats(duration_s=60, median_f0_hz=s.median_f0_hz, artic_rate_sps=s.artic_rate_sps * 1.15,
                        pauses_s=[0.25, 0.3, 0.35, 0.4, 0.3, 0.28, 0.33], accent_sizes_st=s.accent_sizes_st or [4.0],
                        accents_per_speech_min=s.accents_per_speech_min, rise_share=0.5, unit_f0_range_st=s.unit_f0_range_st,
                        n_units=10, rise_delta_st=4.0)
    out, ed = restyle(src, target)
    ev = evaluate_restyle(out, ed, target)
    assert out.duration < src.duration
    assert ev["pause_median_s"]["after"] < ev["pause_median_s"]["before"]
    assert ed["rate_duration_factor"] < 1.0
    rt = StyleStats.from_json(target.to_json())
    assert rt.artic_rate_sps == round(target.artic_rate_sps, 2)


def test_transplant_intonation_moves_register_and_keeps_shape():
    from teacher_prosody.synth.voice_convert import transplant_intonation
    from teacher_prosody.testing import synth_voice

    contour = np.concatenate([np.linspace(110, 160, 80), np.linspace(160, 105, 80)])
    src = synth_voice(contour)                      # source performance at ~130 Hz
    conv = synth_voice(np.full(len(contour), 200.0))  # "converted" audio, flat at 200 Hz
    out = transplant_intonation(conv, src, target_median_hz=200.0)
    f = extract_f0(out, floor=60, ceiling=500)
    v = f.hz[f.voiced]
    assert 170 < np.median(v) < 230                  # teacher register
    assert np.ptp(12 * np.log2(v / np.median(v))) > 5  # source movement (~7 st) carried over


def test_voice_convert_output_stays_sample_aligned():
    torch = __import__("pytest").importorskip("torch")
    from teacher_prosody.audio import Audio
    from teacher_prosody.synth.voice_convert import convert

    class FakeKNN:  # mimics WavLM framing (drops a partial frame) and a 320-sample-hop vocoder
        def get_features(self, x, vad_trigger_level=0):
            n = (x.shape[-1] - 400) // 320 + 1
            return torch.zeros(n, 4)

        def match(self, q, matching_set, topk=4, tgt_loudness_db=None):
            t = torch.arange(q.shape[0] * 320, dtype=torch.float32)
            return 0.1 * torch.sin(2 * np.pi * 150 * t / 16000)

    src = Audio(np.random.default_rng(0).normal(0, 0.1, 16000 * 47 + 123).astype(np.float32), 16000)
    out = convert(FakeKNN(), src, None, chunk_s=20.0)
    assert len(out.y) == len(src.y)


def test_humanize_adds_teacher_dynamics_and_keeps_rate():
    from teacher_prosody.synth.humanize import HumanizeParams, humanize, measure_dynamics
    from teacher_prosody.testing import WordSpec, espeak_utterance

    words = "acceleration is the rate of change of velocity with time and it is a vector".split()
    spec = []
    for k in range(5):  # evenly paced, flat-ish "TTS" phrases with near-identical pauses (MOCK)
        spec += [WordSpec(w, speed=165, pause_after=0.0) for w in (words[:8] if k % 2 else words[6:])]
        spec[-1] = WordSpec(spec[-1].text, speed=165, pause_after=0.5 if k % 2 else 0.35)
    src, _ = espeak_utterance(spec, seed=3)
    before = measure_dynamics(src)
    out, log = humanize(src, [0.25, 0.3, 0.5, 0.6, 0.8, 1.0, 1.4], HumanizeParams(strength=1.0))
    after = measure_dynamics(out)
    assert out.sr == src.sr and log["phrases"] == 5
    assert 0.9 < out.duration / src.duration < 1.25       # a layer, not a slow-down
    assert after["final_slowing"] > before["final_slowing"] + 0.2
    assert after["syllable_timing_cv"] > before["syllable_timing_cv"]
    assert after["pitch_range_st"] > before["pitch_range_st"] + 0.5
    assert after["pause_cv"] > before["pause_cv"] + 0.1  # pauses take on the teacher's variety
    fast, log = humanize(src, None, HumanizeParams(strength=1.0), teacher_rate=before["artic_rate_sps"] * 1.12)
    assert log["speech_duration_factor"] < 0.92            # speech sped up to the teacher's pace ...
    assert fast.duration < out.duration                    # ... on top of the same dynamics
