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
