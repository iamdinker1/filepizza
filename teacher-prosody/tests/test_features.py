import numpy as np

from teacher_prosody.features.energy import extract_energy, speech_mask
from teacher_prosody.features.f0 import (correct_octave_errors, extract_f0, final_contour, normalise, to_semitones)
from teacher_prosody.features.pauses import Pause, classify_pauses, pauses_from_mask
from teacher_prosody.features.rate import syllable_count, words_rate
from teacher_prosody.features.words import emphasis_alignment, fit_prominence_weights, top_prominent, word_prosody
from teacher_prosody.schema import Word
from teacher_prosody.testing import synth_voice


def test_f0_tracks_known_contour():
    f = np.concatenate([np.linspace(110, 190, 120), np.zeros(40), np.full(80, 150.0)])
    tr = extract_f0(synth_voice(f), floor=60, ceiling=400)
    est = np.interp(np.arange(len(f)) * 0.01, tr.times, np.nan_to_num(tr.hz))
    m = (f > 0) & (est > 0)
    assert m.sum() / (f > 0).sum() > 0.9
    assert np.median(np.abs(est[m] - f[m])) < 2.0
    assert np.all(est[125:155] == 0)  # the silent gap stays unvoiced


def test_octave_jump_is_folded_but_smooth_accent_is_kept():
    base = np.full(200, 120.0)
    base[60:90] = 60.0  # halving error, abrupt entry/exit
    accent = 120 * 2 ** (np.sin(np.linspace(0, np.pi, 40)) * 9 / 12)  # smooth +9 st focal accent
    base[130:170] = accent
    fixed, mask = correct_octave_errors(base)
    assert np.allclose(fixed[60:90], 120.0)
    assert mask[60:90].all()
    assert np.allclose(fixed[130:170], accent)  # not flattened


def test_normalisations():
    hz = np.array([100, 200, np.nan, 150.0])
    st = normalise(hz, "st_median", stats={"median_hz": 100.0})
    assert np.isclose(st[1], 12.0) and np.isnan(st[2])
    z = normalise(hz, "z_logf0")
    assert abs(np.nanmean(z)) < 1e-9
    assert np.isfinite(normalise(np.array([100, 110, 120, 130, 140, 150.0]), "st_baseline", times=np.arange(6) * 0.01)).all()
    assert np.isclose(to_semitones(np.array([200.0]), 100)[0], 12)


def test_final_contour_rise_and_step_up():
    t = np.arange(150) * 0.01
    rise = np.where(t < 1.1, 120.0, 120 * 2 ** ((t - 1.1) * 12 / 12))  # rising tail
    assert final_contour(rise, t, 1.5, 120.0, start=0)["final_type"] == "rise"
    step = np.where(t < 1.0, 120.0, 150.0)  # step up onto the last word (~ +3.9 st)
    assert final_contour(step, t, 1.5, 120.0, start=0)["final_type"] == "rise"
    fall = np.where(t < 1.1, 130.0, 130 * 2 ** (-(t - 1.1) * 12 / 12))
    assert final_contour(fall, t, 1.5, 120.0, start=0)["final_type"] == "fall"


def test_syllable_counts():
    assert syllable_count("velocity") == 4
    assert syllable_count("constant") == 2
    assert syllable_count("dekho") == 2
    assert syllable_count("acceleration") == 5
    assert syllable_count("देखो") == 2
    assert syllable_count("नहीं") == 2


def test_words_rate_excludes_pauses():
    ws = [Word("dekho", 0.0, 0.4), Word("yahan", 0.4, 0.8), Word("velocity", 1.8, 2.4)]
    r = words_rate(ws)
    assert r["pause_time_s"] == 1.0
    assert r["artic_rate_sps"] > r["speech_rate_sps"]


def test_pauses_detected_and_typed(not_constant):
    audio, words = not_constant
    f0 = extract_f0(audio)
    e = extract_energy(audio)
    ps = pauses_from_mask(speech_mask(e, f0.voiced), e.times)
    # the scripted 250 ms pre-emphasis pause before NOT is found within 40 ms
    assert any(abs(p.start - words[1].end) < 0.05 and abs(p.dur - 0.25) < 0.04 for p in ps)
    rows = word_prosody(words, f0, e, f0.median())
    typed = classify_pauses([Pause(words[1].end, words[2].start)], words, prominence={r.idx: r.prominence for r in rows})
    assert typed[0].kind == "pre_emphasis"


def test_not_is_most_prominent(not_constant):
    audio, words = not_constant
    f0 = extract_f0(audio)
    e = extract_energy(audio)
    rows = word_prosody(words, f0, e, f0.median())
    assert top_prominent(rows, 1)[0].w == "NOT"
    assert emphasis_alignment(rows, [2])["precision"] == 1.0
    w = fit_prominence_weights([rows], [[2]])
    assert abs(sum(w.values()) - 1) < 1e-6


def test_post_question_pause_type():
    ws = [Word("kya", 0, 0.3), Word("hoga?", 0.3, 0.8), Word("Double", 1.8, 2.3)]
    p = classify_pauses([Pause(0.8, 1.8)], ws, question_ends={1})
    assert p[0].kind == "post_question"
