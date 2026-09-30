import numpy as np
import pytest

from teacher_prosody.director.text_norm import normalise
from teacher_prosody.pedagogy.agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha_nominal, per_class_f1
from teacher_prosody.pedagogy.llm import build_request
from teacher_prosody.pedagogy.rules import annotate
from teacher_prosody.preprocess.lang import cmi, switch_points, tag_tokens, word_langs


@pytest.mark.parametrize("text,spoken", [
    ("F = ma", "F equals M A"),
    ("v = u + at", "V equals U plus A T"),
    ("s = ut + 1/2 at^2", "S equals U T plus half A T square"),
    ("g = 9.8 m/s^2 hota hai", "G equals nine point eight metre per second square hota hai"),
    ("c = 3 × 10^8 m/s", "C equals three into ten to the power eight metre per second"),
    ("a = dv/dt", "A equals d V by d T"),
    ("v0 = 20 m/s, θ = 30°", "V naught equals twenty metre per second, theta equals thirty degree"),
    ("Answer is -5 m/s", "Answer is minus five metre per second"),
    ("Take a ball of mass m.", "Take a ball of mass m."),
    ("Ab dekho, velocity yahan constant nahi hai.", "Ab dekho, velocity yahan constant nahi hai."),
])
def test_normalise(text, spoken):
    assert normalise(text).spoken == spoken


def test_formula_spans_are_reported():
    n = normalise("toh a = F/m = 5 m/s^2 hoga")
    assert [f.written for f in n.formulas] == ["a = F/m = 5 m/s^2"]


def test_lexicon_override():
    assert normalise("F = ma", lexicon={"=": "equal to"}).spoken == "F equal to M A"


def test_language_tags_and_cmi():
    tags = [l for _, _, _, l in tag_tokens("Ab dekho, velocity yahan constant nahi hai.") if l != "other"]
    assert tags == ["hi", "hi", "en", "hi", "en", "hi", "hi"]
    assert word_langs(["अब", "देखो,", "velocity"]) == ["hi", "hi", "en"]
    assert cmi("So the answer is five.") == 0
    assert cmi("Force double kar diya toh acceleration ka kya hoga?") > 20
    assert switch_points("Force double kar diya") == [2]


def test_rule_annotator():
    labs = annotate(["Toh acceleration ka kya hoga?", "Acceleration bhi double ho jaayega.",
                     "Yeh galti mat karna.", "Chalo, next topic dekhte hain.", "a = F/m"])
    assert [l.beat for l in labs] == ["rhetorical_question", "reveal", "common_mistake", "transition", "formula"]


def test_agreement_stats():
    a = ["x", "x", "y", "y", "z", "z"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ["x", "y", "z", "x", "y", "z"]) < 0.2
    assert fleiss_kappa([["x", "x", "x"], ["y", "y", "y"], ["x", "x", "y"]]) > 0.3
    assert krippendorff_alpha_nominal([["x", "x"], ["y", "y"], ["x", None]]) == 1.0
    f1 = per_class_f1(["x", "y"], ["x", "x"])
    assert f1["x"]["precision"] == 0.5 and f1["y"]["recall"] == 0.0


def test_llm_request_shape():
    req = build_request(["Toh kya hoga?", "Double ho jaayega."], "newton's laws")
    assert req["model"] == "claude-opus-5-5"
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "reveal" in fmt["schema"]["properties"]["labels"]["items"]["properties"]["beat"]["enum"]


def test_devanagari_loanwords_and_transliteration():
    from teacher_prosody.preprocess.lang import deva_to_latin, english_in_devanagari, is_question_text

    for w in ("वोल्टेज", "डायरेक्शन", "एक्सेलरेशन", "फिजिक्स", "स्केलर", "वेक्टर"):
        assert english_in_devanagari(w), w
    for w in ("देखो", "क्या", "होगा", "समझते", "बच्चों", "मतलब", "मात्रा"):
        assert english_in_devanagari(w) is None, w
    assert deva_to_latin("समझते") == "samajhte" and deva_to_latin("कहते") == "kahte"
    assert is_question_text("mass की कोई direction है क्या") and not is_question_text("direction तो होती है")
    labs = annotate(["तो acceleration का क्या होगा", "ध्यान रखना यहाँ पर students गलती करते हैं"])
    assert [l.beat for l in labs] == ["rhetorical_question", "common_mistake"]


def test_bracketed_formula_and_prose_brackets():
    assert normalise("R = √(A^2 + B^2 + 2AB cos θ)").spoken == "R equals root A square plus B square plus two A B cos theta"
    assert normalise("Ab dekho (yeh important hai)").spoken == "Ab dekho yeh important hai"


def test_sherpa_whisper_pads_short_windows_to_lift_token_cap():
    from teacher_prosody.preprocess.backends import SherpaWhisperASR

    seen = {}

    class FakeStream:
        def accept_waveform(self, sr, samples):
            seen["n"] = len(samples)

        result = type("R", (), {"text": "{e0a4b9}{e0a482}"})()  # "हं" as hex-token bytes

    class FakeRec:
        def create_stream(self):
            return FakeStream()

        def decode_stream(self, s):
            pass

    asr = SherpaWhisperASR.__new__(SherpaWhisperASR)
    asr.rec = FakeRec()
    assert asr.decode(np.zeros(16000 * 3, np.float32)) == "हं"
    assert seen["n"] == SherpaWhisperASR.PAD_TO  # 3 s of audio decoded as a full 30 s window
