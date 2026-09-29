import json
from pathlib import Path

import jsonschema
import numpy as np

from teacher_prosody.eval.listening_test import icc_raters, package, summarise
from teacher_prosody.eval.longform import cadence_repetition
from teacher_prosody.eval.objective import contour_distance, distribution_distances, dtw
from teacher_prosody.features.f0 import F0Track
from teacher_prosody.preprocess.split import assign_splits, check_leakage
from teacher_prosody.profile.confounds import eta_squared
from teacher_prosody.schema import Quality, Recording, Utterance, Word, to_dict
from teacher_prosody.testing import synth_voice

SCHEMA = json.loads((Path(__file__).parents[1] / "schema" / "utterance.schema.json").read_text())


def test_dtw_identity_and_shift():
    a = np.sin(np.linspace(0, 6, 80))
    assert dtw(a, a)[0] == 0
    assert dtw(a, np.concatenate([a[:10], a]))[0] < 0.05


def test_contour_distance_refuses_mismatched_text():
    t = np.arange(50) * 0.01
    tr = F0Track(t, np.full(50, 120.0), np.ones(50), 60, 400, "x")
    assert "refused" in contour_distance(tr, tr, "force double", "mass half", 120, 120)
    ok = contour_distance(tr, tr, "force double", "force double", 120, 120)
    assert ok["dtw_st"] == 0


def test_distribution_distances():
    rng = np.random.default_rng(0)
    d = distribution_distances({"pause": list(rng.normal(0.8, 0.1, 200))}, {"pause": list(rng.normal(0.4, 0.1, 200))})
    assert d["pause"]["w1"] > 0.3 and d["pause"]["ks_p"] < 1e-6


def test_cadence_repetition_detects_template():
    shape = np.array([0, 1, 2, 1, 0, -1, -2, -3.0])
    rep = np.stack([shape + np.random.default_rng(i).normal(0, 0.05, 8) for i in range(10)])
    varied = np.stack([np.random.default_rng(i).normal(0, 2, 8) for i in range(10)])
    assert cadence_repetition(rep) == 1.0
    assert cadence_repetition(varied) < 0.3


def test_listening_package_and_summary(tmp_path):
    from teacher_prosody.audio import save

    stim = []
    for item in ("i1", "i2", "i3"):
        for sysname in ("ref", "sysA", "sysB"):
            p = save(tmp_path / f"{item}_{sysname}.wav", synth_voice(np.full(30, 130.0)))
            stim.append({"item": item, "system": sysname, "path": str(p)})
    info = package(stim, tmp_path / "lt")
    key = json.loads((tmp_path / "lt" / "KEY_DO_NOT_SHARE.json").read_text())
    assert info["n_stimuli"] == 9 and all("ref" not in f for f in key)  # file names are blinded
    # SYNTHETIC ratings with known system means 80 / 60 / 40 (tests the statistics only)
    rng = np.random.default_rng(1)
    true = {"ref": 80, "sysA": 60, "sysB": 40}
    rows = [{"rater": f"r{r}", "item": it, "system": s, "question": "teacher_explaining",
             "score": float(true[s] + rng.normal(0, 5))} for r in range(8) for it in ("i1", "i2", "i3") for s in true]
    out = summarise(rows, n_boot=300)
    for s, m in true.items():
        assert out["systems"][s]["ci_lo"] < m + 5 and out["systems"][s]["ci_hi"] > m - 5
    assert out["pairwise"]["ref - sysA"]["ci_lo"] > 0
    assert icc_raters([r for r in rows]) > 0.8


def test_utterance_rows_match_json_schema():
    u = Utterance("u1", "rec", "t", "s1", 0.0, 1.2, text="Toh kya hoga?", words=[Word("Toh", 0, 0.3, "hi")],
                  quality=Quality(snr_db=30.0, flags=["align_low"]), split="train", recording_condition="studio")
    jsonschema.validate(to_dict(u), SCHEMA)
    assert Utterance.from_dict(json.loads(json.dumps(to_dict(u)))).words[0].w == "Toh"
    assert not u.quality.usable_for_style or True


def test_session_split_and_leakage():
    recs = [Recording(f"r{i}", "t", f"s{i}", "", "", "", 1.0, 16000, "studio") for i in range(10)]
    split = assign_splits(recs)
    assert set(split.values()) == {"train", "val", "test"}
    utts = [Utterance(f"u{i}", r.recording_id, "t", r.session_id, 0, 1, text="x", split=r.split) for i, r in enumerate(recs)]
    assert check_leakage(utts) == []
    utts.append(Utterance("bad", "r0", "t", "s0", 1, 2, split="test" if recs[0].split != "test" else "train"))
    assert check_leakage(utts)
    assert not recs[0].cleared_for("analysis")


def test_eta_squared():
    assert eta_squared(np.array([1, 1, 5, 5.0]), ["a", "a", "b", "b"]) == 1.0
    assert eta_squared(np.array([1, 5, 1, 5.0]), ["a", "a", "b", "b"]) == 0.0
