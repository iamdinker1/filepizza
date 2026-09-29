"""End-to-end: the analysis must recover behaviours scripted into the MOCK teacher corpus."""
import json

import pytest

from tests.conftest import needs_espeak


@needs_espeak
@pytest.mark.slow
def test_demo_recovers_scripted_behaviour(tmp_path):
    from teacher_prosody.demo import run

    s = run(tmp_path / "demo")
    prof = json.loads((tmp_path / "demo" / "analysis" / "profile.json").read_text())
    assert s["analysis"]["annotator_agreement"]["rule_vs_gold_kappa"] > 0.6
    pb = prof["per_beat"]
    assert pb["rhetorical_question"]["final_types"].get("rise", 0) == 3          # scripted rising questions
    assert 0.85 < prof["questions"]["post_question_pause_s"]["median"] < 1.2      # scripted 0.95 s gap
    assert prof["transitions"]["energy_reset_db"]["median"] > 0.5                # scripted louder transition onset
    assert pb["recap"]["rate_rel"]["median"] > pb["reveal"]["rate_rel"]["median"]  # recaps faster than reveals
    conf = prof["confounds"]
    # per-recording normalisation removes the scripted session gain differences
    assert conf["raw_energy_dbfs"]["session"] > 0.1 > conf["energy_rel_db"]["session"]
    # PSOLA post-edit imposes planned emphasis on flat candidates
    gains = [e["emphasis_after"] - e["emphasis_before"] for e in s["psola_edits"] if e["emphasis_before"] is not None]
    assert sum(g > 0 for g in gains) >= len(gains) - 1
    assert (tmp_path / "demo" / "analysis" / "dashboard.html").stat().st_size > 50_000
