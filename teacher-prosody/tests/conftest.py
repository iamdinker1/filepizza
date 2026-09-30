import numpy as np
import pytest

from teacher_prosody.testing import WordSpec, espeak_utterance, have_espeak

needs_espeak = pytest.mark.skipif(not have_espeak(), reason="espeak-ng not installed")


@pytest.fixture(scope="session")
def not_constant():
    """MOCK teacher utterance with a scripted strong focus on NOT and a pre-emphasis pause."""
    if not have_espeak():
        pytest.skip("espeak-ng not installed")
    spec = [WordSpec("Velocity", pause_after=0.05), WordSpec("is", pause_after=0.25),
            WordSpec("NOT", pitch=85, speed=105, amp=170, pause_after=0.08), WordSpec("constant.", pitch=45)]
    return espeak_utterance(spec, seed=1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
