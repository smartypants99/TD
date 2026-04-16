"""Tests for ensemble scoring feature."""
import sys
from unittest.mock import MagicMock, call

from timedilate.config import TimeDilateConfig, ConfigError

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.controller import DilationController
import pytest


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return engine


# --- Config validation ---

def test_ensemble_scores_default_is_1():
    cfg = TimeDilateConfig()
    assert cfg.ensemble_scores == 1


def test_ensemble_scores_valid_range():
    for n in (1, 2, 3, 4, 5):
        cfg = TimeDilateConfig(ensemble_scores=n)
        cfg.validate()


def test_ensemble_scores_too_low():
    cfg = TimeDilateConfig(ensemble_scores=0)
    with pytest.raises(ConfigError, match="ensemble_scores must be 1-5"):
        cfg.validate()


def test_ensemble_scores_too_high():
    cfg = TimeDilateConfig(ensemble_scores=6)
    with pytest.raises(ConfigError, match="ensemble_scores must be 1-5"):
        cfg.validate()


# --- _parse_score static method ---

def test_parse_score_integer():
    assert DilationController._parse_score("85") == 85


def test_parse_score_with_text():
    assert DilationController._parse_score("The score is 72 out of 100.") == 72


def test_parse_score_fraction_format():
    assert DilationController._parse_score("80/100") == 80


def test_parse_score_clamps_high():
    assert DilationController._parse_score("150") == 100


def test_parse_score_clamps_low():
    assert DilationController._parse_score("-5") == 0


def test_parse_score_unparseable():
    assert DilationController._parse_score("no number here") == 50


# --- Ensemble scoring (ensemble_scores=1 passthrough) ---

def test_single_ensemble_uses_one_call():
    """ensemble_scores=1 should behave like the original single-call path."""
    engine = _mock_engine([
        "initial output",  # initial generation
        "75",              # single score call
    ])
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=1)
    ctrl = DilationController(cfg, engine)
    result = ctrl.run("test prompt")
    assert result.output == "initial output"


# --- Ensemble scoring (ensemble_scores=3) ---

def test_ensemble_3_takes_median():
    """With ensemble_scores=3, _score makes 3 calls and returns median."""
    engine = _mock_engine(["dummy"])  # not used directly
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=3)
    ctrl = DilationController(cfg, engine)
    # Call _score directly to test ensemble logic in isolation.
    engine.generate = MagicMock(side_effect=["60", "80", "70"])
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    score = ctrl._score("test prompt", "test output")
    # sorted: [60, 70, 80], median = 70
    assert score == 70
    assert engine.generate.call_count == 3


def test_ensemble_5_takes_median():
    """With ensemble_scores=5, median of 5 sorted scores."""
    engine = _mock_engine(["dummy"])
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=5)
    ctrl = DilationController(cfg, engine)
    engine.generate = MagicMock(side_effect=["90", "50", "70", "80", "60"])
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    score = ctrl._score("test prompt", "test output")
    # sorted: [50, 60, 70, 80, 90], median = 70
    assert score == 70


def test_ensemble_2_takes_higher_median():
    """With ensemble_scores=2, median index is 1 (higher of two)."""
    engine = _mock_engine(["dummy"])
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=2)
    ctrl = DilationController(cfg, engine)
    engine.generate = MagicMock(side_effect=["60", "80"])
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    score = ctrl._score("test prompt", "test output")
    # sorted: [60, 80], index 1 = 80
    assert score == 80


def test_ensemble_caches_median_not_individual():
    """Second call for same prompt+output should hit cache, not re-run ensemble."""
    engine = _mock_engine(["dummy"])
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=3)
    ctrl = DilationController(cfg, engine)
    engine.generate = MagicMock(side_effect=["60", "80", "70"])
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    score1 = ctrl._score("test prompt", "test output")
    assert score1 == 70
    # Second call should hit cache — no more generate calls
    score2 = ctrl._score("test prompt", "test output")
    assert score2 == 70
    assert engine.generate.call_count == 3  # no extra calls


def test_ensemble_all_parse_failures_default_50():
    """If all ensemble calls fail to parse, each returns 50, median is 50."""
    engine = _mock_engine(["dummy"])
    cfg = TimeDilateConfig(dilation_factor=1.0, ensemble_scores=3)
    ctrl = DilationController(cfg, engine)
    engine.generate = MagicMock(side_effect=["no score", "also no", "still no"])
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    score = ctrl._score("test prompt", "test output")
    assert score == 50
