"""Tests for multi-dimensional scoring rubric."""
import pytest

from timedilate.config import TimeDilateConfig
from timedilate.controller import (
    DEFAULT_SCORING_DIMENSIONS,
    DilationController,
    DilationResult,
    ScoringResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal engine stub that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self._idx = 0
        self.last_usage = None

    def generate(self, prompt, **kwargs):
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return "50"


def _make_controller(responses=None, scoring_dimensions=None, **cfg_kw):
    config = TimeDilateConfig(
        dilation_factor=1.0,
        scoring_dimensions=scoring_dimensions,
        **cfg_kw,
    )
    engine = FakeEngine(responses)
    return DilationController(config, engine=engine)


# ---------------------------------------------------------------------------
# ScoringResult dataclass
# ---------------------------------------------------------------------------

class TestScoringResult:
    def test_fields(self):
        sr = ScoringResult(
            dimensions={"accuracy": 80, "clarity": 70},
            weighted_total=76,
            raw_text="accuracy: 80\nclarity: 70",
        )
        assert sr.dimensions["accuracy"] == 80
        assert sr.weighted_total == 76
        assert "accuracy" in sr.raw_text


# ---------------------------------------------------------------------------
# DEFAULT_SCORING_DIMENSIONS
# ---------------------------------------------------------------------------

class TestDefaultDimensions:
    def test_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_SCORING_DIMENSIONS.values()) - 1.0) < 1e-9

    def test_all_five_present(self):
        expected = {"accuracy", "completeness", "clarity", "reasoning", "efficiency"}
        assert set(DEFAULT_SCORING_DIMENSIONS.keys()) == expected


# ---------------------------------------------------------------------------
# _parse_dimensional_scores
# ---------------------------------------------------------------------------

class TestParseDimensionalScores:
    dims = {"accuracy": 0.3, "completeness": 0.25, "clarity": 0.2,
            "reasoning": 0.15, "efficiency": 0.10}

    def test_all_dimensions_parsed(self):
        raw = (
            "accuracy: 80\n"
            "completeness: 70\n"
            "clarity: 65\n"
            "reasoning: 90\n"
            "efficiency: 85"
        )
        result = DilationController._parse_dimensional_scores(raw, self.dims)
        assert result == {
            "accuracy": 80, "completeness": 70, "clarity": 65,
            "reasoning": 90, "efficiency": 85,
        }

    def test_equals_format(self):
        raw = "accuracy=80\ncompleteness=70\nclarity=65\nreasoning=90\nefficiency=85"
        result = DilationController._parse_dimensional_scores(raw, self.dims)
        assert result is not None
        assert result["accuracy"] == 80

    def test_missing_some_fills_with_50(self):
        raw = "accuracy: 80\ncompleteness: 70\nclarity: 65"
        result = DilationController._parse_dimensional_scores(raw, self.dims)
        assert result is not None
        assert result["reasoning"] == 50
        assert result["efficiency"] == 50

    def test_too_few_returns_none(self):
        raw = "accuracy: 80"
        result = DilationController._parse_dimensional_scores(raw, self.dims)
        assert result is None

    def test_empty_returns_none(self):
        result = DilationController._parse_dimensional_scores("", self.dims)
        assert result is None

    def test_clamps_to_0_100(self):
        raw = "accuracy: 150\ncompleteness: -5\nclarity: 65\nreasoning: 90\nefficiency: 85"
        result = DilationController._parse_dimensional_scores(raw, self.dims)
        assert result is not None
        # -5 won't match \d+ so it won't parse; will get default 50
        assert result["completeness"] == 50
        assert result["accuracy"] == 100  # clamped


# ---------------------------------------------------------------------------
# _compute_weighted_total
# ---------------------------------------------------------------------------

class TestComputeWeightedTotal:
    def test_equal_scores(self):
        dims = {"a": 0.5, "b": 0.5}
        scores = {"a": 80, "b": 80}
        assert DilationController._compute_weighted_total(scores, dims) == 80

    def test_weighted(self):
        dims = {"a": 0.75, "b": 0.25}
        scores = {"a": 100, "b": 0}
        assert DilationController._compute_weighted_total(scores, dims) == 75

    def test_clamps(self):
        dims = {"a": 1.0}
        scores = {"a": 150}
        assert DilationController._compute_weighted_total(scores, dims) == 100


# ---------------------------------------------------------------------------
# _get_scoring_dimensions
# ---------------------------------------------------------------------------

class TestGetScoringDimensions:
    def test_default(self):
        ctrl = _make_controller()
        dims = ctrl._get_scoring_dimensions()
        assert abs(sum(dims.values()) - 1.0) < 1e-9
        assert "accuracy" in dims

    def test_custom_normalized(self):
        ctrl = _make_controller(scoring_dimensions={"foo": 2, "bar": 3})
        dims = ctrl._get_scoring_dimensions()
        assert abs(dims["foo"] - 0.4) < 1e-9
        assert abs(dims["bar"] - 0.6) < 1e-9

    def test_zero_weights_fallback(self):
        ctrl = _make_controller(scoring_dimensions={"foo": 0, "bar": 0})
        dims = ctrl._get_scoring_dimensions()
        assert abs(dims["foo"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _score integration (via FakeEngine)
# ---------------------------------------------------------------------------

class TestScoreIntegration:
    def test_dimensional_response(self):
        response = (
            "accuracy: 80\n"
            "completeness: 70\n"
            "clarity: 65\n"
            "reasoning: 90\n"
            "efficiency: 85"
        )
        ctrl = _make_controller(responses=[response])
        score = ctrl._score("task", "output")
        assert 0 <= score <= 100
        assert ctrl._last_scoring_result is not None
        assert ctrl._last_scoring_result.dimensions["accuracy"] == 80
        assert ctrl._last_scoring_result.weighted_total == score

    def test_fallback_to_single_int(self):
        ctrl = _make_controller(responses=["72"])
        score = ctrl._score("task", "output")
        assert score == 72
        assert ctrl._last_scoring_result is not None
        assert ctrl._last_scoring_result.dimensions == {}
        assert ctrl._last_scoring_result.weighted_total == 72

    def test_cache_hit_skips_scoring(self):
        response = (
            "accuracy: 80\ncompleteness: 70\nclarity: 65\n"
            "reasoning: 90\nefficiency: 85"
        )
        ctrl = _make_controller(responses=[response])
        s1 = ctrl._score("task", "output")
        s2 = ctrl._score("task", "output")
        assert s1 == s2
        assert ctrl._score_cache_hits == 1

    def test_unparseable_defaults_to_50(self):
        ctrl = _make_controller(responses=["garbage no numbers here"])
        score = ctrl._score("task", "output")
        assert score == 50


# ---------------------------------------------------------------------------
# to_report includes dimensional breakdown
# ---------------------------------------------------------------------------

class TestReportDimensional:
    def test_report_includes_dimensions(self):
        sr = ScoringResult(
            dimensions={"accuracy": 80, "clarity": 70},
            weighted_total=76,
            raw_text="...",
        )
        result = DilationResult(
            output="test",
            dilation_factor=2.0,
            cycles_completed=1,
            total_cycles=2,
            elapsed_seconds=1.0,
            model_used="test-model",
            score=76,
            last_scoring_result=sr,
        )
        report = result.to_report()
        assert "scoring_dimensions" in report
        assert report["scoring_dimensions"]["accuracy"] == 80
        assert report["scoring_weighted_total"] == 76

    def test_report_without_dimensions(self):
        result = DilationResult(
            output="test",
            dilation_factor=2.0,
            cycles_completed=1,
            total_cycles=2,
            elapsed_seconds=1.0,
            model_used="test-model",
            score=50,
        )
        report = result.to_report()
        assert "scoring_dimensions" not in report
