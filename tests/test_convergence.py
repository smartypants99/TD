"""Tests for slope-based convergence detection (_detect_trend)."""
import sys
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.controller import DilationController, CycleRecord


class TestDetectTrend:
    """Unit tests for DilationController._detect_trend."""

    def test_too_few_scores_returns_plateau(self):
        assert DilationController._detect_trend([]) == "plateau"
        assert DilationController._detect_trend([50]) == "plateau"
        assert DilationController._detect_trend([50, 55]) == "plateau"

    def test_improving_trend(self):
        # Clear upward slope
        scores = [50, 52, 55, 58, 62, 66, 70, 75]
        assert DilationController._detect_trend(scores) == "improving"

    def test_degrading_trend(self):
        # Clear downward slope
        scores = [80, 77, 73, 70, 66, 62, 58, 55]
        assert DilationController._detect_trend(scores) == "degrading"

    def test_plateau_trend(self):
        # Flat scores
        scores = [60, 60, 60, 60, 60]
        assert DilationController._detect_trend(scores) == "plateau"

    def test_plateau_with_minor_noise(self):
        # Small oscillation within threshold
        scores = [60, 61, 60, 61, 60]
        assert DilationController._detect_trend(scores) == "plateau"

    def test_window_limits_recent_scores(self):
        # Old degrading data followed by improving — window=4 should see improving
        scores = [90, 85, 80, 75, 70, 71, 73, 76, 80]
        result = DilationController._detect_trend(scores, window=4)
        assert result == "improving"

    def test_all_identical_scores(self):
        scores = [50, 50, 50, 50]
        assert DilationController._detect_trend(scores) == "plateau"

    def test_exactly_three_scores_improving(self):
        scores = [50, 55, 61]
        assert DilationController._detect_trend(scores) == "improving"

    def test_exactly_three_scores_degrading(self):
        scores = [70, 65, 59]
        assert DilationController._detect_trend(scores) == "degrading"

    def test_custom_window(self):
        scores = [10, 20, 30, 40, 50, 60]
        # window=3 looks at [40, 50, 60] — improving
        assert DilationController._detect_trend(scores, window=3) == "improving"


class TestTrendInCycleRecord:
    """Verify trend field is populated on CycleRecord."""

    def test_trend_field_default(self):
        rec = CycleRecord(cycle=1, action="refine", improved=True)
        assert rec.trend == ""

    def test_trend_field_set(self):
        rec = CycleRecord(cycle=1, action="refine", improved=True, trend="improving")
        assert rec.trend == "improving"


class TestTrendIntegration:
    """Integration: trend detection influences patience in the main loop."""

    def _mock_engine(self, responses):
        engine = MagicMock()
        engine.generate = MagicMock(side_effect=list(responses))
        engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return engine

    def test_degrading_triggers_fresh_attempt(self):
        """When scores degrade, the controller should force a fresh attempt sooner."""
        # Setup: factor=5 (5 cycles), patience=3
        # Scores: initial=70, then cycle scores degrade: 68, 65, 62
        # With degrading trend, fresh attempt should trigger before patience=3
        responses = [
            "initial output",    # initial generation
            "70",                # score initial
            # cycle 1
            "critique 1",        # critique
            "refined 1",         # refine
            "68",                # score refined (worse)
            # cycle 2
            "critique 2",
            "refined 2",
            "65",                # score (worse again)
            # cycle 3
            "critique 3",
            "refined 3",
            "62",                # score (still degrading)
            # fresh attempt triggered by degrading trend
            "fresh output",      # fresh attempt
            "75",                # score fresh
            # cycle 4
            "critique 4",
            "refined 4",
            "76",                # score
            # cycle 5
            "critique 5",
            "refined 5",
            "77",                # score
        ]
        engine = self._mock_engine(responses)
        config = TimeDilateConfig(dilation_factor=5.0, convergence_patience=3)
        controller = DilationController(config, engine)
        result = controller.run("test prompt")
        # Should have at least one convergence reset due to degrading trend
        assert result.convergence_resets >= 1

    def test_trend_recorded_in_history(self):
        """Cycle records should contain trend information."""
        responses = [
            "initial",
            "60",                # score initial
            # cycle 1
            "critique",
            "better",
            "65",
            # cycle 2
            "critique",
            "even better",
            "70",
        ]
        engine = self._mock_engine(responses)
        config = TimeDilateConfig(dilation_factor=2.0, convergence_patience=3)
        controller = DilationController(config, engine)
        result = controller.run("test")
        # All refine records should have a trend string
        for rec in result.cycle_history:
            if rec.action == "refine":
                assert rec.trend in ("improving", "plateau", "degrading")

    def test_to_report_includes_trend(self):
        """to_report serialization should include trend field."""
        responses = [
            "initial",
            "60",
            "critique",
            "better",
            "70",
        ]
        engine = self._mock_engine(responses)
        config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=3)
        controller = DilationController(config, engine)
        result = controller.run("test")
        report = result.to_report()
        for entry in report["cycle_history"]:
            assert "trend" in entry
