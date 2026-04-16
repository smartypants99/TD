"""Tests for performance optimizations — truncation and token efficiency stats."""
import sys
from unittest.mock import MagicMock

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController, DilationResult


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    engine.last_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return engine


# --- _truncate tests ---

class TestTruncate:
    def test_short_text_unchanged(self):
        result = DilationController._truncate("hello", 100)
        assert result == "hello"

    def test_exact_limit_unchanged(self):
        text = "a" * 100
        assert DilationController._truncate(text, 100) == text

    def test_long_text_truncated(self):
        text = "A" * 5000 + "B" * 5000
        result = DilationController._truncate(text, 8000, 4000, 4000)
        assert len(result) < len(text)
        assert result.startswith("A" * 4000)
        assert result.endswith("B" * 4000)
        assert "...[truncated]..." in result

    def test_default_halves(self):
        text = "X" * 200
        result = DilationController._truncate(text, 100)
        # default keep_start=50, keep_end=50
        assert result.startswith("X" * 50)
        assert result.endswith("X" * 50)
        assert "...[truncated]..." in result

    def test_asymmetric_keep(self):
        text = "A" * 100 + "B" * 100
        result = DilationController._truncate(text, 50, keep_start=30, keep_end=20)
        assert result[:30] == "A" * 30
        assert result[-20:] == "B" * 20

    def test_empty_string(self):
        assert DilationController._truncate("", 100) == ""


# --- _history_summary respects max_history_items ---

class TestHistorySummaryLimit:
    def test_config_default_caps_items(self):
        config = TimeDilateConfig(max_history_items=3)
        engine = _mock_engine([])
        ctrl = DilationController(config, engine)
        from timedilate.controller import CycleRecord
        history = [
            CycleRecord(cycle=i, action="refine", improved=False,
                        score_before=50, score_after=50)
            for i in range(10)
        ]
        summary = ctrl._history_summary(history)
        # Should only include last 3
        assert "c7" in summary
        assert "c8" in summary
        assert "c9" in summary
        assert "c0" not in summary

    def test_override_max_items(self):
        config = TimeDilateConfig(max_history_items=2)
        engine = _mock_engine([])
        ctrl = DilationController(config, engine)
        from timedilate.controller import CycleRecord
        history = [
            CycleRecord(cycle=i, action="refine", improved=False,
                        score_before=50, score_after=50)
            for i in range(5)
        ]
        summary = ctrl._history_summary(history)
        assert "c3" in summary
        assert "c4" in summary
        assert "c2" not in summary


# --- Score truncation doesn't break uuid fence ---

class TestScoreTruncation:
    def test_score_truncates_long_output(self):
        """When output > max_score_input_chars, the scoring prompt uses truncated text."""
        config = TimeDilateConfig(max_score_input_chars=100)
        engine = _mock_engine(["85"])
        ctrl = DilationController(config, engine)
        long_output = "X" * 500
        score = ctrl._score("task", long_output)
        assert score == 85
        # Verify the prompt passed to engine.generate was truncated
        call_args = engine.generate.call_args
        prompt_sent = call_args[0][0]
        assert "...[truncated]..." in prompt_sent
        # The uuid fence delimiters must still be present
        assert "<<<RESPONSE" in prompt_sent or "RESPONSE" in prompt_sent

    def test_score_short_output_not_truncated(self):
        config = TimeDilateConfig(max_score_input_chars=8000)
        engine = _mock_engine(["90"])
        ctrl = DilationController(config, engine)
        score = ctrl._score("task", "short output")
        assert score == 90
        call_args = engine.generate.call_args
        prompt_sent = call_args[0][0]
        assert "...[truncated]..." not in prompt_sent


# --- Token efficiency stats ---

class TestTokenEfficiencyStats:
    def test_tokens_per_cycle_zero_cycles(self):
        r = DilationResult(
            output="x", dilation_factor=1.0, cycles_completed=0,
            total_cycles=0, elapsed_seconds=1.0, model_used="m", score=50,
            total_input_tokens=100, total_output_tokens=50,
        )
        assert r.tokens_per_cycle == 0.0

    def test_tokens_per_cycle(self):
        r = DilationResult(
            output="x", dilation_factor=2.0, cycles_completed=5,
            total_cycles=5, elapsed_seconds=1.0, model_used="m", score=80,
            total_input_tokens=500, total_output_tokens=250,
            initial_score=60,
        )
        assert r.tokens_per_cycle == 150.0  # 750 / 5

    def test_tokens_per_score_point(self):
        r = DilationResult(
            output="x", dilation_factor=2.0, cycles_completed=5,
            total_cycles=5, elapsed_seconds=1.0, model_used="m", score=80,
            total_input_tokens=500, total_output_tokens=250,
            initial_score=60,
        )
        # gain = 20, total tokens = 750
        assert r.tokens_per_score_point == 37.5

    def test_tokens_per_score_point_no_gain(self):
        r = DilationResult(
            output="x", dilation_factor=2.0, cycles_completed=5,
            total_cycles=5, elapsed_seconds=1.0, model_used="m", score=50,
            total_input_tokens=500, total_output_tokens=250,
            initial_score=50,
        )
        assert r.tokens_per_score_point == 0.0

    def test_tokens_per_score_point_negative_gain(self):
        r = DilationResult(
            output="x", dilation_factor=2.0, cycles_completed=5,
            total_cycles=5, elapsed_seconds=1.0, model_used="m", score=40,
            total_input_tokens=500, total_output_tokens=250,
            initial_score=50,
        )
        assert r.tokens_per_score_point == 0.0


# --- Config defaults ---

class TestPerfConfigDefaults:
    def test_defaults_exist(self):
        c = TimeDilateConfig()
        assert c.max_score_input_chars == 8000
        assert c.max_history_items == 5
        assert c.max_critique_chars == 16000
        assert c.max_refine_output_chars == 16000
