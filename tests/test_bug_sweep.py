"""Bug sweep tests — edge cases found during deep audit."""
import sys
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig, ConfigError

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.controller import DilationController


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return engine


# ── 1. Score parse edge cases ───────────────────────────────────────

class TestScoreParsing:
    def _score_with_reply(self, reply_text):
        """Helper: build a controller, call _score, return parsed score."""
        engine = _mock_engine([reply_text])
        config = TimeDilateConfig(dilation_factor=1.0)
        ctrl = DilationController(config, engine)
        return ctrl._score("task", "output")

    def test_pure_integer(self):
        assert self._score_with_reply("85") == 85

    def test_negative_clamps_to_zero(self):
        assert self._score_with_reply("-5") == 0

    def test_float_rounds(self):
        assert self._score_with_reply("3.7") == 4

    def test_na_defaults_to_50(self):
        assert self._score_with_reply("N/A") == 50

    def test_score_colon_prefix(self):
        assert self._score_with_reply("score: 80") == 80

    def test_fraction_format(self):
        assert self._score_with_reply("80/100") == 80

    def test_over_100_clamps(self):
        assert self._score_with_reply("150") == 100

    def test_float_over_100_clamps(self):
        assert self._score_with_reply("100.5") == 100


# ── 2. Empty prompt ─────────────────────────────────────────────────

def test_empty_prompt_no_crash():
    engine = _mock_engine(["some output"])
    config = TimeDilateConfig(dilation_factor=1.0)
    ctrl = DilationController(config, engine)
    result = ctrl.run("")
    assert result.output == "some output"


# ── 3. Unicode in cache key ─────────────────────────────────────────

def test_unicode_cache_key_consistent():
    config = TimeDilateConfig(dilation_factor=1.0)
    engine = _mock_engine([])
    ctrl = DilationController(config, engine)
    key1 = ctrl._cache_key("Write \u00e9m\u00f6ji \U0001f600", "output \u4e16\u754c")
    key2 = ctrl._cache_key("Write \u00e9m\u00f6ji \U0001f600", "output \u4e16\u754c")
    assert key1 == key2
    # Different input -> different key
    key3 = ctrl._cache_key("different", "output \u4e16\u754c")
    assert key1 != key3


# ── 4. Engine returns None ──────────────────────────────────────────

def test_engine_returns_none_coerced_to_empty():
    engine = _mock_engine([None])
    config = TimeDilateConfig(dilation_factor=1.0)
    ctrl = DilationController(config, engine)
    result = ctrl.run("")
    assert result.output == ""


# ── 5. history_summary with special chars ───────────────────────────

def test_history_summary_special_chars():
    from timedilate.controller import CycleRecord
    config = TimeDilateConfig(dilation_factor=1.0)
    engine = _mock_engine([])
    ctrl = DilationController(config, engine)
    history = [
        CycleRecord(cycle=1, action="refine", improved=True,
                    score_before=50, score_after=60),
    ]
    summary = ctrl._history_summary(history)
    assert "c1" in summary
    assert "50->60" in summary


# ── 6. Token counter — Python ints don't overflow, verify accumulation ─

def test_token_accumulation_large():
    engine = _mock_engine(["ok", "80"])
    engine.last_usage = {"input_tokens": 10**15, "output_tokens": 10**15, "total_tokens": 2 * 10**15}
    config = TimeDilateConfig(dilation_factor=1.0)
    ctrl = DilationController(config, engine)
    ctrl.run("test")
    assert ctrl._total_input_tokens >= 10**15


# ── 7. _reject_short with best_output="" ────────────────────────────

def test_reject_short_empty_best():
    config = TimeDilateConfig(dilation_factor=1.0)
    engine = _mock_engine([])
    ctrl = DilationController(config, engine)
    # Empty candidate should be rejected
    assert ctrl._reject_short("", "") is True
    # Non-empty candidate with empty best should not be rejected
    assert ctrl._reject_short("some text", "") is False


# ── 8. _pairwise_break with single candidate ───────────────────────

def test_pairwise_break_single_candidate():
    engine = _mock_engine(["A"])
    config = TimeDilateConfig(dilation_factor=1.0)
    ctrl = DilationController(config, engine)
    # Single item — should return 0 (the only index)
    result = ctrl._pairwise_break("task", [(80, "text", 0.7)])
    assert result == 0


# ── 9. Config: dilation_factor=1.0 AND time_budget ─────────────────

def test_factor_1_with_time_budget():
    """factor=1.0 + time_budget should still run cycles (time-budget mode)."""
    engine = _mock_engine([
        "initial",  # gen
        "70",       # score
        "critique", # critique
        "better",   # refine
        "85",       # score
    ])
    config = TimeDilateConfig(dilation_factor=1.0, time_budget_seconds=60)
    ctrl = DilationController(config, engine)
    result = ctrl.run("test")
    # Should have run at least one cycle via time budget
    assert result.score >= 70


# ── 10. CLI dtype aliases ───────────────────────────────────────────

def test_cli_dtype_half_alias():
    from click.testing import CliRunner
    from timedilate.cli import run, main
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hello", "--dtype", "half", "--dry-run"])
    # Should not fail with ConfigError about dtype
    assert result.exit_code == 0
    assert "ConfigError" not in (result.output or "")


def test_cli_dtype_bf16_alias():
    from click.testing import CliRunner
    from timedilate.cli import run, main
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hello", "--dtype", "bf16", "--dry-run"])
    assert result.exit_code == 0
    assert "ConfigError" not in (result.output or "")


def test_cli_dtype_canonical_passthrough():
    from click.testing import CliRunner
    from timedilate.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hello", "--dtype", "float16", "--dry-run"])
    assert result.exit_code == 0
