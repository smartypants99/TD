"""Edge-case gap coverage: score parser negatives, time-budget extremes,
early_stop boundaries, generate-path OOM classification, cache eviction.

These fill small gaps left by the main suites — no overlap with existing tests.
"""
import sys
from unittest.mock import MagicMock

import pytest

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController
from timedilate.engine import DilationEngine, _looks_like_oom


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    return engine


# --- Score parser: negative numbers must not be accepted ---

def test_score_parser_rejects_negative_number():
    """'-50' should NOT parse as a valid score (isdigit() rejects the '-')."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="-50")
    controller = DilationController(TimeDilateConfig(), engine)
    s = controller._score("prompt", "output")
    assert s == 50  # parser gave up, defaulted to 50


def test_score_parser_zero_is_valid():
    """'0' is a legitimate score — must not be mis-rejected as falsy."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="0")
    controller = DilationController(TimeDilateConfig(), engine)
    s = controller._score("prompt", "output")
    assert s == 0


def test_score_parser_100_is_valid_at_upper_bound():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="100")
    controller = DilationController(TimeDilateConfig(), engine)
    assert controller._score("p", "o") == 100


# --- early_stop_score boundary: exactly equal triggers stop ---

def test_early_stop_triggers_at_exact_threshold():
    """Score == early_stop_score must trigger early stop (>= comparison)."""
    engine = _mock_engine(["initial", "98"])  # score exactly 98
    config = TimeDilateConfig(dilation_factor=50, early_stop_score=98)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 98
    assert result.cycles_completed == 0  # stopped immediately


def test_early_stop_score_100_never_triggers_on_99():
    """early_stop_score=100 should keep running even at 99."""
    responses = ["initial", "99"]
    for _ in range(3):
        responses.extend(["critique", "refine", "99"])
    engine = _mock_engine(responses)
    config = TimeDilateConfig(
        dilation_factor=3, early_stop_score=100, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.cycles_completed == 3  # ran all cycles


# --- Time budget: zero means "no cycles after initial" in budget mode ---

def test_time_budget_zero_still_returns_initial():
    """time_budget_seconds=0 means: produce initial only, no cycles."""
    engine = _mock_engine(["initial", "70"])
    config = TimeDilateConfig(dilation_factor=100, time_budget_seconds=0.0)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.output == "initial"
    assert result.elapsed_seconds < 1.0


# --- OOM classification on generate path (not just init) ---

def test_looks_like_oom_catches_generate_kv_cache_message():
    """vLLM's KV-cache allocation message at inference time must classify as OOM."""
    msg = "not enough kv cache blocks; reduce max_model_len"
    err = RuntimeError(msg)
    assert _looks_like_oom(err)


def test_looks_like_oom_does_not_fire_on_generic_value_error():
    err = ValueError("bad input shape")
    assert not _looks_like_oom(err)


# --- batched generate with single prompt in list returns list ---

def test_batched_input_single_element_still_returns_list(vllm_mock, make_vllm_output):
    """generate(['one']) must return a list even though it has one element."""
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("only")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate(["one"])
    assert isinstance(result, list)
    assert result == ["only"]


def test_scalar_input_returns_scalar(vllm_mock, make_vllm_output):
    """generate('one') must return a str, preserving legacy contract."""
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("single")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate("one")
    assert isinstance(result, str)
    assert result == "single"


# --- Score cache eviction: LRU bound must hold ---

def test_score_cache_respects_max_size():
    """_score_cache must not grow without bound — LRU eviction keeps it capped."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="70")
    controller = DilationController(TimeDilateConfig(), engine)
    max_size = getattr(controller, "_SCORE_CACHE_MAX", None)
    if max_size is None:
        pytest.skip("no _SCORE_CACHE_MAX attribute on controller")
    # Fill beyond the cap
    for i in range(max_size + 10):
        controller._score(f"prompt_{i}", f"output_{i}")
    assert len(controller._score_cache) <= max_size


# --- Config: extreme dilation factor does not overflow num_cycles ---

def test_num_cycles_handles_int_boundary():
    """Large factors stay as ints without overflow."""
    config = TimeDilateConfig(dilation_factor=2**62)
    assert config.num_cycles == 2**62
    assert isinstance(config.num_cycles, int)


# --- Controller: branch_factor=1 must never spread temperature ---

def test_branch_factor_1_ignores_temperature_spread_setting():
    """With a single branch, branch_temperature_spread must be a no-op
    (spread requires >=2 branches to make sense)."""
    engine = _mock_engine([
        "initial", "60",
        "critique", "refined", "75",
    ])
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=1,
        branch_temperature_spread=0.9, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Should complete one cycle successfully without crashing on spread math
    assert result.cycles_completed == 1
    assert result.score == 75
