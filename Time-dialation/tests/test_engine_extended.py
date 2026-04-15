"""Extended engine tests: OOM retry, kwarg propagation, malformed responses."""
import sys
from unittest.mock import MagicMock, call
import pytest

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.engine import DilationEngine, InferenceError


def _reset():
    mock_vllm.reset_mock()
    mock_vllm.LLM.reset_mock()
    mock_vllm.SamplingParams.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = None
    mock_vllm.LLM.return_value.generate.return_value = None


def _output(text):
    o = MagicMock()
    o.outputs = [MagicMock(text=text)]
    return o


# --- OOM retry behavior ---

def test_oom_then_success_retries_and_returns():
    _reset()
    mock_vllm.LLM.return_value.generate.side_effect = [
        RuntimeError("CUDA out of memory"),
        [_output("recovered")],
    ]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate("prompt", retries=2)
    assert result == "recovered"
    assert engine._failed_calls == 1
    assert engine._total_calls == 1  # only success counts


def test_oom_exhausts_retries_raises_inference_error():
    _reset()
    mock_vllm.LLM.return_value.generate.side_effect = RuntimeError("OOM OOM OOM")
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError) as exc_info:
        engine.generate("prompt", retries=1)
    assert "OOM" in str(exc_info.value)
    assert engine._failed_calls == 2  # retries=1 => 2 attempts


def test_multiple_transient_errors_then_success():
    _reset()
    mock_vllm.LLM.return_value.generate.side_effect = [
        RuntimeError("transient 1"),
        RuntimeError("transient 2"),
        [_output("finally")],
    ]
    engine = DilationEngine(TimeDilateConfig())
    assert engine.generate("p", retries=3) == "finally"


# --- kwarg propagation (current API: only some kwargs forwarded) ---

def test_max_model_len_propagates_to_llm_kwargs():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    config = TimeDilateConfig(max_model_len=2048)
    engine = DilationEngine(config)
    engine.generate("p")
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert kwargs["max_model_len"] == 2048


def test_max_model_len_none_not_passed():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    config = TimeDilateConfig(max_model_len=None)
    engine = DilationEngine(config)
    engine.generate("p")
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert "max_model_len" not in kwargs


def test_gpu_memory_utilization_propagates():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    config = TimeDilateConfig(gpu_memory_utilization=0.45)
    engine = DilationEngine(config)
    engine.generate("p")
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert kwargs["gpu_memory_utilization"] == 0.45


def test_model_name_propagates():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    config = TimeDilateConfig(model="some/custom-model")
    engine = DilationEngine(config)
    engine.generate("p")
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert kwargs["model"] == "some/custom-model"
    assert kwargs["trust_remote_code"] is True


def test_initialize_called_once_across_multiple_generates():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p1")
    engine.generate("p2")
    engine.generate("p3")
    assert mock_vllm.LLM.call_count == 1


# --- SamplingParams propagation ---

def test_sampling_params_uses_config_defaults():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    config = TimeDilateConfig(max_tokens=1234, temperature=0.42)
    engine = DilationEngine(config)
    engine.generate("p")
    kwargs = mock_vllm.SamplingParams.call_args.kwargs
    assert kwargs["max_tokens"] == 1234
    assert kwargs["temperature"] == 0.42


def test_sampling_params_overrides_per_call():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    engine = DilationEngine(TimeDilateConfig(max_tokens=4096, temperature=0.7))
    engine.generate("p", max_tokens=32, temperature=0.0)
    kwargs = mock_vllm.SamplingParams.call_args.kwargs
    assert kwargs["max_tokens"] == 32
    assert kwargs["temperature"] == 0.0


def test_temperature_zero_override_not_treated_as_none():
    """temperature=0.0 must be honored, not fall back to config default."""
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("ok")]
    engine = DilationEngine(TimeDilateConfig(temperature=0.9))
    engine.generate("p", temperature=0.0)
    kwargs = mock_vllm.SamplingParams.call_args.kwargs
    assert kwargs["temperature"] == 0.0


# --- Malformed / empty model responses ---

def test_whitespace_only_treated_as_empty_and_retries():
    _reset()
    mock_vllm.LLM.return_value.generate.side_effect = [
        [_output("   \n\t  ")],
        [_output("real")],
    ]
    engine = DilationEngine(TimeDilateConfig())
    assert engine.generate("p", retries=1) == "real"


def test_all_empty_responses_raise_inference_error():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_output("")]
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError):
        engine.generate("p", retries=2)


def test_stats_after_failure():
    _reset()
    mock_vllm.LLM.return_value.generate.side_effect = RuntimeError("boom")
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError):
        engine.generate("p", retries=1)
    stats = engine.stats
    assert stats["failed_calls"] == 2
    assert stats["total_calls"] == 0
    assert "total_latency_s" in stats
