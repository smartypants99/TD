import sys
from unittest.mock import MagicMock, patch
from timedilate.config import TimeDilateConfig

# Mock vllm before importing engine
mock_vllm = MagicMock()
sys.modules["vllm"] = mock_vllm


from timedilate.engine import InferenceEngine


def test_engine_init():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    engine = InferenceEngine(config)
    assert engine.config == config


def test_generate_calls_model():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="def hello(): pass")]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]

    engine = InferenceEngine(config)
    result = engine.generate("Write a function")
    assert result == "def hello(): pass"


def test_estimate_tokens():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    engine = InferenceEngine(config)
    assert engine.estimate_tokens("a" * 400) == 100
