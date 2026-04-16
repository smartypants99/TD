"""Tests for checkpoint/resume functionality."""
import json
import pytest
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController, CycleRecord


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    call_count = 0

    def gen_side_effect(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if "Rate the RESPONSE" in prompt or "strict reviewer" in prompt:
            return "75"
        return f"response {call_count}"

    engine.generate.side_effect = gen_side_effect
    engine.last_usage = None
    return engine


def test_checkpoint_saves_file(tmp_path, mock_engine):
    config = TimeDilateConfig(
        dilation_factor=3.0,
        checkpoint_interval=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_self_critique=False,
    )
    ctrl = DilationController(config, engine=mock_engine)
    ctrl.run("test prompt")

    ckpt_dir = tmp_path / "ckpts"
    assert ckpt_dir.exists()
    ckpts = list(ckpt_dir.glob("checkpoint_cycle_*.json"))
    assert len(ckpts) >= 1

    data = json.loads(ckpts[0].read_text())
    assert "cycle" in data
    assert "best_output" in data
    assert "best_score" in data
    assert "config" in data
    assert "history" in data
    assert "prompt" in data


def test_checkpoint_interval_zero_no_save(tmp_path, mock_engine):
    config = TimeDilateConfig(
        dilation_factor=3.0,
        checkpoint_interval=0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_self_critique=False,
    )
    ctrl = DilationController(config, engine=mock_engine)
    ctrl.run("test prompt")

    ckpt_dir = tmp_path / "ckpts"
    assert not ckpt_dir.exists()


def test_checkpoint_interval_2_saves_every_other(tmp_path, mock_engine):
    config = TimeDilateConfig(
        dilation_factor=5.0,
        checkpoint_interval=2,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_self_critique=False,
    )
    ctrl = DilationController(config, engine=mock_engine)
    ctrl.run("test prompt")

    ckpts = sorted((tmp_path / "ckpts").glob("checkpoint_cycle_*.json"))
    cycle_nums = []
    for cp in ckpts:
        data = json.loads(cp.read_text())
        cycle_nums.append(data["cycle"])
    # All saved cycles should be multiples of 2
    for c in cycle_nums:
        assert c % 2 == 0


def test_resume_restores_config(tmp_path):
    cp_data = {
        "cycle": 5,
        "best_output": "best so far",
        "best_score": 80,
        "prompt": "do something",
        "history": [],
        "config": {
            "model": "test-model",
            "dilation_factor": 10.0,
            "temperature": 0.5,
        },
    }
    cp_file = tmp_path / "checkpoint.json"
    cp_file.write_text(json.dumps(cp_data))

    mock_engine = MagicMock()
    mock_engine.generate.return_value = "75"
    mock_engine.last_usage = None

    ctrl = DilationController.resume(str(cp_file), engine=mock_engine)
    assert ctrl.config.model == "test-model"
    assert ctrl.config.dilation_factor == 10.0
    assert ctrl.config.temperature == 0.5
    assert ctrl._checkpoint_state["best_score"] == 80


def test_resume_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DilationController.resume("/nonexistent/checkpoint.json")


def test_checkpoint_config_excludes_prompt_templates(tmp_path, mock_engine):
    config = TimeDilateConfig(
        dilation_factor=2.0,
        checkpoint_interval=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_self_critique=False,
    )
    ctrl = DilationController(config, engine=mock_engine)
    ctrl.run("test prompt")

    ckpts = list((tmp_path / "ckpts").glob("*.json"))
    assert len(ckpts) >= 1
    data = json.loads(ckpts[0].read_text())
    assert "prompt_templates" not in data["config"]
