"""Tests for TimeDilateConfig.from_file() — JSON and YAML config loading."""
import json
import pytest

from timedilate.config import TimeDilateConfig, ConfigError


def test_from_json_file(tmp_path):
    cfg = {"dilation_factor": 10.0, "temperature": 0.5, "model": "test-model"}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    config = TimeDilateConfig.from_file(str(f))
    assert config.dilation_factor == 10.0
    assert config.temperature == 0.5
    assert config.model == "test-model"
    # Defaults preserved for unspecified fields
    assert config.max_tokens == 4096


def test_from_json_with_cli_overrides(tmp_path):
    cfg = {"dilation_factor": 10.0, "temperature": 0.5}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    config = TimeDilateConfig.from_file(str(f), temperature=0.9, model="override-model")
    assert config.temperature == 0.9
    assert config.model == "override-model"
    assert config.dilation_factor == 10.0


def test_from_json_none_overrides_ignored(tmp_path):
    cfg = {"temperature": 0.3}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    config = TimeDilateConfig.from_file(str(f), temperature=None)
    assert config.temperature == 0.3


def test_from_yaml_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = {"dilation_factor": 50.0, "branch_factor": 3}
    f = tmp_path / "config.yaml"
    f.write_text(yaml.dump(cfg))
    config = TimeDilateConfig.from_file(str(f))
    assert config.dilation_factor == 50.0
    assert config.branch_factor == 3


def test_from_yml_extension(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = {"dilation_factor": 5.0}
    f = tmp_path / "config.yml"
    f.write_text(yaml.dump(cfg))
    config = TimeDilateConfig.from_file(str(f))
    assert config.dilation_factor == 5.0


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        TimeDilateConfig.from_file("/nonexistent/path.json")


def test_invalid_json_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        TimeDilateConfig.from_file(str(f))


def test_non_dict_json_raises(tmp_path):
    f = tmp_path / "list.json"
    f.write_text("[1, 2, 3]")
    with pytest.raises(ConfigError, match="must contain a JSON/YAML object"):
        TimeDilateConfig.from_file(str(f))


def test_unknown_keys_warned_and_ignored(tmp_path, caplog):
    import logging
    cfg = {"dilation_factor": 2.0, "bogus_key": 999}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    logger = logging.getLogger("timedilate.config")
    orig = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="timedilate.config"):
            config = TimeDilateConfig.from_file(str(f))
    finally:
        logger.propagate = orig
    assert config.dilation_factor == 2.0
    assert "bogus_key" in caplog.text


def test_yaml_without_pyyaml_raises(tmp_path, monkeypatch):
    """If PyYAML is not installed, loading .yaml should give a clear error."""
    f = tmp_path / "config.yaml"
    f.write_text("dilation_factor: 10.0\n")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigError, match="PyYAML is required"):
        TimeDilateConfig.from_file(str(f))
