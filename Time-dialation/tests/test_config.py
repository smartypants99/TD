"""Tests for TimeDilateConfig."""
import pytest
from timedilate.config import TimeDilateConfig, ConfigError


def test_default_config():
    config = TimeDilateConfig()
    assert config.dilation_factor == 1.0
    assert config.model == "Qwen/Qwen3-8B"


def test_validate_rejects_negative_factor():
    config = TimeDilateConfig(dilation_factor=0.5)
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_rejects_bad_temperature():
    config = TimeDilateConfig(temperature=3.0)
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_rejects_bad_gpu_memory():
    config = TimeDilateConfig(gpu_memory_gb=0)
    with pytest.raises(ConfigError):
        config.validate()


def test_num_cycles_factor_1():
    config = TimeDilateConfig(dilation_factor=1.0)
    assert config.num_cycles == 0


def test_num_cycles_factor_10():
    config = TimeDilateConfig(dilation_factor=10)
    assert config.num_cycles == 10


def test_num_cycles_factor_1000():
    config = TimeDilateConfig(dilation_factor=1000)
    assert config.num_cycles == 1000


def test_num_cycles_factor_million():
    config = TimeDilateConfig(dilation_factor=1_000_000)
    assert config.num_cycles == 1_000_000


def test_num_cycles_scales_infinitely():
    """No ceiling — any factor works."""
    config = TimeDilateConfig(dilation_factor=1_000_000_000_000)
    assert config.num_cycles == 1_000_000_000_000


def test_describe():
    config = TimeDilateConfig(dilation_factor=100)
    desc = config.describe()
    assert "100" in desc
    assert "100" in desc  # cycles


def test_describe_with_branch_factor():
    config = TimeDilateConfig(dilation_factor=10, branch_factor=3)
    desc = config.describe()
    assert "3" in desc


def test_validate_rejects_bad_gpu_util():
    config = TimeDilateConfig(gpu_memory_utilization=1.5)
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_rejects_zero_branch_factor():
    config = TimeDilateConfig(branch_factor=0)
    with pytest.raises(ConfigError):
        config.validate()


def test_default_gpu_util_is_safe():
    """Default must leave headroom for other GPU processes."""
    config = TimeDilateConfig()
    assert config.gpu_memory_utilization <= 0.75


def _capture_config_warnings(config):
    """Attach a direct handler to timedilate.config logger and capture records."""
    import logging
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("timedilate.config")
    handler = _Capture(level=logging.WARNING)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        config.validate()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    return records


def test_seed_with_zero_spread_warns():
    """seed + branch_factor>1 + spread=0 collapses branches — should warn."""
    config = TimeDilateConfig(
        seed=42, branch_factor=3, branch_temperature_spread=0.0,
    )
    records = _capture_config_warnings(config)
    msgs = [r.getMessage() for r in records]
    assert any("collapse" in m and "branch_temperature_spread" in m for m in msgs)


def test_seed_with_spread_no_warning():
    """seed + spread>0 is fine — branches diversify via temperature."""
    config = TimeDilateConfig(
        seed=42, branch_factor=3, branch_temperature_spread=0.3,
    )
    records = _capture_config_warnings(config)
    assert not any("collapse" in r.getMessage() for r in records)


def test_no_seed_no_warning():
    """No seed → no collapse risk even with spread=0."""
    config = TimeDilateConfig(
        seed=None, branch_factor=3, branch_temperature_spread=0.0,
    )
    records = _capture_config_warnings(config)
    assert not any("collapse" in r.getMessage() for r in records)
