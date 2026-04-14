import json
import logging
from timedilate.logging_config import setup_logging, log_cycle_summary, StructuredFormatter


def test_setup_logging_verbose():
    setup_logging(verbose=True)
    logger = logging.getLogger("timedilate")
    assert logger.level == logging.DEBUG


def test_setup_logging_quiet():
    setup_logging(verbose=False)
    logger = logging.getLogger("timedilate")
    assert logger.level == logging.WARNING


def test_structured_formatter():
    fmt = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
    output = fmt.format(record)
    data = json.loads(output)
    assert data["msg"] == "hello"
    assert data["level"] == "INFO"


def test_structured_formatter_with_cycle_data():
    fmt = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "cycle 1", (), None)
    record.cycle_data = {"cycle": 1, "score": 80}
    output = fmt.format(record)
    data = json.loads(output)
    assert data["cycle"]["score"] == 80


def test_log_cycle_summary():
    test_logger = logging.getLogger("timedilate.test_cycle")
    test_logger.setLevel(logging.DEBUG)
    handler = logging.handlers.MemoryHandler(capacity=10) if hasattr(logging, 'handlers') else logging.StreamHandler()
    test_logger.handlers.clear()
    # Just verify it doesn't crash
    log_cycle_summary(test_logger, 1, 80, 70, "builtin", 0.5, 3)


def test_setup_logging_structured():
    setup_logging(verbose=True, structured=True)
    logger = logging.getLogger("timedilate")
    assert isinstance(logger.handlers[0].formatter, StructuredFormatter)
