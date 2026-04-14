import json
import logging
import sys


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable output."""

    def format(self, record):
        log_entry = {
            "ts": self.formatTime(record, "%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "cycle_data"):
            log_entry["cycle"] = record.cycle_data
        return json.dumps(log_entry)


def setup_logging(verbose: bool = False, structured: bool = False) -> None:
    """Configure logging for the timedilate package."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    root = logging.getLogger("timedilate")
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def log_cycle_summary(logger: logging.Logger, cycle: int, score: int, previous_score: int,
                      directive_source: str, elapsed: float, branch_count: int) -> None:
    """Emit a structured cycle summary log."""
    data = {
        "cycle": cycle,
        "score": score,
        "delta": score - previous_score,
        "source": directive_source,
        "elapsed_s": round(elapsed, 3),
        "branches": branch_count,
    }
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"Cycle {cycle}: {score} ({score - previous_score:+d}) [{directive_source}] {elapsed:.2f}s",
        (), None
    )
    record.cycle_data = data
    logger.handle(record)
