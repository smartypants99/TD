"""Time Dilation Runtime — give AI more subjective thinking time."""
from timedilate.config import TimeDilateConfig, ConfigError
from timedilate.controller import DilationController, DilationResult, CycleRecord, DilationEvent
from timedilate.engine import DilationEngine, InferenceError

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "TimeDilateConfig",
    "ConfigError",
    "DilationController",
    "DilationResult",
    "CycleRecord",
    "DilationEvent",
    "DilationEngine",
    "InferenceError",
]
