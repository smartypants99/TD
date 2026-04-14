from dataclasses import dataclass


@dataclass
class TimeDilateConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dilation_factor: int = 2
    budget_seconds: float = 30.0
    branch_factor: int = 3
    max_tokens: int = 4096
    temperature: float = 0.7
    scoring_temperature: float = 0.0
    checkpoint_dir: str = ".timedilate_checkpoints"
    checkpoint_interval: int = 10
    convergence_threshold: int = 5
    context_window: int = 32768
    use_reflection: bool = False
    score_weights: dict | None = None  # e.g. {"correctness": 40, "completeness": 20, "quality": 20, "elegance": 20}
