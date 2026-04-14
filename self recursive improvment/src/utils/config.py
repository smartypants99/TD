"""Central configuration for the recursive self-improvement system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_path: str = ""  # must be set before use
    quantization_config: Optional[dict] = None
    device_map: str = "auto"
    max_seq_length: int = 4096
    dtype: str = "bfloat16"


@dataclass
class DiagnosticsConfig:
    questions_per_domain: int = 200
    domains: list[str] = field(default_factory=lambda: [
        "reasoning", "math", "code", "science", "logic",
        "common_sense", "language_understanding", "abstraction",
    ])
    batch_size: int = 32
    confidence_threshold: float = 0.7
    activation_analysis: bool = True
    weak_layer_percentile: float = 0.2
    code_execution_timeout: int = 10  # seconds for code execution checks in diagnostics


@dataclass
class GeneratorConfig:
    min_reasoning_steps: int = 3
    max_reasoning_steps: int = 50
    require_explicit_assumptions: bool = True
    require_step_justification: bool = True
    samples_per_weakness: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3  # retries for insufficient reasoning


@dataclass
class VerifierConfig:
    check_logical_validity: bool = True
    check_step_completeness: bool = True
    check_assumption_grounding: bool = True
    reject_on_any_gap: bool = False  # True is too strict — rejects nearly all samples early on
    min_confidence_for_accept: float = 0.85
    use_model_verification: bool = False  # escalation: let model assist
    min_chain_steps: int = 2  # minimum steps for chain-level check


@dataclass
class TrainerConfig:
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    target_modules: list[str] = field(default_factory=lambda: [
        # LLaMA/Mistral/Qwen-style
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # GPT-2/GPT-J-style (Conv1D layers)
        "c_attn", "c_proj", "c_fc",
    ])
    # Custom: weakness-adaptive rank scaling
    min_rank: int = 8
    max_rank: int = 256
    weakness_rank_scale: float = 2.0  # how much extra rank for weak layers


@dataclass
class OrchestratorConfig:
    max_cycles: int = 100
    min_improvement_threshold: float = 0.01
    plateau_patience: int = 3
    escalation_schedule: dict = field(default_factory=lambda: {
        "verification": 4,
        "diagnosis": 7,
        "generation": 10,
    })
    output_dir: Path = Path("./outputs")
    log_dir: Path = Path("./logs")
    checkpoint_every: int = 1  # save every N cycles
    resume_from: Optional[str] = None  # resume from checkpoint path


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
