"""Entry point for the recursive self-improvement system."""

import argparse
import logging
from pathlib import Path

from src.utils.config import SystemConfig, ModelConfig
from src.orchestrator.loop import ImprovementLoop


def main():
    parser = argparse.ArgumentParser(description="Recursive Self-Improvement System")
    parser.add_argument("--model", required=True, help="Path to the base model")
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum improvement cycles")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (use float16 for GPUs without bf16 support)")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size (default: 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Training epochs per cycle (default: 3)")
    parser.add_argument("--questions-per-domain", type=int, default=None, help="Diagnostic questions per domain (default: 200)")
    parser.add_argument("--samples-per-weakness", type=int, default=None, help="Training samples per weakness (default: 100)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for 5-10x faster inference (pip install vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory fraction (default: 0.90)")
    args = parser.parse_args()

    # Create output dir BEFORE setting up file logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )

    config = SystemConfig()
    config.model = ModelConfig(model_path=args.model, dtype=args.dtype)
    config.orchestrator.max_cycles = args.max_cycles
    config.orchestrator.output_dir = output_dir
    config.orchestrator.log_dir = output_dir / "logs"
    if args.resume:
        config.orchestrator.resume_from = args.resume
    if args.lora_rank is not None:
        config.trainer.lora_rank = args.lora_rank
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.trainer.num_epochs = args.num_epochs
    if args.questions_per_domain is not None:
        config.diagnostics.questions_per_domain = args.questions_per_domain
    if args.samples_per_weakness is not None:
        config.generator.samples_per_weakness = args.samples_per_weakness

    loop = ImprovementLoop(config)

    if args.use_vllm:
        from src.utils.vllm_backend import VLLMBackend, patch_model_loader_with_vllm
        vllm_backend = VLLMBackend(
            model_path=args.model,
            dtype=args.dtype,
            max_model_len=config.model.max_seq_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        # Patch AFTER loop creation but BEFORE run — loop.run() calls _setup()
        # which loads the HF model (needed for LoRA), then vLLM handles generation.
        loop._vllm_backend = vllm_backend

    try:
        loop.run()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("\nInterrupted — cleaning up and saving checkpoint + report")
        # Strip any injected LoRA layers so the model is in a clean state.
        # If interrupted mid-training, LoRA params are partially trained —
        # merging would corrupt the base model, so just discard them.
        try:
            loop.trainer.strip_lora()
        except Exception:
            pass
        # Save checkpoint so resume picks up where we left off — without this,
        # work since the last checkpoint_every boundary is lost on interrupt.
        if loop.history:
            try:
                loop._save_checkpoint(loop.history[-1].cycle)
            except Exception:
                pass
        loop._save_final_report()


if __name__ == "__main__":
    main()
