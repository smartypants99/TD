"""Optional vLLM backend for 5-10x faster inference across all phases.

Drop-in replacement for ModelLoader's generate/generate_batch methods.
Uses vLLM's continuous batching, PagedAttention, and optimized CUDA kernels.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VLLMBackend:
    """Fast inference backend using vLLM. Replaces HuggingFace generate()."""

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90):
        from vllm import LLM, SamplingParams  # noqa: F811
        self.SamplingParams = SamplingParams

        logger.info(f"Loading model with vLLM: {model_path}")
        self._llm = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            # Disable log spam
            disable_log_stats=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("vLLM backend ready")

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a single response."""
        results = self.generate_batch([prompt], max_new_tokens, temperature, top_p)
        return results[0] if results else ""

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048,
                       temperature: float = 0.7, top_p: float = 0.9) -> list[str]:
        """Generate responses for multiple prompts — all at once via vLLM."""
        if not prompts:
            return []

        params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p if temperature > 0 else 1.0,
        )

        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def get_engine(self):
        """Return the underlying vLLM engine (for advanced use)."""
        return self._llm


def patch_model_loader_with_vllm(model_loader, vllm_backend: VLLMBackend):
    """Monkey-patch a ModelLoader to use vLLM for generation while keeping
    the HF model for LoRA training/merging/activation capture.

    This is the cleanest approach: diagnostics, generation, and verification
    all call model_loader.generate/generate_batch, so patching those two
    methods speeds up ALL phases with zero code changes elsewhere.
    """
    model_loader._vllm = vllm_backend
    model_loader._hf_generate = model_loader.generate
    model_loader._hf_generate_batch = model_loader.generate_batch

    # Replace methods directly — these are simple function objects bound to the instance.
    # We capture vllm_backend in the closure.
    def vllm_generate(self, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        return vllm_backend.generate(prompt, max_new_tokens, temperature, top_p)

    def vllm_generate_batch(self, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        return vllm_backend.generate_batch(prompts, max_new_tokens, temperature, top_p)

    import types
    model_loader.generate = types.MethodType(vllm_generate, model_loader)
    model_loader.generate_batch = types.MethodType(vllm_generate_batch, model_loader)

    logger.info("ModelLoader patched to use vLLM for all generation")


def unpatch_model_loader(model_loader):
    """Restore original HF generate methods (needed during LoRA training)."""
    if hasattr(model_loader, '_hf_generate'):
        model_loader.generate = model_loader._hf_generate
        model_loader.generate_batch = model_loader._hf_generate_batch
        logger.info("ModelLoader restored to HuggingFace generation (for training)")
