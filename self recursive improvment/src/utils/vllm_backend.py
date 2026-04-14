"""Optional vLLM backend for 5-10x faster inference across all phases.

Strategy: vLLM and HF model can't coexist on one GPU (both need ~16GB for 8B).
So we swap between them:
  - Inference phases (diagnose, generate, verify): vLLM loaded, HF unloaded
  - Training phase: vLLM destroyed, HF loaded, LoRA trained+merged
  - After training: HF unloaded, vLLM reloaded (with merged weights from saved checkpoint)
"""

import gc
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class VLLMModelLoader:
    """Drop-in replacement for ModelLoader that uses vLLM for inference
    and swaps to HF only for training. Has the same interface as ModelLoader."""

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90):
        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._tokenizer = None
        self._sampling_params_cls = None
        # HF model/tokenizer for training phases
        self._hf_model = None
        self._hf_tokenizer = None
        # Track the current model path (changes after checkpoint save)
        self._current_model_path = model_path

    def load(self):
        """Load vLLM engine (called by ImprovementLoop._setup)."""
        self._load_vllm()
        return self

    def _load_vllm(self):
        """Load vLLM engine."""
        from vllm import LLM, SamplingParams
        self._sampling_params_cls = SamplingParams

        logger.info(f"Loading model with vLLM: {self._current_model_path}")
        self._llm = LLM(
            model=self._current_model_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("vLLM backend ready")

    def _unload_vllm(self):
        """Destroy vLLM engine and free GPU memory."""
        if self._llm is not None:
            logger.info("Unloading vLLM to free GPU memory for training")
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()

    def _load_hf(self):
        """Load HF model for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading HF model for training: {self._current_model_path}")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self._current_model_path, trust_remote_code=True)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        target_dtype = getattr(torch, self.dtype)
        from transformers import __version__ as _tf_version
        dtype_key = "dtype" if int(_tf_version.split(".")[0]) >= 5 else "torch_dtype"
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self._current_model_path,
            device_map="auto",
            trust_remote_code=True,
            **{dtype_key: target_dtype},
        )

    def _unload_hf(self):
        """Unload HF model and free GPU memory."""
        if self._hf_model is not None:
            logger.info("Unloading HF model to free GPU memory for vLLM")
            del self._hf_model
            self._hf_model = None
            del self._hf_tokenizer
            self._hf_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Inference interface (used by diagnostics, generator, verifier) ----

    @property
    def tokenizer(self):
        return self._tokenizer or self._hf_tokenizer

    @property
    def model(self):
        """HF model (only available during training phase)."""
        if self._hf_model is not None:
            return self._hf_model
        raise RuntimeError("HF model not loaded — currently in vLLM inference mode")

    @property
    def device(self):
        if self._hf_model is not None:
            return next(self._hf_model.parameters()).device
        return torch.device("cuda:0")

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        results = self.generate_batch([prompt], max_new_tokens, temperature, top_p)
        return results[0] if results else ""

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048,
                       temperature: float = 0.7, top_p: float = 0.9) -> list[str]:
        if not prompts:
            return []

        if self._llm is not None:
            # vLLM path (fast)
            params = self._sampling_params_cls(
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                top_p=top_p if temperature > 0 else 1.0,
            )
            outputs = self._llm.generate(prompts, params)
            return [o.outputs[0].text for o in outputs]
        else:
            # HF fallback (during training phase eval)
            return self._hf_generate_batch(prompts, max_new_tokens, temperature, top_p)

    def _hf_generate_batch(self, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        """HF generate_batch fallback."""
        if not self._hf_model:
            raise RuntimeError("Neither vLLM nor HF model is loaded")
        original_side = self._hf_tokenizer.padding_side
        try:
            self._hf_tokenizer.padding_side = "left"
            inputs = self._hf_tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_model_len,
            ).to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._hf_tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                outputs = self._hf_model.generate(**inputs, **gen_kwargs)
            responses = []
            for i, output in enumerate(outputs):
                prompt_len = inputs["attention_mask"][i].sum()
                responses.append(self._hf_tokenizer.decode(
                    output[prompt_len:], skip_special_tokens=True))
            del inputs, outputs
            return responses
        finally:
            self._hf_tokenizer.padding_side = original_side

    # ---- Training swap interface ----

    def swap_to_hf_for_training(self):
        """Unload vLLM, load HF model for LoRA training."""
        self._unload_vllm()
        self._load_hf()

    def swap_to_vllm_after_training(self, checkpoint_path: str | None = None):
        """Save HF model to checkpoint, unload HF, reload vLLM with merged weights."""
        if checkpoint_path:
            self._current_model_path = checkpoint_path
        self._unload_hf()
        self._load_vllm()

    # ---- Compatibility with ModelLoader interface ----

    def capture_activations(self, layer_names=None):
        """Activation capture requires HF model — used during diagnosis.
        For vLLM mode, return a dummy context manager with empty activations."""
        from contextlib import contextmanager
        from .model_loader import ActivationCapture

        @contextmanager
        def dummy():
            yield ActivationCapture()

        if self._hf_model is not None:
            capture = ActivationCapture()
            capture.register(self._hf_model, layer_names)
            @contextmanager
            def real():
                try:
                    yield capture
                finally:
                    capture.remove_hooks()
            return real()
        else:
            return dummy()

    def get_layer_info(self) -> dict:
        """Layer info requires HF model. Return empty dict in vLLM mode."""
        if self._hf_model is not None:
            from ..utils.model_loader import _chunked_norm
            skip_suffixes = ("embed_tokens.weight", "lm_head.weight",
                             "layernorm.weight", "layer_norm.weight",
                             "norm.weight", "ln_f.weight")
            layers = {}
            for name, param in self._hf_model.named_parameters():
                if any(name.endswith(s) for s in skip_suffixes):
                    continue
                layers[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                    "norm": _chunked_norm(param.data),
                }
            return layers
        return {}

    def save_checkpoint(self, path: Path, cycle: int):
        """Save model checkpoint (only works in HF mode)."""
        if self._hf_model is None:
            logger.warning("Cannot save checkpoint — HF model not loaded")
            return
        save_path = path / f"cycle_{cycle}"
        save_path.mkdir(parents=True, exist_ok=True)
        self._hf_model.save_pretrained(save_path)
        self._hf_tokenizer.save_pretrained(save_path)
