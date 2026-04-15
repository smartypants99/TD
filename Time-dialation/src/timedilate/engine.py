"""Inference engine — thin wrapper around vLLM.

No acceleration tricks, no quality tradeoffs. Just runs the model
at full precision and full output. The dilation (more thinking)
happens in the controller, not here.
"""
import logging
import time
from typing import Sequence

from timedilate.config import TimeDilateConfig

logger = logging.getLogger(__name__)

# Fallback utilizations tried in order when the first init OOMs.
_OOM_FALLBACK_UTILS = (0.45, 0.30)


class InferenceError(RuntimeError):
    """Raised when inference fails after retries."""
    pass


def _looks_like_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = ("out of memory", "oom", "cuda error: out", "no memory", "cuda oom")
    return any(n in msg for n in needles) or exc.__class__.__name__ in (
        "OutOfMemoryError", "CUDAOutOfMemoryError",
    )


class DilationEngine:
    """Runs model inference. Full precision, full output, no shortcuts."""

    def __init__(self, config: TimeDilateConfig):
        self.config = config
        self._total_calls = 0
        self._total_latency = 0.0
        self._failed_calls = 0
        self._oom_retries = 0
        self._model = None
        self._initialized = False
        self._effective_gpu_util: float | None = None
        self._total_output_tokens = 0
        self._last_token_counts: list[int] = []

    def _build_llm_kwargs(self, gpu_util: float) -> dict:
        kwargs = dict(
            model=self.config.model,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_util,
            dtype=self.config.dtype,
            enforce_eager=self.config.enforce_eager,
            swap_space=self.config.swap_space_gb,
        )
        if self.config.max_model_len is not None:
            kwargs["max_model_len"] = self.config.max_model_len
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        return kwargs

    def initialize(self) -> None:
        """Initialize the vLLM model, with OOM fallback to lower gpu_memory_utilization."""
        if self._initialized:
            return

        from vllm import LLM

        utils_to_try = [self.config.gpu_memory_utilization, *_OOM_FALLBACK_UTILS]
        last_error: BaseException | None = None
        for idx, util in enumerate(utils_to_try):
            logger.info(
                "Initializing model: %s (gpu_mem_util=%.2f, max_model_len=%s, dtype=%s, "
                "enforce_eager=%s, swap=%.1fGiB, seed=%s)",
                self.config.model, util, self.config.max_model_len,
                self.config.dtype, self.config.enforce_eager,
                self.config.swap_space_gb, self.config.seed,
            )
            try:
                self._model = LLM(**self._build_llm_kwargs(util))
                self._effective_gpu_util = util
                self._initialized = True
                if idx > 0:
                    logger.warning(
                        "Model loaded after OOM fallback at gpu_mem_util=%.2f", util,
                    )
                return
            except Exception as e:
                last_error = e
                if not _looks_like_oom(e) or idx == len(utils_to_try) - 1:
                    break
                self._oom_retries += 1
                logger.warning(
                    "OOM at gpu_mem_util=%.2f, retrying at %.2f: %s",
                    util, utils_to_try[idx + 1], e,
                )
        raise InferenceError(f"Model initialization failed: {last_error}")

    def health_check(self) -> bool:
        """Tiny generation to confirm the model is live. Returns True on success."""
        try:
            self.initialize()
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=4, temperature=0.0)
            outputs = self._model.generate(["ping"], params)
            return bool(outputs and outputs[0].outputs)
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def _make_sampling_params(
        self,
        max_tokens: int | None,
        temperature: float | None,
        stop: Sequence[str] | None,
    ):
        from vllm import SamplingParams
        kwargs = dict(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if stop:
            kwargs["stop"] = list(stop)
        return SamplingParams(**kwargs)

    def generate(
        self,
        prompt: "str | Sequence[str]",
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
        stop: Sequence[str] | None = None,
    ) -> "str | list[str]":
        """Generate text from a prompt, or a list of texts from a list of prompts.

        Back-compat: passing a `str` returns a `str`. Passing a list returns a list.
        """
        self.initialize()

        batched = not isinstance(prompt, str)
        prompts: list[str] = list(prompt) if batched else [prompt]
        if batched and not prompts:
            return []

        params = self._make_sampling_params(max_tokens, temperature, stop)

        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                start = time.time()
                outputs = self._model.generate(prompts, params)
                elapsed = time.time() - start

                texts = [o.outputs[0].text for o in outputs]
                token_counts: list[int] = []
                for o in outputs:
                    out0 = o.outputs[0]
                    tids = getattr(out0, "token_ids", None)
                    try:
                        token_counts.append(len(tids) if tids is not None else 0)
                    except TypeError:
                        token_counts.append(0)
                self._last_token_counts = token_counts
                self._total_output_tokens += sum(token_counts)

                self._total_calls += len(texts)
                self._total_latency += elapsed

                empty = [i for i, t in enumerate(texts) if not t or not t.strip()]
                if empty:
                    if attempt < retries:
                        logger.warning(
                            "Empty response at indices %s on attempt %d, retrying",
                            empty, attempt + 1,
                        )
                        continue
                    raise InferenceError(
                        f"Model returned empty response after retries (indices={empty})"
                    )

                return texts if batched else texts[0]
            except InferenceError:
                raise
            except Exception as e:
                last_error = e
                self._failed_calls += 1
                if attempt < retries:
                    logger.warning("Inference failed (attempt %d/%d): %s",
                                   attempt + 1, retries + 1, e)
                    continue
        raise InferenceError(f"Inference failed after {retries + 1} attempts: {last_error}")

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "total_latency_s": round(self._total_latency, 3),
            "oom_retries": self._oom_retries,
            "effective_gpu_util": self._effective_gpu_util,
        }
