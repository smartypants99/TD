from timedilate.config import TimeDilateConfig


class InferenceEngine:
    def __init__(self, config: TimeDilateConfig):
        from vllm import LLM
        self.config = config
        self.llm = LLM(
            model=config.model,
            speculative_model=config.draft_model,
            num_speculative_tokens=5,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4
