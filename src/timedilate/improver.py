from timedilate.config import TimeDilateConfig
from timedilate.scorer import Scorer


class ImprovementEngine:
    def __init__(self, engine, config: TimeDilateConfig):
        self.engine = engine
        self.config = config
        self.scorer = Scorer()

    def _build_improvement_prompt(
        self, original_prompt: str, current_best: str, directive: str
    ) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution:\n{current_best}\n\n"
            f"Improvement directive: {directive}\n\n"
            f"Produce an improved version of the solution. "
            f"Output ONLY the improved solution, nothing else."
        )

    def _maybe_summarize(self, text: str, original_prompt: str) -> str:
        """If text exceeds 75% of context window, ask model to summarize."""
        token_est = self.engine.estimate_tokens(text)
        limit = int(self.config.context_window * 0.75)
        if token_est <= limit:
            return text
        summary_prompt = (
            f"The following is a solution to this task: {original_prompt}\n\n"
            f"Solution:\n{text}\n\n"
            f"The solution is very long. Produce a condensed version that preserves "
            f"all key functionality and correctness. Output ONLY the condensed solution."
        )
        return self.engine.generate(summary_prompt)

    def run_cycle(
        self,
        original_prompt: str,
        current_best: str,
        current_score: int,
        directive: str,
    ) -> tuple[str, int]:
        current_best = self._maybe_summarize(current_best, original_prompt)

        variants = []
        for _ in range(self.config.branch_factor):
            prompt = self._build_improvement_prompt(
                original_prompt, current_best, directive
            )
            variant = self.engine.generate(prompt)
            variants.append(variant)

        best_variant = current_best
        best_score = current_score

        for variant in variants:
            score_prompt = self.scorer.build_scoring_prompt(original_prompt, variant)
            raw_score = self.engine.generate(
                score_prompt, temperature=self.config.scoring_temperature
            )
            score = self.scorer.parse_score(raw_score)
            if score > best_score:
                best_variant = variant
                best_score = score

        return best_variant, best_score
