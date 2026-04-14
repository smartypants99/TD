import time
from dataclasses import dataclass
from timedilate.config import TimeDilateConfig
from timedilate.improver import ImprovementEngine
from timedilate.scorer import Scorer
from timedilate.directives import DirectiveGenerator
from timedilate.checkpoint import CheckpointManager


@dataclass
class DilationResult:
    output: str
    score: int
    cycles_completed: int
    elapsed_seconds: float
    convergence_detected: bool
    interrupted: bool = False


class DilationController:
    def __init__(self, config: TimeDilateConfig, engine):
        self.config = config
        self.engine = engine
        self.improver = ImprovementEngine(engine, config)
        self.scorer = Scorer()
        self.directives = DirectiveGenerator()
        self.checkpoint = CheckpointManager(config.checkpoint_dir)

    def run(self, prompt: str, on_cycle=None) -> DilationResult:
        start = time.time()
        task_type = self.directives.classify_task(prompt)
        refinement_cycles = self.config.dilation_factor - 1

        # Initial generation
        current_best = self.engine.generate(prompt)

        if refinement_cycles <= 0:
            return DilationResult(
                output=current_best,
                score=0,
                cycles_completed=0,
                elapsed_seconds=time.time() - start,
                convergence_detected=False,
            )

        # Score the initial output
        score_prompt = self.scorer.build_scoring_prompt(prompt, current_best)
        raw_score = self.engine.generate(
            score_prompt, temperature=self.config.scoring_temperature
        )
        current_score = self.scorer.parse_score(raw_score)

        no_improvement_count = 0
        convergence_detected = False
        built_in_exhausted = False
        directive_offset = 0
        built_in_count = len(self.directives.get_directives(task_type))

        try:
            for cycle in range(refinement_cycles):
                # Choose directive: built-in or self-generated
                if built_in_exhausted:
                    custom_prompt = self.directives.generate_custom_directive_prompt(
                        task_type, prompt, current_best
                    )
                    directive = self.engine.generate(custom_prompt)
                else:
                    directive = self.directives.next_directive(
                        task_type, cycle + directive_offset
                    )

                new_best, new_score = self.improver.run_cycle(
                    original_prompt=prompt,
                    current_best=current_best,
                    current_score=current_score,
                    directive=directive,
                )

                if new_score > current_score:
                    current_best = new_best
                    current_score = new_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.config.convergence_threshold:
                    convergence_detected = True
                    if not built_in_exhausted:
                        directive_offset += built_in_count
                        # If we've cycled through all built-ins twice, switch to self-generated
                        if directive_offset >= built_in_count * 2:
                            built_in_exhausted = True
                    no_improvement_count = 0

                if (cycle + 1) % self.config.checkpoint_interval == 0:
                    self.checkpoint.save(cycle + 1, current_best, current_score)

                if on_cycle:
                    elapsed = time.time() - start
                    on_cycle(cycle + 1, refinement_cycles, current_score, elapsed)

        except KeyboardInterrupt:
            return DilationResult(
                output=current_best,
                score=current_score,
                cycles_completed=cycle + 1 if "cycle" in dir() else 0,
                elapsed_seconds=time.time() - start,
                convergence_detected=convergence_detected,
                interrupted=True,
            )

        self.checkpoint.cleanup()

        return DilationResult(
            output=current_best,
            score=current_score,
            cycles_completed=refinement_cycles,
            elapsed_seconds=time.time() - start,
            convergence_detected=convergence_detected,
        )
