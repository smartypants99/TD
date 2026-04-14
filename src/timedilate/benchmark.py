"""Benchmark harness for measuring time dilation effectiveness.

Run with a real model to measure quality improvement across dilation factors.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController


BENCHMARK_PROMPTS = [
    {
        "name": "sort_function",
        "prompt": "Write a Python function that sorts a list of integers using merge sort.",
        "type": "code",
    },
    {
        "name": "fibonacci",
        "prompt": "Write a Python function that returns the nth Fibonacci number efficiently.",
        "type": "code",
    },
    {
        "name": "web_scraper",
        "prompt": "Write a Python script that scrapes a webpage and extracts all links.",
        "type": "code",
    },
    {
        "name": "climate_essay",
        "prompt": "Write a short essay explaining the greenhouse effect to a high school student.",
        "type": "prose",
    },
    {
        "name": "api_design",
        "prompt": "Design a REST API for a todo list application. Include endpoints, methods, and example request/response bodies.",
        "type": "general",
    },
]


@dataclass
class BenchmarkResult:
    prompt_name: str
    dilation_factor: int
    score: int
    peak_score: int
    cycles_completed: int
    elapsed_seconds: float
    improvement_rate: float
    output_length: int
    total_inference_calls: int = 0
    initial_score: int = 0

    @property
    def points_per_second(self) -> float:
        """Score improvement per second of compute."""
        if self.elapsed_seconds <= 0:
            return 0.0
        gain = max(0, self.score - self.initial_score)
        return gain / self.elapsed_seconds

    @property
    def points_per_inference(self) -> float:
        """Score improvement per inference call."""
        if self.total_inference_calls <= 0:
            return 0.0
        gain = max(0, self.score - self.initial_score)
        return gain / self.total_inference_calls


def run_benchmark(
    engine,
    factors: list[int] | None = None,
    prompts: list[dict] | None = None,
    output_dir: str = "benchmark_results",
) -> list[BenchmarkResult]:
    """Run benchmark across multiple dilation factors and prompts."""
    if factors is None:
        factors = [1, 2, 5, 10]
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for prompt_info in prompts:
        for factor in factors:
            config = TimeDilateConfig(
                dilation_factor=factor,
                branch_factor=min(3, factor),
            )
            controller = DilationController(config, engine)

            start = time.time()
            result = controller.run(prompt_info["prompt"])
            elapsed = time.time() - start

            initial = result.metrics.cycles[0].previous_score if result.metrics and result.metrics.cycles else 0
            bench_result = BenchmarkResult(
                prompt_name=prompt_info["name"],
                dilation_factor=factor,
                score=result.score,
                peak_score=result.metrics.peak_score if result.metrics else result.score,
                cycles_completed=result.cycles_completed,
                elapsed_seconds=elapsed,
                improvement_rate=result.metrics.improvement_rate if result.metrics else 0,
                output_length=len(result.output),
                total_inference_calls=result.metrics.total_inference_calls if result.metrics else 0,
                initial_score=initial,
            )
            results.append(bench_result)

    # Save results
    results_data = [
        {
            "prompt": r.prompt_name,
            "factor": r.dilation_factor,
            "score": r.score,
            "cycles": r.cycles_completed,
            "elapsed_s": round(r.elapsed_seconds, 2),
            "improvement_rate": round(r.improvement_rate, 3),
            "peak_score": r.peak_score,
            "output_chars": r.output_length,
            "inference_calls": r.total_inference_calls,
            "initial_score": r.initial_score,
            "points_per_second": round(r.points_per_second, 3),
            "points_per_inference": round(r.points_per_inference, 3),
        }
        for r in results
    ]
    (output_path / "results.json").write_text(json.dumps(results_data, indent=2))

    return results


def format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table."""
    lines = [
        f"{'Prompt':<20} {'Factor':>6} {'Score':>5} {'Peak':>5} {'Cycles':>6} {'Time':>8} {'Imp%':>5} {'pts/s':>6} {'pts/inf':>7}",
        "-" * 88,
    ]
    for r in results:
        lines.append(
            f"{r.prompt_name:<20} {r.dilation_factor:>5}x {r.score:>5} {r.peak_score:>5} "
            f"{r.cycles_completed:>6} {r.elapsed_seconds:>7.1f}s {r.improvement_rate:>4.0%} "
            f"{r.points_per_second:>5.1f} {r.points_per_inference:>6.2f}"
        )
    return "\n".join(lines)
