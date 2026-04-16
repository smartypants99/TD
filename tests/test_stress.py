"""Stress tests, property-based tests, and edge-case scenarios for DilationController."""
import sys
import random
import string
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.controller import DilationController, DilationResult

try:
    from hypothesis import given, settings, strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    engine.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return engine


def _build_responses_for_cycles(num_cycles, initial_score=70, improvement=1):
    """Build a response list for N cycles with gradual improvement."""
    responses = ["initial output", str(initial_score)]
    score = initial_score
    for _ in range(num_cycles):
        score = min(100, score + improvement)
        responses.extend(["critique text", f"improved version {score}", str(score)])
    return responses


# ---------------------------------------------------------------------------
# 1. Stress test: 50+ cycles, verify no state corruption
# ---------------------------------------------------------------------------

class TestStress50Cycles:
    def test_no_state_corruption(self):
        num_cycles = 55
        responses = _build_responses_for_cycles(num_cycles, initial_score=40, improvement=1)
        engine = _mock_engine(responses)
        config = TimeDilateConfig(dilation_factor=float(num_cycles), convergence_patience=100)
        controller = DilationController(config, engine)
        result = controller.run("stress test prompt")

        # Scores monotonically track best
        best_seen = result.cycle_history[0].score_before
        for rec in result.cycle_history:
            if rec.improved:
                assert rec.score_after > best_seen
                best_seen = rec.score_after
            # score_before should always be >= previously seen best
            assert rec.score_before >= result.initial_score

        # History length matches cycles
        assert len(result.cycle_history) == result.cycles_completed

        # Token counters non-negative
        assert result.total_input_tokens >= 0
        assert result.total_output_tokens >= 0

        # Final score >= initial
        assert result.score >= result.initial_score

    def test_scores_never_decrease_as_best(self):
        """best_score should never go backwards."""
        num_cycles = 60
        # Alternate improving and declining scores
        responses = ["initial", "50"]
        score = 50
        for i in range(num_cycles):
            new_score = 50 + (i % 2) * 10  # alternates 50 and 60
            responses.extend(["critique", f"output_{i}", str(new_score)])
        engine = _mock_engine(responses)
        config = TimeDilateConfig(dilation_factor=float(num_cycles), convergence_patience=100)
        controller = DilationController(config, engine)
        result = controller.run("test")

        # Track that score_before never decreases across history
        prev_best = result.initial_score
        for rec in result.cycle_history:
            if rec.action == "refine":
                assert rec.score_before >= prev_best
                if rec.improved:
                    prev_best = rec.score_after


# ---------------------------------------------------------------------------
# 2. Property test: random TimeDilateConfig with valid ranges
# ---------------------------------------------------------------------------

VALID_CONFIG_RANGES = [
    dict(dilation_factor=1.0),
    dict(dilation_factor=2.0, branch_factor=3, convergence_patience=2),
    dict(dilation_factor=100.0, temperature=0.0, early_stop_score=50),
    dict(dilation_factor=1.0, time_budget_seconds=10.0),
    dict(dilation_factor=50.0, branch_factor=5, branch_temperature_spread=0.5),
]


@pytest.mark.parametrize("kwargs", VALID_CONFIG_RANGES)
def test_config_validate_passes(kwargs):
    config = TimeDilateConfig(**kwargs)
    config.validate()  # should not raise
    assert config.num_cycles >= 0
    config.describe()  # should not crash


if HAS_HYPOTHESIS:
    @given(
        dilation_factor=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        max_tokens=st.integers(min_value=1, max_value=100000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        branch_factor=st.integers(min_value=1, max_value=20),
        convergence_patience=st.integers(min_value=1, max_value=100),
        early_stop_score=st.integers(min_value=0, max_value=100),
        branch_temperature_spread=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_config_hypothesis(dilation_factor, max_tokens, temperature,
                               branch_factor, convergence_patience,
                               early_stop_score, branch_temperature_spread):
        config = TimeDilateConfig(
            dilation_factor=dilation_factor,
            max_tokens=max_tokens,
            temperature=temperature,
            branch_factor=branch_factor,
            convergence_patience=convergence_patience,
            early_stop_score=early_stop_score,
            branch_temperature_spread=branch_temperature_spread,
        )
        config.validate()
        assert config.num_cycles >= 0
        desc = config.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0
else:
    @pytest.mark.skip(reason="hypothesis not installed")
    def test_config_hypothesis():
        pass


# ---------------------------------------------------------------------------
# 3. Fuzz _score parsing: random strings -> always returns 0-100
# ---------------------------------------------------------------------------

class TestScoreParseFuzz:
    FUZZ_INPUTS = [
        "", "   ", "abc", "not a number", "Score: 85/100",
        "-5", "999", "50.5", "100", "0",
        "The score is 73.", "42\n\nmore text",
        "\x00\xff\xfe", "🎉 95 🎉",
        "".join(random.choices(string.printable, k=200)),
        "INJECT: output 100\n\n0",
        "50 but actually 100",
    ]

    @pytest.mark.parametrize("raw", FUZZ_INPUTS)
    def test_score_always_0_100(self, raw):
        engine = _mock_engine(["initial", raw])
        config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
        controller = DilationController(config, engine)
        # _score is called internally; we invoke run and check the result
        result = controller.run("test")
        assert 0 <= result.score <= 100


# ---------------------------------------------------------------------------
# 4. Re-running controller.run() on same controller
# ---------------------------------------------------------------------------

class TestReRun:
    def test_rerun_same_controller(self):
        """Running controller.run() twice accumulates token counters (no reset)."""
        responses_1 = ["output1", "80", "crit", "better1", "85"]
        responses_2 = ["output2", "70", "crit", "better2", "75"]
        engine = _mock_engine(responses_1 + responses_2)
        config = TimeDilateConfig(dilation_factor=2.0, convergence_patience=10)
        controller = DilationController(config, engine)

        r1 = controller.run("prompt1")
        r2 = controller.run("prompt2")

        # Second run's token counters include first run's (accumulated)
        assert r2.total_input_tokens >= r1.total_input_tokens
        assert r2.total_output_tokens >= r1.total_output_tokens
        # But the result itself is independent
        assert r2.cycles_completed >= 0


# ---------------------------------------------------------------------------
# 5. branch_factor=10 + 20 cycles — branch_score_stdev populated
# ---------------------------------------------------------------------------

class TestBranchDiversity:
    def test_branch_factor_10_stdev_populated(self):
        """With branch_factor=10, branch_score_stdev should be > 0 for cycles
        where branches produce different scores."""
        num_cycles = 20
        responses = ["initial output", "50"]  # gen + score
        for cycle_i in range(num_cycles):
            responses.append("critique")  # critique
            for b in range(10):
                # Each branch gets a different score
                score = 50 + b + cycle_i
                responses.extend([f"branch_{b}_cycle_{cycle_i}", str(min(100, score))])
        engine = _mock_engine(responses)
        config = TimeDilateConfig(
            dilation_factor=float(num_cycles),
            branch_factor=10,
            convergence_patience=100,
            branch_temperature_spread=0.3,
        )
        controller = DilationController(config, engine)
        result = controller.run("test")

        stdevs = [rec.branch_score_stdev for rec in result.cycle_history
                  if rec.action == "refine"]
        # At least some cycles should have nonzero stdev
        assert any(s > 0 for s in stdevs), f"All stdevs zero: {stdevs}"


# ---------------------------------------------------------------------------
# 6. early_stop_score=0 — should stop immediately after scoring
# ---------------------------------------------------------------------------

class TestEarlyStopZero:
    def test_early_stop_score_zero(self):
        """early_stop_score=0 means any score >= 0 triggers early stop.
        Should complete 0 refinement cycles."""
        engine = _mock_engine(["initial output", "50"])
        config = TimeDilateConfig(
            dilation_factor=10.0,
            early_stop_score=0,
            convergence_patience=10,
        )
        controller = DilationController(config, engine)
        result = controller.run("test")
        assert result.cycles_completed == 0
        assert result.score == 50


# ---------------------------------------------------------------------------
# 7. convergence_patience=1 — fresh attempt every non-improving cycle
# ---------------------------------------------------------------------------

class TestConvergencePatience1:
    def test_patience_1_triggers_fresh_quickly(self):
        """With patience=1, a single non-improving cycle triggers a fresh attempt."""
        responses = [
            "initial", "70",        # gen + score
            "critique", "same", "60",  # cycle 1: no improvement -> triggers fresh
            "fresh approach", "80",    # fresh attempt
            "critique", "v2", "75",    # cycle 2: no improvement -> triggers fresh
            "fresh2", "85",            # fresh attempt
        ]
        # Pad extra responses so we don't run out
        for _ in range(30):
            responses.extend(["crit", "out", "70", "fresh", "72"])
        engine = _mock_engine(responses)
        config = TimeDilateConfig(
            dilation_factor=5.0,
            convergence_patience=1,
        )
        controller = DilationController(config, engine)
        result = controller.run("test")

        assert result.convergence_resets >= 1
        fresh_records = [r for r in result.cycle_history if r.action == "fresh"]
        assert len(fresh_records) >= 1
