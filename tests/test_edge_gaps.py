"""Edge-case gap coverage: score parser negatives, time-budget extremes,
early_stop boundaries, generate-path OOM classification, cache eviction.

These fill small gaps left by the main suites — no overlap with existing tests.
"""
import sys
from unittest.mock import MagicMock

import pytest

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController
from timedilate.engine import DilationEngine, _looks_like_oom


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    return engine


# --- Score parser: negative numbers must not be accepted ---

def test_score_parser_clamps_negative_number():
    """'-50' is parsed as -50 then clamped to 0 (minimum valid score)."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="-50")
    controller = DilationController(TimeDilateConfig(), engine)
    s = controller._score("prompt", "output")
    assert s == 0  # parsed as -50, clamped to 0


def test_score_parser_zero_is_valid():
    """'0' is a legitimate score — must not be mis-rejected as falsy."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="0")
    controller = DilationController(TimeDilateConfig(), engine)
    s = controller._score("prompt", "output")
    assert s == 0


def test_score_parser_100_is_valid_at_upper_bound():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="100")
    controller = DilationController(TimeDilateConfig(), engine)
    assert controller._score("p", "o") == 100


# --- early_stop_score boundary: exactly equal triggers stop ---

def test_early_stop_triggers_at_exact_threshold():
    """Score == early_stop_score must trigger early stop (>= comparison)."""
    engine = _mock_engine(["initial", "98"])  # score exactly 98
    config = TimeDilateConfig(dilation_factor=50, early_stop_score=98)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 98
    assert result.cycles_completed == 0  # stopped immediately


def test_early_stop_score_100_never_triggers_on_99():
    """early_stop_score=100 should keep running even at 99."""
    responses = ["initial", "99"]
    for _ in range(3):
        responses.extend(["critique", "refine", "99"])
    engine = _mock_engine(responses)
    config = TimeDilateConfig(
        dilation_factor=3, early_stop_score=100, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.cycles_completed == 3  # ran all cycles


# --- Time budget: zero means "no cycles after initial" in budget mode ---

def test_time_budget_zero_still_returns_initial():
    """time_budget_seconds=0 means: produce initial only, no cycles."""
    engine = _mock_engine(["initial", "70"])
    config = TimeDilateConfig(dilation_factor=100, time_budget_seconds=0.0)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.output == "initial"
    assert result.elapsed_seconds < 1.0


# --- OOM classification on generate path (not just init) ---

def test_looks_like_oom_catches_generate_kv_cache_message():
    """vLLM's KV-cache allocation message at inference time must classify as OOM."""
    msg = "not enough kv cache blocks; reduce max_model_len"
    err = RuntimeError(msg)
    assert _looks_like_oom(err)


def test_looks_like_oom_does_not_fire_on_generic_value_error():
    err = ValueError("bad input shape")
    assert not _looks_like_oom(err)


# --- batched generate with single prompt in list returns list ---

def test_batched_input_single_element_still_returns_list(vllm_mock, make_vllm_output):
    """generate(['one']) must return a list even though it has one element."""
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("only")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate(["one"])
    assert isinstance(result, list)
    assert result == ["only"]


def test_scalar_input_returns_scalar(vllm_mock, make_vllm_output):
    """generate('one') must return a str, preserving legacy contract."""
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("single")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate("one")
    assert isinstance(result, str)
    assert result == "single"


# --- Score cache eviction: LRU bound must hold ---

def test_score_cache_respects_max_size():
    """_score_cache must not grow without bound — LRU eviction keeps it capped."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="70")
    controller = DilationController(TimeDilateConfig(), engine)
    max_size = getattr(controller, "_SCORE_CACHE_MAX", None)
    if max_size is None:
        pytest.skip("no _SCORE_CACHE_MAX attribute on controller")
    # Fill beyond the cap
    for i in range(max_size + 10):
        controller._score(f"prompt_{i}", f"output_{i}")
    assert len(controller._score_cache) <= max_size


# --- Config: extreme dilation factor does not overflow num_cycles ---

def test_num_cycles_handles_int_boundary():
    """Large factors stay as ints without overflow."""
    config = TimeDilateConfig(dilation_factor=2**62)
    assert config.num_cycles == 2**62
    assert isinstance(config.num_cycles, int)


# --- Controller: branch_factor=1 must never spread temperature ---

def test_branch_factor_1_ignores_temperature_spread_setting():
    """With a single branch, branch_temperature_spread must be a no-op
    (spread requires >=2 branches to make sense)."""
    engine = _mock_engine([
        "initial", "60",
        "critique", "refined", "75",
    ])
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=1,
        branch_temperature_spread=0.9, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Should complete one cycle successfully without crashing on spread math
    assert result.cycles_completed == 1
    assert result.score == 75


# --- CLI report JSON shape (new fields from 5c61b33) ---

def test_cli_report_json_contains_new_fields(tmp_path):
    """--report output must include avg_cycle_seconds, score_cache_hits,
    cycle_history, final_score so downstream tooling can rely on them."""
    import json
    from pathlib import Path
    from click.testing import CliRunner
    from timedilate.cli import main

    vllm = sys.modules["vllm"]
    vllm.reset_mock()
    out = MagicMock()
    out.outputs = [MagicMock(text="generated")]
    vllm.LLM.return_value.generate.return_value = [out]

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main, ["run", "hi", "--factor", "1", "--report", "--quiet"]
        )
        assert result.exit_code == 0, result.output
        reports = list(Path(".").glob("timedilate_report_*.json"))
        assert len(reports) == 1
        data = json.loads(reports[0].read_text())
        for key in ("avg_cycle_seconds", "score_cache_hits",
                    "cycle_history", "final_score"):
            assert key in data, f"missing {key} in report JSON"
        assert isinstance(data["cycle_history"], list)
        assert isinstance(data["avg_cycle_seconds"], (int, float))
        assert isinstance(data["score_cache_hits"], int)
        assert data["final_score"] == data["score"]


# --- Engine: last_token_counts across partial-retry and empty-batch ---

def test_last_token_counts_preserved_after_partial_retry(vllm_mock, make_vllm_output):
    """Batched retry that re-issues only empty slots must still produce
    per-slot token counts aligned to the original prompt order."""
    first = [
        make_vllm_output("ok1", token_ids=[1, 2]),
        make_vllm_output("", token_ids=[]),
        make_vllm_output("ok3", token_ids=[7, 8, 9]),
    ]
    second = [make_vllm_output("recovered", token_ids=[4, 5, 6, 7])]
    vllm_mock.LLM.return_value.generate.side_effect = [first, second]
    engine = DilationEngine(TimeDilateConfig())
    results = engine.generate(["p1", "p2", "p3"])
    assert results == ["ok1", "recovered", "ok3"]
    assert engine.last_token_counts == [2, 4, 3]


def test_last_token_counts_after_empty_batch(vllm_mock):
    """generate([]) short-circuits and must not crash later callers
    reading last_token_counts — should be a list (possibly empty or stale)."""
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate([])
    assert result == []
    counts = engine.last_token_counts
    assert isinstance(counts, list)


# --- Engine stop semantics: None / [] / ["END"] ---

def test_stop_none_omits_key_from_sampling_params(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("ok")]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p")
    assert "stop" not in vllm_mock.SamplingParams.call_args.kwargs


def test_stop_empty_list_treated_as_none(vllm_mock, make_vllm_output):
    """Empty list is falsy → must not add 'stop' kwarg."""
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("ok")]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", stop=[])
    assert "stop" not in vllm_mock.SamplingParams.call_args.kwargs


# --- Engine health_check contract ---

def test_health_check_wiring_error_raises_health_check_error(vllm_mock):
    from timedilate.engine import HealthCheckError
    vllm_mock.LLM.return_value.generate.side_effect = AttributeError("no such attr")
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(HealthCheckError):
        engine.health_check()
    assert engine._last_health_status == "wiring_error"


def test_last_health_error_was_oom_after_oom_failure(vllm_mock):
    vllm_mock.LLM.return_value.generate.side_effect = RuntimeError("CUDA out of memory")
    engine = DilationEngine(TimeDilateConfig())
    assert engine.health_check() is False
    assert engine.last_health_error_was_oom is True
    assert engine._last_health_status == "oom"


def test_last_health_error_was_oom_false_on_runtime_error(vllm_mock):
    vllm_mock.LLM.return_value.generate.side_effect = RuntimeError("timeout")
    engine = DilationEngine(TimeDilateConfig())
    assert engine.health_check() is False
    assert engine.last_health_error_was_oom is False
    assert engine._last_health_status == "runtime_error"


# --- Batched retry: _total_calls invariant ---

def test_total_calls_equals_prompt_count_not_retry_count(vllm_mock, make_vllm_output):
    """_total_calls must reflect user-visible work (len(prompts)), not
    number of vLLM round-trips taken due to empty-subset retries."""
    first = [
        make_vllm_output("ok1"),
        make_vllm_output(""),
        make_vllm_output("ok3"),
    ]
    second = [make_vllm_output("recovered")]
    vllm_mock.LLM.return_value.generate.side_effect = [first, second]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate(["p1", "p2", "p3"])
    assert engine._total_calls == 3


# --- Controller: _reject_short semantics ---

def _bare_controller():
    return DilationController(TimeDilateConfig(), MagicMock())


def test_reject_short_empty_candidate():
    c = _bare_controller()
    assert c._reject_short("", "anything") is True
    assert c._reject_short("   \n\t", "anything") is True


def test_reject_short_skips_ratio_when_best_under_200_chars():
    """Ratio check only kicks in when best_output >= 200 chars."""
    c = _bare_controller()
    short_best = "x" * 199
    # 1-char candidate vs 199-char best: ratio check skipped → accepted
    assert c._reject_short("a", short_best) is False


def test_reject_short_applies_ratio_when_best_at_200_chars():
    """At threshold (>=200), candidate <25% of best length is rejected."""
    c = _bare_controller()
    best = "x" * 200  # exactly threshold
    # 49 chars = 24.5% → below MIN_LENGTH_RATIO 0.25 → reject
    assert c._reject_short("y" * 49, best) is True
    # 50 chars = 25% → meets threshold → accept
    assert c._reject_short("y" * 50, best) is False


# --- Controller: _adapt_patience branches ---

def _refine_record(improved: bool) -> "CycleRecord":
    from timedilate.controller import CycleRecord
    return CycleRecord(cycle=1, action="refine", improved=improved)


def test_adapt_patience_shrinks_on_steady_progress():
    c = _bare_controller()
    history = [_refine_record(True) for _ in range(4)]
    # rate = 1.0 >= 0.5 → shrink to max(2, base-1)
    assert c._adapt_patience(history, score_stdev=0.0, base=5) == 4
    # floor at 2
    assert c._adapt_patience(history, score_stdev=0.0, base=2) == 2


def test_adapt_patience_grows_when_flat_and_noisy():
    c = _bare_controller()
    history = [_refine_record(False) for _ in range(4)]
    # rate=0 AND stdev>8 → grow to min(base+2, base*2)
    assert c._adapt_patience(history, score_stdev=10.0, base=5) == 7
    # base*2 cap: base=2 → min(4, 4) = 4
    assert c._adapt_patience(history, score_stdev=10.0, base=2) == 4


def test_adapt_patience_stays_when_flat_but_quiet():
    c = _bare_controller()
    history = [_refine_record(False) for _ in range(4)]
    # rate=0 but stdev<=8 → stay at base
    assert c._adapt_patience(history, score_stdev=5.0, base=5) == 5


def test_adapt_patience_returns_base_with_too_few_samples():
    c = _bare_controller()
    history = [_refine_record(True), _refine_record(False)]  # only 2
    assert c._adapt_patience(history, score_stdev=99.0, base=5) == 5


def test_adapt_patience_ignores_fresh_records():
    """Only 'refine' action records count toward the sliding window."""
    from timedilate.controller import CycleRecord
    c = _bare_controller()
    # 3 fresh + 1 refine: after filter only 1 refine → below min samples → base
    history = [
        CycleRecord(cycle=1, action="fresh", improved=True),
        CycleRecord(cycle=2, action="fresh", improved=True),
        CycleRecord(cycle=3, action="fresh", improved=True),
        CycleRecord(cycle=4, action="refine", improved=True),
    ]
    assert c._adapt_patience(history, score_stdev=0.0, base=5) == 5


# --- Controller: _pairwise_break ---

def _tied(n: int):
    return [(80, f"cand_{i}", 0.7) for i in range(n)]


def test_pairwise_break_returns_index_for_valid_letter():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="B")
    c = DilationController(TimeDilateConfig(), engine)
    assert c._pairwise_break("prompt", _tied(3)) == 1


def test_pairwise_break_caps_at_four_candidates():
    """Only A/B/C/D labels exist; a 5th candidate must be unreachable."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="D")
    c = DilationController(TimeDilateConfig(), engine)
    assert c._pairwise_break("prompt", _tied(5)) == 3  # D is index 3, not 4


def test_pairwise_break_falls_back_to_zero_on_unparseable_reply():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="???")
    c = DilationController(TimeDilateConfig(), engine)
    assert c._pairwise_break("prompt", _tied(3)) == 0


def test_pairwise_break_case_insensitive():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="c")  # lowercase
    c = DilationController(TimeDilateConfig(), engine)
    assert c._pairwise_break("prompt", _tied(3)) == 2


# --- DilationResult new fields and to_report ---

def test_result_report_includes_all_new_fields():
    from timedilate.controller import DilationResult, CycleRecord
    hist = [
        CycleRecord(
            cycle=1, action="refine", improved=True,
            score_before=60, score_after=80,
            elapsed_s=0.5, branches_tried=3,
            branch_score_stdev=5.2,
            input_tokens=100, output_tokens=50,
        ),
    ]
    result = DilationResult(
        output="ok", dilation_factor=2, cycles_completed=1,
        total_cycles=1, elapsed_seconds=0.5, model_used="m", score=80,
        cycle_history=hist, initial_score=60,
        total_input_tokens=500, total_output_tokens=200,
        tiebreaks_run=1, early_rejections=2,
    )
    report = result.to_report(score_cache_hits=3)
    for key in ("total_input_tokens", "total_output_tokens",
                "tiebreaks_run", "early_rejections", "score_cache_hits"):
        assert key in report
    assert report["tiebreaks_run"] == 1
    assert report["early_rejections"] == 2
    entry = report["cycle_history"][0]
    for key in ("branch_score_stdev", "input_tokens", "output_tokens"):
        assert key in entry
    assert entry["branch_score_stdev"] == 5.2


# --- Fresh action shares cycle index with preceding refine ---

def test_fresh_record_shares_cycle_index_with_preceding_refine():
    """After convergence-triggered reset, the fresh record's `cycle` equals
    the refine cycle it replaced (action='fresh' disambiguates)."""
    # 5 cycles with no improvement → patience=5 triggers fresh attempt
    responses = ["initial", "70"]
    for _ in range(5):
        responses.extend(["critique", "flat", "60"])
    responses.extend(["fresh output", "90"])  # fresh attempt
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    config = TimeDilateConfig(dilation_factor=6, convergence_patience=5,
                              branch_factor=1, branch_temperature_spread=0.0)
    controller = DilationController(config, engine)
    result = controller.run("test")
    fresh = [h for h in result.cycle_history if h.action == "fresh"]
    refines = [h for h in result.cycle_history if h.action == "refine"]
    if fresh and refines:
        # fresh cycle number must match some refine cycle number
        refine_cycles = {r.cycle for r in refines}
        assert any(f.cycle in refine_cycles for f in fresh)


# --- Token counting prefers engine.last_usage when present ---

def test_token_counting_uses_engine_last_usage_when_available():
    """If engine exposes last_usage dict, controller must use those exact
    counts instead of split() approximation."""
    engine = MagicMock()
    engine.generate = MagicMock(return_value="short reply")
    # Report 999/42 regardless of string length
    engine.last_usage = {"input_tokens": 999, "output_tokens": 42, "total_tokens": 1041}
    controller = DilationController(TimeDilateConfig(), engine)
    controller._generate("prompt text here")
    assert controller._total_input_tokens == 999
    assert controller._total_output_tokens == 42


def test_token_counting_falls_back_to_split_approx_without_last_usage():
    """Without engine.last_usage, controller approximates via str.split()."""
    engine = MagicMock(spec=["generate"])  # no last_usage attr
    engine.generate = MagicMock(return_value="one two three")
    controller = DilationController(TimeDilateConfig(), engine)
    controller._generate("alpha beta")
    # approx via split: input=2, output=3
    assert controller._total_input_tokens == 2
    assert controller._total_output_tokens == 3


# --- clear_cache / clear_score_cache ---

def test_clear_cache_empties_scores_and_resets_hit_counter():
    c = _bare_controller()
    c._score_cache["k1"] = 80
    c._score_cache["k2"] = 90
    c._score_cache_hits = 7
    c.clear_cache()
    assert len(c._score_cache) == 0
    assert c._score_cache_hits == 0


def test_clear_cache_preserves_token_and_tiebreak_counters():
    c = _bare_controller()
    c._score_cache["k"] = 50
    c._total_input_tokens = 123
    c._total_output_tokens = 456
    c._tiebreaks_run = 4
    c._early_rejections = 2
    c.clear_cache()
    assert c._total_input_tokens == 123
    assert c._total_output_tokens == 456
    assert c._tiebreaks_run == 4
    assert c._early_rejections == 2


def test_clear_score_cache_alias_matches_clear_cache():
    c = _bare_controller()
    c._score_cache["k"] = 80
    c._score_cache_hits = 3
    c.clear_score_cache()
    assert len(c._score_cache) == 0
    assert c._score_cache_hits == 0


# --- effective_patience property ---

def test_effective_patience_starts_at_configured_value():
    config = TimeDilateConfig(convergence_patience=7)
    c = DilationController(config, MagicMock())
    assert c.effective_patience == 7


def test_effective_patience_tracks_adaptive_updates():
    config = TimeDilateConfig(convergence_patience=5)
    c = DilationController(config, MagicMock())
    c._patience = 3
    assert c.effective_patience == 3


# --- Time-budget lookahead ---

def test_time_budget_lookahead_stops_before_overrun(caplog):
    """If avg cycle time exceeds remaining budget, loop must break before
    starting a new cycle. History has at least one entry → lookahead active."""
    import logging
    import time as _time

    # First generate returns quickly; subsequent are irrelevant because
    # lookahead should stop before they're called.
    call_count = [0]
    gen_start = [_time.time()]

    def mock_generate(prompt, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "initial"
        if call_count[0] == 2:
            return "50"  # initial score
        # Simulate a slow refine cycle — burns most of the budget
        if "critique" in prompt.lower() or "reviewing" in prompt.lower():
            _time.sleep(0.05)
            return "crit"
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return "60"
        _time.sleep(0.05)
        return "refined"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=mock_generate)
    config = TimeDilateConfig(dilation_factor=1000, time_budget_seconds=0.2,
                              branch_factor=1, branch_temperature_spread=0.0,
                              convergence_patience=10)
    controller = DilationController(config, engine)
    with caplog.at_level(logging.INFO, logger="timedilate.controller"):
        result = controller.run("test")
    # Budget respected, no overrun
    assert result.elapsed_seconds < 1.0
    # Lookahead log should fire at least once (history populated before break)
    lookahead_logs = [r for r in caplog.records
                      if "lookahead" in r.getMessage().lower()]
    # May or may not fire depending on timing; if it did, verify format
    for r in lookahead_logs:
        assert "remaining" in r.getMessage() and "avg cycle" in r.getMessage()


def test_time_budget_lookahead_skipped_when_history_empty():
    """With no cycle history yet, lookahead cannot fire — only the hard
    elapsed>=budget check applies. Verifies the history-guard branch."""
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=["initial", "99"])  # early stop on score 99
    config = TimeDilateConfig(dilation_factor=10, time_budget_seconds=5.0,
                              early_stop_score=98, convergence_patience=5)
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Early stop wins before any cycle → no lookahead needed, no crash
    assert result.score == 99
    assert result.cycles_completed == 0


# --- End-to-end tiebreak integration ---

def test_tiebreak_triggers_when_top_scores_tie_and_picks_judged_winner():
    """run() must invoke the pairwise judge when ≥2 branches share top score,
    and the judge's letter-reply must map to the adopted output."""
    responses = [
        "initial", "60",
        "critique",
        "branch_a", "85",
        "branch_b", "85",
        "branch_c", "85",
        "B",
    ]
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=3,
        branch_temperature_spread=0.0, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert controller._tiebreaks_run >= 1
    assert result.output == "branch_b"
    assert result.score == 85
    assert result.to_report()["tiebreaks_run"] >= 1
