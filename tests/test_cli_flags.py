"""CLI flag tests for flags shipped in commit 5c61b33.

Uses the shared vllm mock from conftest.py (autouse-reset between tests).
"""
from click.testing import CliRunner
from timedilate.cli import main, format_subjective_time, sparkline


# --- Hardware flags propagate to LLM kwargs ---

def test_gpu_mem_util_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("result")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--gpu-mem-util", "0.35", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert vllm_mock.LLM.call_args.kwargs["gpu_memory_utilization"] == 0.35


def test_dtype_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--dtype", "bfloat16", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert vllm_mock.LLM.call_args.kwargs["dtype"] == "bfloat16"


def test_enforce_eager_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--enforce-eager", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert vllm_mock.LLM.call_args.kwargs["enforce_eager"] is True


def test_enforce_eager_default_off(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hi", "--factor", "1", "--quiet"])
    assert result.exit_code == 0
    # Default config has enforce_eager=False; kwarg will be False (not omitted)
    assert vllm_mock.LLM.call_args.kwargs.get("enforce_eager") is False


def test_swap_space_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--swap-space", "16", "--quiet"]
    )
    assert result.exit_code == 0
    assert vllm_mock.LLM.call_args.kwargs["swap_space"] == 16


def test_max_model_len_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--max-model-len", "2048", "--quiet"]
    )
    assert result.exit_code == 0
    assert vllm_mock.LLM.call_args.kwargs["max_model_len"] == 2048


# --- Sampling flags ---

def test_seed_flag_propagates_to_llm(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--seed", "42", "--quiet"]
    )
    assert result.exit_code == 0
    assert vllm_mock.LLM.call_args.kwargs["seed"] == 42


def test_temperature_flag_propagates(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--temperature", "0.25", "--quiet"]
    )
    assert result.exit_code == 0
    assert vllm_mock.SamplingParams.call_args.kwargs["temperature"] == 0.25


def test_temperature_invalid_rejected(vllm_mock):
    """config.validate() rejects temperature > 2.0 via BadParameter."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--temperature", "3.5", "--quiet"]
    )
    assert result.exit_code != 0


# --- Dilation strategy flags: dry-run to avoid invoking engine loop ---

def test_branch_factor_flag_in_dry_run_output():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "10", "--branch-factor", "4", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "4" in result.output


def test_patience_flag_accepted_in_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "10", "--patience", "12", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "12" in result.output


def test_early_stop_flag_accepted_in_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "hi", "--factor", "10", "--early-stop", "85", "--dry-run"],
    )
    assert result.exit_code == 0


def test_no_critique_flag_accepted():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "10", "--no-critique", "--dry-run"]
    )
    assert result.exit_code == 0


def test_no_cot_flag_accepted():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "10", "--no-cot", "--dry-run"]
    )
    assert result.exit_code == 0
    # CoT line should NOT appear
    assert "Chain-of-thought" not in result.output


def test_no_critique_omits_self_critique_line():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "10", "--no-critique", "--dry-run"]
    )
    assert "Self-critique" not in result.output


def test_default_shows_critique_and_cot():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hi", "--factor", "10", "--dry-run"])
    assert result.exit_code == 0
    assert "Self-critique" in result.output
    assert "Chain-of-thought" in result.output


# --- --stream-progress flag ---

def test_stream_progress_flag_accepted(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("r")]
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--stream-progress", "--quiet"]
    )
    # factor=1 means no cycles, but flag must not crash
    assert result.exit_code == 0


# --- Invalid flag values surface as BadParameter ---

def test_invalid_gpu_mem_util_rejected():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--gpu-mem-util", "2.0", "--quiet"]
    )
    assert result.exit_code != 0


def test_invalid_dtype_rejected_by_click_choice():
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--dtype", "int8", "--quiet"]
    )
    assert result.exit_code != 0


def test_invalid_branch_factor_rejected():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "hi", "--factor", "10", "--branch-factor", "0", "--dry-run"],
    )
    assert result.exit_code != 0


# --- explain --time-budget formatting ---

def test_explain_time_budget_shows_breakdown():
    runner = CliRunner()
    result = runner.invoke(main, ["explain", "--factor", "1000", "--time-budget", "5"])
    assert result.exit_code == 0
    assert "Subjective time breakdown" in result.output
    assert "days" in result.output
    assert "weeks" in result.output
    assert "months" in result.output
    assert "years" in result.output


def test_explain_large_factor_reaches_years():
    runner = CliRunner()
    result = runner.invoke(
        main, ["explain", "--factor", "1000000", "--time-budget", "5"]
    )
    assert result.exit_code == 0
    # 5s * 1M = 5M sec ≈ 0.16 years — should show a years line
    assert "years" in result.output


def test_explain_without_time_budget_shows_cycles():
    runner = CliRunner()
    result = runner.invoke(main, ["explain", "--factor", "100"])
    assert result.exit_code == 0
    assert "reasoning cycles" in result.output
    assert "100" in result.output


# --- format_subjective_time pure-function tests ---

def test_format_subjective_time_seconds():
    assert "seconds" in format_subjective_time(30)


def test_format_subjective_time_minutes():
    assert "minute" in format_subjective_time(120)


def test_format_subjective_time_hours():
    assert "hour" in format_subjective_time(7200)


def test_format_subjective_time_days():
    assert "day" in format_subjective_time(3 * 86400)


def test_format_subjective_time_weeks():
    assert "week" in format_subjective_time(14 * 86400)


def test_format_subjective_time_months():
    assert "month" in format_subjective_time(60 * 86400)


def test_format_subjective_time_years():
    assert "year" in format_subjective_time(2 * 365 * 86400)


# --- sparkline helper ---

def test_sparkline_empty_returns_empty_string():
    assert sparkline([]) == ""


def test_sparkline_scales_within_bounds():
    result = sparkline([0, 50, 100])
    assert len(result) == 3
    # lowest value should map to lowest spark char
    assert result[0] == "▁"
    assert result[-1] == "█"


def test_sparkline_clips_out_of_range_values():
    # Values outside [lo, hi] should be clipped, not crash
    result = sparkline([-50, 200], lo=0, hi=100)
    assert len(result) == 2


# --- Combined flags: ensure CLI stitches everything together ---

def test_run_quiet_with_many_flags(vllm_mock, make_vllm_output):
    vllm_mock.LLM.return_value.generate.return_value = [make_vllm_output("output")]
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "Solve x^2=4",
        "--factor", "1",
        "--model", "test/model",
        "--dtype", "float16",
        "--enforce-eager",
        "--swap-space", "4",
        "--gpu-mem-util", "0.55",
        "--temperature", "0.3",
        "--seed", "99",
        "--max-tokens", "256",
        "--quiet",
    ])
    assert result.exit_code == 0, result.output
    kw = vllm_mock.LLM.call_args.kwargs
    assert kw["model"] == "test/model"
    assert kw["dtype"] == "float16"
    assert kw["enforce_eager"] is True
    assert kw["swap_space"] == 4
    assert kw["gpu_memory_utilization"] == 0.55
    assert kw["seed"] == 99
    sp_kw = vllm_mock.SamplingParams.call_args.kwargs
    assert sp_kw["max_tokens"] == 256
    assert sp_kw["temperature"] == 0.3
