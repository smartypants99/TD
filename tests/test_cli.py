import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from timedilate.cli import main, parse_budget


def test_parse_budget_seconds():
    assert parse_budget("5s") == 5.0
    assert parse_budget("30s") == 30.0


def test_parse_budget_minutes():
    assert parse_budget("5m") == 300.0


def test_parse_budget_hours():
    assert parse_budget("2h") == 7200.0


def test_parse_budget_bare_number():
    assert parse_budget("10") == 10.0


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "budget" in result.output
    assert "factor" in result.output


def test_cli_parses_args():
    runner = CliRunner()
    with patch("timedilate.cli.run_dilation") as mock_run:
        mock_metrics = MagicMock()
        mock_metrics.improvement_rate = 0.8
        mock_metrics.total_improvement = 35
        mock_run.return_value = MagicMock(
            output="result",
            score=85,
            cycles_completed=5,
            elapsed_seconds=2.5,
            convergence_detected=False,
            interrupted=False,
            metrics=mock_metrics,
        )
        result = runner.invoke(main, ["Write hello", "--factor", "5", "--budget", "10s"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
