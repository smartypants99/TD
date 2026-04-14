from unittest.mock import MagicMock
from timedilate.controller import DilationController
from timedilate.config import TimeDilateConfig


def make_mock_engine(responses: list[str]):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    engine.estimate_tokens = MagicMock(return_value=100)
    return engine


def test_controller_runs_correct_number_of_cycles():
    config = TimeDilateConfig(dilation_factor=3, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial output",       # first generation
        "75",                   # score initial
        "improved v1", "90",    # cycle 1
        "improved v2", "95",    # cycle 2
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("Write hello world")
    assert result.cycles_completed == 2
    assert result.output == "improved v2"


def test_controller_dilation_1_means_no_refinement():
    config = TimeDilateConfig(dilation_factor=1, branch_factor=1)
    mock_engine = make_mock_engine(["initial output"])
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.output == "initial output"
    assert result.cycles_completed == 0


def test_controller_convergence_detection():
    config = TimeDilateConfig(dilation_factor=6, branch_factor=1, convergence_threshold=3)
    responses = ["initial output", "80"]
    # 5 cycles where nothing improves
    for _ in range(5):
        responses.extend(["same output", "70"])
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.cycles_completed == 5
    assert result.convergence_detected


def test_controller_on_cycle_callback():
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial", "80",
        "improved", "90",
    ])
    callback_calls = []

    def on_cycle(cycle, total, score, elapsed):
        callback_calls.append((cycle, total, score))

    controller = DilationController(config, mock_engine)
    controller.run("test", on_cycle=on_cycle)
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == 1
