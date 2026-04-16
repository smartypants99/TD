"""Tests for DilationEvent structured event system."""
from unittest.mock import MagicMock, patch

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController, DilationEvent


def _make_controller(factor=2.0, patience=2, branch_factor=1, early_stop=100):
    config = TimeDilateConfig(
        dilation_factor=factor,
        convergence_patience=patience,
        branch_factor=branch_factor,
        early_stop_score=early_stop,
    )
    engine = MagicMock()
    engine.last_usage = None
    return config, engine


class TestDilationEventDataclass:
    def test_basic_creation(self):
        e = DilationEvent(event_type="cycle_start", cycle=1)
        assert e.event_type == "cycle_start"
        assert e.cycle == 1
        assert e.data == {}

    def test_with_data(self):
        e = DilationEvent(event_type="branch_scored", cycle=3,
                          data={"score": 85, "branch": 0})
        assert e.data["score"] == 85

    def test_all_event_types_are_strings(self):
        for t in ("cycle_start", "cycle_end", "branch_scored",
                   "fresh_attempt", "convergence_reset", "early_stop", "tiebreak"):
            e = DilationEvent(event_type=t, cycle=0)
            assert isinstance(e.event_type, str)


class TestOnEventCallback:
    def test_on_event_receives_cycle_start_and_end(self):
        """With factor=2 (1 cycle), we should get cycle_start and cycle_end."""
        config, engine = _make_controller(factor=2.0)
        # generate -> initial output, score -> "75", critique -> "ok",
        # refine -> "better", score refined -> "80"
        engine.generate.side_effect = [
            "initial",  # initial generation
            "75",       # score initial
            "critique", # critique
            "better",   # refine
            "80",       # score refined
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test prompt", on_event=lambda e: events.append(e))

        types = [e.event_type for e in events]
        assert "cycle_start" in types
        assert "cycle_end" in types

    def test_on_event_none_does_not_crash(self):
        """on_event=None (default) should work fine."""
        config, engine = _make_controller(factor=2.0)
        engine.generate.side_effect = ["initial", "75", "critique", "better", "80"]
        ctrl = DilationController(config, engine=engine)
        result = ctrl.run("test prompt")  # no on_event
        assert result.output is not None

    def test_branch_scored_events(self):
        """With branch_factor=2, should see branch_scored events."""
        config, engine = _make_controller(factor=2.0, branch_factor=2)
        engine.generate.side_effect = [
            "initial",    # initial gen
            "75",         # score initial
            "critique",   # critique
            "branch_a",   # refine branch 0
            "80",         # score branch 0
            "branch_b",   # refine branch 1
            "78",         # score branch 1
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test", on_event=lambda e: events.append(e))

        branch_events = [e for e in events if e.event_type == "branch_scored"]
        assert len(branch_events) == 2
        assert branch_events[0].data["branch"] == 0
        assert branch_events[1].data["branch"] == 1

    def test_early_stop_event(self):
        """early_stop event fires when score >= threshold."""
        config, engine = _make_controller(factor=3.0, early_stop=75)
        engine.generate.side_effect = [
            "initial",    # initial gen
            "60",         # score initial
            "critique",   # critique
            "better",     # refine
            "80",         # score refined -> 80 >= 75, next cycle triggers early_stop
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test", on_event=lambda e: events.append(e))

        early = [e for e in events if e.event_type == "early_stop"]
        assert len(early) == 1
        assert early[0].data["score"] >= 75

    def test_tiebreak_event(self):
        """Tiebreak event fires when branches tie."""
        config, engine = _make_controller(factor=2.0, branch_factor=2)
        engine.generate.side_effect = [
            "initial",    # initial gen
            "60",         # score initial
            "critique",   # critique
            "branch_a",   # refine branch 0
            "70",         # score branch 0
            "branch_b",   # refine branch 1
            "70",         # score branch 1 (tie!)
            "A",          # pairwise tiebreak
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test", on_event=lambda e: events.append(e))

        tie_events = [e for e in events if e.event_type == "tiebreak"]
        assert len(tie_events) == 1
        assert tie_events[0].data["num_tied"] == 2

    def test_fresh_attempt_and_convergence_reset(self):
        """After patience exhausted, fresh_attempt + convergence_reset fire."""
        config, engine = _make_controller(factor=5.0, patience=1)
        engine.generate.side_effect = [
            "initial",    # initial gen
            "60",         # score initial
            # cycle 1: no improvement -> patience hit
            "critique",   # critique
            "same",       # refine
            "60",         # score (no improve, patience=1 -> converge)
            "fresh_out",  # fresh attempt
            "65",         # score fresh
            # cycle 2: improved, triggers early cycle end
            "critique2",  # critique
            "better2",    # refine
            "70",         # score
            # cycle 3: no improvement again
            "critique3",
            "same3",
            "70",         # no improve -> patience hit again
            "fresh2",     # fresh
            "72",         # score fresh
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test", on_event=lambda e: events.append(e))

        fresh_events = [e for e in events if e.event_type == "fresh_attempt"]
        reset_events = [e for e in events if e.event_type == "convergence_reset"]
        assert len(fresh_events) >= 1
        assert len(reset_events) >= 1
        assert "improved" in fresh_events[0].data

    def test_on_cycle_still_works_with_on_event(self):
        """Both on_cycle and on_event can coexist."""
        config, engine = _make_controller(factor=2.0)
        engine.generate.side_effect = ["initial", "75", "critique", "better", "80"]

        cycle_calls = []
        event_calls = []

        ctrl = DilationController(config, engine=engine)
        ctrl.run(
            "test",
            on_cycle=lambda c, t, s, e: cycle_calls.append(c),
            on_event=lambda e: event_calls.append(e),
        )
        assert len(cycle_calls) >= 1
        assert len(event_calls) >= 1

    def test_event_cycle_numbers_match(self):
        """cycle field on events should match the loop cycle."""
        config, engine = _make_controller(factor=3.0)
        engine.generate.side_effect = [
            "initial", "60",
            "c1", "r1", "65",
            "c2", "r2", "70",
        ]
        events = []
        ctrl = DilationController(config, engine=engine)
        ctrl.run("test", on_event=lambda e: events.append(e))

        starts = [e for e in events if e.event_type == "cycle_start"]
        for i, s in enumerate(starts):
            assert s.cycle == i + 1


class TestDilationEventExport:
    def test_importable_from_package(self):
        from timedilate import DilationEvent
        e = DilationEvent(event_type="test", cycle=0)
        assert e.event_type == "test"
