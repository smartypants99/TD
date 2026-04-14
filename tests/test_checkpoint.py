import tempfile
from timedilate.checkpoint import CheckpointManager


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="hello world", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 5
        assert result["output"] == "hello world"
        assert result["score"] == 85


def test_load_latest_picks_highest_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.save(cycle=10, output="v2", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 10
        assert result["output"] == "v2"


def test_load_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        assert mgr.load_latest() is None


def test_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.cleanup()
        assert mgr.load_latest() is None
