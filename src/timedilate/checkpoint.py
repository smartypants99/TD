import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, cycle: int, output: str, score: int) -> None:
        path = self.dir / f"cycle_{cycle:06d}.json"
        path.write_text(json.dumps({
            "cycle": cycle,
            "output": output,
            "score": score,
        }))

    def load_latest(self) -> dict | None:
        files = sorted(self.dir.glob("cycle_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())

    def cleanup(self) -> None:
        for f in self.dir.glob("cycle_*.json"):
            f.unlink()
