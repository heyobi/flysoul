"""Keep everything a run produces, in one folder per run.

A day of live fights was lost because the only record was a console log parsed with
regular expressions and a checkpoint that each night overwrote. This writes, per run:

    checkpoints/runs/<run_id>/
        run.json            flags, host, git revision, start time
        episodes.jsonl      one line per episode, timestamped, plus sleep and other events
        archive.npz         the replay archive (rolling), never shared between runs
        weights_epNNNNN.npz periodic snapshots of the learned synapses
        clip_epNNNNN_<tag>/ JPEG frames of fights worth watching (victories, near misses)

Nothing here is needed to fight; if any of it fails the fight goes on.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def _git_revision(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if hasattr(value, "item"):  # numpy scalars
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RunRecorder:
    def __init__(self, root: Path, args: Any = None, tag: str = "", fingerprint: Optional[str] = None):
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
        self.dir = Path(root) / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(root)
        self._episodes = self.dir / "episodes.jsonl"
        header = {
            "run_id": self.run_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "git": _git_revision(self.root.parent),
            "fingerprint": fingerprint,
            "args": _jsonable(vars(args)) if args is not None and hasattr(args, "__dict__") else _jsonable(args),
        }
        (self.dir / "run.json").write_text(json.dumps(header, indent=2), encoding="utf-8")

    @staticmethod
    def prior_series(root: Path, fingerprint: str) -> tuple[list, list]:
        """Boss HP and hits of every earlier fight recorded on the same circuit.

        Restarts are frequent and each starts a new run folder; the learning curve
        belongs to the circuit, not to the process, so earlier runs with the same
        wiring fingerprint are read back in chronological order.
        """
        boss, hits = [], []
        for run_dir in sorted((Path(root) / "runs").glob("*/")):
            try:
                header = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if header.get("fingerprint") != fingerprint:
                continue
            try:
                for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("type") == "episode":
                        boss.append(round(float(e.get("boss_hp_pct", 1.0)) * 100))
                        hits.append(int(e.get("hits", 0)))
            except Exception:
                continue
        return boss, hits

    # ------------------------------------------------------------------ events

    def event(self, kind: str, **fields: Any) -> None:
        line = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": kind, **_jsonable(fields)}
        try:
            with self._episodes.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except Exception:
            pass

    def episode(self, ep: int, summary: dict, **extra: Any) -> None:
        self.event("episode", episode=ep, **{k: v for k, v in summary.items() if k != "episode"}, **extra)

    # ---------------------------------------------------------------- payloads

    def snapshot_weights(self, plasticity, ep: int) -> Optional[Path]:
        path = self.dir / f"weights_ep{ep:05d}.npz"
        try:
            return path if plasticity.save(path) else None
        except Exception:
            return None

    def save_frames(self, ep: int, frames: Iterable[bytes], tag: str) -> Optional[Path]:
        frames = list(frames)
        if not frames:
            return None
        folder = self.dir / f"clip_ep{ep:05d}_{tag}"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for i, jpeg in enumerate(frames):
                (folder / f"{i:04d}.jpg").write_bytes(jpeg)
            return folder
        except Exception:
            return None

    def archive(self, sleep, ep: int) -> int:
        path = self.dir / "archive.npz"
        written = sleep.dump(path)
        if written:
            try:  # a convenience copy of the latest archive, whatever run it came from
                shutil.copyfile(path, self.root / "replay_archive.npz")
            except Exception:
                pass
        return written
