"""Record the visualizer's live event stream and topology to static files, for a replay
that needs no server (GitHub Pages).

    python scripts/record_stream.py http://192.168.2.118:8080 docs/replay --seconds 120

Writes <out>/topology.json (the /api/topology payload, with real MaleCNS coordinates and
skeletons) and <out>/events.jsonl (one SSE event per line with a relative timestamp), plus
<out>/frames/NNNNN.jpg for the live capture if the server serves /api/frame.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--frames", action="store_true", help="also save the live-capture JPEGs (2 per second)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topo = urllib.request.urlopen(args.base.rstrip("/") + "/api/topology", timeout=60).read()
    (out / "topology.json").write_bytes(topo)
    print(f"topology {len(topo) / 1e6:.1f} MB")
    t0 = time.time()
    n = 0
    frames = 0
    last_frame = 0.0
    with urllib.request.urlopen(args.base.rstrip("/") + "/api/stream", timeout=60) as resp, \
            open(out / "events.jsonl", "w", encoding="utf-8") as fh:
        buf = ""
        while time.time() - t0 < args.seconds:
            line = resp.readline().decode("utf-8", errors="replace")
            if not line:
                break
            line = line.rstrip("\n")
            if line.startswith("data:"):
                buf += line[5:].strip()
            elif line == "" and buf:
                try:
                    ev = json.loads(buf)
                    fh.write(json.dumps({"t": round(time.time() - t0, 3), "e": ev}) + "\n")
                    n += 1
                except json.JSONDecodeError:
                    pass
                buf = ""
                if args.frames and time.time() - last_frame >= 0.5:
                    try:
                        jpg = urllib.request.urlopen(args.base.rstrip("/") + "/api/frame.jpg", timeout=5).read()
                        (out / "frames").mkdir(exist_ok=True)
                        (out / "frames" / f"{frames:05d}.jpg").write_bytes(jpg)
                        frames += 1
                        last_frame = time.time()
                    except Exception:
                        pass
    (out / "manifest.json").write_text(json.dumps({"events": n, "frames": frames, "seconds": round(time.time() - t0, 1)}))
    print(f"recorded {n} events, {frames} frames in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
