"""Where does the time between fights go? Reads a run's episodes.jsonl.

    python scripts/reset_timing.py [checkpoints/runs/<run>/]   (default: newest run)
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import statistics as st
import sys


def main() -> int:
    if len(sys.argv) > 1:
        d = sys.argv[1].rstrip("/\\") + "/"
    else:
        runs = sorted(glob.glob("checkpoints/runs/*/"), key=os.path.getmtime)
        if not runs:
            print("no runs recorded yet")
            return 1
        d = runs[-1]
    ev = [json.loads(l) for l in open(d + "episodes.jsonl", encoding="utf-8") if l.strip()]
    rt = [e for e in ev if e["type"] == "reset_timing"]
    eps = [e for e in ev if e["type"] == "episode"]
    sl = [e for e in ev if e["type"] == "sleep"]
    print(f"run {d}: {len(eps)} episodes, {len(rt)} timed resets, {len(sl)} nights")
    if rt:
        keys = [k for k in rt[0] if k not in ("t", "type", "episode")]
        print("\nbetween-fight phases, median seconds (max):")
        for k in keys:
            vals = [e[k] for e in rt]
            print(f"  {k:24} {st.median(vals):6.1f}  ({max(vals):5.1f})")
        print(f"  {'sum of phases':24} {st.median(sum(e[k] for k in keys) for e in rt):6.1f}")
    if len(eps) > 2:
        ts = [dt.datetime.fromisoformat(e["t"]) for e in eps]
        gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
        steps = [e["steps"] for e in eps]
        print(f"\nwall per episode: median {st.median(gaps):.1f}s; fight steps median {st.median(steps):.0f} "
              f"(= {st.median(steps) * 0.1:.1f}s of game at 1x)")
    if sl:
        print(f"sleep (runs in parallel with the reset): median {st.median(e['duration_s'] for e in sl):.1f}s, "
              f"passes {st.median(e['passes'] for e in sl):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
