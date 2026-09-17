"""How much does a badly aimed start actually cost?

The agent logs a warning when an episode opens with the camera off the boss or the
character's back turned. That tells you the condition happens; it does not tell you
whether it matters. This pairs each warning with the episode that followed it and
compares the outcomes, so the fix can be prioritised on its measured cost rather than on
how annoying it looks.

    python scripts/aim_impact.py [logfile]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

EPISODE_RE = re.compile(r"EPISODE #(\d+).*?Boss HP left:\s*(\d+)%.*?Steps:\s*(\d+)", re.S)
HITS_RE = re.compile(r"Hits\s*landed:\s*(\d+)")
AIM_RE = re.compile(
    r"Episode #(\d+) starts poorly aimed:.*?body\s*([0-9.]+)\s*deg off", re.S
)


def mannwhitney(a, b):
    """U test by normal approximation; scipy is not installed on the game host."""
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    ra = sum(ranks[k] for k, (_, g) in enumerate(comb) if g == 0)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 1.0
    u = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = (na * nb * (na + nb + 1) / 12) ** 0.5
    z = (u - mu) / sd if sd else 0.0
    return z, math.erfc(abs(z) / 2 ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="flysoul_run.log")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    # A warning is printed before the episode it refers to completes, so walk the log in
    # order and attach the most recent warning to the next episode result.
    pending = None
    episodes = []
    for chunk in re.split(r"(?=EPISODE #|Episode #)", text):
        aim = AIM_RE.search(chunk)
        if aim:
            pending = float(aim.group(2))
            continue
        m = EPISODE_RE.search(chunk)
        if not m:
            continue
        h = HITS_RE.search(chunk)
        episodes.append({
            "boss": int(m.group(2)),
            "steps": int(m.group(3)),
            "hits": int(h.group(1)) if h else 0,
            "aim_off": pending,
        })
        pending = None

    bad = [e for e in episodes if e["aim_off"] is not None]
    good = [e for e in episodes if e["aim_off"] is None]
    if not bad or not good:
        print(f"{len(episodes)} episodes, {len(bad)} badly aimed - need both groups to compare")
        return 0

    print(f"{len(episodes)} episodes: {len(good)} aimed well, {len(bad)} aimed badly "
          f"({len(bad) / len(episodes):.0%})\n")
    print(f"{'metric':16} {'aimed well':>12} {'aimed badly':>12} {'p':>8}")
    print("-" * 52)
    for name, key in (("boss HP left %", "boss"), ("hits landed", "hits"), ("steps", "steps")):
        a = [e[key] for e in good]
        b = [e[key] for e in bad]
        _, p = mannwhitney(a, b)
        print(f"{name:16} {statistics.mean(a):12.1f} {statistics.mean(b):12.1f} {p:8.3f}")

    back = [e for e in bad if e["aim_off"] > 90]
    if back:
        print(f"\nOf the badly aimed, {len(back)} had the back turned (>90 deg): "
              f"boss HP left {statistics.mean([e['boss'] for e in back]):.1f}%, "
              f"{statistics.mean([e['hits'] for e in back]):.1f} hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
