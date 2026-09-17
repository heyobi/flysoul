"""Summarise a live FlySoul run from its log.

Reports only what the environment reported: boss health at the end of each episode,
hits landed, and what the fly spent its actions on. Split into blocks so a trend is
visible rather than inferred from the last episode, which is mostly noise.

    python scripts/progress.py [logfile] [--block 20] [--since-restart]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A victory line reads "VICTORY ... Fly HP left: 13%" and carries no boss HP, which is
# zero by definition; the pattern must not drop exactly the episodes that matter most.
EPISODE_RE = re.compile(
    r"EPISODE #(\d+).*?(?:(?:Boss HP left|Remaining Boss HP):\s*(\d+)%|VICTORY).*?"
    r"Steps:\s*(\d+).*?Reward:\s*([-+0-9.]+)",
    re.S,
)
HITS_RE = re.compile(r"Hits\s*landed:\s*(\d+)")
MIX_RE = re.compile(r"([A-Za-z]+):(\d+)%")


def parse(text: str):
    """Pull one record per episode out of the rich-wrapped console log."""
    # rich wraps lines, so an episode's text can span several physical lines.
    chunks = re.split(r"(?=EPISODE #)", text)
    episodes = []
    for chunk in chunks:
        m = EPISODE_RE.search(chunk)
        if not m:
            continue
        hits = HITS_RE.search(chunk)
        tail = chunk.split("Hits", 1)[-1]
        mix = {k: int(v) for k, v in MIX_RE.findall(tail)}
        episodes.append({
            "ep": int(m.group(1)),
            "boss": int(m.group(2)) if m.group(2) else 0,
            "steps": int(m.group(3)),
            "reward": float(m.group(4)),
            "hits": int(hits.group(1)) if hits else 0,
            "victory": "VICTORY" in chunk,
            "mix": mix,
        })
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="flysoul_run.log")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--since-restart", action="store_true",
                    help="Only episodes after the last agent restart.")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"no log at {path}")
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    if args.since_restart:
        marker = max(text.rfind("Running in continuous combat loop"), 0)
        text = text[marker:]

    eps = parse(text)
    # The agent restarts its episode numbering, so a log spanning restarts contains
    # several #1s. Renumber sequentially or the block labels come out as "ep 181-11".
    for i, e in enumerate(eps, start=1):
        e["ep"] = i
    if not eps:
        print("no completed episodes yet")
        return 0

    wins = sum(e["victory"] for e in eps)
    best = min(e["boss"] for e in eps)
    print(f"{len(eps)} episodes | {wins} victories | best boss HP left: {best}%\n")

    b = args.block
    print(f"{'block':>12} {'boss HP':>8} {'hits':>6} {'steps':>7}  top actions")
    print("-" * 74)
    for i in range(0, len(eps), b):
        blk = eps[i:i + b]
        n = len(blk)
        totals: dict[str, int] = {}
        for e in blk:
            for k, v in e["mix"].items():
                totals[k] = totals.get(k, 0) + v
        top = " ".join(
            f"{k}:{v // n}%" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:4]
        )
        label = f"ep {blk[0]['ep']}-{blk[-1]['ep']}"
        boss = sum(e["boss"] for e in blk) / n
        hits = sum(e["hits"] for e in blk) / n
        steps = sum(e["steps"] for e in blk) / n
        print(f"{label:>12} {boss:7.1f}% {hits:6.1f} {steps:7.1f}  {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
