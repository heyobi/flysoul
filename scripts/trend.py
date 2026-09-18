"""Is the fly improving, or is that noise? Read it off the episodes, not off a block table.

Block means wander because single fights are noisy; the question is how noisy, and
whether a trend survives that noise. This reports, for a log or a run record:

  - the per-fight standard deviation of boss HP left and the resulting standard error
    of a block mean (so you know which block-to-block differences mean nothing);
  - a least-squares slope of boss HP against fight number with a 95% interval;
  - first half vs second half with a Mann-Whitney U test (no normality assumed);
  - how many fights per arm are needed to detect a given improvement.

    python scripts/trend.py flysoul_run.log --from-line 22037          # a slice of the log
    python scripts/trend.py checkpoints/runs/<run>/episodes.jsonl
    python scripts/trend.py A.log --compare B.log                        # two arms
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aim_impact import mannwhitney  # noqa: E402  (hand-rolled; no scipy on the host)
from progress import parse  # noqa: E402


def load(path: str, from_line: int = 0) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        eps = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [{"boss": round(e["boss_hp_pct"] * 100), "hits": e.get("hits", 0), "steps": e["steps"],
                 "victory": e.get("victory", False)} for e in eps if e.get("type") == "episode"]
    text = p.read_text(encoding="utf-8", errors="replace")
    if from_line:
        text = "\n".join(text.splitlines()[from_line - 1:])
    return parse(text)


def slope_ci(y: list[float]) -> tuple[float, float]:
    n = len(y)
    x = list(range(n))
    mx, my = (n - 1) / 2, st.mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    resid = [yi - (my + b * (xi - mx)) for xi, yi in zip(x, y)]
    se = math.sqrt(sum(r * r for r in resid) / max(1, n - 2) / sxx)
    return b, 1.96 * se


def describe(name: str, eps: list[dict], block: int) -> None:
    y = [e["boss"] for e in eps]
    n = len(y)
    sd = st.pstdev(y)
    print(f"\n== {name}: {n} fights, boss HP left mean {st.mean(y):.1f}%, per-fight SD {sd:.1f}")
    print(f"   standard error of a {block}-fight block mean: {sd / math.sqrt(block):.1f} points "
          f"-> two blocks differ by chance up to ~{2.8 * sd / math.sqrt(block):.0f} points (95%)")
    b, ci = slope_ci(y)
    per100 = b * 100
    verdict = "significant" if abs(per100) > ci * 100 else "not distinguishable from zero"
    print(f"   trend: {per100:+.1f} points per 100 fights (95% CI +/-{ci * 100:.1f}) -> {verdict}")
    half = n // 2
    a, bb = y[:half], y[half:]
    _, p = mannwhitney(a, bb)
    print(f"   first half {st.mean(a):.1f}% vs second half {st.mean(bb):.1f}%: Mann-Whitney p = {p:.3f}")
    hits = [e["hits"] for e in eps]
    print(f"   hits per fight: first half {st.mean(hits[:half]):.2f}, second half {st.mean(hits[half:]):.2f}; "
          f"victories {sum(e['victory'] for e in eps)}")
    for delta in (3, 5, 10):
        need = math.ceil((2 * 1.96 * sd / delta) ** 2)
        print(f"   to detect a {delta}-point improvement between two arms at 95%: ~{need} fights per arm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--from-line", type=int, default=0)
    ap.add_argument("--compare", default=None, help="second log/jsonl to compare against")
    ap.add_argument("--compare-from-line", type=int, default=0)
    ap.add_argument("--compare-first", type=int, default=0, help="use only the first N fights of the comparison")
    ap.add_argument("--block", type=int, default=50)
    args = ap.parse_args()

    a = load(args.path, args.from_line)
    if len(a) < 10:
        print("too few fights"); return 1
    describe(args.path, a, args.block)
    if args.compare:
        b = load(args.compare, args.compare_from_line)
        if args.compare_first:
            b = b[: args.compare_first]
        describe(args.compare, b, args.block)
        ya, yb = [e["boss"] for e in a], [e["boss"] for e in b]
        _, p = mannwhitney(ya, yb)
        print(f"\n== A vs B: boss HP {st.mean(ya):.1f}% vs {st.mean(yb):.1f}% "
              f"(difference {st.mean(yb) - st.mean(ya):+.1f} points), Mann-Whitney p = {p:.4f}; "
              f"hits {st.mean(e['hits'] for e in a):.2f} vs {st.mean(e['hits'] for e in b):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
