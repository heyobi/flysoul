"""Is the rule's *form* the limit, or its memory? Batch replay against the ceiling.

Replays the training fights through the live plasticity rule many passes in shuffled
order, with no forgetting between passes, and scores held-out knowledge after each pass
(scripts/diagnose_learning.py's measure). If the score climbs towards the least-squares
ceiling, the rule can express the mapping and the live gap is recency/forgetting. If it
saturates far below, the form of the rule (one scalar dopamine per step, eligibility on
the executed compartment) is what caps it.

Also prints the ceiling's own learning curve against the number of training fights, and
the ceiling when credit goes to the compartment the circuit *wanted* (motor winner)
instead of the one the body executed.

    python scripts/offline_rule_limit.py archive.npz --passes 12 --lr 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit  # noqa: E402
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402
from diagnose_learning import drives, knowledge_score, ranks01  # noqa: E402
from offline_ceiling import ridge, spearman  # noqa: E402
from offline_search import outcomes  # noqa: E402


def ceiling(X, label, outc, train, test, lam=3.0):
    px, py = [], []
    for k in range(len(ACTION_CHANNELS)):
        m = (label == k) & (outc != 0)
        if (m & train).sum() < 15 or (m & test).sum() < 10:
            continue
        w = ridge(X[m & train], outc[m & train], lam)
        px.append(ranks01(X[m & test] @ w)); py.append(outc[m & test])
    return spearman(np.concatenate(px), np.concatenate(py)) if px else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--credit", type=float, default=0.90)
    ap.add_argument("--wmin", type=float, default=0.05)
    ap.add_argument("--wmax", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-batch", action="store_true")
    args = ap.parse_args()
    z = dict(np.load(args.archive).items())
    fights = np.unique(z["fight"])
    ch = z["channel"].astype(np.int64)
    outc = outcomes(z)
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    kc = topo.kenyon_indices
    X = (z["spikes"][:, kc] > 0).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    train_f, test_f = fights[::2], fights[1::2]
    train, test = np.isin(z["fight"], train_f), np.isin(z["fight"], test_f)
    rng = np.random.default_rng(args.seed)

    print(f"archive {len(ch)} steps / {len(fights)} fights; train {len(train_f)} test {len(test_f)}")
    print("\nA. least-squares ceiling vs number of training fights (held-out on the same test half)")
    for n in (15, 30, 60, 120, len(train_f)):
        sub = np.isin(z["fight"], rng.choice(train_f, size=min(n, len(train_f)), replace=False))
        print(f"  {n:4d} fights: {ceiling(X, ch, outc, sub, test):+.3f}")

    names = list(z["extra_names"])
    rates = np.stack([z["extra"][:, names.index(f'rate_{c}')] for c in ACTION_CHANNELS], 1)
    winner = rates.argmax(1)
    print("\nD. ceiling when credit is assigned by executed channel vs by the motor winner (intended)")
    print(f"  executed: {ceiling(X, ch, outc, train, test):+.3f}   intended: {ceiling(X, winner, outc, train, test):+.3f}"
          f"   (steps where they differ: {np.mean(winner[ch >= 0] != ch[ch >= 0]):.0%})")

    if args.skip_batch:
        return 0
    print(f"\nB. batch replay of the live rule (lr {args.lr}, credit {args.credit}, w_min {args.wmin}, "
          f"w_max {args.wmax}), shuffled fights, no forgetting; held-out knowledge after each pass")
    p = DopaminePlasticity(topo, learning_rate=args.lr, credit_decay=args.credit,
                           w_min_factor=args.wmin, w_max_factor=args.wmax)
    innate = topo.weight[p.edges].astype(np.float32).copy()
    d0 = drives(z["spikes"], p, innate)
    channels = list(z["channels"])
    for ps in range(1, args.passes + 1):
        for f in rng.permutation(train_f):
            idx = np.flatnonzero(z["fight"] == f)
            p.reset()
            for i in idx:
                s = z["spikes"][i].astype(np.int32)
                c = channels[ch[i]] if ch[i] >= 0 else None
                p.update_traces(s, executed_channel=c)
                p.apply_reinforcement(float(z["reward"][i]), s, terminal=bool(z["terminal"][i]))
        w = topo.weight[p.edges].astype(np.float32)
        dd = drives(z["spikes"], p, w) - d0
        per, held = knowledge_score(dd, ch, outc, test)
        _, ins = knowledge_score(dd, ch, outc, train)
        lo = float(np.mean(w <= p.w_min * 1.001))
        print(f"  pass {ps:2d}: held-out {held:+.3f}  in-sample {ins:+.3f}  at floor {lo:.0%}  "
              + "  ".join(f"{k[:6]} {v[0]:+.2f}" for k, v in per.items()), flush=True)
    topo.weight[p.edges] = innate
    return 0


if __name__ == "__main__":
    sys.exit(main())
