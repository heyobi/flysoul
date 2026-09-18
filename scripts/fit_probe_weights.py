"""DIAGNOSTIC ONLY: fit KC->MBON weights by least squares and write them as a checkpoint.

This is learning outside the brain, which FlySoul does not do for its results. It exists
to answer one question the live fly cannot: if the plastic synapses held the *best*
linear readout of the Kenyon code that the archived fights allow (the "ceiling" of
scripts/offline_ceiling.py, +0.41 held-out), would the fly win? Loaded into the same
circuit with learning and sleep off (`run.py --probe-weights`), for ~200 fights:

  - if the fly wins often, the learning rule is the bottleneck and worth structural work;
  - if it still plateaus, the sensory code or circuit capacity is, and no rule will help.

The output is never merged into the learned checkpoint and is reported as a diagnostic.

    python scripts/fit_probe_weights.py checkpoints/offline/archive_probe_fit.npz --out checkpoints/probe_ls.npz
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="checkpoints/probe_ls.npz")
    ap.add_argument("--gain", type=float, default=1.5,
                    help="synaptic spread: weight = mean * (1 + gain * z-scored readout), clipped to the rule's bounds")
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--label", choices=("outcome", "reward"), default="outcome",
                    help="outcome: hit/damage within 3 steps (+1/-1, damage wins). reward: the learning "
                         "reward summed over the next 3 steps (aggression-weighted, what the fly chases).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    z = dict(np.load(args.archive).items())
    ch = z["channel"].astype(np.int64)
    outc = outcomes(z)
    fights = np.unique(z["fight"])
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    p = DopaminePlasticity(topo)
    innate = topo.weight[p.edges].astype(np.float32).copy()
    kc = topo.kenyon_indices
    kc_pos = {int(n): i for i, n in enumerate(kc)}
    A = (z["spikes"][:, kc] > 0).astype(np.float32)
    X = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-6)
    train = np.isin(z["fight"], fights[::2])
    test = ~train
    if args.label == "reward":
        r = z["reward"].astype(np.float64)
        fid = z["fight"]
        y = np.zeros(len(r))
        for d in (1, 2, 3):
            sh = np.zeros(len(r)); sh[:-d] = r[d:]
            same = np.zeros(len(r), bool); same[:-d] = fid[d:] == fid[:-d]
            y += sh * same
        score_mask = np.ones(len(r), bool)
    else:
        y = outc.astype(np.float64)
        score_mask = outc != 0
    print(f"{len(ch)} steps, {len(fights)} fights; outcomes + {int((outc > 0).sum())} / - {int((outc < 0).sum())}")

    # held-out sanity of the fit itself, then the fit on all fights
    px, py = [], []
    readout = np.zeros((p.num_channels, len(kc)), dtype=np.float64)
    print(f"  {'compartment':13} {'n':>6} {'held-out':>9}")
    for k, name in enumerate(ACTION_CHANNELS):
        m = (ch == k) & score_mask
        if m.sum() < 40:
            print(f"  {name:13} {int(m.sum()):6d}   (too few; innate kept)")
            continue
        if (m & train).sum() >= 15 and (m & test).sum() >= 10:
            w = ridge(X[m & train], y[m & train], args.lam)
            pred = X[m & test] @ w
            px.append(ranks01(pred)); py.append(y[m & test])
            print(f"  {name:13} {int(m.sum()):6d} {spearman(pred, y[m & test]):+9.3f}")
        readout[k] = ridge(X[m], y[m], args.lam)
    print(f"  pooled held-out ceiling {spearman(np.concatenate(px), np.concatenate(py)):+.3f}")

    # map each compartment's readout onto its synapses: same mean as innate, spread by gain
    w = innate.copy()
    for k in range(p.num_channels):
        pos = p._channel_edge_positions[k]
        if len(pos) == 0 or not np.any(readout[k]):
            continue
        r = readout[k]
        zr = (r - r.mean()) / (r.std() + 1e-9)
        pre_pos = np.array([kc_pos[int(n)] for n in p.pre[pos]])
        mean_w = float(np.mean(innate[pos]))
        w[pos] = np.clip(mean_w * (1.0 + args.gain * zr[pre_pos]), p.w_min, p.w_max).astype(np.float32)
    dd = drives(z["spikes"], p, w) - drives(z["spikes"], p, innate)
    per, pooled = knowledge_score(dd, ch, outc, np.ones(len(ch), dtype=bool))
    print(f"  knowledge score of the probe weights on all fights (in-sample): {pooled:+.3f}  "
          + "  ".join(f"{k[:6]} {v[0]:+.2f}" for k, v in per.items()))
    print(f"  weight range {w.min() / p._w_scale:.2f}..{w.max() / p._w_scale:.2f} of the innate scale; "
          f"at bounds {np.mean((w <= p.w_min * 1.001) | (w >= p.w_max * 0.999)):.0%}")
    np.savez_compressed(
        args.out, weights=w, reward_baseline=0.0, value_weights=np.zeros(len(kc), dtype=np.float32),
        delta_magnitude=0.1, total_ltp=0, total_ltd=0, num_edges=len(p.edges),
    )
    print(f"wrote {args.out} ({len(w)} synapses) - DIAGNOSTIC PROBE, not a learned checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
