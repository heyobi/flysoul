"""What is the best any local learning rule could do on these fights?

The plastic synapses are a linear map from Kenyon cell activity to compartment drive.
Whatever the dopamine rule does, the most it can express is that linear map. So fit
that map directly by least squares on half of the archived fights - Kenyon spikes in,
outcome of the executed action out - and score it on the other half, the same way the
parameter search scores a rule. That number is the ceiling. A rule cannot beat it, and
if it is low the fights do not contain the signal and no parameter will help.

The fit is a diagnostic only. It is never loaded into the fly: a readout fitted from data
is learning outside the brain, which this project does not do.

Also fits the critic by least-squares TD (LSTD) in one shot, and reports how much of the
reward variance a linear value function over Kenyon cells can explain at all.

    python scripts/offline_ceiling.py checkpoints/replay_archive.npz
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
from offline_search import outcomes  # noqa: E402


def spearman(a, b) -> float:
    def rank(x):
        r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?", default="checkpoints/replay_archive.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    z = np.load(args.archive)
    fights = np.unique(z["fight"])
    train_f, test_f = fights[::2], fights[1::2]
    outc = outcomes(z)
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    kc = topo.kenyon_indices
    X = (z["spikes"][:, kc] > 0).astype(np.float32)          # the critic's own features
    norms = np.linalg.norm(X, axis=1, keepdims=True); X = X / np.maximum(norms, 1e-6)
    train = np.isin(z["fight"], train_f)
    test = np.isin(z["fight"], test_f)
    print(f"archive: {len(outc)} steps, {len(fights)} fights; Kenyon features {X.shape[1]}")

    # ---- ceiling for credit assignment, per compartment and pooled ------------------
    print("\nleast-squares ceiling: outcome of the executed action predicted from Kenyon "
          "activity (held-out Spearman)")
    print(f"  {'compartment':14} {'n_train':>8} {'n_test':>7} {'ceiling':>9}")
    pooled_pred, pooled_true = [], []
    for k, ch in enumerate(ACTION_CHANNELS):
        m = (z["channel"] == k) & (outc != 0)
        tr, te = m & train, m & test
        if tr.sum() < 15 or te.sum() < 10:
            print(f"  {ch:14} {int(tr.sum()):8d} {int(te.sum()):7d}   (too few)")
            continue
        best = None
        for lam in (0.3, 1.0, 3.0, 10.0):
            w = ridge(X[tr], outc[tr], lam)
            pred = X[te] @ w
            c = spearman(pred, outc[te])
            if best is None or c > best[0]:
                best = (c, lam, pred)
        pooled_pred.append(best[2]); pooled_true.append(outc[te])
        print(f"  {ch:14} {int(tr.sum()):8d} {int(te.sum()):7d} {best[0]:+9.3f}   (ridge {best[1]})")
    if pooled_pred:
        c = spearman(np.concatenate(pooled_pred), np.concatenate(pooled_true))
        print(f"  {'pooled':14} {'':>8} {'':>7} {c:+9.3f}")

    # ---- roll timing ceiling ---------------------------------------------------------
    k = ACTION_CHANNELS.index("roll")
    m = (z["channel"] == k) & (outc != 0)
    if m.sum() >= 25:
        w = ridge(X[m & train], outc[m & train], 3.0)
        pred = X @ w
        att = z["attacking"]
        early = att & (z["anim_t"] >= 0.3) & (z["anim_t"] < 0.6) & test
        late = att & (z["anim_t"] >= 0.6) & (z["anim_t"] < 1.0) & test
        idle = (~att) & test
        print(f"\nroll fit, held-out mean predicted outcome: early {pred[early].mean():+.3f}  "
              f"late {pred[late].mean():+.3f}  idle {pred[idle].mean():+.3f}"
              f"   (late > early means the data does say 'roll later')")

    # ---- LSTD critic -----------------------------------------------------------------
    r = z["reward"].astype(np.float64)
    gamma = 0.95
    nxt = np.roll(X, -1, axis=0); nxt[z["terminal"]] = 0.0
    same = np.roll(z["fight"], -1) == z["fight"]
    valid = train & (same | z["terminal"])
    A = X[valid].T @ (X[valid] - gamma * nxt[valid]) + 1.0 * np.eye(X.shape[1])
    b = X[valid].T @ r[valid]
    wv = np.linalg.solve(A, b)
    V = X @ wv
    td = r + gamma * np.where(z["terminal"], 0.0, np.roll(V, -1)) - V
    te = test & (same | z["terminal"])
    print(f"\nLSTD critic (one-shot least-squares TD): V range {V.min():+.2f}..{V.max():+.2f}; "
          f"held-out |TD error| {np.abs(td[te]).mean():.3f} vs |reward| {np.abs(r[te]).mean():.3f} "
          f"({1 - np.abs(td[te]).mean() / max(1e-6, np.abs(r[te]).mean()):.0%} of the reward "
          f"magnitude is predictable from Kenyon activity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
