"""3b - SYNTHETIC: fit a Q readout on the fly's Kenyon code by least-squares Q-iteration.

This is not the fly learning. It is option 3b of the project, agreed on 2026-09-19 after
every measurable in-brain lever came back empty: the biological sensory circuit stays,
and a clearly labelled synthetic readout is fitted *outside the brain* on the archived
fights - fitted Q-iteration (the batch form of Q-learning, the objective SoulsAI's DQN
optimised), linear in the Kenyon features, one weight vector per action. It is run live
under its own fingerprint ("3b"), never merged with the fly's synapses, and reported
separately.

    Q(s, a) = w_a . phi(s)      phi = L2-normalised binary Kenyon activity + bias
    y_i     = r_i + gamma * max_a' Q(s'_i, a')     (0 at fight end)
    w_a     = argmin sum_{i: a_i = a} (y_i - w_a . phi_i)^2 + lam |w_a|^2,  repeated

Reward is the environment's own reward plus (aggression - 1) x damage dealt, the same
signal the fly was given, recomputed so that every archived run is on one scale.

    python scripts/fit_q_readout.py checkpoints/offline/archive_q_fit.npz --out checkpoints/q_readout_3b.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit  # noqa: E402
from flysoul.synthetic.features import raw_features_from_archive  # noqa: E402
from flysoul.synthetic.actions import CHANNEL_ACTIONS, SYNTH_ACTIONS, relabel_archive  # noqa: E402

ACTIONS = list(ACTION_CHANNELS) + ["idle"]


def features(spikes_kc: np.ndarray) -> np.ndarray:
    A = (spikes_kc > 0).astype(np.float32)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-6)
    return np.hstack([A, np.ones((len(A), 1), dtype=np.float32)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="checkpoints/q_readout_3b.npz")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--aggression", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", choices=("kc", "kc+raw", "raw"), default="kc",
                    help="kc: the fly's Kenyon code; raw: binned game state; kc+raw: both")
    ap.add_argument("--action-set", choices=("channels", "synth12"), default="channels",
                    help="channels: the 8 descending channels + idle; synth12: rolls split by direction")
    args = ap.parse_args()
    actions = SYNTH_ACTIONS if args.action_set == "synth12" else CHANNEL_ACTIONS

    z = dict(np.load(args.archive).items())
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    kc = topo.kenyon_indices
    X = features(z["spikes"][:, kc])
    if args.features != "kc":
        R = raw_features_from_archive(z)
        X = np.hstack([X, R]) if args.features == "kc+raw" else np.hstack([R, np.ones((len(R), 1), np.float32)])
    n, d = X.shape
    ch = z["channel"].astype(np.int64)
    a = relabel_archive(z, actions)
    names = list(z["extra_names"])
    r_env = z["extra"][:, names.index("reward_env")].astype(np.float64)
    r = r_env + z["hit"].astype(np.float64) * (args.aggression - 1.0)
    fid = z["fight"]
    term = z["terminal"].astype(bool).copy()
    nxt = np.arange(n) + 1
    last = (nxt >= n) | (np.roll(fid, -1) != fid)
    term |= last
    nxt[last] = 0  # unused where terminal
    fights = np.unique(fid)
    train = np.isin(fid, fights[::2])
    test = ~train
    print(f"{n} steps, {len(fights)} fights, {d} features, {len(actions)} actions; "
          f"reward mean {r.mean():+.3f}, > 0 on {np.mean(r > 0):.0%} of steps")

    def fit(mask, W0=None):
        W = np.zeros((len(actions), d)) if W0 is None else W0.copy()
        for it in range(args.iters):
            Qn = X[nxt] @ W.T  # Q(s', .)
            y = r + np.where(term, 0.0, args.gamma * Qn.max(axis=1))
            Wn = W.copy()
            for k in range(len(actions)):
                m = mask & (a == k)
                if m.sum() < 30:
                    continue
                Xa = X[m]
                Wn[k] = np.linalg.solve(Xa.T @ Xa + args.lam * np.eye(d), Xa.T @ y[m])
            delta = float(np.abs(Wn - W).max())
            W = Wn
            if it % 10 == 0 or it == args.iters - 1:
                Q = np.einsum("ij,ij->i", X, W[a])
                be = y - Q
                print(f"  iter {it:2d}: max|dW| {delta:.4f}  Bellman RMSE train {np.sqrt(np.mean(be[mask] ** 2)):.3f} "
                      f"test {np.sqrt(np.mean(be[~mask] ** 2)):.3f}  Q range {Q.min():+.2f}..{Q.max():+.2f}", flush=True)
        return W

    print("fit on the training half (diagnostic):")
    W = fit(train)
    Qall = X @ W.T
    greedy = Qall.argmax(axis=1)
    print("  greedy action mix on held-out steps: " + ", ".join(
        f"{actions[k]} {np.mean(greedy[test] == k):.0%}" for k in range(len(actions)) if np.mean(greedy[test] == k) >= 0.01))
    print(f"  agreement with the action the fly executed (held-out): {np.mean(greedy[test] == a[test]):.0%}")
    # where they disagree, was the fly's action worse by Q?
    adv = Qall[np.arange(n), greedy] - Qall[np.arange(n), a]
    print(f"  mean Q advantage of greedy over executed (held-out): {adv[test].mean():+.3f}")
    print("fit on all fights:")
    W = fit(np.ones(n, dtype=bool), W)
    np.savez_compressed(args.out, W=W.astype(np.float32), gamma=args.gamma, actions=np.array(actions),
                        num_kc=len(kc), aggression=args.aggression, fights=len(fights), steps=n,
                        features=args.features)
    print(f"wrote {args.out}  - 3b SYNTHETIC readout, not the fly's learning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
