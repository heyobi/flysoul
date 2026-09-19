"""3b - SYNTHETIC: evaluate a saved Q readout (linear or MLP) on an archive.

Reports the Bellman RMSE against the readout's own bootstrapped targets, the greedy
action mix, agreement with the executed action and the mean Q advantage of greedy over
executed. Used to compare readouts fitted on different training sets on one common
held-out set (e.g. fit on all data vs on recent on-policy data only).

    python scripts/eval_q_readout.py checkpoints/q_readout_3b.npz checkpoints/offline/archive_test.npz
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
from flysoul.synthetic.features import cont_features_from_archive, raw_features_from_archive  # noqa: E402
from flysoul.synthetic.actions import relabel_archive  # noqa: E402
from fit_q_readout import features  # noqa: E402


def load_model(path):
    b = np.load(path, allow_pickle=False)
    spec = str(b["features"]) if "features" in b.files else "kc"
    if "W1" in b.files:
        layers = []
        i = 1
        while f"W{i}" in b.files:
            layers.append((b[f"W{i}"].astype(np.float64), b[f"b{i}"].astype(np.float64)))
            i += 1
        def q(X):
            h = X.astype(np.float64)
            for j, (W, bb) in enumerate(layers):
                h = h @ W.T + bb
                if j < len(layers) - 1:
                    h = np.maximum(h, 0.0)
            return h
    else:
        W = b["W"].astype(np.float64)
        def q(X):
            return X.astype(np.float64) @ W.T
    actions = [str(x) for x in b["actions"]]
    return q, spec, float(b["gamma"]), float(b["aggression"]) if "aggression" in b.files else 8.0, actions


def build_X(z, spec, kc):
    if spec == "kc":
        return features(z["spikes"][:, kc])
    R = raw_features_from_archive(z)
    if spec == "raw":
        return np.hstack([R, np.ones((len(R), 1), np.float32)])
    if spec == "raw+cont":
        return np.hstack([R, cont_features_from_archive(z), np.ones((len(R), 1), np.float32)])
    return np.hstack([features(z["spikes"][:, kc]), R])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    z = dict(np.load(args.archive).items())
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    kc = topo.kenyon_indices
    names = list(z["extra_names"])
    fid = z["fight"]
    n = len(fid)
    term = z["terminal"].astype(bool).copy()
    nxt = np.arange(n) + 1
    last = (nxt >= n) | (np.roll(fid, -1) != fid)
    term |= last
    nxt[last] = 0
    print(f"archive {args.archive}: {n} steps, {len(np.unique(fid))} fights")
    cache = {}
    for m in args.model:
        q, spec, gamma, aggr, ACTIONS = load_model(m)
        a = relabel_archive(z, ACTIONS)
        if spec not in cache:
            cache[spec] = build_X(z, spec, kc)
        X = cache[spec]
        r = z["extra"][:, names.index("reward_env")].astype(np.float64) + z["hit"].astype(np.float64) * (aggr - 1.0)
        Q = q(X)
        y = r + np.where(term, 0.0, gamma * q(X[nxt]).max(axis=1))
        be = y - Q[np.arange(n), a]
        greedy = Q.argmax(1)
        adv = Q[np.arange(n), greedy] - Q[np.arange(n), a]
        mix = ", ".join(f"{ACTIONS[k]} {np.mean(greedy == k):.0%}" for k in range(len(ACTIONS)) if np.mean(greedy == k) >= 0.02)
        print(f"  {Path(m).name:34} [{spec:6}] Bellman RMSE {np.sqrt(np.mean(be ** 2)):.3f}  agreement {np.mean(greedy == a):.0%}  "
              f"advantage {adv.mean():+.3f}  mix: {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
