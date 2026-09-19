"""3b - SYNTHETIC: fit a small MLP Q function on the fly's Kenyon code (offline, torch).

Same objective and data as scripts/fit_q_readout.py (fitted Q-iteration on archived
fights, reward = env reward + (aggression - 1) x damage dealt), but the readout is a
two-hidden-layer network instead of a linear map, trained DQN-style against a slowly
updated target network with a Huber loss. Exported as plain numpy weights so the live
agent evaluates it without torch (flysoul/synthetic/q_readout.py).

Everything about this is outside the fly's brain and is labelled so.

    python scripts/fit_q_mlp.py checkpoints/offline/archive_q_fit3.npz --out checkpoints/q_mlp_3b.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit  # noqa: E402
from fit_q_readout import ACTIONS, features  # noqa: E402
from flysoul.synthetic.features import raw_features_from_archive  # noqa: E402


def main() -> int:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="checkpoints/q_mlp_3b.npz")
    ap.add_argument("--gamma", type=float, default=0.90)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--outer", type=int, default=25, help="target-network updates")
    ap.add_argument("--epochs", type=int, default=3, help="epochs per target update")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--aggression", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", choices=("kc", "kc+raw", "raw"), default="kc")
    ap.add_argument("--wd", type=float, default=1e-4) if False else None
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = dict(np.load(args.archive).items())
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    kc = topo.kenyon_indices
    X = features(z["spikes"][:, kc]).astype(np.float32)
    if args.features != "kc":
        R = raw_features_from_archive(z)
        X = (np.hstack([X, R]) if args.features == "kc+raw" else np.hstack([R, np.ones((len(R), 1), np.float32)])).astype(np.float32)
    n, d = X.shape
    ch = z["channel"].astype(np.int64)
    a = np.where(ch >= 0, ch, len(ACTION_CHANNELS))
    names = list(z["extra_names"])
    r = z["extra"][:, names.index("reward_env")].astype(np.float32) + z["hit"].astype(np.float32) * (args.aggression - 1.0)
    fid = z["fight"]
    term = z["terminal"].astype(bool).copy()
    nxt = np.arange(n) + 1
    last = (nxt >= n) | (np.roll(fid, -1) != fid)
    term |= last
    nxt[last] = 0
    fights = np.unique(fid)
    train = np.isin(fid, fights[::2])
    print(f"{n} steps, {len(fights)} fights, {d} features, {len(ACTIONS)} actions")

    Xt = torch.from_numpy(X); Xn = torch.from_numpy(X[nxt]); At = torch.from_numpy(a); Rt = torch.from_numpy(r)
    Tt = torch.from_numpy(term.astype(np.float32))

    def make():
        return nn.Sequential(nn.Linear(d, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden), nn.ReLU(),
                             nn.Linear(args.hidden, len(ACTIONS)))

    def run(mask: np.ndarray, label: str):
        idx = torch.from_numpy(np.flatnonzero(mask))
        test_idx = torch.from_numpy(np.flatnonzero(~mask)) if (~mask).any() else None
        q, tq = make(), make()
        tq.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_fn = nn.SmoothL1Loss()
        bs = 512
        t0 = time.perf_counter()
        for outer in range(args.outer):
            with torch.no_grad():
                y_all = Rt + (1.0 - Tt) * args.gamma * tq(Xn).max(dim=1).values
            for _ in range(args.epochs):
                perm = idx[torch.randperm(len(idx))]
                for s in range(0, len(perm), bs):
                    b = perm[s:s + bs]
                    pred = q(Xt[b]).gather(1, At[b, None]).squeeze(1)
                    loss = loss_fn(pred, y_all[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            tq.load_state_dict(q.state_dict())
            if outer % 5 == 0 or outer == args.outer - 1:
                with torch.no_grad():
                    Q = q(Xt)
                    y = Rt + (1.0 - Tt) * args.gamma * q(Xn).max(dim=1).values
                    be = (y - Q.gather(1, At[:, None]).squeeze(1))
                    msg = f"  [{label}] outer {outer:2d}: Bellman RMSE train {be[idx].pow(2).mean().sqrt():.3f}"
                    if test_idx is not None:
                        msg += f" test {be[test_idx].pow(2).mean().sqrt():.3f}"
                    msg += f"  Q range {Q.min():+.2f}..{Q.max():+.2f}  ({time.perf_counter() - t0:.0f}s)"
                    print(msg, flush=True)
        return q

    q = run(train, "half")
    with torch.no_grad():
        Q = q(Xt).numpy()
    greedy = Q.argmax(1)
    test = ~train
    print("  greedy mix held-out: " + ", ".join(f"{ACTIONS[k]} {np.mean(greedy[test] == k):.0%}" for k in range(len(ACTIONS)) if np.mean(greedy[test] == k) >= 0.01))
    print(f"  agreement with executed (held-out): {np.mean(greedy[test] == a[test]):.0%}; "
          f"mean Q advantage of greedy over executed: {(Q[np.arange(n), greedy] - Q[np.arange(n), a])[test].mean():+.3f}")
    q = run(np.ones(n, dtype=bool), "all")
    layers = [m for m in q if isinstance(m, nn.Linear)]
    out = {"type": "mlp", "gamma": args.gamma, "actions": np.array(ACTIONS), "num_kc": len(kc),
           "aggression": args.aggression, "fights": len(fights), "steps": n, "features": args.features}
    for i, m in enumerate(layers, start=1):
        out[f"W{i}"] = m.weight.detach().numpy().astype(np.float32)
        out[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}  - 3b SYNTHETIC MLP readout ({len(layers)} layers, hidden {args.hidden})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
