"""3b - SYNTHETIC: offline Double-DQN on the archived fights (torch), exported to numpy.

The SoulsAI route, done as batch RL on our own archive: a two-hidden-layer network over
the raw game state (one-hot bins + smooth values), 12 actions with directional rolls,
n-step returns, a target network, Double-DQN action selection for the bootstrap, Huber
loss, and early stopping on a held-out half by Bellman error. Nothing here is the fly's
learning; runs that use it carry the fingerprint "3b".

    python scripts/fit_q_dqn.py checkpoints/offline/archive_q_fit20.npz --out checkpoints/q_dqn_3b.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flysoul.synthetic.actions import CHANNEL_ACTIONS, SYNTH_ACTIONS, relabel_archive  # noqa: E402
from flysoul.synthetic.features import cont_features_from_archive, raw_features_from_archive  # noqa: E402


def build_features(z, spec: str) -> np.ndarray:
    R = raw_features_from_archive(z)
    if spec == "raw":
        return np.hstack([R, np.ones((len(R), 1), np.float32)])
    C = cont_features_from_archive(z)
    return np.hstack([R, C, np.ones((len(R), 1), np.float32)])


def nstep_targets_info(r, term, fid, n, gamma):
    """For each step i: sum_{k<n} gamma^k r_{i+k} (stopping at fight end / terminal),
    the index of the bootstrap state s_{i+n} (or -1), and gamma^n_eff."""
    N = len(r)
    ret = np.zeros(N, dtype=np.float32)
    boot = np.full(N, -1, dtype=np.int64)
    disc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        g = 1.0
        acc = 0.0
        j = i
        ended = False
        for k in range(n):
            acc += g * r[j]
            g *= gamma
            if term[j] or j + 1 >= N or fid[j + 1] != fid[i]:
                ended = True
                break
            j += 1
        ret[i] = acc
        if not ended:
            boot[i] = j + 1 if (j + 1 < N and fid[j + 1] == fid[i]) else -1
            disc[i] = g if boot[i] >= 0 else 0.0
    return ret, boot, disc


def main() -> int:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="checkpoints/q_dqn_3b.npz")
    ap.add_argument("--features", choices=("raw", "raw+cont"), default="raw+cont")
    ap.add_argument("--action-set", choices=("channels", "synth12"), default="synth12")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--outer", type=int, default=30, help="target-network updates (max)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--aggression", type=float, default=4.0)
    ap.add_argument("--terminal-bonus", type=float, default=0.0,
                    help="add +bonus to the last step of a won fight and -bonus to the last step of a lost one, "
                         "so the network sees the outcome itself, not only the running damage balance")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep the final network after --outer updates (Bellman error rises with the "
                         "targets, so early stopping on it tends to stop at update 0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = dict(np.load(args.archive).items())
    actions = SYNTH_ACTIONS if args.action_set == "synth12" else CHANNEL_ACTIONS
    X = build_features(z, args.features).astype(np.float32)
    n, d = X.shape
    a = relabel_archive(z, actions)
    names = list(z["extra_names"])
    r = (z["extra"][:, names.index("reward_env")].astype(np.float32)
         + z["hit"].astype(np.float32) * (args.aggression - 1.0))
    fid = z["fight"]
    term = z["terminal"].astype(bool).copy()
    last = (np.arange(n) + 1 >= n) | (np.roll(fid, -1) != fid)
    term |= last
    if args.terminal_bonus > 0:
        # a fight is won when its last step's hit brings the boss (as seen before the step) to zero
        boss_hp = z["extra"][:, names.index("boss_hp")]
        ends = np.flatnonzero(last)
        won = (z["hit"][ends] > 0) & (boss_hp[ends] - z["hit"][ends] <= 0.02)
        r[ends] += np.where(won, args.terminal_bonus, -args.terminal_bonus).astype(np.float32)
        print(f"terminal bonus +/-{args.terminal_bonus}: {int(won.sum())} of {len(ends)} fights end in a win")
    ret, boot, disc = nstep_targets_info(r, term, fid, args.nstep, args.gamma)
    fights = np.unique(fid)
    train = np.isin(fid, fights[::2])
    print(f"{n} steps, {len(fights)} fights, {d} features, {len(actions)} actions, n-step {args.nstep}, gamma {args.gamma}")

    Xt = torch.from_numpy(X)
    At = torch.from_numpy(a)
    Rt = torch.from_numpy(ret)
    Bt = torch.from_numpy(np.maximum(boot, 0))
    Dt = torch.from_numpy(disc)

    def make():
        return nn.Sequential(nn.Linear(d, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden), nn.ReLU(),
                             nn.Linear(args.hidden, len(actions)))

    def bellman(q, tq, idx):
        with torch.no_grad():
            qn_online = q(Xt[Bt[idx]])
            astar = qn_online.argmax(dim=1, keepdim=True)
            qn_target = tq(Xt[Bt[idx]]).gather(1, astar).squeeze(1)
            y = Rt[idx] + Dt[idx] * qn_target
            pred = q(Xt[idx]).gather(1, At[idx, None]).squeeze(1)
            return float((y - pred).pow(2).mean().sqrt())

    def run(mask, label):
        idx = torch.from_numpy(np.flatnonzero(mask))
        test_idx = torch.from_numpy(np.flatnonzero(~mask)) if (~mask).any() else None
        q, tq = make(), make()
        tq.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_fn = nn.SmoothL1Loss()
        bs = 512
        best = (float("inf"), None, -1)
        bad = 0
        t0 = time.perf_counter()
        for outer in range(args.outer):
            with torch.no_grad():
                astar = q(Xt[Bt]).argmax(dim=1, keepdim=True)
                y_all = Rt + Dt * tq(Xt[Bt]).gather(1, astar).squeeze(1)
            for _ in range(args.epochs):
                perm = idx[torch.randperm(len(idx))]
                for s in range(0, len(perm), bs):
                    b = perm[s:s + bs]
                    pred = q(Xt[b]).gather(1, At[b, None]).squeeze(1)
                    loss = loss_fn(pred, y_all[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            tq.load_state_dict(q.state_dict())
            tr_be = bellman(q, tq, idx)
            te_be = bellman(q, tq, test_idx) if test_idx is not None else tr_be
            print(f"  [{label}] update {outer:2d}: Bellman RMSE train {tr_be:.3f} test {te_be:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if args.no_early_stop:
                best = (te_be, {k: v.clone() for k, v in q.state_dict().items()}, outer)
            elif te_be < best[0] - 1e-4:
                best = (te_be, {k: v.clone() for k, v in q.state_dict().items()}, outer)
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        q.load_state_dict(best[1])
        print(f"  [{label}] kept update {best[2]} (test Bellman {best[0]:.3f})")
        return q, best[2]

    q, best_outer = run(train, "half")
    with torch.no_grad():
        Q = q(Xt).numpy()
    greedy = Q.argmax(1)
    test = ~train
    print("  greedy mix held-out: " + ", ".join(f"{actions[k]} {np.mean(greedy[test] == k):.0%}" for k in range(len(actions)) if np.mean(greedy[test] == k) >= 0.01))
    print(f"  agreement with executed (held-out): {np.mean(greedy[test] == a[test]):.0%}; "
          f"mean Q advantage of greedy over executed: {(Q[np.arange(n), greedy] - Q[np.arange(n), a])[test].mean():+.3f}")
    # final fit on everything, for as many updates as the held-out run found best (+1)
    args.outer = args.outer if args.no_early_stop else max(1, best_outer + 1)
    args.patience = 10 ** 6
    q, _ = run(np.ones(n, dtype=bool), "all")
    layers = [m for m in q if isinstance(m, nn.Linear)]
    out = {"type": "mlp", "gamma": args.gamma, "actions": np.array(actions), "num_kc": 0,
           "aggression": args.aggression, "fights": len(fights), "steps": n, "features": args.features,
           "nstep": args.nstep, "learner": "double-dqn", "terminal_bonus": args.terminal_bonus}
    for i, m in enumerate(layers, start=1):
        out[f"W{i}"] = m.weight.detach().numpy().astype(np.float32)
        out[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}  - 3b SYNTHETIC Double-DQN ({len(layers)} layers, hidden {args.hidden}, {len(actions)} actions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
