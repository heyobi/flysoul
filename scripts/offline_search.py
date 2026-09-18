"""Search the plasticity parameters offline, on archived fights, before touching the live run.

What can and cannot be measured without the game. The archive holds what the fly saw,
what it did and what followed, under the behaviour it had at the time. A different
learning rule cannot be scored on how the fly *would* have fought - that needs the game.
It can be scored on whether it assigns credit where the outcomes say it belongs:

    For every archived step, the action taken had an outcome - a hit landed within the
    next few steps (+), damage taken (-), or nothing. A rule that learns from half of the
    fights should, on the other half, have raised the synaptic drive of the executed
    compartment in the states where that action worked and lowered it where it did not.
    The score is the correlation between that change in drive and the outcome, on fights
    the rule never saw. It rewards generalising credit assignment and punishes both
    noise and overfitting.

Also reported: the roll-timing ratio (drive at 0.6-1.0 s over 0.3-0.6 s of the boss's
swing) because that is the specific thing the live fly has not learned.

    python scripts/offline_search.py checkpoints/replay_archive.npz --trials 80
    python scripts/offline_search.py archive.npz --only live      # score the live settings
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit  # noqa: E402
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_replay import old_update_traces  # noqa: E402

WINDOW = 3
LIVE = dict(rule="tagged", elig=0.92, credit=0.90, lr=0.08, critic_lr=0.05, discount=0.95, gain=0.5,
            passes=3)


def outcomes(z) -> np.ndarray:
    """+1 if the action was followed by a hit and no damage within WINDOW steps, -1 if
    damage arrived, 0 otherwise. Windows do not cross fights."""
    n = len(z["reward"])
    out = np.zeros(n, dtype=np.float32)
    fight = z["fight"]
    hit, dmg = z["hit"], z["damage_taken"]
    for i in range(n):
        j_end = i + 1
        while j_end < n and j_end <= i + WINDOW and fight[j_end] == fight[i]:
            j_end += 1
        h = float(hit[i:j_end].sum()) > 0
        d = float(dmg[i:j_end].sum()) > 0
        out[i] = -1.0 if d else (1.0 if h else 0.0)
    return out


def make(topo, cfg: dict) -> DopaminePlasticity:
    p = DopaminePlasticity(
        topo, learning_rate=cfg["lr"], eligibility_decay=cfg["elig"], credit_decay=cfg["credit"],
        critic_lr=cfg["critic_lr"], discount=cfg["discount"],
    )
    if cfg["rule"] == "old":
        p.update_traces = old_update_traces.__get__(p, DopaminePlasticity)
    return p


def replay(p: DopaminePlasticity, z, fights, gain: float, passes: int) -> None:
    lr0, c0 = p.lr, p.critic_lr
    p.lr, p.critic_lr = lr0 * gain, c0 * gain
    channels = list(z["channels"])
    for _ in range(passes):
        for f in fights:
            idx = np.flatnonzero(z["fight"] == f)
            p.reset()
            for i in idx:
                s = z["spikes"][i].astype(np.int32)
                ch = channels[z["channel"][i]] if z["channel"][i] >= 0 else None
                p.update_traces(s, executed_channel=ch)
                p.apply_reinforcement(float(z["reward"][i]), s, terminal=bool(z["terminal"][i]))
    p.lr, p.critic_lr = lr0, c0
    p.reset()


def drive_matrix(p: DopaminePlasticity, spikes: np.ndarray) -> np.ndarray:
    """[steps, channels] plastic drive into each compartment for each archived step."""
    out = np.zeros((len(spikes), len(ACTION_CHANNELS)), dtype=np.float32)
    w = p.topology.weight[p.edges]
    for k in range(len(ACTION_CHANNELS)):
        sel = np.flatnonzero(p.channel == k)
        out[:, k] = spikes[:, p.pre[sel]] @ w[sel]
    return out


def score(cfg: dict, z, base_drive: np.ndarray, outc: np.ndarray, train, test, topo_factory) -> dict:
    topo = topo_factory()
    p = make(topo, cfg)
    replay(p, z, train, cfg["gain"], cfg["passes"])
    spikes = z["spikes"].astype(np.float32)
    drive = drive_matrix(p, spikes)
    delta = drive - base_drive
    test_mask = np.isin(z["fight"], test) & (z["channel"] >= 0) & (outc != 0)
    idx = np.flatnonzero(test_mask)
    if len(idx) < 20:
        return {"credit_corr": float("nan"), "n": int(len(idx))}
    d_exec = delta[idx, z["channel"][idx]]
    o = outc[idx]
    # Spearman: rank-based, so a few huge deltas cannot dominate.
    def rank(a):
        r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
    corr = float(np.corrcoef(rank(d_exec), rank(o))[0, 1]) if np.std(d_exec) > 0 else 0.0
    # Roll timing ratio on test fights.
    k = ACTION_CHANNELS.index("roll")
    att = np.isin(z["fight"], test) & z["attacking"]
    early = att & (z["anim_t"] >= 0.3) & (z["anim_t"] < 0.6)
    late = att & (z["anim_t"] >= 0.6) & (z["anim_t"] < 1.0)
    ratio = (float(drive[late, k].mean()) / max(1e-6, float(drive[early, k].mean()))
             if early.any() and late.any() else float("nan"))
    return {"credit_corr": corr, "roll_late_early": ratio, "n": int(len(idx)),
            "mean_w": float(np.mean(p.topology.weight[p.edges]))}


def sample(rng) -> dict:
    return dict(
        rule=rng.choice(["tagged", "old"]),
        elig=float(rng.uniform(0.70, 0.97)),
        credit=float(rng.uniform(0.70, 0.95)),
        lr=float(10 ** rng.uniform(-1.7, -0.5)),
        critic_lr=float(10 ** rng.uniform(-2.0, -0.7)),
        discount=float(rng.uniform(0.90, 0.99)),
        gain=float(rng.uniform(0.2, 1.0)),
        passes=int(rng.integers(1, 6)),
    )


def fmt(cfg: dict) -> str:
    return (f"{cfg['rule']:6} elig={cfg['elig']:.2f} credit={cfg['credit']:.2f} lr={cfg['lr']:.3f} "
            f"critic={cfg['critic_lr']:.3f} gamma={cfg['discount']:.3f} gain={cfg['gain']:.2f} "
            f"passes={cfg['passes']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?", default="checkpoints/replay_archive.npz")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["live"], default=None)
    args = ap.parse_args()

    # NpzFile re-reads and decompresses an array on every access; indexing it inside
    # a loop turns a one-second replay into half an hour. Materialise once.
    z = {k: v for k, v in np.load(args.archive).items()}
    fights = np.unique(z["fight"])
    if len(fights) < 4:
        print(f"only {len(fights)} fights in the archive; need at least 4 to hold some out")
        return 1
    train, test = fights[::2], fights[1::2]
    outc = outcomes(z)
    print(f"archive: {len(z['reward'])} steps, {len(fights)} fights -> train {len(train)} / test {len(test)}; "
          f"outcomes: +{int((outc > 0).sum())} / -{int((outc < 0).sum())} / 0 {int((outc == 0).sum())}")

    cfg_c, bio = CircuitConfig(), BioPhysicsConfig()

    def topo_factory():
        t = build_fly_circuit(cfg_c, seed=args.seed)
        calibrate_or_load(t, cfg_c, bio, seed=args.seed)
        return t

    base_drive = drive_matrix(DopaminePlasticity(topo_factory()), z["spikes"].astype(np.float32))

    t0 = time.perf_counter()
    live = score(LIVE, z, base_drive, outc, train, test, topo_factory)
    print(f"\nlive settings  credit_corr={live['credit_corr']:+.3f}  roll late/early={live.get('roll_late_early', float('nan')):.2f}  "
          f"(n={live['n']}, {time.perf_counter() - t0:.1f}s per evaluation)")
    if args.only == "live":
        return 0

    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(args.trials):
        cfg = sample(rng)
        r = score(cfg, z, base_drive, outc, train, test, topo_factory)
        results.append((r["credit_corr"], r.get("roll_late_early", float("nan")), cfg))
        if (i + 1) % 10 == 0:
            best = max(results, key=lambda x: (x[0] if x[0] == x[0] else -9))
            print(f"  {i + 1}/{args.trials} best so far credit_corr={best[0]:+.3f}")
    results.sort(key=lambda x: -(x[0] if x[0] == x[0] else -9))
    print(f"\ntop settings by credit_corr on held-out fights (live = {live['credit_corr']:+.3f}):")
    for corr, ratio, cfg in results[:8]:
        print(f"  {corr:+.3f}  roll {ratio:5.2f}  {fmt(cfg)}")
    worst = results[-3:]
    print("worst:")
    for corr, ratio, cfg in worst:
        print(f"  {corr:+.3f}  roll {ratio:5.2f}  {fmt(cfg)}")
    print("\nA setting only deserves a live trial if it beats the live score by a clear margin "
          "across archives, not by one lucky draw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
