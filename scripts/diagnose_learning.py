"""Why is the score flat? Separate the four places learning can fail, on real data.

The trend statistics (scripts/trend.py, the visualizer) say *whether* the score moves.
This says *where* the chain from experience to behaviour breaks. The chain is:

    fights -> dopamine writes KC->MBON weights -> weights carry knowledge -> knowledge
    reaches the motor pools -> the executed action changes -> the score changes

Each link is measured separately, from the run's weight snapshots and step archive:

  1. Weight trajectory. Distance between successive snapshots against the distance
     from first to last. A learning system drifts (far >> near); a noisy one random-walks
     (far ~ sqrt(k) * near); a homeostatically pinned one goes nowhere (far << near).
     Also how many synapses sit on the clip bounds.
  2. Knowledge in the weights. For each snapshot, the change in MBON drive it produces
     (learned minus innate drive of the executed compartment) is correlated with the
     outcome of the executed action, on archived steps. Compared with the least-squares
     ceiling on the same steps. Rising with fights = the rule is extracting the signal.
  3. Knowledge reaching behaviour. Does the compartment the learned weights favour win
     the motor competition? Agreement between argmax learned drive and the executed
     action, against chance, and the outcome when they agree vs disagree.
  4. The teaching signal. Sign agreement between the dopamine the fly received and the
     eventual outcome; how the critic changed the plain reward; class balance.

    python scripts/diagnose_learning.py checkpoints/runs/<run>/archive.npz \
        --weights checkpoints/runs/<run>/weights_ep*.npz checkpoints/learned_<fp>.npz
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
from offline_ceiling import ridge, spearman  # noqa: E402
from offline_search import outcomes  # noqa: E402


def drives(spikes_u8: np.ndarray, p: DopaminePlasticity, w: np.ndarray) -> np.ndarray:
    """MBON synaptic drive per compartment for every archived step: sum over that
    compartment's KC->MBON synapses of presynaptic spikes times weight."""
    n = len(spikes_u8)
    out = np.zeros((n, p.num_channels), dtype=np.float64)
    for c in range(p.num_channels):
        pos = p._channel_edge_positions[c]
        if len(pos) == 0:
            continue
        pre = p.pre[pos]
        # spikes[:, pre] can repeat columns (several synapses from one KC); that is right.
        out[:, c] = spikes_u8[:, pre].astype(np.float32) @ w[pos].astype(np.float32)
    return out


def ranks01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / max(1, len(x) - 1)


def knowledge_score(delta_drive, channel, outc, mask):
    """Spearman between the learned change in drive of the executed compartment and the
    outcome of that action, per compartment (min 40 scored steps), and pooled with
    per-compartment rank normalisation so one big compartment does not dominate."""
    per = {}
    pooled_x, pooled_y = [], []
    for k, name in enumerate(ACTION_CHANNELS):
        m = mask & (channel == k) & (outc != 0)
        if m.sum() < 40:
            continue
        x, y = delta_drive[m, k], outc[m]
        per[name] = (spearman(x, y), int(m.sum()))
        pooled_x.append(ranks01(x))
        pooled_y.append(y)
    pooled = spearman(np.concatenate(pooled_x), np.concatenate(pooled_y)) if pooled_x else 0.0
    return per, pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="weight snapshots in chronological order (weights_ep*.npz, learned_*.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=0.90)
    args = ap.parse_args()

    z = dict(np.load(args.archive).items())
    spikes = z["spikes"]
    channel = z["channel"].astype(np.int64)
    outc = outcomes(z)
    fights = np.unique(z["fight"])
    extra = {n: z["extra"][:, i] for i, n in enumerate(list(z["extra_names"]))}
    n = len(channel)
    print(f"archive: {n} steps, {len(fights)} fights; outcomes + {int((outc > 0).sum())} / "
          f"- {int((outc < 0).sum())} / 0 {int((outc == 0).sum())}")

    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    p = DopaminePlasticity(topo)
    innate = topo.weight[p.edges].astype(np.float32).copy()

    snaps = []
    for path in args.weights:
        b = np.load(path)
        w = b["weights"]
        if w.shape != innate.shape:
            print(f"  skip {path}: {w.shape[0]} synapses, circuit has {innate.shape[0]}")
            continue
        snaps.append((Path(path).parent.name[-11:] + "/" + Path(path).stem, w.astype(np.float32)))
    if not snaps:
        print("no compatible snapshots")
        return 1

    # ---- 1. weight trajectory -----------------------------------------------------
    print("\n1. WEIGHT TRAJECTORY  (distances in units of the innate mean weight)")
    ws = [innate] + [w for _, w in snaps]
    names = ["innate"] + [nm for nm, _ in snaps]
    scale = float(np.mean(np.abs(innate)))
    near = [float(np.linalg.norm(ws[i + 1] - ws[i])) / scale for i in range(len(ws) - 1)]
    print(f"  {'snapshot':40} {'|dw| from prev':>14} {'mean w/innate':>14} {'at w_min':>9} {'at w_max':>9}")
    for i, (nm, w) in enumerate(zip(names, ws)):
        d = "" if i == 0 else f"{near[i - 1]:14.2f}"
        lo = float(np.mean(w <= p.w_min * 1.001))
        hi = float(np.mean(w >= p.w_max * 0.999))
        print(f"  {nm:40} {d:>14} {float(np.mean(w)) / float(np.mean(innate)):14.3f} {lo:9.1%} {hi:9.1%}")
    steps_between = near[1:]
    if len(steps_between) >= 2:
        far = float(np.linalg.norm(ws[-1] - ws[1])) / scale
        rw = float(np.sqrt(np.sum(np.square(steps_between))))
        drift = float(np.sum(steps_between))
        verdict = ("DRIFT: a direction is kept" if far > 1.3 * rw
                   else "RANDOM WALK or mean reversion: no direction kept" if far < 1.1 * rw
                   else "in between")
        print(f"  first learned -> last: {far:.2f};  random walk would give ~{rw:.2f}, straight drift {drift:.2f}"
              f"  ->  {verdict}")
        cos = []
        for i in range(1, len(ws) - 1):
            a, b = ws[i + 1] - ws[i], ws[i] - ws[i - 1]
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        if cos:
            print(f"  cosine between successive weight changes: {' '.join(f'{c:+.2f}' for c in cos)}"
                  f"   (+1 same direction, 0 unrelated, negative = undoing the previous change)")
    print("  per-compartment mean weight relative to innate (x):")
    print(f"  {'':40}" + "".join(f"{c[:9]:>10}" for c in ACTION_CHANNELS))
    for nm, w in zip(names[1:], ws[1:]):
        row = []
        for c in range(p.num_channels):
            pos = p._channel_edge_positions[c]
            row.append(float(np.mean(w[pos]) / np.mean(innate[pos])) if len(pos) else float("nan"))
        print(f"  {nm:40}" + "".join(f"{v:10.2f}" for v in row))

    # ---- 2. knowledge in the weights ---------------------------------------------
    print("\n2. KNOWLEDGE IN THE WEIGHTS  (Spearman of the learned change in drive of the "
          "executed compartment vs the outcome of that action)")
    d0 = drives(spikes, p, innate)
    allmask = np.ones(n, dtype=bool)
    print(f"  {'snapshot':40} {'pooled':>8}  per compartment")
    for nm, w in snaps:
        dd = drives(spikes, p, w) - d0
        per, pooled = knowledge_score(dd, channel, outc, allmask)
        detail = "  ".join(f"{k[:6]} {v[0]:+.2f}(n{v[1]})" for k, v in per.items())
        print(f"  {nm:40} {pooled:+8.3f}  {detail}")
    kc = topo.kenyon_indices
    X = (spikes[:, kc] > 0).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    tr = np.isin(z["fight"], fights[::2])
    te = ~tr
    px, py = [], []
    for k in range(p.num_channels):
        m = (channel == k) & (outc != 0)
        if (m & tr).sum() < 15 or (m & te).sum() < 10:
            continue
        w = ridge(X[m & tr], outc[m & tr], 3.0)
        pred = X[m & te] @ w
        px.append(ranks01(pred))
        py.append(outc[m & te])
    ceiling = spearman(np.concatenate(px), np.concatenate(py)) if px else 0.0
    print(f"  {'least-squares ceiling (held-out, same steps)':40} {ceiling:+8.3f}")
    per0, pooled0 = knowledge_score(d0, channel, outc, allmask)
    print(f"  {'innate drive itself (KC-count confound)':40} {pooled0:+8.3f}")
    print("  note: snapshots are scored on the whole archive, which includes fights they were "
          "trained on; the ceiling is held-out. A snapshot scoring far below the ceiling even "
          "in-sample means the rule is not extracting the signal.")

    # ---- 3. knowledge reaching behaviour -----------------------------------------
    print("\n3. KNOWLEDGE REACHING BEHAVIOUR  (latest snapshot)")
    w_last = snaps[-1][1]
    dd = drives(spikes, p, w_last) - d0
    rates = np.stack([extra[f"rate_{c}"] for c in ACTION_CHANNELS], axis=1)
    executed = channel >= 0
    pref = np.argmax(dd, axis=1)
    winner = np.argmax(rates, axis=1)
    agree_exec = float(np.mean(pref[executed] == channel[executed]))
    agree_win = float(np.mean(pref[executed] == winner[executed]))
    mix = np.bincount(channel[executed], minlength=p.num_channels) / executed.sum()
    chance = float(np.sum(mix * mix))
    explore_share = float(np.mean(winner[executed] != channel[executed]))
    print(f"  executed action == compartment with the largest learned drive change: {agree_exec:.1%}"
          f"   (chance from the action mix {chance:.1%})")
    print(f"  motor winner (argmax pool rate) == that compartment: {agree_win:.1%};"
          f"   executed != motor winner (exploration/veto): {explore_share:.1%}")
    rc = []
    for i in np.flatnonzero(executed):
        if np.std(dd[i]) > 0 and np.std(rates[i]) > 0:
            rc.append(spearman(dd[i], rates[i]))
    print(f"  per-step Spearman(learned drive change, pool rates) across the 8 pools: "
          f"mean {np.mean(rc):+.2f}  (0 = the learned preference does not show in the motor competition)")
    m = executed & (outc != 0)
    a = m & (pref == channel)
    b = m & (pref != channel)
    if a.sum() > 20 and b.sum() > 20:
        print(f"  outcome when the executed action IS the learned favourite: {outc[a].mean():+.3f} (n={int(a.sum())}); "
              f"when it is NOT: {outc[b].mean():+.3f} (n={int(b.sum())})")
    thirds = np.array_split(fights, 3)
    atk = ACTION_CHANNELS.index("attack_light")
    for fs in thirds:
        mm = np.isin(z["fight"], fs) & executed
        print(f"  fights {fs[0]:3d}-{fs[-1]:3d}: agreement {float(np.mean(pref[mm] == channel[mm])):.1%}, "
              f"mean outcome {outc[mm].mean():+.3f}, attack_light share {float(np.mean(channel[mm] == atk)):.0%}")

    # ---- 4. the teaching signal --------------------------------------------------
    print("\n4. TEACHING SIGNAL")
    r = z["reward"].astype(np.float64)
    dop = extra["dopamine"]
    td = extra["td_error"]
    V = extra["value"]
    nz = np.abs(dop) > 1e-3
    print(f"  steps with any dopamine: {nz.mean():.0%}; positive {float(np.mean(dop > 1e-3)):.0%}, "
          f"negative {float(np.mean(dop < -1e-3)):.0%}")
    print(f"  Spearman(dopamine, learning reward) = {spearman(dop, r):+.2f}; "
          f"Spearman(TD error, reward) = {spearman(td, r):+.2f}   (1.0 would mean the critic adds nothing)")
    G = np.zeros(n)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = r[i] + (0.0 if z["terminal"][i] else args.gamma * acc)
        if i + 1 < n and z["fight"][i + 1] != z["fight"][i]:
            acc = r[i]
        G[i] = acc
    print(f"  critic: Spearman(V(s), realised discounted return) = {spearman(V, G):+.2f}; "
          f"V range {V.min():+.2f}..{V.max():+.2f}; return range {G.min():+.2f}..{G.max():+.2f}")
    idx = np.flatnonzero(m)
    burst = np.array([dop[i + 1:i + 4].sum() for i in idx])
    agree = float(np.mean(np.sign(burst) == outc[m]))
    print(f"  sign agreement: dopamine over the next 3 steps vs outcome of the action: {agree:.0%} "
          f"(50% = uninformative)")
    print(f"  |reward| mean {np.abs(r).mean():.3f}; reward > 0 on {float(np.mean(r > 1e-6)):.0%} of steps, "
          f"< 0 on {float(np.mean(r < -1e-6)):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
