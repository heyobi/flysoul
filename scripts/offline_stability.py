"""Do the weights accumulate what the fights teach, or overwrite it? Tested offline.

scripts/diagnose_learning.py found that successive 50-fight weight changes on the live
run point in opposite directions (cosine -0.4 to -0.6): what one block writes the next
half undoes, and the knowledge in the weights stays at half the least-squares ceiling.
This replays the archived fights *sequentially*, exactly as the live agent experiences
them - the online pass through plasticity, then a night of sleep replay - for several
learning configurations, and tracks along the way:

  - held-out knowledge: Spearman between the learned change in MBON drive of the
    executed compartment and the outcome of that action, on fights never replayed;
  - in-sample knowledge (same, on the replayed fights);
  - the cosine between successive weight changes (negative = undoing);
  - how far the weights have travelled against how far a random walk of the same steps
    would have.

A configuration whose held-out knowledge keeps rising while the cosine stays near zero
or positive is accumulating; one that oscillates is not. Both halves of the archive
serve as the training sequence in turn, so a gain has to show on both splits.

    python scripts/offline_stability.py checkpoints/runs/<run>/archive.npz --variant live --variant nosleep
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
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402
from flysoul.connectome.sleep import SleepConsolidation  # noqa: E402
from diagnose_learning import drives, knowledge_score  # noqa: E402
from offline_search import outcomes  # noqa: E402

VARIANTS = {
    # name: (learning_rate, critic_lr, sleep memory fights, sleep passes, sleep gain)
    "live":        dict(lr=0.08, critic_lr=0.05, memory=20,  passes=30, gain=0.5),
    "nosleep":     dict(lr=0.08, critic_lr=0.05, memory=0,   passes=0,  gain=0.0),
    "sleep_light": dict(lr=0.08, critic_lr=0.05, memory=20,  passes=30, gain=0.1),
    "lr_low":      dict(lr=0.02, critic_lr=0.05, memory=20,  passes=30, gain=0.5),
    "lr_low_nosleep": dict(lr=0.02, critic_lr=0.05, memory=0, passes=0, gain=0.0),
    "broad":       dict(lr=0.08, critic_lr=0.05, memory=100, passes=30, gain=0.5),
    "broad_light": dict(lr=0.08, critic_lr=0.05, memory=100, passes=30, gain=0.1),
    # dopamine scaling: live adapts tanh(delta / running mean |delta|), so a critic-noise
    # step of typical size yields dopamine ~0.76 while a real hit taken yields ~1.0
    "rpe_0.3":     dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rpe_scale=0.3),
    "rpe_1.0":     dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rpe_scale=1.0),
    "nocritic":    dict(lr=0.08, critic_lr=0.0,  memory=20, passes=30, gain=0.5),
    # less recency weighting: smaller steps and/or a longer sleep memory
    "lr_low_broad":   dict(lr=0.02, critic_lr=0.05, memory=100, passes=30, gain=0.5),
    "lr_low_mem50":   dict(lr=0.02, critic_lr=0.05, memory=50,  passes=30, gain=0.5),
    "lr_0.04":        dict(lr=0.04, critic_lr=0.05, memory=20,  passes=30, gain=0.5),
    "lr_0.04_broad":  dict(lr=0.04, critic_lr=0.05, memory=100, passes=30, gain=0.5),
    "lr_0.01_broad":  dict(lr=0.01, critic_lr=0.05, memory=100, passes=30, gain=0.5),
    # dopamine centred against a running baseline (scripts/centered_rule.py)
    "center_comp":    dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rule="comp"),
    "center_global":  dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rule="global"),
    "center_comp_lr02_mem50": dict(lr=0.02, critic_lr=0.05, memory=50, passes=30, gain=0.5, rule="comp"),
    "center_comp_lr02_broad": dict(lr=0.02, critic_lr=0.05, memory=100, passes=30, gain=0.5, rule="comp"),
    # what the tagged synapses receive: reward-only with a baseline (no critic noise),
    # gamma-matched traces (credit x eligibility = discount, so TD terms telescope),
    # linear dopamine (no tanh compression of big events)
    "reward_center_global": dict(lr=0.08, critic_lr=0.0, memory=20, passes=30, gain=0.5, rule="global"),
    "reward_center_comp":   dict(lr=0.08, critic_lr=0.0, memory=20, passes=30, gain=0.5, rule="comp"),
    "gamma_matched":        dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, credit=0.98),
    "linear_dop":           dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rule="none", saturate=False),
    "linear_dop_center":    dict(lr=0.08, critic_lr=0.05, memory=20, passes=30, gain=0.5, rule="global", saturate=False),
    "reward_center_global_linear": dict(lr=0.08, critic_lr=0.0, memory=20, passes=30, gain=0.5, rule="global", saturate=False),
    # prioritised replay: sleep memory holds the best fights so far (lowest boss HP),
    # not the most recent ones; the latest fight is always included
    "best20":         dict(lr=0.08, critic_lr=0.05, memory=20,  passes=30, gain=0.5, select="best"),
    "best50":         dict(lr=0.08, critic_lr=0.05, memory=50,  passes=30, gain=0.5, select="best"),
    "mix10_10":       dict(lr=0.08, critic_lr=0.05, memory=20,  passes=30, gain=0.5, select="mix"),
    "best20_lr02":    dict(lr=0.02, critic_lr=0.05, memory=20,  passes=30, gain=0.5, select="best"),
    "best50_lr02":    dict(lr=0.02, critic_lr=0.05, memory=50,  passes=30, gain=0.5, select="best"),
    "nocritic_rpe_0.3": dict(lr=0.08, critic_lr=0.0, memory=20, passes=30, gain=0.5, rpe_scale=0.3),
}


def run_variant(name, v, z, topo, train_f, test_f, every, seed, log):
    common = dict(learning_rate=v["lr"], critic_lr=v["critic_lr"], rpe_scale=v.get("rpe_scale", 0.0),
                  credit_decay=v.get("credit", 0.90), eligibility_decay=v.get("elig", 0.92))
    if v.get("rule") in ("comp", "global", "none"):
        from centered_rule import CenteredPlasticity
        p = CenteredPlasticity(topo, mode=v["rule"], saturate=v.get("saturate", True), **common)
    else:
        p = DopaminePlasticity(topo, **common)
    innate = topo.weight[p.edges].astype(np.float32).copy()
    sleep = None
    if v["memory"] > 0 and v["passes"] > 0:
        sleep = SleepConsolidation(memory_episodes=v["memory"], passes=v["passes"],
                                   gain=v["gain"], seed=seed)
    channels = list(z["channels"])
    outc = outcomes(z)
    ch = z["channel"].astype(np.int64)
    d0 = drives(z["spikes"], p, innate)
    test_mask = np.isin(z["fight"], test_f)
    train_mask = np.isin(z["fight"], train_f)
    names = list(z["extra_names"])
    boss_hp = z["extra"][:, names.index("boss_hp")]
    fight_score = {int(f): float(boss_hp[z["fight"] == f].min()) for f in np.unique(z["fight"])}  # lower = better
    seen = []  # fights fought so far, as (score, memory entry)
    prev_w = innate.copy()
    prev_dw = None
    steps, cosines, rows = [], [], []
    t0 = time.perf_counter()
    for n_done, f in enumerate(train_f, start=1):
        idx = np.flatnonzero(z["fight"] == f)
        p.reset()
        if sleep is not None:
            sleep.begin_episode()
        for i in idx:
            s = z["spikes"][i].astype(np.int32)
            c = channels[ch[i]] if ch[i] >= 0 else None
            p.update_traces(s, executed_channel=c)
            p.apply_reinforcement(float(z["reward"][i]), s, terminal=bool(z["terminal"][i]))
            if sleep is not None:
                sleep.record(z["spikes"][i], c, float(z["reward"][i]), bool(z["terminal"][i]))
        if sleep is not None:
            sleep.end_episode()
            if v.get("select") in ("best", "mix") and sleep.memory:
                latest = sleep.memory[-1]
                seen.append((fight_score[int(f)], latest))
                ranked = sorted(seen, key=lambda t: t[0])
                k = v["memory"]
                if v["select"] == "best":
                    chosen = [e for _, e in ranked[:k] if e is not latest] + [latest]
                else:
                    recent = [e for _, e in seen[-(k // 2):]]
                    best = [e for _, e in ranked if e not in recent][: k - len(recent)]
                    chosen = [e for e in best + recent if e is not latest] + [latest]
                sleep.memory = chosen[-k:] if len(chosen) > k else chosen
            sleep.consolidate(p)
        if n_done % every == 0 or n_done == len(train_f):
            w = topo.weight[p.edges].astype(np.float32).copy()
            dw = w - prev_w
            steps.append(float(np.linalg.norm(dw)) / p._w_scale)
            if prev_dw is not None and np.linalg.norm(dw) > 0 and np.linalg.norm(prev_dw) > 0:
                cosines.append(float(dw @ prev_dw / (np.linalg.norm(dw) * np.linalg.norm(prev_dw))))
            dd = drives(z["spikes"], p, w) - d0
            _, held = knowledge_score(dd, ch, outc, test_mask)
            _, insample = knowledge_score(dd, ch, outc, train_mask)
            rows.append((n_done, held, insample, cosines[-1] if cosines else float("nan")))
            prev_w, prev_dw = w, dw
    far = float(np.linalg.norm(prev_w - innate)) / p._w_scale
    rw = float(np.sqrt(np.sum(np.square(steps))))
    per = {c: float(np.mean(prev_w[p._channel_edge_positions[k]]) / np.mean(innate[p._channel_edge_positions[k]]))
           for k, c in enumerate(ACTION_CHANNELS) if len(p._channel_edge_positions[k])}
    # restore innate weights for the next variant
    topo.weight[p.edges] = innate
    log(f"\n== {name}  {v}  ({time.perf_counter() - t0:.0f}s)")
    log(f"  {'fights':>6} {'held-out':>9} {'in-sample':>9} {'cos(prev)':>9}")
    for n_done, held, ins, cs in rows:
        log(f"  {n_done:6d} {held:+9.3f} {ins:+9.3f} {cs:+9.2f}")
    log(f"  travelled {far:.1f} vs random walk of the same steps {rw:.1f} -> ratio {far / max(1e-6, rw):.2f}"
        f"  (>1.3 accumulates, <1.1 wanders); mean cosine {np.mean(cosines) if cosines else float('nan'):+.2f}")
    log("  final mean weight / innate: " + "  ".join(f"{c[:8]} {v_:.2f}" for c, v_ in per.items()))
    return rows[-1][1], rows[-1][2], far / max(1e-6, rw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--variant", action="append", default=None, help=f"one of {list(VARIANTS)}")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="append output to this file as well")
    args = ap.parse_args()
    names = args.variant or list(VARIANTS)

    out = open(args.out, "a", encoding="utf-8") if args.out else None

    def log(s):
        print(s, flush=True)
        if out:
            out.write(s + "\n")
            out.flush()

    z = dict(np.load(args.archive).items())
    fights = np.unique(z["fight"])
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    log(f"archive {args.archive}: {len(z['channel'])} steps, {len(fights)} fights")
    summary = {}
    for name in names:
        v = VARIANTS[name]
        res = []
        for split, (tr, te) in enumerate(((fights[::2], fights[1::2]), (fights[1::2], fights[::2]))):
            log(f"\n#### {name}, split {split}: train on {len(tr)} fights, test on {len(te)}")
            res.append(run_variant(f"{name}/split{split}", v, z, topo, tr, te, args.every, args.seed, log))
        summary[name] = res
    log("\n==== SUMMARY (final held-out knowledge, split 0 / split 1; travel ratio)")
    for name, res in summary.items():
        log(f"  {name:16} held-out {res[0][0]:+.3f} / {res[1][0]:+.3f}   in-sample {res[0][1]:+.3f} / {res[1][1]:+.3f}"
            f"   travel ratio {res[0][2]:.2f} / {res[1][2]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
