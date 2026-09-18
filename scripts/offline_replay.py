"""Try a learning rule on real fights before trusting it live.

The agent archives every step it lives (spikes, the action it took, the reinforcement
that followed, and what the boss was doing) to `checkpoints/replay_archive.npz`. This
replays those fights through candidate plasticity settings and reports what each one
would have learned - in seconds, from the same experience, with nothing changed live.

It cannot say how the fly would then behave (the fights were lived under the old
behaviour), but it answers the question that decided most of this project's failed
interventions: does the rule write credit where it belongs? For roll timing that means
the roll compartment should end up driven more by the Kenyon cells that fire late in
the boss's swing (0.6-1.0 s, where a roll works) than by the ones that fire early.

    python scripts/offline_replay.py checkpoints/replay_archive.npz
    python scripts/offline_replay.py archive.npz --variant tagged:elig=0.92 --variant old:elig=0.92
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
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402

PHASES = ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9.9))


def old_update_traces(self, spike_counts, executed_channel=None):
    """The rule before decision-time tagging: eligibility on every compartment, every step."""
    if len(self.edges) == 0:
        return
    self.eligibility *= self.eligibility_decay
    pre = spike_counts[self.pre].astype(np.float32)
    post = spike_counts[self.post].astype(np.float32)
    np.add(self.eligibility, pre * (post + 0.1), out=self.eligibility)
    self.credit *= self.credit_decay
    if executed_channel is not None and executed_channel in ACTION_CHANNELS:
        self.credit[ACTION_CHANNELS.index(executed_channel)] = 1.0


def parse_variant(spec: str) -> tuple[str, dict]:
    """'tagged:elig=0.92,credit=0.9,gain=0.5' -> ('tagged', {...})."""
    name, _, opts = spec.partition(":")
    kw = {}
    for item in filter(None, opts.split(",")):
        k, v = item.split("=")
        kw[k] = float(v)
    return name, kw


def make_plasticity(topo, rule: str, kw: dict) -> DopaminePlasticity:
    p = DopaminePlasticity(
        topo,
        learning_rate=kw.get("lr", CircuitConfig().learning_rate),
        eligibility_decay=kw.get("elig", CircuitConfig().eligibility_decay),
        credit_decay=kw.get("credit", 0.90),
    )
    if rule == "old":
        p.update_traces = old_update_traces.__get__(p, DopaminePlasticity)
    elif rule != "tagged":
        raise SystemExit(f"unknown rule {rule!r}; use 'tagged' or 'old'")
    return p


def roll_drive_by_phase(p: DopaminePlasticity, z) -> dict:
    """Mean KC->roll synaptic drive for steps grouped by boss phase.

    Uses the archived spike patterns as probes: for each step, sum w * spikes over the
    plastic synapses into the roll compartment. Higher drive at 0.6-1.0 s than at
    0.3-0.6 s means the rule taught 'roll later'.
    """
    k = ACTION_CHANNELS.index("roll")
    sel = np.flatnonzero(p.channel == k)
    pre = p.pre[sel]
    w = p.topology.weight[p.edges[sel]]
    spikes = z["spikes"].astype(np.float32)
    drive = spikes[:, pre] @ w
    out = {}
    attacking = z["attacking"]
    for lo, hi in PHASES:
        m = attacking & (z["anim_t"] >= lo) & (z["anim_t"] < hi)
        out[f"{lo:.1f}-{hi:.1f}s"] = (float(drive[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)
    m = ~attacking
    out["idle"] = (float(drive[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)
    return out


def replay(p: DopaminePlasticity, z, gain: float, passes: int) -> None:
    lr0, c0 = p.lr, p.critic_lr
    p.lr, p.critic_lr = lr0 * gain, c0 * gain
    fights = np.unique(z["fight"])
    channels = list(z["channels"])
    for _ in range(passes):
        for f in fights:
            idx = np.flatnonzero(z["fight"] == f)
            p.reset()
            for i in idx:
                spikes = z["spikes"][i].astype(np.int32)
                ch = channels[z["channel"][i]] if z["channel"][i] >= 0 else None
                p.update_traces(spikes, executed_channel=ch)
                p.apply_reinforcement(float(z["reward"][i]), spikes, terminal=bool(z["terminal"][i]))
    p.lr, p.critic_lr = lr0, c0
    p.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?", default="checkpoints/replay_archive.npz")
    ap.add_argument("--variant", action="append", default=None,
                    help="rule[:k=v,...]; rule is 'tagged' or 'old'; keys elig, credit, lr, gain")
    ap.add_argument("--passes", type=int, default=1, help="passes over the archive")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # NpzFile re-reads and decompresses an array on every access; indexing it inside
    # a loop turns a one-second replay into half an hour. Materialise once.
    z = {k: v for k, v in np.load(args.archive).items()}
    n, fights = len(z["reward"]), len(np.unique(z["fight"]))
    print(f"archive: {n} steps from {fights} fights; "
          f"{int((z['reward'] > 0).sum())} rewarded, {int((z['reward'] < 0).sum())} punished steps")

    variants = args.variant or ["tagged:elig=0.92", "old:elig=0.92", "tagged:elig=0.80", "old:elig=0.80"]
    cfg, bio = CircuitConfig(), BioPhysicsConfig()

    # Innate reference: the drive curve before any learning.
    topo = build_fly_circuit(cfg, seed=args.seed)
    calibrate_or_load(topo, cfg, bio, seed=args.seed)
    base = roll_drive_by_phase(DopaminePlasticity(topo), z)
    keys = list(base.keys())
    print(f"\n{'variant':24}" + "".join(f"{k:>12}" for k in keys) + "   late/early")
    print(f"{'(steps per bucket)':24}" + "".join(f"{base[k][1]:>12d}" for k in keys))
    print(f"{'innate':24}" + "".join(f"{base[k][0]:12.2f}" for k in keys)
          + f"   {base['0.6-1.0s'][0] / max(1e-6, base['0.3-0.6s'][0]):8.2f}")

    for spec in variants:
        rule, kw = parse_variant(spec)
        topo = build_fly_circuit(cfg, seed=args.seed)
        calibrate_or_load(topo, cfg, bio, seed=args.seed)
        p = make_plasticity(topo, rule, kw)
        replay(p, z, gain=kw.get("gain", 1.0), passes=args.passes)
        cur = roll_drive_by_phase(p, z)
        ratio = cur["0.6-1.0s"][0] / max(1e-6, cur["0.3-0.6s"][0])
        print(f"{spec:24}" + "".join(f"{cur[k][0]:12.2f}" for k in keys) + f"   {ratio:8.2f}")

    print("\nlate/early > innate means the rule taught the roll compartment to prefer the "
          "window where a roll works; the bigger the ratio, the sharper the timing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
