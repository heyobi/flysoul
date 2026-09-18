"""Which reward shaping should the fly learn from? Ask the archived fights first.

`--aggression` and `--impatience` reshape the dopamine signal (damage dealt weighted
up; a small cost for steps without damage). They were chosen by argument, never by
measurement. The archive stores the raw environment reward and the damage dealt at each
step, so any shaping can be recomputed exactly and replayed through the live plasticity
settings; the score is the same as in offline_search: on held-out fights, did the rule
raise the drive of actions that were followed by a hit and lower it where damage came?

Several run archives can be merged (fight ids are offset), so the whole life of one
circuit is used.

    python scripts/offline_shaping.py run1/archive.npz run2/archive.npz ...
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
from flysoul.connectome.graph import build_fly_circuit  # noqa: E402
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402
from offline_search import LIVE, drive_matrix, outcomes, score  # noqa: E402


def merge(paths: list[str]) -> dict:
    parts = [{k: v for k, v in np.load(p).items()} for p in paths]
    out = {}
    for k in parts[0]:
        if k in ("channels", "extra_names"):
            out[k] = parts[0][k]
        elif k != "fight":
            out[k] = np.concatenate([z[k] for z in parts])
    # fight ids restart in every run; offset them so fights stay distinct
    fights = []
    offset = 0
    for z in parts:
        fights.append(z["fight"] + offset)
        offset += int(z["fight"].max()) + 1
    out["fight"] = np.concatenate(fights)
    return out


def shaped_reward(z, aggression: float, impatience: float) -> np.ndarray:
    names = list(z["extra_names"])
    r_env = z["extra"][:, names.index("reward_env")].astype(np.float32)
    hit = z["hit"].astype(np.float32)
    r = r_env + hit * (aggression - 1.0)
    r = r - impatience * (hit <= 1e-6).astype(np.float32)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggression", type=float, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--impatience", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    args = ap.parse_args()

    z = merge(args.archives)
    fights = np.unique(z["fight"])
    outc = outcomes(z)
    print(f"merged {len(args.archives)} archives: {len(outc)} steps, {len(fights)} fights; "
          f"outcomes +{int((outc > 0).sum())} / -{int((outc < 0).sum())}")
    cfg_c, bio = CircuitConfig(), BioPhysicsConfig()

    def topo_factory():
        t = build_fly_circuit(cfg_c, seed=args.seed)
        calibrate_or_load(t, cfg_c, bio, seed=args.seed)
        return t

    base = drive_matrix(DopaminePlasticity(topo_factory()), z["spikes"].astype(np.float32))
    splits = {"A": (fights[::2], fights[1::2]), "B": (fights[1::2], fights[::2])}
    print(f"\n{'aggression':>10} {'impatience':>10} {'corr A':>8} {'corr B':>8} {'mean':>7} {'roll late/early':>16}")
    results = []
    for a in args.aggression:
        for im in args.impatience:
            zz = dict(z); zz["reward"] = shaped_reward(z, a, im)
            vals, ratios = [], []
            for tr, te in splits.values():
                r = score(dict(LIVE), zz, base, outc, tr, te, topo_factory)
                vals.append(r["credit_corr"]); ratios.append(r.get("roll_late_early", float("nan")))
            results.append((np.mean(vals), a, im))
            flag = "  <- live" if (a == 8 and im == 0.02) else ""
            print(f"{a:10.0f} {im:10.2f} {vals[0]:+8.3f} {vals[1]:+8.3f} {np.mean(vals):+7.3f} {np.nanmean(ratios):16.2f}{flag}")
    results.sort(reverse=True)
    print(f"\nbest: aggression {results[0][1]:.0f}, impatience {results[0][2]:.2f} -> {results[0][0]:+.3f}")
    print("Only a setting that beats the live one on both splits by a clear margin is worth a live trial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
