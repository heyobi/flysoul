"""Does a change to the sensory encoding raise what the mushroom body can learn?

Replays archived fights through the sensory encoder and the circuit *without* the game:
each archived step's context (distance, bearing, boss animation and phase, health,
action lock) is turned back into a CombatState, encoded, and run through the
connectome for one 100 ms step, fight by fight, exactly as live. The Kenyon cell code
that comes out is the code the plastic synapses would have to learn from. Its ceiling
- the best held-out prediction of the action's outcome that any linear readout of it
can reach - is then compared between encoder variants.

This is how an encoding change is judged before it is trusted live: the fights are real,
the brain is the real model, only what it is shown differs.

    python scripts/kc_ceiling_sim.py checkpoints/replay_archive.npz --pattern 0 --pattern 64
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.engine import ConnectomeEngine  # noqa: E402
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit  # noqa: E402
from flysoul.env.obs import CombatState  # noqa: E402
from flysoul.sensory.encoder import SensoryEncoder  # noqa: E402
from offline_ceiling import ridge, spearman  # noqa: E402
from offline_search import outcomes  # noqa: E402


def states_from_archive(z) -> list[CombatState]:
    names = list(z["extra_names"])
    ex = z["extra"]
    col = {n: ex[:, i] for i, n in enumerate(names)}
    out = []
    for i in range(len(z["reward"])):
        out.append(CombatState(
            player_hp=float(col["player_hp"][i]), player_sp=1.0, boss_hp=float(col["boss_hp"][i]),
            distance=float(col["distance"][i]), angle=float(col["angle"][i]),
            boss_attacking=bool(z["attacking"][i]), boss_staggered=bool(col["boss_staggered"][i] > 0.5),
            boss_anim_time=float(z["anim_t"][i]), boss_anim_id=int(col["boss_anim_id"][i]),
            player_can_act=bool(col["player_can_act"][i] > 0.5), lock_on=True,
        ))
    return out


def simulate_kc(z, states, n_pattern: int, seed: int, kc_inh: float | None = None,
                kc_target: float | None = None) -> np.ndarray:
    cfg = dataclasses.replace(CircuitConfig(), num_retina_pattern=n_pattern)
    if kc_inh is not None:
        cfg = dataclasses.replace(cfg, kc_inhibition_ratio=kc_inh)
    bio = BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=seed)
    t0 = time.perf_counter()
    if kc_target is not None:
        # a different Kenyon sparsity target: calibrate directly, bypassing the cache
        from flysoul.connectome.calibration import calibrate_circuit
        rep, cached = calibrate_circuit(topo, bio, kc_sparsity_target=kc_target), False
    else:
        rep, cached = calibrate_or_load(topo, cfg, bio, seed=seed)
    print(f"  pattern={n_pattern}: {topo.num_neurons} neurons, calibration "
          f"{'cached' if cached else f'{time.perf_counter() - t0:.0f}s'} ({'converged' if rep.converged else 'NOT converged'})")
    engine = ConnectomeEngine(topo.ptr, topo.post, topo.weight, bio)
    engine.seed(seed)
    enc = SensoryEncoder(topo, bio)
    channels = list(z["channels"])
    kc = np.zeros((len(states), len(topo.kenyon_indices)), dtype=np.float32)
    fight = z["fight"]
    t0 = time.perf_counter()
    for i, st in enumerate(states):
        if i == 0 or fight[i] != fight[i - 1]:
            engine.reset_state(); enc.reset()
            engine.step(enc.encode(st), duration_ms=300.0)  # settle, as live does
        counts = engine.step(enc.encode(st), duration_ms=100.0)
        kc[i] = counts[topo.kenyon_indices] > 0
        ch = z["channel"][i]
        enc.set_efference(channels[ch] if ch >= 0 else None)
    freq = kc.mean(0)
    print(f"  simulated {len(states)} steps in {time.perf_counter() - t0:.0f}s; "
          f"KC sparsity {kc.mean():.1%}; cells firing on >50% of steps {np.mean(freq > 0.5):.0%}, never {np.mean(freq == 0):.0%}")
    return kc


def hebbian(A, z, outc, train, test) -> float:
    """What a three-factor rule computes: w = sum of input x (outcome - mean outcome),
    scored held-out the same way. The gap to the ceiling is what decorrelation buys."""
    px, py = [], []
    for ch in ("attack_light", "roll", "advance", "retreat"):
        k = ACTION_CHANNELS.index(ch)
        m = (z["channel"] == k) & (outc != 0)
        tr, te = m & train, m & test
        if tr.sum() < 15 or te.sum() < 10:
            continue
        w = A[tr].T @ (outc[tr] - outc[tr].mean())
        pred = A[te] @ w
        r = np.empty(len(pred)); r[np.argsort(pred)] = np.arange(len(pred)); px.append(r / max(1, len(pred) - 1)); py.append(outc[te])
    return spearman(np.concatenate(px), np.concatenate(py)) if px else float("nan")


def ceiling(X, z, outc, train, test) -> dict:
    res = {}
    pooled_p, pooled_t = [], []
    for ch in ("attack_light", "roll", "advance", "retreat"):
        k = ACTION_CHANNELS.index(ch)
        m = (z["channel"] == k) & (outc != 0)
        tr, te = m & train, m & test
        if tr.sum() < 15 or te.sum() < 10:
            res[ch] = float("nan"); continue
        best = max((spearman(X[te] @ ridge(X[tr], outc[tr], lam), outc[te]), lam) for lam in (0.3, 1, 3, 10, 30))
        res[ch] = best[0]
        pooled_p.append(X[te] @ ridge(X[tr], outc[tr], best[1])); pooled_t.append(outc[te])
    res["pooled"] = spearman(np.concatenate(pooled_p), np.concatenate(pooled_t)) if pooled_p else float("nan")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--pattern", type=int, action="append", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fights", type=int, default=0, help="use only the first N fights (speed)")
    ap.add_argument("--kc-sparsity", type=float, default=None,
                    help="Kenyon sparsity target for calibration (live 0.10); calibrates without the cache")
    ap.add_argument("--kc-inhibition", type=float, action="append", default=None,
                    help="APL feedback strength(s) to simulate (CircuitConfig.kc_inhibition_ratio, live 1.35)")
    args = ap.parse_args()
    z = {k: v for k, v in np.load(args.archive).items()}
    if args.max_fights:
        keep = z["fight"] < np.unique(z["fight"])[: args.max_fights].max() + 1
        z = {k: (v[keep] if hasattr(v, "shape") and len(v.shape) and v.shape[0] == len(keep) else v) for k, v in z.items()}
    states = states_from_archive(z)
    outc = outcomes(z)
    fights = np.unique(z["fight"])
    train, test = np.isin(z["fight"], fights[::2]), np.isin(z["fight"], fights[1::2])
    print(f"{len(states)} steps, {len(fights)} fights, {len(np.unique(z['extra'][:, list(z['extra_names']).index('boss_anim_id')]))} boss animations")
    variants = args.pattern or [0, 64]
    inhs = args.kc_inhibition or [None]
    print(f"\n{'encoder':24} {'ceiling':>8} {'hebbian':>8} {'attack_light':>13} {'roll':>7} {'advance':>9} {'retreat':>9}")
    for n in variants:
        for inh in inhs:
            A = simulate_kc(z, states, n, args.seed, inh, args.kc_sparsity)
            X = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-6)
            r = ceiling(X, z, outc, train, test)
            h = hebbian(A, z, outc, train, test)
            label = f"pattern={n} apl={inh if inh is not None else 'live'}" + (f" ks={args.kc_sparsity}" if args.kc_sparsity else "")
            print(f"{label:24} {r['pooled']:+8.3f} {h:+8.3f} {r['attack_light']:+13.3f} {r['roll']:+7.3f} "
                  f"{r['advance']:+9.3f} {r['retreat']:+9.3f}", flush=True)
    print("\nHigher is a code the plastic synapses can learn more from. The archived KC code "
          "(what the live fly actually saw) is the reference for pattern=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
