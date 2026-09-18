"""Can the mushroom body tell *when* in the boss's attack it is?

Live measurement: the fly rolls a third of the time too early in the attack animation
(0.3-0.6 s, 74-83% still hit) and does not improve over a hundred fights, although the
plasticity rule can in principle learn timing. It can only learn it if the Kenyon cell
population code differs between attack phases - the KC->MBON synapses are the only
plastic ones, so whatever the KCs do not distinguish, the fly cannot learn to treat
differently.

This drives the encoder with the same combat state at different points in the attack
animation, lets the circuit settle, and compares the sets of active Kenyon cells. High
overlap between 0.4 s and 0.8 s means "roll now" and "too early" look the same to the
learner and the timing failure is a coding problem, not a learning problem.

    python scripts/kc_phase_code.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig  # noqa: E402
from flysoul.connectome.calibration import calibrate_or_load  # noqa: E402
from flysoul.connectome.engine import ConnectomeEngine  # noqa: E402
from flysoul.connectome.graph import build_fly_circuit  # noqa: E402
from flysoul.env.obs import CombatState  # noqa: E402
from flysoul.sensory.encoder import SensoryEncoder  # noqa: E402

PHASES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def kc_set(engine, encoder, topo, state, settle_ms=300.0, read_ms=100.0, seed=0):
    engine.reset_state()
    engine.seed(seed)
    encoder.reset()
    drive = encoder.encode(state)
    engine.step(drive, duration_ms=settle_ms)
    counts = engine.step(encoder.encode(state), duration_ms=read_ms)
    kc = counts[topo.kenyon_indices]
    return set(np.flatnonzero(kc > 0).tolist()), int((kc > 0).sum())


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(cfg, seed=42)
    calibrate_or_load(topo, cfg, bio, seed=42)
    engine = ConnectomeEngine(topo.ptr, topo.post, topo.weight, bio)
    encoder = SensoryEncoder(topo, bio)

    base = dict(player_hp=0.8, player_sp=0.9, boss_hp=0.7, distance=2.5, angle=0.0,
                lock_on=True, player_can_act=True)
    idle = CombatState(boss_attacking=False, boss_anim_time=0.0, **base)
    sets = {}
    sizes = {}
    for t in PHASES:
        s, n = kc_set(engine, encoder, topo, CombatState(boss_attacking=True, boss_anim_time=t, **base))
        sets[t] = s
        sizes[t] = n
    idle_set, idle_n = kc_set(engine, encoder, topo, idle)
    # Same state, different noise seed: the floor of "how similar is identical".
    same_again, _ = kc_set(engine, encoder, topo, CombatState(boss_attacking=True, boss_anim_time=0.8, **base), seed=1)

    print(f"{len(topo.kenyon_indices)} Kenyon cells; active per phase: "
          + ", ".join(f"{t:.1f}s={sizes[t]}" for t in PHASES) + f"; idle={idle_n}")
    print("\nJaccard overlap of active KC sets (1.0 = identical code):")
    print("        " + "".join(f"{t:>7.1f}" for t in PHASES) + "   idle")
    for a in PHASES:
        row = "".join(f"{jaccard(sets[a], sets[b]):7.2f}" for b in PHASES)
        print(f"  {a:.1f}s {row}{jaccard(sets[a], idle_set):7.2f}")
    print(f"\n  same state, other noise seed (0.8s vs 0.8s): {jaccard(sets[0.8], same_again):.2f}")

    early_vs_late = jaccard(sets[0.4], sets[0.8])
    attack_vs_idle = np.mean([jaccard(sets[t], idle_set) for t in PHASES])
    print()
    if early_vs_late > 0.8:
        print(f"[!] 0.4s and 0.8s share {early_vs_late:.0%} of their Kenyon cells: 'too early' and "
              "'now' are the same pattern to the plastic synapses. Roll timing cannot be learned "
              "from this code, however long the fly trains.")
    elif early_vs_late > 0.5:
        print(f"[*] 0.4s and 0.8s share {early_vs_late:.0%} of their Kenyon cells: timing is "
              "weakly separable; learning it will be slow.")
    else:
        print(f"[+] 0.4s and 0.8s share only {early_vs_late:.0%}: the phase code is separable; "
              "the timing failure is a learning-speed problem, not a coding problem.")
    print(f"    attack vs idle overlap: {attack_vs_idle:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
