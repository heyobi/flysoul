"""How much wall-clock does one environment step cost the brain?

SoulsGym advances the game 100 ms per step. If simulating 100 ms of the circuit plus the
plasticity update takes longer than that, the game speed multiplier cannot buy anything;
if it is much shorter, running the game at 3x (as the SoulsGym author did) triples the
samples per hour for free. Measures the pieces separately so the bottleneck is named.

    python scripts/bench_step.py [--steps 60]
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
from flysoul.connectome.engine import ConnectomeEngine  # noqa: E402
from flysoul.connectome.graph import build_fly_circuit  # noqa: E402
from flysoul.connectome.plasticity import DopaminePlasticity  # noqa: E402
from flysoul.env.mock_env import MockIudexEnv  # noqa: E402
from flysoul.env.obs import parse_obs  # noqa: E402
from flysoul.motor.decoder import MotorDecoder  # noqa: E402
from flysoul.sensory.encoder import SensoryEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    circuit_cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topo = build_fly_circuit(circuit_cfg, seed=42)
    calibrate_or_load(topo, circuit_cfg, bio, seed=42)
    engine = ConnectomeEngine(topo.ptr, topo.post, topo.weight, bio)
    encoder = SensoryEncoder(topo, bio)
    decoder = MotorDecoder(topo, step_ms=100.0)
    plast = DopaminePlasticity(topo, learning_rate=circuit_cfg.learning_rate,
                               eligibility_decay=circuit_cfg.eligibility_decay)
    env = MockIudexEnv()
    obs, info = env.reset()
    state = parse_obs(obs, info)

    # Warm the JIT before timing anything.
    for _ in range(5):
        engine.step(encoder.encode(state), duration_ms=100.0)

    t_enc = t_sim = t_dec = t_pl = 0.0
    for _ in range(args.steps):
        t0 = time.perf_counter(); drive = encoder.encode(state); t1 = time.perf_counter()
        spikes = engine.step(drive, duration_ms=100.0); t2 = time.perf_counter()
        channel, rates = decoder.decode(spikes, explore=False, valid_channels=None); t3 = time.perf_counter()
        encoder.set_efference(channel)
        plast.update_traces(spikes, executed_channel=channel)
        plast.apply_reinforcement(0.0, spikes, terminal=False); t4 = time.perf_counter()
        t_enc += t1 - t0; t_sim += t2 - t1; t_dec += t3 - t2; t_pl += t4 - t3
        action, channel = decoder.to_game_action(channel, rates, mock=True, lock_on=True)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
        state = parse_obs(obs, info)

    n = args.steps
    total = (t_enc + t_sim + t_dec + t_pl) / n * 1000
    print(f"{topo.num_neurons:,} neurons, {len(topo.post):,} synapses, {n} steps of 100 ms")
    print(f"  encode      {t_enc / n * 1000:7.1f} ms")
    print(f"  LIF sim     {t_sim / n * 1000:7.1f} ms")
    print(f"  decode      {t_dec / n * 1000:7.1f} ms")
    print(f"  plasticity  {t_pl / n * 1000:7.1f} ms")
    print(f"  brain total {total:7.1f} ms per 100 ms game step  "
          f"({100.0 / max(total, 1e-6):.1f}x real time)")
    if total < 33:
        print("[+] The brain keeps up with the game at 3x speed; the game is the bottleneck.")
    elif total < 100:
        print(f"[*] The brain allows at most {100.0 / total:.1f}x game speed.")
    else:
        print("[!] The brain is slower than the game; speeding the game up cannot help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
