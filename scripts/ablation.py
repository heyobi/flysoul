"""Lesion study: is the behaviour actually coming from the connectome?

The honest way to answer "is this the fly deciding, or is it hand-written logic?" is not
to read the code and take its word for it. It is to damage the circuit and see whether
the behaviour changes. Logic outside the brain survives a lesion; behaviour produced by
the brain does not.

Run:
    python scripts/ablation.py --episodes 40

Each condition runs the same agent against the same mock arena with one part of the
circuit cut, and reports what the fly did. If the action distributions come out the same
across conditions, the connectome is not driving the behaviour and something else is.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.calibration import calibrate_or_load
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.obs import parse_obs
from flysoul.env.mock_env import MockIudexEnv
from flysoul.env.souls_wrapper import ValidActionInfo
from flysoul.motor.decoder import IDLE, MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder


def _incoming_mask(topology, targets, excitatory_only=True):
    is_target = np.zeros(topology.num_neurons, dtype=bool)
    is_target[np.concatenate(targets)] = True
    mask = is_target[topology.post]
    if excitatory_only:
        mask &= topology.weight > 0
    return mask


def apply_lesion(topology, baseline_weight, name, rng):
    """Restore the intact weights, then cut one part of the circuit."""
    topology.weight[:] = baseline_weight

    if name == "intact":
        return "the circuit as built"

    if name == "mb_lesion":
        # Silence the mushroom body's output: the learned-valence pathway is gone, the
        # innate reflexes and the central complex remain.
        mbon = np.concatenate([topology.mbon_compartments[c] for c in ACTION_CHANNELS])
        is_mbon = np.zeros(topology.num_neurons, dtype=bool)
        is_mbon[mbon] = True
        row_len = np.diff(topology.ptr)
        pre = np.repeat(np.arange(len(row_len)), row_len)
        topology.weight[is_mbon[pre]] = 0.0
        return "MBON output cut (no learned valence)"

    if name == "dn_deaf":
        # Cut every excitatory afferent to the descending pools. The brain can still
        # think; it just cannot reach the motor output.
        pools = [topology.motor_indices[c] for c in ACTION_CHANNELS]
        topology.weight[_incoming_mask(topology, pools)] = 0.0
        return "descending pools deafened (brain cannot reach the motor output)"

    if name == "sensory_deaf":
        # Cut the sensory afferents: the circuit runs, but on no information.
        sensory = [
            topology.retina_indices, topology.motion_indices,
            topology.compass_indices, topology.proprioceptor_indices,
        ]
        is_sensory = np.zeros(topology.num_neurons, dtype=bool)
        is_sensory[np.concatenate(sensory)] = True
        row_len = np.diff(topology.ptr)
        pre = np.repeat(np.arange(len(row_len)), row_len)
        topology.weight[is_sensory[pre]] = 0.0
        return "sensory afferents cut (circuit intact, no information)"

    if name == "shuffled":
        # Keep every synapse and every weight, destroy only which cell talks to which.
        perm = rng.permutation(len(topology.weight))
        topology.weight[:] = baseline_weight[perm]
        return "weights shuffled across synapses (same statistics, no structure)"

    raise ValueError(name)


def run_condition(topology, bio, cfg, env, episodes, seed):
    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, bio)
    encoder = SensoryEncoder(topology, bio)
    decoder = MotorDecoder(topology, step_ms=100.0)
    plasticity = DopaminePlasticity(topology, learning_rate=cfg.learning_rate)
    rng = np.random.default_rng(seed)

    actions = collections.Counter()
    hits = steps = 0
    boss_hp_end = []
    for _ in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        engine.reset_state()
        encoder.reset()
        plasticity.reset()
        state = parse_obs(obs, info)
        for _ in range(20):
            engine.step(encoder.encode(state), 100.0)
        encoder.reset(state.player_hp, state.distance)

        terminated = truncated = False
        while not (terminated or truncated):
            steps += 1
            spikes = engine.step(encoder.encode(state), 100.0)
            valid = decoder.channels_for_valid_actions(state.valid_actions, mock=True)
            channel, rates = decoder.decode(spikes, valid_channels=valid)
            actions[channel] += 1
            prev = state.boss_hp
            obs, reward, terminated, truncated, info = env.step(decoder.to_mock_action(channel))
            state = parse_obs(obs, info)
            if state.boss_hp < prev - 1e-6:
                hits += 1
            encoder.set_efference(channel)
            plasticity.update_traces(spikes, channel)
            plasticity.apply_reinforcement(reward)
        boss_hp_end.append(state.boss_hp)

    total = max(1, sum(actions.values()))
    return {
        "hits_per_episode": hits / episodes,
        "boss_hp": float(np.mean(boss_hp_end)),
        "steps_per_episode": steps / episodes,
        "mix": {k: v / total for k, v in actions.most_common()},
        "idle_frac": actions[IDLE] / total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=250,
                    help="Step budget per episode (a lesioned fly may never die).")
    args = ap.parse_args()

    cfg, bio = CircuitConfig(), BioPhysicsConfig()
    topology = build_fly_circuit(cfg, seed=args.seed)
    report, cached = calibrate_or_load(topology, cfg, bio, seed=args.seed)
    print(f"calibration: {report.format()}{' (cached)' if cached else ''}\n")

    baseline = topology.weight.copy()
    # A lesioned fly can stand still for the whole episode, so the step budget has to
    # be short or the silent conditions dominate the runtime.
    env = ValidActionInfo(MockIudexEnv(max_steps=args.max_steps))
    rng = np.random.default_rng(args.seed)

    conditions = ["intact", "mb_lesion", "dn_deaf", "sensory_deaf", "shuffled"]
    results = {}
    for name in conditions:
        description = apply_lesion(topology, baseline, name, rng)
        results[name] = run_condition(topology, bio, cfg, env, args.episodes, args.seed)
        print(f"{name:14} {description}")
    topology.weight[:] = baseline

    print(f"\n{'condition':14} {'hits/ep':>8} {'boss HP':>8} {'steps':>7} {'idle':>7}  top actions")
    print("-" * 86)
    for name in conditions:
        r = results[name]
        top = ", ".join(f"{k} {100*v:.0f}%" for k, v in list(r["mix"].items())[:3])
        print(f"{name:14} {r['hits_per_episode']:8.2f} {100*r['boss_hp']:7.1f}% "
              f"{r['steps_per_episode']:7.1f} {100*r['idle_frac']:6.0f}%  {top}")

    print(
        "\nIf the connectome is driving the behaviour, these rows must differ: deafening the\n"
        "descending pools should leave the fly idle, cutting the sensory afferents should\n"
        "make it act without regard to the boss, and shuffling the weights should destroy\n"
        "the action mix. Rows that come out identical would mean the behaviour is being\n"
        "produced somewhere other than the circuit."
    )


if __name__ == "__main__":
    main()
