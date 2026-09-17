"""Homeostatic scaling must hold the level of the plastic synapses, not their pattern."""

from __future__ import annotations

import numpy as np
import pytest

from flysoul.config import CircuitConfig
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity


@pytest.fixture
def topology():
    # Fresh per test: these tests write to the shared weight array.
    return build_fly_circuit(CircuitConfig(), seed=11)


def _reward_channel(plasticity, topology, channel, reward, lessons):
    """Execute `channel` and be rewarded for it, `lessons` times."""
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.kenyon_indices[::3]] = 2
    for _ in range(lessons):
        plasticity.reset()
        plasticity.update_traces(spikes, executed_channel=channel)
        plasticity.apply_reinforcement(0.0, spikes)
        plasticity.update_traces(spikes, executed_channel=channel)
        plasticity.apply_reinforcement(reward, spikes)


def test_a_learned_preference_survives_long_reinforcement(topology):
    """Reward one action a great many times: it must end up clearly above the others.

    Under the old per-compartment cap, every compartment the fly used drifted to the same
    ceiling and the differences vanished - measured live, five channels sat at exactly
    +53% and could not be told apart. That is the failure this test pins down.
    """
    p = DopaminePlasticity(topology, learning_rate=0.3)
    innate = p.channel_weights()
    _reward_channel(p, topology, "attack_light", +1.0, lessons=300)
    _reward_channel(p, topology, "retreat", +1.0, lessons=300)

    learned = p.channel_weights()
    rel = {c: learned[c] / innate[c] for c in ACTION_CHANNELS}
    # Both rewarded channels grew, and grew alike; the untouched ones did not.
    assert rel["attack_light"] > 1.2 and rel["retreat"] > 1.2
    untouched = [rel[c] for c in ACTION_CHANNELS if c not in ("attack_light", "retreat")]
    assert max(untouched) < min(rel["attack_light"], rel["retreat"]) - 0.1, (
        "rewarded channels must remain distinguishable from unrewarded ones"
    )


def test_scaling_holds_the_total_level(topology):
    """However much is written, the total plastic drive stays near where it started."""
    p = DopaminePlasticity(topology, learning_rate=0.5)
    start = float(np.sum(p.topology.weight[p.edges]))
    for ch in ("attack_light", "roll", "parry", "advance"):
        _reward_channel(p, topology, ch, +1.0, lessons=200)
    end = float(np.sum(p.topology.weight[p.edges]))
    assert end / start < 1.0 + p.scaling_deadband + 0.15


def test_scaling_is_uniform_across_compartments(topology):
    """One dopamine event must not move one compartment's share relative to another's
    except through the eligibility and credit that event carried."""
    p = DopaminePlasticity(topology)
    # Inflate everything artificially past the dead band, then trigger scaling alone.
    p.topology.weight[p.edges] *= 1.6
    before = p.channel_weights()
    p._homeostatic_scaling()
    after = p.channel_weights()
    ratios = [after[c] / before[c] for c in ACTION_CHANNELS]
    assert max(ratios) - min(ratios) < 1e-4
    assert ratios[0] < 1.0
