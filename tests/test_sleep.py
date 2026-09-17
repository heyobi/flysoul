"""Sleep consolidation: the fly must learn more from a fight than the one pass it lived."""

from __future__ import annotations

import numpy as np
import pytest

from flysoul.config import CircuitConfig
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.connectome.sleep import SleepConsolidation


@pytest.fixture(scope="module")
def topology():
    return build_fly_circuit(CircuitConfig(), seed=7)


def _episode(sleep, topology, channel, reward, steps=6, kc_slice=slice(0, 30)):
    sleep.begin_episode()
    for i in range(steps):
        spikes = np.zeros(topology.num_neurons, dtype=np.int32)
        spikes[topology.kenyon_indices[kc_slice]] = 2
        sleep.record(spikes, channel, reward, terminal=(i == steps - 1))
    return sleep.end_episode()


def test_memory_keeps_only_recent_fights(topology):
    sleep = SleepConsolidation(memory_episodes=5, passes=3, min_transitions=4)
    for _ in range(8):
        _episode(sleep, topology, "advance", 0.0)
    assert len(sleep.memory) == 5
    # A fight too short to carry a lesson is not filed.
    assert _episode(sleep, topology, "advance", 0.0, steps=2) == 5


def test_empty_memory_is_a_quiet_night(topology):
    sleep = SleepConsolidation()
    plasticity = DopaminePlasticity(topology)
    before = plasticity.topology.weight.copy()
    report = sleep.consolidate(plasticity)
    assert report.transitions == 0 and report.passes == 0
    assert np.array_equal(before, plasticity.topology.weight)


def test_replay_reinforces_the_rewarded_channel(topology):
    """A rewarded action replayed in sleep must strengthen its compartment and no other."""
    plasticity = DopaminePlasticity(topology, learning_rate=0.3)
    sleep = SleepConsolidation(memory_episodes=10, passes=6, gain=1.0, seed=1)
    _episode(sleep, topology, "attack_light", reward=1.0)

    before = plasticity.channel_weights()
    report = sleep.consolidate(plasticity)
    after = plasticity.channel_weights()

    assert report.passes == 6 and report.transitions == 36
    assert report.weight_shift > 0.0
    gain = {c: after[c] - before[c] for c in ACTION_CHANNELS}
    assert gain["attack_light"] > 0.0, "replayed reward must strengthen the executed channel"
    others = [gain[c] for c in ACTION_CHANNELS if c != "attack_light"]
    assert gain["attack_light"] > max(others) + 1e-6


def test_sleep_restores_waking_learning_rates(topology):
    plasticity = DopaminePlasticity(topology, learning_rate=0.2, critic_lr=0.05)
    sleep = SleepConsolidation(passes=2, gain=0.25)
    _episode(sleep, topology, "roll", reward=-0.5)
    sleep.consolidate(plasticity)
    assert plasticity.lr == pytest.approx(0.2)
    assert plasticity.critic_lr == pytest.approx(0.05)
    # Traces are clean for the next waking fight.
    assert float(np.max(plasticity.eligibility)) == 0.0
    assert plasticity._prev_features is None


def test_schedule_starts_with_the_latest_fight_and_draws_from_memory(topology):
    sleep = SleepConsolidation(memory_episodes=4, passes=12, seed=3)
    for _ in range(4):
        _episode(sleep, topology, "parry", 0.0)
    order = sleep.schedule()
    assert order[0] == 3
    assert len(order) == 12
    assert set(order) <= {0, 1, 2, 3}


def test_a_night_fits_its_budget_of_replayed_steps(topology):
    """Long fights must shorten the night, not lengthen it past the loading screen."""
    sleep = SleepConsolidation(memory_episodes=5, passes=30, max_transitions=100, seed=2)
    for _ in range(3):
        _episode(sleep, topology, "advance", 0.0, steps=40)
    order = sleep.schedule()
    assert order[0] == 2
    assert 1 <= len(order) <= 100 // 40
    assert sum(len(sleep.memory[i]) for i in order) <= 100


def test_on_step_sees_every_replayed_transition(topology):
    plasticity = DopaminePlasticity(topology)
    sleep = SleepConsolidation(passes=3)
    _episode(sleep, topology, "advance", 0.1, steps=5)
    seen = []
    sleep.consolidate(plasticity, on_step=lambda p, n, m, tr, done, total: seen.append((p, done, total)))
    assert len(seen) == 15
    assert seen[-1] == (2, 15, 15)
