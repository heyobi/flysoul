"""The synaptic tag is set when the command is issued, on the cells active then."""

from __future__ import annotations

import numpy as np
import pytest

from flysoul.config import CircuitConfig
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity


@pytest.fixture
def topology():
    return build_fly_circuit(CircuitConfig(), seed=5)


def _spikes(topology, kc_slice):
    s = np.zeros(topology.num_neurons, dtype=np.int32)
    s[topology.kenyon_indices[kc_slice]] = 3
    return s


def _mean_weight(p, channel, kc_slice):
    """Mean plastic weight from the given Kenyon cells into one compartment."""
    kcs = set(p.topology.kenyon_indices[kc_slice].tolist())
    k = ACTION_CHANNELS.index(channel)
    sel = [i for i in np.flatnonzero(p.channel == k) if int(p.pre[i]) in kcs]
    return float(np.mean(p.topology.weight[p.edges[sel]])) if sel else float("nan")


def test_blame_lands_on_the_cells_active_at_the_decision(topology):
    """Roll early (cells A active), wait locked in the animation (cells B active), get
    hit: the A->roll synapses must be depressed more than the B->roll synapses.

    Under the old always-accumulating trace the opposite happened - B was freshest
    when the punishment arrived - and the fly could never learn to roll later.
    """
    p = DopaminePlasticity(topology, learning_rate=0.5)
    A, B = slice(0, 40), slice(500, 540)
    a0, b0 = _mean_weight(p, "roll", A), _mean_weight(p, "roll", B)

    p.reset()
    p.update_traces(_spikes(topology, A), executed_channel="roll")   # the decision
    p.apply_reinforcement(0.0, _spikes(topology, A))
    for _ in range(4):                                                 # locked in the roll
        p.update_traces(_spikes(topology, B), executed_channel=None)
        p.apply_reinforcement(0.0, _spikes(topology, B))
    p.update_traces(_spikes(topology, B), executed_channel=None)
    p.apply_reinforcement(-1.0, _spikes(topology, B))                  # hit anyway

    da, db = _mean_weight(p, "roll", A) - a0, _mean_weight(p, "roll", B) - b0
    assert da < 0.0, "the cells active at the decision must be depressed"
    assert da < db - 1e-6, "they must carry more of the blame than cells active later"


def test_unexecuted_compartments_get_no_tag(topology):
    p = DopaminePlasticity(topology)
    p.reset()
    p.update_traces(_spikes(topology, slice(0, 60)), executed_channel="attack_light")
    k_attack = ACTION_CHANNELS.index("attack_light")
    for k, ch in enumerate(ACTION_CHANNELS):
        sel = np.flatnonzero(p.channel == k)
        total = float(np.sum(p.eligibility[sel]))
        if k == k_attack:
            assert total > 0.0
        else:
            assert total == 0.0, f"{ch} was not executed and must not be tagged"


def test_idle_steps_only_decay_the_tag(topology):
    p = DopaminePlasticity(topology)
    p.reset()
    p.update_traces(_spikes(topology, slice(0, 60)), executed_channel="parry")
    before = float(np.sum(p.eligibility))
    p.update_traces(_spikes(topology, slice(100, 160)), executed_channel=None)
    after = float(np.sum(p.eligibility))
    assert 0.0 < after < before
    assert after == pytest.approx(before * p.eligibility_decay, rel=1e-4)
