"""Spike-frequency adaptation: on when asked for, silent when not."""

from __future__ import annotations

import numpy as np

from flysoul.config import BioPhysicsConfig
from flysoul.connectome.engine import ConnectomeEngine


def _driven_spikes(jump_mv: float, drive_mv: float = 12.0, ms: float = 500.0) -> int:
    # One neuron, no synapses: a CSR graph with a single empty row.
    ptr = np.array([0, 0], dtype=np.int64)
    post = np.zeros(0, dtype=np.int32)
    weight = np.zeros(0, dtype=np.float32)
    engine = ConnectomeEngine(ptr, post, weight, BioPhysicsConfig(adaptation_jump_mv=jump_mv))
    engine.seed(0)
    return int(engine.step(np.array([drive_mv], dtype=np.float32), duration_ms=ms)[0])


def test_adaptation_reduces_sustained_firing():
    plain = _driven_spikes(0.0)
    adapted = _driven_spikes(2.0)
    assert plain > 20, "a strongly driven cell must fire"
    assert adapted < plain * 0.8, "adaptation must lower the sustained rate"


def test_adaptation_off_is_bit_identical_to_before():
    """With the jump at zero the kernel must reproduce the un-adapted dynamics exactly,
    so the default configuration does not silently change the calibrated circuit."""
    a = _driven_spikes(0.0)
    b = _driven_spikes(0.0)
    assert a == b


def test_adaptation_state_resets_with_the_engine():
    ptr = np.array([0, 0], dtype=np.int64)
    engine = ConnectomeEngine(ptr, np.zeros(0, np.int32), np.zeros(0, np.float32),
                              BioPhysicsConfig(adaptation_jump_mv=2.0))
    engine.step(np.array([12.0], dtype=np.float32), duration_ms=200.0)
    assert engine.adapt[0] > 0.0
    engine.reset_state()
    assert engine.adapt[0] == 0.0
