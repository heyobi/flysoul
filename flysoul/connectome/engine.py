"""High-performance Leaky Integrate-and-Fire (LIF) simulation engine for MaleCNS v1.0.

Implements analytic subthreshold integration, 1.8 ms axonal transmission delay,
and sparse spike propagation compiled via Numba JIT.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from flysoul.config import BioPhysicsConfig


@njit(cache=True, fastmath=True)
def advance_lif(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    refractory: np.ndarray,
    drive: np.ndarray,
    queue: np.ndarray,
    queue_count: np.ndarray,
    cursor: int,
    steps: int,
    dt: float,
    v_rest: float,
    v_thresh: float,
    v_reset: float,
    refractory_ticks: int,
    delay_ticks: int,
    counts: np.ndarray,
    active: np.ndarray,
    active_flag: np.ndarray,
    nactive: np.ndarray,
) -> int:
    """Core Numba-jitted LIF integration loop with axonal delay ring buffer."""
    av = math.exp(-dt / 20.0)
    ag = math.exp(-dt / 5.0)
    coupling = (av - ag) / 3.0
    delay_slots = queue.shape[0]

    for _ in range(steps):
        slot = cursor % delay_slots

        # 1. Update active neurons: integrate membrane potential & check firing threshold
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = v_rest + (v[i] - v_rest) * av + drive[i] * (1.0 - av) + g[i] * coupling
                g[i] *= ag
                if v[i] > v_thresh:
                    counts[i] += 1
                    # Schedule spike delivery at t + delay_ticks
                    future = (cursor + delay_ticks) % delay_slots
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1

        # 2. Deliver delayed spikes arriving at current slot
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1
                    active[nactive[0]] = j
                    nactive[0] += 1
        queue_count[slot] = 0

        # 3. Post-spike reset & refractory setting for neurons that fired at t - delay_ticks
        future = (cursor + delay_ticks) % delay_slots
        for q in range(queue_count[future]):
            i = queue[future, q]
            v[i] = v_reset
            g[i] = 0.0
            refractory[i] = refractory_ticks

        cursor += 1

    return cursor


class ConnectomeEngine:
    """Manages the full connectome LIF state and step execution."""

    def __init__(
        self,
        ptr: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray,
        config: BioPhysicsConfig | None = None,
    ):
        self.config = config or BioPhysicsConfig()
        self.num_neurons = len(ptr) - 1

        # Topology (CSR)
        self.ptr = np.ascontiguousarray(ptr, dtype=np.int64)
        self.post = np.ascontiguousarray(post, dtype=np.int32)
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)

        # State vectors
        self.v = np.full(self.num_neurons, self.config.v_rest, dtype=np.float32)
        self.g = np.zeros(self.num_neurons, dtype=np.float32)
        self.refractory = np.zeros(self.num_neurons, dtype=np.int32)
        self.drive = np.zeros(self.num_neurons, dtype=np.float32)
        self.counts = np.zeros(self.num_neurons, dtype=np.int32)

        # Delay ring buffer
        delay_slots = max(32, self.config.delay_steps + 1)
        max_spikes_per_slot = max(1024, self.num_neurons // 2)
        self.queue = np.zeros((delay_slots, max_spikes_per_slot), dtype=np.int32)
        self.queue_count = np.zeros(delay_slots, dtype=np.int32)
        self.cursor = 0

        # Active tracking arrays
        self.active = np.zeros(self.num_neurons, dtype=np.int32)
        self.active_flag = np.zeros(self.num_neurons, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)

        # Pre-seed active neurons with all indices initially so driven neurons are processed
        self._reset_active()

    def _reset_active(self):
        """Seed all neurons as active initially so external drives can be captured."""
        self.active[:] = np.arange(self.num_neurons, dtype=np.int32)
        self.active_flag[:] = 1
        self.nactive[0] = self.num_neurons

    def reset_state(self):
        """Reset neural potentials, conductances, and queues to baseline."""
        self.v.fill(self.config.v_rest)
        self.g.fill(0.0)
        self.refractory.fill(0)
        self.drive.fill(0.0)
        self.counts.fill(0)
        self.queue.fill(0)
        self.queue_count.fill(0)
        self.cursor = 0
        self._reset_active()

    def step(self, drive: np.ndarray, duration_ms: float = 16.67) -> np.ndarray:
        """Advance the connectome for a specified real-time duration (e.g. 1 frame = 16.67 ms).

        Args:
            drive: Vector of input currents [num_neurons].
            duration_ms: Duration in milliseconds (default: 1 frame at 60 FPS).

        Returns:
            counts: Array of spike counts per neuron produced during this interval.
        """
        steps = int(round(duration_ms / self.config.dt))
        self.counts.fill(0)
        np.copyto(self.drive, drive)

        self.cursor = advance_lif(
            self.ptr,
            self.post,
            self.weight,
            self.v,
            self.g,
            self.refractory,
            self.drive,
            self.queue,
            self.queue_count,
            self.cursor,
            steps,
            self.config.dt,
            self.config.v_rest,
            self.config.v_threshold,
            self.config.v_reset,
            self.config.refractory_steps,
            self.config.delay_steps,
            self.counts,
            self.active,
            self.active_flag,
            self.nactive,
        )

        return self.counts.copy()
