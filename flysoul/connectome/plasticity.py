"""Dopamine-gated three-factor synaptic plasticity for FlySoul.

Implements the biological reinforcement learning rule found in the Drosophila
mushroom body (Aso et al., Cell 2014; Hige et al., Nature 2015):
  - Damage taken triggers PPL101 aversive dopamine burst -> LTD on active KC->MBON synapses.
  - Successful boss hit triggers PAM rewarding dopamine burst -> LTP on active KC->MBON synapses.
"""

from __future__ import annotations

import numpy as np

from flysoul.connectome.graph import CircuitTopology


class DopaminePlasticity:
    """Manages dopamine-modulated synaptic plasticity on KC -> MBON synapses."""

    def __init__(
        self,
        topology: CircuitTopology,
        learning_rate: float = 0.05,
        w_min: float = 0.1,
        w_max: float = 5.0,
    ):
        self.topology = topology
        self.lr = learning_rate
        self.w_min = w_min
        self.w_max = w_max

        # Synaptic eligibility trace for Kenyon Cell -> MBON synapses
        self.eligibility = np.zeros_like(self.topology.weight, dtype=np.float32)
        self.plastic_mask = self.topology.plastic_synapse_mask

        # Dopamine reservoir state
        self.dopamine_level = 0.0  # Normalized [-1.0, 1.0]: negative = aversive, positive = reward
        self.total_ltp_events = 0
        self.total_ltd_events = 0
        self.initial_mean_weight = self.mean_plastic_weight

    def update_traces(self, spike_counts: np.ndarray, decay: float = 0.85):
        """Update synaptic eligibility traces based on co-activation of pre- and post-neurons."""
        self.eligibility *= decay

        # For plastic synapses: pre is KC, post is MBON
        ptr = self.topology.ptr
        post = self.topology.post

        for kc in self.topology.kenyon_indices:
            pre_spikes = spike_counts[kc]
            if pre_spikes == 0:
                continue
            start, end = ptr[kc], ptr[kc + 1]
            for edge_idx in range(start, end):
                if self.plastic_mask[edge_idx]:
                    mbon = post[edge_idx]
                    post_spikes = spike_counts[mbon]
                    # Hebbian coincidence trace
                    self.eligibility[edge_idx] += float(pre_spikes * (post_spikes + 0.1))

    def apply_reinforcement(self, reward: float):
        """Apply third factor (dopamine signal) to modulate synaptic weights.

        Args:
            reward: Float signal.
                < 0: Aversive stimulus (Player took damage / died) -> PPL1 burst.
                > 0: Rewarding stimulus (Boss damaged / hit landed) -> PAM burst.
        """
        # Update smoothed dopamine tracking
        self.dopamine_level = np.clip(reward, -1.0, 1.0)

        if abs(reward) < 1e-4:
            return

        # Modulate plastic synapses: Delta W = lr * Dopamine * Eligibility
        delta_w = self.lr * self.dopamine_level * self.eligibility[self.plastic_mask]
        current_w = self.topology.weight[self.plastic_mask]
        new_w = np.clip(current_w + delta_w, self.w_min, self.w_max)
        self.topology.weight[self.plastic_mask] = new_w.astype(np.float32)

        if self.dopamine_level > 0:
            self.total_ltp_events += int(np.count_nonzero(delta_w > 1e-4))
        elif self.dopamine_level < 0:
            self.total_ltd_events += int(np.count_nonzero(delta_w < -1e-4))

        # Clear eligibility upon reinforcement delivery
        self.eligibility.fill(0.0)

    @property
    def mean_plastic_weight(self) -> float:
        """Return average synaptic weight across all KC -> MBON synapses."""
        if not hasattr(self, "plastic_mask") or np.count_nonzero(self.plastic_mask) == 0:
            return 1.5
        return float(np.mean(self.topology.weight[self.plastic_mask]))

    def reset(self):
        """Reset eligibility traces and dopamine level."""
        self.eligibility.fill(0.0)
        self.dopamine_level = 0.0
