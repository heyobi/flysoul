"""Dopamine-gated three-factor synaptic plasticity for FlySoul.

Implements the biological reinforcement learning rule found in the Drosophila
mushroom body (Aso et al., Cell 2014; Hige et al., Nature 2015):
  - Damage taken triggers PPL101 aversive dopamine burst -> LTD on active KC->MBON synapses.
  - Successful boss hit triggers PAM rewarding dopamine burst -> LTP on active KC->MBON synapses.

Credit assignment is **compartment-specific**. Each action channel owns one mushroom
body compartment with its own dopaminergic neurons, and an efference copy of the action
the fly actually executed keeps that compartment eligible for a short window. Without
this, one scalar reward scaled every plastic synapse in the same direction and learning
could only make the whole mushroom body louder or quieter, never prefer one action
over another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flysoul.connectome.graph import ACTION_CHANNELS, CircuitTopology


class DopaminePlasticity:
    """Manages dopamine-modulated synaptic plasticity on KC -> MBON synapses."""

    def __init__(
        self,
        topology: CircuitTopology,
        learning_rate: float = 0.08,
        w_min_factor: float = 0.05,
        w_max_factor: float = 4.0,
        eligibility_decay: float = 0.92,
        credit_decay: float = 0.82,
        scaling_deadband: float = 0.35,
        scaling_rate: float = 0.02,
        rpe_scale: float = 1.5,
        rpe_baseline_rate: float = 0.02,
    ):
        self.topology = topology
        self.lr = learning_rate
        self.eligibility_decay = eligibility_decay
        self.credit_decay = credit_decay
        self.scaling_deadband = scaling_deadband
        self.scaling_rate = scaling_rate
        self.rpe_scale = rpe_scale
        self.rpe_baseline_rate = rpe_baseline_rate
        # Running expectation of what a reinforcing event is worth in this fight.
        self.reward_baseline = 0.0

        self.plastic_mask = topology.plastic_synapse_mask
        self.edges = np.flatnonzero(self.plastic_mask).astype(np.int64)

        # Presynaptic index per edge, recovered from the CSR row pointers.
        row_lengths = np.diff(topology.ptr)
        pre_of_edge = np.repeat(np.arange(len(row_lengths), dtype=np.int64), row_lengths)
        self.pre = pre_of_edge[self.edges]
        self.post = topology.post[self.edges].astype(np.int64)
        self.channel = topology.plastic_channel[self.edges].astype(np.int64)

        self.num_channels = len(ACTION_CHANNELS)
        self.eligibility = np.zeros(len(self.edges), dtype=np.float32)
        # Per-channel responsibility: how recently the fly executed that action.
        self.credit = np.zeros(self.num_channels, dtype=np.float32)

        # Baseline synaptic drive per compartment, used for homeostatic scaling so that
        # a long punishing streak cannot silence a compartment permanently.
        self._channel_edges = [
            self.edges[self.channel == c] for c in range(self.num_channels)
        ]
        self._baseline = np.array(
            [
                float(np.sum(topology.weight[sel])) if len(sel) else 0.0
                for sel in self._channel_edges
            ],
            dtype=np.float32,
        )

        # Bounds and step size are relative to the initial synaptic scale, which the
        # fan-in normalisation sets; absolute constants would be meaningless here.
        init = self.topology.weight[self.edges] if len(self.edges) else np.ones(1, dtype=np.float32)
        self._w_scale = float(np.mean(np.abs(init))) or 1.0
        self.w_min = w_min_factor * self._w_scale
        self.w_max = w_max_factor * self._w_scale

        self.dopamine_level = 0.0
        self.total_ltp_events = 0
        self.total_ltd_events = 0
        self.initial_mean_weight = self.mean_plastic_weight

    # ------------------------------------------------------------------ traces

    def update_traces(self, spike_counts: np.ndarray, executed_channel: str | None = None):
        """Accumulate pre/post coincidence and register the efference copy."""
        if len(self.edges) == 0:
            return
        self.eligibility *= self.eligibility_decay
        pre_spikes = spike_counts[self.pre].astype(np.float32)
        post_spikes = spike_counts[self.post].astype(np.float32)
        np.add(self.eligibility, pre_spikes * (post_spikes + 0.1), out=self.eligibility)

        self.credit *= self.credit_decay
        if executed_channel is not None and executed_channel in ACTION_CHANNELS:
            self.credit[ACTION_CHANNELS.index(executed_channel)] = 1.0

    # ----------------------------------------------------------- reinforcement

    def apply_reinforcement(self, reward: float):
        """Apply the third factor (dopamine) to the eligible synapses.

        Dopamine carries a reward *prediction error*, not the raw reward. In this fight
        the two are very different quantities: a clean hit from Iudex is worth about
        -2.8 and lands three times as often as the +0.5 the fly earns for a hit of its
        own, so feeding the raw signal through delivers a net negative burst on every
        reinforcing event. That depresses whichever compartment was active, which
        quietens the descending pools, which makes the fly act less - and doing nothing
        is the one behaviour that can never be punished, because it owns no compartment.
        Subtracting a running baseline makes "better than expected" the thing that gets
        reinforced, which is both what dopaminergic neurons actually encode and what
        stops the circuit from learning to stand still.

        Args:
            reward: < 0 aversive (PPL1 burst), > 0 rewarding (PAM burst).
        """
        if len(self.edges) == 0 or abs(reward) < 1e-4:
            return
        prediction_error = reward - self.reward_baseline
        self.reward_baseline += self.rpe_baseline_rate * prediction_error
        self.dopamine_level = float(np.tanh(prediction_error / self.rpe_scale))
        if abs(self.dopamine_level) < 1e-3:
            return

        # Only the compartments the fly is currently responsible for get the credit.
        credit_per_edge = self.credit[self.channel]
        # Eligibility is in spike-count units and spans orders of magnitude; normalise
        # it so the learning rate means the same thing regardless of firing rates.
        peak = float(np.max(self.eligibility))
        if peak <= 1e-6:
            return
        elig = self.eligibility / peak
        delta_w = (self.lr * self.dopamine_level * self._w_scale) * elig * credit_per_edge
        if not np.any(np.abs(delta_w) > 1e-6):
            return

        current = self.topology.weight[self.edges]
        updated = np.clip(current + delta_w, self.w_min, self.w_max)
        self.topology.weight[self.edges] = updated.astype(np.float32)

        if self.dopamine_level > 0:
            self.total_ltp_events += int(np.count_nonzero(delta_w > 1e-4))
        else:
            self.total_ltd_events += int(np.count_nonzero(delta_w < -1e-4))

        self._homeostatic_scaling()
        # Dopamine delivery consumes the eligibility trace.
        self.eligibility *= 0.1

    def _homeostatic_scaling(self):
        """Guard rail keeping a compartment from saturating or going permanently silent.

        This must not be a constant pull towards baseline. The descending readout depends
        on how much total drive a compartment delivers, so renormalising that total after
        every dopamine event erases precisely the difference learning just created: run
        for a hundred episodes that way and every compartment ends up back within a
        percent of every other. It therefore only engages outside a wide dead band, and
        gently even then.
        """
        weights = self.topology.weight
        for c in range(self.num_channels):
            baseline = self._baseline[c]
            sel = self._channel_edges[c]
            if baseline <= 0.0 or len(sel) == 0:
                continue
            total = float(np.sum(weights[sel]))
            if total <= 1e-6:
                continue
            ratio = baseline / total
            if 1.0 - self.scaling_deadband <= ratio <= 1.0 + self.scaling_deadband:
                continue
            scale = 1.0 + self.scaling_rate * (ratio - 1.0)
            weights[sel] = np.clip(weights[sel] * scale, self.w_min, self.w_max).astype(np.float32)

    # ------------------------------------------------------------------ readout

    @property
    def mean_plastic_weight(self) -> float:
        """Average synaptic weight across all KC -> MBON synapses."""
        if len(self.edges) == 0:
            return 1.5
        return float(np.mean(self.topology.weight[self.edges]))

    def channel_weights(self) -> dict:
        """Mean plastic weight per action channel, for telemetry."""
        out = {}
        for c, name in enumerate(ACTION_CHANNELS):
            sel = self.edges[self.channel == c]
            out[name] = float(np.mean(self.topology.weight[sel])) if len(sel) else 0.0
        return out

    def reset(self):
        """Reset per-episode state. Weights and the reward baseline are kept: both are
        what the fly has learned, and resetting the baseline between episodes would
        re-teach the same lesson from scratch every time."""
        self.eligibility.fill(0.0)
        self.credit.fill(0.0)
        self.dopamine_level = 0.0

    # -------------------------------------------------------------- persistence

    def save(self, path) -> bool:
        """Write the learned synaptic weights and the reward baseline to disk.

        Plastic weights live only in memory otherwise, so every restart of the agent -
        for a code change, a crash, a machine reboot - throws away everything the fly
        has learned and starts it again from the innate circuit.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                weights=self.topology.weight[self.edges],
                reward_baseline=self.reward_baseline,
                total_ltp=self.total_ltp_events,
                total_ltd=self.total_ltd_events,
                num_edges=len(self.edges),
            )
            return True
        except Exception:
            return False

    def load(self, path) -> bool:
        """Restore weights saved by :meth:`save`. Returns False if there is nothing to load."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            blob = np.load(path, allow_pickle=False)
            weights = blob["weights"]
            if weights.shape != self.edges.shape:
                return False  # A different circuit; the weights are not transferable.
            self.topology.weight[self.edges] = weights.astype(np.float32)
            self.reward_baseline = float(blob["reward_baseline"])
            self.total_ltp_events = int(blob["total_ltp"])
            self.total_ltd_events = int(blob["total_ltd"])
            return True
        except Exception:
            return False
