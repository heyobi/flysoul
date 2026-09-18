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
        # How long an executed action stays responsible for what follows. At 0.82 an
        # action four steps back kept 45% of the credit; measured live over 135 fights
        # that left "advance" at -71% of its innate weight while attacks rose to +148%:
        # closing the distance is what precedes both the hits and the damage, and the
        # damage arrives sooner. At 0.90 the same action keeps 66%, so the approach
        # that set up a hit shares in it.
        credit_decay: float = 0.90,
        scaling_deadband: float = 0.15,
        scaling_rate: float = 0.02,
        rpe_scale: float = 0.0,
        rpe_baseline_rate: float = 0.02,
        # Horizon of the value estimate. 0.90 is about ten steps, one second of the
        # fight, which is how long the consequences of an action take to arrive here.
        # Chosen offline (scripts/offline_search.py) on 90 archived fights with a
        # held-out split: at 0.95 the rule's credit assignment correlated with the real
        # outcomes at +0.04 / -0.03 on the two splits, at 0.90 at +0.18 / +0.15, at 0.98
        # negative. The least-squares ceiling on the same fights was +0.33. Every other
        # parameter the search varied made no consistent difference.
        discount: float = 0.90,
        critic_lr: float = 0.05,
    ):
        self.topology = topology
        self.lr = learning_rate
        self.eligibility_decay = eligibility_decay
        self.credit_decay = credit_decay
        self.scaling_deadband = scaling_deadband
        self.scaling_rate = scaling_rate
        # Dopamine saturation point. Fixed at a constant it encodes an assumption about
        # how big rewards are, and the two environments differ by an order of magnitude:
        # a value tuned on the mock delivers a three times weaker signal in the game for
        # no reason other than the constant. Zero means track it from experience.
        self.rpe_scale = rpe_scale
        self._adaptive_scale = rpe_scale <= 0.0
        self._delta_magnitude = 0.1  # running mean of |delta|, seeded small
        self.rpe_baseline_rate = rpe_baseline_rate
        self.discount = discount
        self.critic_lr = critic_lr
        # Value of the current situation, read out linearly from the Kenyon cell
        # population. Starts at zero everywhere, so before it has learned anything
        # the prediction error is just the reward and behaviour is unchanged.
        self.value_weights = np.zeros(len(topology.kenyon_indices), dtype=np.float32)
        self._prev_features: np.ndarray | None = None
        self._prev_value = 0.0
        self.last_td_error = 0.0
        self.last_value = 0.0
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
        # Positions within the plastic arrays, for tagging one compartment at a time.
        self._channel_edge_positions = [
            np.flatnonzero(self.channel == c) for c in range(self.num_channels)
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
        """Set the synaptic tag on the compartment whose command was just issued.

        The tag is set at the moment of the decision, on the synapses from the Kenyon
        cells active *then*, and only in the compartment that was executed. It used to
        accumulate on every compartment every step. Measured live, that misassigned
        blame in exactly the way that blocks timing from being learned: the fly rolls
        early in the boss's swing, is locked in the roll animation for half a second,
        and is hit at the end of the swing; when the punishment arrives, the freshest
        eligibility in the roll compartment belongs to the cells active during the
        swing's late phase - the "roll now" cells - and they are depressed for a roll
        that was chosen too early. Over 1,400 rolls the early ones never got rarer.

        Tagging at decision time is also what the biology describes: the KC->MBON
        eligibility that dopamine later reads is created by the coincidence at the
        time of the event, not by whatever the Kenyon cells do afterwards.
        """
        if len(self.edges) == 0:
            return
        self.eligibility *= self.eligibility_decay
        self.credit *= self.credit_decay
        if executed_channel is not None and executed_channel in ACTION_CHANNELS:
            k = ACTION_CHANNELS.index(executed_channel)
            self.credit[k] = 1.0
            sel = self._channel_edge_positions[k]
            pre_spikes = spike_counts[self.pre[sel]].astype(np.float32)
            post_spikes = spike_counts[self.post[sel]].astype(np.float32)
            self.eligibility[sel] += pre_spikes * (post_spikes + 0.1)

    # ----------------------------------------------------------- reinforcement

    def _features(self, spike_counts: np.ndarray) -> np.ndarray:
        """Sparse Kenyon cell activity, the substrate the value readout is built on.

        Scaled to unit L2 norm, which is what makes the critic's step size mean what the
        learning rate says it means. Dividing by the sum instead - so the features add to
        one - looks equally reasonable and is not: it shrinks every update by the number
        of active cells. With about a hundred Kenyon cells firing, that left the value
        estimate two orders of magnitude below the rewards it was supposed to predict
        (|V| ~ 0.004 against a hit worth -0.44), so the prediction error collapsed back
        to the plain reward and the whole mechanism was inert.
        """
        active = (spike_counts[self.topology.kenyon_indices] > 0).astype(np.float32)
        norm = float(np.linalg.norm(active))
        return active / norm if norm > 0 else active

    def value_of(self, spike_counts: np.ndarray) -> float:
        """Estimated value of the situation the mushroom body is currently representing."""
        return float(np.dot(self.value_weights, self._features(spike_counts)))

    def apply_reinforcement(
        self,
        reward: float,
        spike_counts: np.ndarray | None = None,
        terminal: bool = False,
    ):
        """Apply the third factor (dopamine) to the eligible synapses.

        Dopamine carries a temporal-difference error, not the raw reward:

            delta = r + gamma * V(s') - V(s)

        The distinction decides whether the fly can learn to dodge at all. SoulsGym pays
        for damage dealt and damage taken and for nothing else, so a successful dodge
        produces a reward of exactly zero - indistinguishable, to a plain reward signal,
        from standing still. Every attack survived teaches nothing. With a value
        estimate, being in front of a winding-up boss is worth less than being past the
        swing, so coming through it unhurt is a positive prediction error and the roll
        that achieved it gets reinforced. This is also what midbrain dopamine neurons
        actually encode.

        Args:
            reward: The environment's reward for the step just taken.
            spike_counts: Current spikes, used to evaluate the new situation. Without
                them this falls back to the plain reward signal.
            terminal: True on the last step of an episode, where there is no future.
        """
        if len(self.edges) == 0:
            return

        if spike_counts is None:
            # No state available: fall back to a running-baseline prediction error.
            if abs(reward) < 1e-4:
                return
            error = reward - self.reward_baseline
            self.reward_baseline += self.rpe_baseline_rate * error
        else:
            features = self._features(spike_counts)
            value_now = 0.0 if terminal else float(np.dot(self.value_weights, features))
            self.last_value = value_now
            if self._prev_features is None:
                # Nothing to compare against yet; remember this step and wait.
                self._prev_features = features
                self._prev_value = value_now
                return
            error = reward + self.discount * value_now - self._prev_value
            # Critic update: move the previous state's value towards what followed it.
            self.value_weights += (self.critic_lr * error) * self._prev_features
            self._prev_features = None if terminal else features
            self._prev_value = 0.0 if terminal else value_now

        self.last_td_error = float(error)
        if self._adaptive_scale:
            self._delta_magnitude += 0.01 * (abs(error) - self._delta_magnitude)
            scale = max(1e-3, self._delta_magnitude)
        else:
            scale = self.rpe_scale
        self.dopamine_level = float(np.tanh(error / scale))
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

    def _homeostatic_scaling(self):
        """Guard rail keeping the plastic synapses from inflating or going silent.

        It scales every KC->MBON synapse by the same factor, so that what learning has
        made different stays different and only the overall level is held. The readout
        that turns these weights into behaviour is a competition between compartments,
        and a competition only carries information in the differences.

        The earlier version of this capped each compartment on its own. Measured after
        two hundred live fights, five of the eight compartments sat at exactly the cap,
        1/(1 - deadband) times their baseline, indistinguishable from one another: every
        action the fly executed had been pumped up until it hit the ceiling, and the
        ceiling then erased the very preferences the dopamine had written. From that
        point on nothing the fly experienced could change what it did, and more
        experience (or replaying it in sleep) only pressed harder against the wall.

        Uniform scaling still must not be a constant pull, because the descending pools
        are tuned to an absolute level of drive; it engages only outside the dead band,
        and gently even then.
        """
        weights = self.topology.weight
        if len(self.edges) == 0:
            return
        baseline = float(np.sum(self._baseline))
        total = float(np.sum(weights[self.edges]))
        if baseline <= 0.0 or total <= 1e-6:
            return
        ratio = baseline / total
        if 1.0 - self.scaling_deadband <= ratio <= 1.0 + self.scaling_deadband:
            return
        scale = 1.0 + self.scaling_rate * (ratio - 1.0)
        weights[self.edges] = np.clip(
            weights[self.edges] * scale, self.w_min, self.w_max
        ).astype(np.float32)

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
        self._prev_features = None
        self._prev_value = 0.0

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
                value_weights=self.value_weights,
                delta_magnitude=self._delta_magnitude,
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
            if "delta_magnitude" in blob:
                self._delta_magnitude = float(blob["delta_magnitude"])
            if "value_weights" in blob:
                vw = blob["value_weights"]
                if vw.shape == self.value_weights.shape:
                    self.value_weights = vw.astype(np.float32)
            self.total_ltp_events = int(blob["total_ltp"])
            self.total_ltd_events = int(blob["total_ltd"])
            return True
        except Exception:
            return False
