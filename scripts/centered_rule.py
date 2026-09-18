"""Candidate rule: dopamine centred per compartment (offline test only, not yet live).

Measured on 580 archived fights, the live three-factor rule computes a Hebbian sum of
Kenyon input times dopamine. On the same data that sum, taken against the raw outcome,
predicts held-out outcomes at +0.09; taken against the outcome minus its mean, at +0.22;
least squares reaches +0.37. Two thirds of the reward steps are punishments, so an
uncentred signal mostly writes "which Kenyon cells fire often" into every compartment.

Biology has a mechanism for the centring: MBON->DAN feedback in each compartment sets
the dopaminergic neuron's tonic baseline against which bursts and dips are read. Here
each compartment keeps a running mean of the prediction error it has been receiving
while it held credit, and the dopamine it applies is the deviation from that mean.
"""

from __future__ import annotations

import numpy as np

from flysoul.connectome.plasticity import DopaminePlasticity


class CenteredPlasticity(DopaminePlasticity):
    def __init__(self, *a, baseline_rate: float = 0.02, mode: str = "comp", saturate: bool = True, **kw):
        super().__init__(*a, **kw)
        self.baseline_rate = baseline_rate
        self.mode = mode  # "comp": per compartment; "global": one baseline; "none": no centring
        self.saturate = saturate  # False: dopamine linear in the error (clipped at 2 scales), not tanh
        self.comp_baseline = np.zeros(self.num_channels, dtype=np.float64)
        self.global_baseline = 0.0

    def apply_reinforcement(self, reward, spike_counts=None, terminal=False):
        if len(self.edges) == 0 or spike_counts is None:
            return super().apply_reinforcement(reward, spike_counts, terminal)
        features = self._features(spike_counts)
        value_now = 0.0 if terminal else float(np.dot(self.value_weights, features))
        self.last_value = value_now
        if self._prev_features is None:
            self._prev_features = features
            self._prev_value = value_now
            return
        error = reward + self.discount * value_now - self._prev_value
        self.value_weights += (self.critic_lr * error) * self._prev_features
        self._prev_features = None if terminal else features
        self._prev_value = 0.0 if terminal else value_now
        self.last_td_error = float(error)

        if self.mode == "none":
            centred = np.full(self.num_channels, error)
        elif self.mode == "global":
            self.global_baseline += self.baseline_rate * (error - self.global_baseline)
            centred = np.full(self.num_channels, error - self.global_baseline)
        else:
            # each compartment tracks the error it receives, weighted by how much
            # credit it held at the time
            self.comp_baseline += self.baseline_rate * self.credit * (error - self.comp_baseline)
            centred = error - self.comp_baseline

        if self._adaptive_scale:
            self._delta_magnitude += 0.01 * (abs(error) - self._delta_magnitude)
            scale = max(1e-3, self._delta_magnitude)
        else:
            scale = self.rpe_scale
        dop = np.tanh(centred / scale) if self.saturate else np.clip(centred / scale, -2.0, 2.0) / 2.0
        self.dopamine_level = float(dop[int(np.argmax(self.credit))]) if self.credit.max() > 0 else 0.0
        peak = float(np.max(self.eligibility))
        if peak <= 1e-6:
            return
        elig = self.eligibility / peak
        credit_per_edge = self.credit[self.channel]
        delta_w = (self.lr * self._w_scale) * elig * credit_per_edge * dop[self.channel]
        if not np.any(np.abs(delta_w) > 1e-6):
            return
        current = self.topology.weight[self.edges]
        self.topology.weight[self.edges] = np.clip(current + delta_w, self.w_min, self.w_max).astype(np.float32)
        self.total_ltp_events += int(np.count_nonzero(delta_w > 1e-4))
        self.total_ltd_events += int(np.count_nonzero(delta_w < -1e-4))
        self._homeostatic_scaling()
