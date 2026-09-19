"""3b - SYNTHETIC readout. Not part of the fly's brain.

A linear Q function over the fly's Kenyon cell code, fitted outside the brain by
least-squares Q-iteration on archived fights (scripts/fit_q_readout.py). When it is
active, it - not the mushroom body - chooses the action. The biological circuit still
runs (its sensory code is the input), its learned synapses are loaded for comparison and
never updated or saved, and every run is recorded under the fingerprint "3b" so that
nothing here is ever mistaken for the fly's own learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from flysoul.connectome.graph import ACTION_CHANNELS
from flysoul.synthetic.features import CONT_DIM, RAW_DIM, cont_features, raw_features
from flysoul.synthetic.actions import channel_index, to_channel

IDLE = "idle"


class SyntheticQ:
    def __init__(self, path, kenyon_indices: np.ndarray, epsilon: float = 0.05, seed: int = 0):
        blob = np.load(Path(path), allow_pickle=False)
        self.actions = [str(a) for a in blob["actions"]]
        self.gamma = float(blob["gamma"])
        self.kc = np.asarray(kenyon_indices)
        self.feature_spec = str(blob["features"]) if "features" in blob.files else "kc"
        self.layers = None
        if "W1" in blob.files:
            # MLP exported by scripts/fit_q_mlp.py: ReLU between layers, linear output
            self.layers = []
            i = 1
            while f"W{i}" in blob.files:
                self.layers.append((blob[f"W{i}"].astype(np.float64), blob[f"b{i}"].astype(np.float64)))
                i += 1
            width = self.layers[0][0].shape[1]
            self.W = None
        else:
            self.W = blob["W"].astype(np.float64)  # [actions, num_kc + 1]
            width = self.W.shape[1]
        expected = {"kc": len(self.kc) + 1, "kc+raw": len(self.kc) + 1 + RAW_DIM, "raw": RAW_DIM + 1,
                    "raw+cont": RAW_DIM + CONT_DIM + 1}[self.feature_spec]
        self.learner = str(blob["learner"]) if "learner" in blob.files else ("mlp" if self.layers else "linear")
        self.last_hidden = None
        if width != expected:
            raise ValueError(f"Q readout ({self.feature_spec}) expects {expected} features, got {width}")
        self.prev_action = -1
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.source = Path(path).name
        self.fitted_on = int(blob["fights"]) if "fights" in blob else 0
        self.last_q: Dict[str, float] = {}
        self.last_choice = IDLE
        self.last_explored = False
        self.decisions = 0
        self.agreements = 0

    def features(self, spike_counts: np.ndarray, state=None) -> np.ndarray:
        a = (spike_counts[self.kc] > 0).astype(np.float64)
        n = float(np.linalg.norm(a))
        if n > 0:
            a /= n
        kcf = np.concatenate([a, [1.0]])
        if self.feature_spec == "kc":
            return kcf
        if state is None:
            raise ValueError("this Q readout needs the combat state for its raw features")
        raw = raw_features(state.distance, state.angle, state.player_hp, state.boss_hp, state.boss_staggered,
                           state.player_can_act, state.boss_anim_id, state.boss_anim_time,
                           state.boss_attacking, self.prev_action, state.player_sp).astype(np.float64)
        if self.feature_spec == "kc+raw":
            return np.concatenate([kcf, raw])
        if self.feature_spec == "raw+cont":
            cont = cont_features(state.distance, state.angle, state.player_hp, state.boss_hp,
                                 state.boss_anim_time, state.player_sp).astype(np.float64)
            return np.concatenate([raw, cont, [1.0]])
        return np.concatenate([raw, [1.0]])

    def new_fight(self) -> None:
        self.prev_action = -1

    def act(self, spike_counts: np.ndarray, valid_channels: Optional[Iterable[str]],
            mb_winner: Optional[str] = None, state=None) -> Tuple[str, Dict[str, float]]:
        phi = self.features(spike_counts, state)
        if self.layers is not None:
            h = phi
            for j, (W, b) in enumerate(self.layers):
                h = W @ h + b
                if j < len(self.layers) - 1:
                    h = np.maximum(h, 0.0)
                    self.last_hidden = h  # last hidden layer, for the visualizer
            q = h
        else:
            q = self.W @ phi
        allowed = None if valid_channels is None else set(valid_channels) | {IDLE}
        cand = [i for i, a in enumerate(self.actions) if allowed is None or to_channel(a) in allowed]
        if not cand:
            cand = [self.actions.index(IDLE)]
        if self.rng.random() < self.epsilon:
            k = int(self.rng.choice(cand))
            self.last_explored = True
        else:
            k = max(cand, key=lambda i: q[i])
            self.last_explored = False
        self.last_q = {a: float(q[i]) for i, a in enumerate(self.actions)}
        self.last_choice = self.actions[k]
        self.prev_action = channel_index(self.last_choice)
        self.decisions += 1
        if mb_winner is not None and mb_winner == to_channel(self.last_choice):
            self.agreements += 1
        return self.last_choice, self.last_q

    def _hidden_summary(self, bins: int = 32):
        """The last hidden layer's activity folded into `bins` values in [0, 1] (mean of
        each consecutive group), so the visualizer can show the network working."""
        if self.last_hidden is None:
            return None
        h = np.asarray(self.last_hidden, dtype=np.float64)
        if h.size < bins:
            return [round(float(v), 3) for v in h]
        parts = np.array_split(h, bins)
        vals = np.array([p.mean() for p in parts])
        top = float(vals.max()) if vals.max() > 0 else 1.0
        return [round(float(v / top), 3) for v in vals]

    def report(self, mb_winner: Optional[str] = None) -> dict:
        return {
            "source": self.source + f" ({self.learner}) [{self.feature_spec}]",
            "hidden": self._hidden_summary(),
            "fitted_on_fights": self.fitted_on,
            "gamma": self.gamma,
            "q": {a: round(v, 3) for a, v in self.last_q.items()},
            "chosen": self.last_choice,
            "explored": self.last_explored,
            "mb_winner": mb_winner,
            "agreement": round(self.agreements / max(1, self.decisions), 3),
        }
