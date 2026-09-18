"""Sensory encoder: Maps SoulsGym combat state into fruit fly neural drive currents.

Every drive level here is expressed as a multiple of the cell's distance to threshold
(``v_threshold - v_rest``) rather than as a bare number. The previous version used
absolute currents that happened to sit below threshold for the compass (5.0 against a
7.0 mV gap, so the heading system never fired at all) and below threshold for the
retina beyond ~4.3 m, which left the fly blind to a boss it was walking towards.
"""

from __future__ import annotations

import math
import numpy as np

from flysoul.config import BioPhysicsConfig
from flysoul.connectome.graph import CircuitTopology
from flysoul.env.obs import CombatState, parse_obs

# Visual range over which proximity is coded, in metres.
_D_NEAR = 1.5
_D_FAR = 14.0
# Longest boss attack animation the telegraph bank tiles, in seconds.
_TELEGRAPH_SPAN = 1.4
# Locomotor channels whose execution produces self-generated optic flow.
_SELF_MOTION_CHANNELS = frozenset(
    {'advance', 'retreat', 'roll', 'strafe_left', 'strafe_right'}
)
# Learning rate of the forward model that predicts the fly's own displacement.
_FORWARD_MODEL_RATE = 0.25


class SensoryEncoder:
    """Encodes combat state into sensory drive currents [num_neurons]."""

    def __init__(self, topology: CircuitTopology, biophysics: BioPhysicsConfig | None = None):
        self.topology = topology
        self.num_neurons = topology.num_neurons
        bio = biophysics or BioPhysicsConfig()
        # Drive needed to take a resting cell to threshold on its own.
        self.thr = float(bio.v_threshold - bio.v_rest)
        self.step_seconds = 0.1  # SoulsGym advances 0.1 s of game time per step

        sub = topology.sensory_subpools
        self.loom_idx = sub.get("motion_looming", topology.motion_indices)
        self.telegraph_idx = sub.get("motion_telegraph", topology.motion_indices[:0])
        self.size_idx = sub.get("motion_size", topology.motion_indices[:0])
        self.pattern_idx = sub.get("motion_pattern", topology.motion_indices[:0])
        self._pattern_cache: dict = {}
        self.sp_low_idx = sub.get("proprio_stamina_low", topology.proprioceptor_indices[:0])
        self.sp_high_idx = sub.get("proprio_stamina_high", topology.proprioceptor_indices[:0])
        self.hp_low_idx = sub.get("proprio_health_low", topology.proprioceptor_indices[:0])
        self.lock_idx = sub.get("proprio_action_lock", topology.proprioceptor_indices[:0])

        # Pre-compute tuning curves; they never change.
        n_ret = len(topology.retina_indices)
        self._retina_centers = (
            np.linspace(-math.pi, math.pi, n_ret, endpoint=False) if n_ret else np.zeros(0)
        )
        # ~20 deg receptive fields. The old value was 1.5x the inter-column spacing
        # (~4 deg), so only three or four columns were ever active and the Kenyon cells,
        # which sample five to seven inputs at random, essentially never saw a coincidence.
        self._retina_sigma = 0.35
        n_comp = len(topology.compass_indices)
        self._compass_centers = (
            np.linspace(-math.pi, math.pi, n_comp, endpoint=False) if n_comp else np.zeros(0)
        )
        self._compass_sigma = 0.5
        n_tel = len(self.telegraph_idx)
        self._telegraph_centers = (
            (np.arange(n_tel) + 0.5) / n_tel * _TELEGRAPH_SPAN if n_tel else np.zeros(0)
        )
        self._telegraph_sigma = 0.22

        self.reset()

    def set_efference(self, channel):
        """Register the motor command the body is currently executing.

        A fly walking towards something sees that thing expand on its retina. Without an
        efference copy the looming pathway reads the fly's own approach as an object
        rushing in and fires the escape reflex, so the agent walks forward, frightens
        itself, rolls away and repeats. Cancelling self-generated optic flow during
        locomotion is a well-characterised mechanism in Drosophila.

        What gets cancelled is only the *predicted* component. Damping the whole looming
        signal whenever the body moves throws away the boss's approach along with the
        fly's own, and since the fly is walking most of the time, the one pathway that
        innately drives escape is suppressed for most of the fight.
        """
        self._last_command = channel if channel in _SELF_MOTION_CHANNELS else None

    def _pattern_cells(self, anim_id: int) -> np.ndarray:
        """The fixed subset of pattern cells that answers one attack animation."""
        cells = self._pattern_cache.get(anim_id)
        if cells is None:
            n = len(self.pattern_idx)
            k = max(1, n // 8)
            pick = np.random.default_rng(100_003 + anim_id).choice(n, size=k, replace=False)
            cells = self.pattern_idx[np.sort(pick)]
            self._pattern_cache[anim_id] = cells
        return cells

    def reset(self, initial_player_hp: float = 1.0, initial_distance: float = 8.0):
        self._last_command = None
        # Forward model: how much closing each locomotor command produces, learned from
        # the fly's own experience rather than assumed. Persists across episodes.
        if not hasattr(self, "_forward_model"):
            self._forward_model = {c: 0.0 for c in _SELF_MOTION_CHANNELS}
        self.prev_player_hp = initial_player_hp
        self.prev_boss_hp = 1.0
        self.prev_distance = initial_distance
        self.last_angle = 0.0
        self.last_distance = initial_distance

    def encode(self, obs) -> np.ndarray:
        """Translate combat state into bio-electric drive current."""
        state = obs if isinstance(obs, CombatState) else parse_obs(obs)
        drive = np.zeros(self.num_neurons, dtype=np.float32)
        thr = self.thr

        distance = float(state.distance)
        angle = float(state.angle)
        # 0 at maximum visual range, 1 in melee.
        proximity = float(np.clip((_D_FAR - distance) / (_D_FAR - _D_NEAR), 0.0, 1.0))

        # 1. Visual hemifield / retinal activation (compound eye).
        # The bump always crosses threshold, so bearing is legible at every distance;
        # its amplitude, and therefore the number of recruited columns, grows as the
        # boss's retinal image expands.
        retina = self.topology.retina_indices
        if len(retina) > 0:
            # A pure bearing map: constant amplitude and width at every range. Distance
            # is carried by the looming and object-size populations instead.
            #
            # Letting distance modulate this map at all puts a threshold cliff in front
            # of the whole circuit. Every downstream population is a threshold element,
            # so a retina that dims with range does not degrade gracefully - it takes
            # the central complex, the mushroom body and the descending pools down with
            # it. Measured against the previous encoding, every single step on which the
            # motor system fell silent was a step spent standing at spawn distance,
            # waiting for the boss to walk close enough to switch the brain on.
            amplitude = thr * 1.85
            delta = np.angle(np.exp(1j * (self._retina_centers - angle)))
            drive[retina] += (
                amplitude * np.exp(-0.5 * (delta / self._retina_sigma) ** 2)
            ).astype(np.float32)

        # 2. Looming detectors (LPTC / LC11 in the lobula).
        # Angular expansion rate, the actual looming cue: closing speed over distance.
        observed_closing = self.prev_distance - distance
        if self._last_command is not None:
            predicted = self._forward_model[self._last_command]
            self._forward_model[self._last_command] = predicted + _FORWARD_MODEL_RATE * (
                observed_closing - predicted
            )
        else:
            predicted = 0.0

        if len(self.loom_idx) > 0:
            # Only the part of the closing the fly did not cause itself counts as looming.
            external = max(0.0, observed_closing - predicted) / self.step_seconds
            loom_rate = external / max(1.0, distance)
            drive[self.loom_idx] += np.float32(
                thr * (0.30 + 1.45 * float(np.clip(loom_rate, 0.0, 1.5)))
            )

        # 3. Telegraph bank: cells tiling time-since-attack-onset. This hands the circuit
        # a representation of *when* in the swing the boss is, without prescribing what
        # to do about it. A poise break drives the whole bank instead, which is a
        # distinguishable pattern the mushroom body can learn the punish window from.
        if len(self.telegraph_idx) > 0:
            if state.boss_staggered:
                drive[self.telegraph_idx] += np.float32(thr * 1.30)
            elif state.boss_attacking:
                dt = state.boss_anim_time - self._telegraph_centers
                drive[self.telegraph_idx] += (
                    thr * 1.55 * np.exp(-0.5 * (dt / self._telegraph_sigma) ** 2)
                ).astype(np.float32)

        # 3a. Attack-pattern cells: which swing this is. Each animation the boss can
        # perform drives a fixed, sparse subset of these cells for as long as it lasts,
        # the way a lobula columnar type answers one visual motion pattern. Combined
        # at random in the Kenyon cells with the telegraph bank, this yields a code for
        # 'this attack, at this point' - without saying anything about what to do.
        if len(self.pattern_idx) > 0 and state.boss_attacking and state.boss_anim_id >= 0:
            drive[self._pattern_cells(int(state.boss_anim_id))] += np.float32(thr * 1.45)

        # 3b. Object-size cells: monotonic in how large the boss looks, which is the
        # signal that the approach has arrived. They are subthreshold at range and
        # fire in melee.
        if len(self.size_idx) > 0:
            drive[self.size_idx] += np.float32(thr * (0.25 + 1.55 * proximity))

        # 4. Central complex compass (EPG heading ring).
        compass = self.topology.compass_indices
        if len(compass) > 0:
            delta = np.angle(np.exp(1j * (self._compass_centers - angle)))
            drive[compass] += (
                thr * 1.65 * np.exp(-0.5 * (delta / self._compass_sigma) ** 2)
            ).astype(np.float32)

        # 5. Nociceptors: the aversive afferent that drives PPL1.
        nociceptors = self.topology.nociceptor_indices
        if len(nociceptors) > 0:
            damage = max(0.0, self.prev_player_hp - state.player_hp)
            if damage > 0.001:
                drive[nociceptors] += np.float32(thr * (1.0 + 3.0 * min(1.0, damage * 4.0)))

        # 6. PAM reward neurons: fire when the boss loses health, i.e. when the fly's own
        # swing connected. Previously these cells had no inputs and no outputs at all.
        pam = self.topology.pam_indices
        if len(pam) > 0:
            boss_damage = max(0.0, self.prev_boss_hp - state.boss_hp)
            if boss_damage > 0.001:
                drive[pam] += np.float32(thr * (1.0 + 3.0 * min(1.0, boss_damage * 8.0)))

        # 7. Interoception: stamina, health and whether the body can act at all.
        if len(self.sp_low_idx) > 0:
            drive[self.sp_low_idx] += np.float32(thr * (0.35 + 1.55 * (1.0 - state.player_sp)))
        if len(self.sp_high_idx) > 0:
            drive[self.sp_high_idx] += np.float32(thr * (0.35 + 1.55 * state.player_sp))
        if len(self.hp_low_idx) > 0:
            drive[self.hp_low_idx] += np.float32(thr * (0.35 + 1.55 * (1.0 - state.player_hp)))
        if len(self.lock_idx) > 0:
            drive[self.lock_idx] += np.float32(thr * (1.70 if not state.player_can_act else 0.30))

        self.prev_player_hp = state.player_hp
        self.prev_boss_hp = state.boss_hp
        self.prev_distance = distance
        self.last_distance = distance
        self.last_angle = angle

        return drive
