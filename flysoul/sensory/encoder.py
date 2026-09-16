"""Sensory encoder: Maps SoulsGym combat state into fruit fly neural drive currents."""

from __future__ import annotations

import math
import numpy as np

from flysoul.connectome.graph import CircuitTopology


class SensoryEncoder:
    """Encodes SoulsGym observation vector into sensory drive currents [num_neurons]."""

    def __init__(self, topology: CircuitTopology):
        self.topology = topology
        self.num_neurons = topology.num_neurons
        self.prev_player_hp = 1.0
        self.prev_distance = 10.0
        self.last_angle = 0.0
        self.last_distance = 8.0

    def reset(self, initial_player_hp: float = 1.0, initial_distance: float = 8.0):
        self.prev_player_hp = initial_player_hp
        self.prev_distance = initial_distance
        self.last_angle = 0.0
        self.last_distance = initial_distance

    def encode(self, obs: dict | np.ndarray) -> np.ndarray:
        """Translate combat state into bio-electric drive current.

        Supports both SoulsGym dictionary observations and vector observations.
        """
        drive = np.zeros(self.num_neurons, dtype=np.float32)

        def _to_float(v, default=0.0) -> float:
            if v is None:
                return default
            if isinstance(v, (np.ndarray, list)):
                return float(v[0]) if len(v) > 0 else default
            return float(v)

        # Parse observation fields
        if isinstance(obs, dict):
            # Check if official SoulsGym pose format
            if "player_pose" in obs and "boss_pose" in obs:
                p_pose = np.asarray(obs["player_pose"], dtype=np.float32)
                b_pose = np.asarray(obs["boss_pose"], dtype=np.float32)
                dx = float(b_pose[0] - p_pose[0])
                dy = float(b_pose[1] - p_pose[1])
                distance = float(math.hypot(dx, dy))
                # Angle relative to player orientation
                world_angle = math.atan2(dy, dx)
                player_heading = float(p_pose[3]) if len(p_pose) > 3 else 0.0
                angle = float((world_angle - player_heading + math.pi) % (2.0 * math.pi) - math.pi)

                p_max_hp = max(1.0, _to_float(obs.get("player_max_hp"), 1037.0))
                b_max_hp = max(1.0, _to_float(obs.get("boss_max_hp"), 1037.0))
                p_max_sp = max(1.0, _to_float(obs.get("player_max_sp"), 100.0))

                player_hp = float(np.clip(_to_float(obs.get("player_hp"), p_max_hp) / p_max_hp, 0.0, 1.0))
                player_sp = float(np.clip(_to_float(obs.get("player_sp"), p_max_sp) / p_max_sp, 0.0, 1.0))
                boss_hp = float(np.clip(_to_float(obs.get("boss_hp"), b_max_hp) / b_max_hp, 0.0, 1.0))

                boss_anim = obs.get("boss_animation", -1)
                boss_attacking = bool(boss_anim > 0 or obs.get("boss_attacking", False))
            else:
                player_hp = _to_float(obs.get("player_hp"), 1.0)
                player_sp = _to_float(obs.get("player_sp"), 1.0)
                boss_hp = _to_float(obs.get("boss_hp"), 1.0)
                distance = _to_float(obs.get("boss_distance"), 5.0)
                angle = _to_float(obs.get("boss_rel_angle"), 0.0)  # [-pi, pi]
                boss_attacking = bool(obs.get("boss_attacking", False))
        elif isinstance(obs, (np.ndarray, list)):
            # Vector fallback: [player_hp, player_sp, boss_hp, distance, angle, boss_attacking]
            player_hp = float(obs[0]) if len(obs) > 0 else 1.0
            player_sp = float(obs[1]) if len(obs) > 1 else 1.0
            boss_hp = float(obs[2]) if len(obs) > 2 else 1.0
            distance = float(obs[3]) if len(obs) > 3 else 5.0
            angle = float(obs[4]) if len(obs) > 4 else 0.0
            boss_attacking = bool(obs[5] > 0.5) if len(obs) > 5 else False
        else:
            player_hp, player_sp, boss_hp, distance, angle, boss_attacking = 1.0, 1.0, 1.0, 5.0, 0.0, False

        # 1. Visual Hemifield / Retinal Activation (Compound Eye)
        # Receptive fields span azimuth from -pi (far left) to +pi (far right)
        retina = self.topology.retina_indices
        num_retina = len(retina)
        if num_retina > 0:
            azimuth_centers = np.linspace(-math.pi, math.pi, num_retina)
            # Receptive field tuning width
            sigma = (2.0 * math.pi) / num_retina * 1.5
            # Angular visual tuning (von Mises / Gaussian profile)
            angular_dist = np.angle(np.exp(1j * (azimuth_centers - angle)))
            angular_response = np.exp(-0.5 * (angular_dist / sigma) ** 2)

            # Looming: Closer distance dramatically increases retinal brightness/extent
            # Normal range: distance in [1.5m (melee), 15m (far)]
            looming_factor = float(np.clip(12.0 / max(1.0, distance), 0.5, 6.0))
            retinal_drive = angular_response * looming_factor * 2.5
            drive[retina] += retinal_drive.astype(np.float32)

        # 2. Looming Motion Detectors (LPTC / LC11 cells in Lobula)
        # Triggers heavily when boss is in attack windup or closing distance rapidly
        motion = self.topology.motion_indices
        if len(motion) > 0:
            closing_speed = max(0.0, self.prev_distance - distance)
            motion_stimulus = (closing_speed * 3.0) + (8.0 if boss_attacking else 0.0)
            drive[motion] += float(motion_stimulus)

        # 3. Central Complex Compass (EPG Heading Neurons)
        # Ring attractor representing angle relative to enemy
        compass = self.topology.compass_indices
        num_compass = len(compass)
        if num_compass > 0:
            compass_angles = np.linspace(-math.pi, math.pi, num_compass, endpoint=False)
            compass_dist = np.angle(np.exp(1j * (compass_angles - angle)))
            # Sharp directional tuning curve
            compass_tuning = np.exp(-0.5 * (compass_dist / 0.4) ** 2) * 5.0
            drive[compass] += compass_tuning.astype(np.float32)

        # 4. Nociceptor Activation (Pain / Damage Reflex)
        # Injects massive burst into nociceptors when player HP drops
        nociceptors = self.topology.nociceptor_indices
        if len(nociceptors) > 0:
            damage_taken = max(0.0, self.prev_player_hp - player_hp)
            if damage_taken > 0.001:
                pain_current = min(25.0, damage_taken * 60.0)
                drive[nociceptors] += float(pain_current)

        # Update historical state
        self.prev_player_hp = player_hp
        self.prev_distance = distance
        self.last_distance = distance
        self.last_angle = angle

        return drive
