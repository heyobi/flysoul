"""Motor decoder: Decodes Descending Neuron (DN) spiking into SoulsGym combat actions."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from flysoul.connectome.graph import CircuitTopology


class MotorDecoder:
    """Decodes spiking activity of Descending Motor Neurons into discrete game actions."""

    # SoulsGym action mapping
    ACTION_NAMES = [
        "idle",          # 0
        "dodge_roll",    # 1
        "attack_light",  # 2
        "attack_heavy",  # 3
        "block_parry",   # 4
        "step_back",     # 5
        "run_forward",   # 6
    ]

    def __init__(
        self,
        topology: CircuitTopology,
        threshold: float = 1.0,
        temperature: float = 0.5,
    ):
        self.topology = topology
        self.threshold = threshold
        self.temperature = temperature
        self.motor_indices = topology.motor_indices

    def decode(
        self,
        spike_counts: np.ndarray,
        explore: bool = False,
        distance: float = 10.0,
        boss_attacking: bool = False,
    ) -> Tuple[int, str, Dict[str, float]]:
        """Determine game action from descending neuron spike counts.

        Args:
            spike_counts: Array of spike counts per neuron over the simulation step.
            explore: If True, uses softmax temperature sampling instead of argmax.
            distance: Distance to enemy in meters.
            boss_attacking: Whether enemy is currently executing an attack.

        Returns:
            action_id: Integer action for SoulsGym.
            action_name: Human-readable action name.
            firing_rates: Dict mapping motor pool names to their mean spike count.
        """
        # Calculate mean firing rate per motor pool
        rates: Dict[str, float] = {}
        for pool_name, idx_list in self.motor_indices.items():
            if len(idx_list) > 0:
                rates[pool_name] = float(np.mean(spike_counts[idx_list]))
            else:
                rates[pool_name] = 0.0

        # Melee strike opportunity: if in striking range and not dodging an attack
        melee_window = (distance <= 3.0) and (not boss_attacking)
        if melee_window and (np.random.random() < 0.28):
            rates["attack_light"] = max(rates.get("attack_light", 0.0), 1.8)

        # Approach drive: if far from enemy (>3.0m), actively close distance to engage in combat
        approach_drive = float(np.clip((distance - 2.5) * 0.45, 0.0, 2.2))
        forward_score = (rates.get("turn_left", 0.0) + rates.get("turn_right", 0.0)) * 0.8 + approach_drive

        # Score the discrete actions
        action_scores = np.array([
            0.4,                                       # 0: Idle baseline
            rates.get("dodge_roll", 0.0) * (1.6 if boss_attacking else 1.1),  # 1: Escape / dodge roll
            rates.get("attack_light", 0.0) * (1.5 if melee_window else 0.9),  # 2: Light attack
            rates.get("attack_heavy", 0.0) * (1.2 if melee_window else 0.8),  # 3: Heavy attack
            rates.get("block_parry", 0.0) * (1.3 if boss_attacking else 0.8),  # 4: Block / parry
            rates.get("step_back", 0.0) * (1.2 if boss_attacking else 0.8),    # 5: Step back
            forward_score,                             # 6: Forward advance / approach target
        ], dtype=np.float32)

        if explore:
            # Softmax Boltzmann distribution
            exp_scores = np.exp((action_scores - np.max(action_scores)) / max(0.1, self.temperature))
            probs = exp_scores / np.sum(exp_scores)
            action_id = int(np.random.choice(len(action_scores), p=probs))
        else:
            # Winner-take-all (Argmax with threshold gate)
            max_idx = int(np.argmax(action_scores))
            if max_idx == 0 or action_scores[max_idx] >= self.threshold:
                action_id = max_idx
            else:
                action_id = 0  # Below threshold -> idle

        action_name = self.ACTION_NAMES[action_id]
        return action_id, action_name, rates

    @staticmethod
    def to_soulsgym_action(action_id: int) -> int:
        """Translate internal agent action ID to SoulsGym discrete action space.

        SoulsGym 1.2.0 discrete action IDs:
            0: forward
            4: backward
            8: forward roll
            12: backward roll
            16: light attack
            17: heavy attack
            18: parry
            19: idle / do nothing
        """
        mapping = {
            0: 19,  # idle
            1: 12,  # dodge_roll (backward roll)
            2: 16,  # light attack
            3: 17,  # heavy attack
            4: 18,  # parry
            5: 4,   # step backward
            6: 0,   # run forward
        }
        return mapping.get(action_id, 19)
