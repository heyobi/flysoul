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

    def decode(self, spike_counts: np.ndarray, explore: bool = False) -> Tuple[int, str, Dict[str, float]]:
        """Determine game action from descending neuron spike counts.

        Args:
            spike_counts: Array of spike counts per neuron over the simulation step.
            explore: If True, uses softmax temperature sampling instead of argmax.

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

        # Score the discrete actions
        # Action 0: idle
        # Action 1: dodge_roll
        # Action 2: attack_light
        # Action 3: attack_heavy
        # Action 4: block_parry
        # Action 5: step_back
        # Action 6: run_forward (synthesized if approaching or attacking)

        action_scores = np.array([
            0.5,                                       # 0: Idle baseline
            rates.get("dodge_roll", 0.0) * 1.4,        # 1: Escape / dodge roll (high priority reflex)
            rates.get("attack_light", 0.0) * 1.1,      # 2: Light attack
            rates.get("attack_heavy", 0.0) * 1.0,      # 3: Heavy attack
            rates.get("block_parry", 0.0) * 1.2,       # 4: Block / parry
            rates.get("step_back", 0.0) * 1.1,         # 5: Step back
            (rates.get("turn_left", 0.0) + rates.get("turn_right", 0.0)) * 0.8,  # 6: Forward advance
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
