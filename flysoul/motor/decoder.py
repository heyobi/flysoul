"""Motor decoder: Decodes Descending Neuron (DN) spiking into SoulsGym combat actions.

This module deliberately contains **no policy**. It reads one firing rate per descending
pool and reports the winner. Choosing between approaching, rolling and swinging is the
circuit's job, resolved by premotor cross-inhibition in :mod:`flysoul.connectome.graph`.

The previous version scored actions with hand-written constants (``advance_drive = 3.0``
when far, a forced ``attack_light`` rate in melee range, a 1.8x multiplier on dodging
whenever the boss was deemed to be attacking). Those constants, not the connectome,
produced the observed behaviour: the descending pools other than ``dodge_roll`` never
fired at all, so the agent ran forward on the hard-coded advance term and rolled
backwards the moment it expired.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from flysoul.connectome.graph import ACTION_CHANNELS, CircuitTopology

IDLE = "idle"

# SoulsGym DarkSoulsIII discrete action IDs (soulsgym/core/data/darksouls3/actions.yaml):
#   0-7   walk    forward, fwd-right, right, right-back, backward, back-left, left, left-fwd
#   8-15  roll    same eight directions
#   16    light attack   17  heavy attack   18  parry   19  do nothing
SOULSGYM_WALK = {"advance": 0, "strafe_right": 2, "retreat": 4, "strafe_left": 6}
SOULSGYM_ROLL = {"advance": 8, "strafe_right": 10, "retreat": 12, "strafe_left": 14}
SOULSGYM_ATTACK = {"attack_light": 16, "attack_heavy": 17, "parry": 18}
SOULSGYM_IDLE = 19

# Mock environment action IDs (flysoul.env.mock_env.MockIudexEnv)
MOCK_ACTIONS = {
    IDLE: 0,
    "roll": 1,
    "attack_light": 2,
    "attack_heavy": 3,
    "parry": 4,
    "retreat": 5,
    "advance": 6,
    "strafe_left": 7,
    "strafe_right": 8,
}


class MotorDecoder:
    """Reads out the descending motor pools. Reports the winner; does not pick it."""

    ACTION_NAMES = (IDLE,) + ACTION_CHANNELS

    def __init__(
        self,
        topology: CircuitTopology,
        threshold_hz: float = 8.0,
        temperature: float = 4.0,
        step_ms: float = 100.0,
    ):
        self.topology = topology
        # Minimum firing rate for a pool to count as committed rather than noise,
        # in Hz so it does not silently change meaning with the step length.
        self.threshold = threshold_hz
        self.step_s = step_ms / 1000.0
        self.temperature = temperature
        self.motor_indices = topology.motor_indices

    def pool_rates(self, spike_counts: np.ndarray) -> Dict[str, float]:
        """Mean firing rate in Hz per descending pool over the simulated interval."""
        scale = 1.0 / max(1e-6, self.step_s)
        return {
            name: float(np.mean(spike_counts[idx])) * scale if len(idx) else 0.0
            for name, idx in self.motor_indices.items()
        }

    def decode(
        self,
        spike_counts: np.ndarray,
        explore: bool = False,
        valid_channels: Iterable[str] | None = None,
        **_ignored,
    ) -> Tuple[str, Dict[str, float]]:
        """Return the winning action channel and the full rate vector.

        Args:
            spike_counts: Spike counts per neuron over the simulation step.
            explore: Sample from a Boltzmann distribution over pool rates instead of
                taking the maximum. Exploration is the only stochastic element; it does
                not reweight any channel.
            valid_channels: Channels the environment can currently execute. Pools that
                cannot be executed are excluded from the readout rather than being
                issued and silently dropped by the game.

        Returns:
            ``(channel_name, rates)`` where ``channel_name`` is an action channel or
            ``"idle"``.
        """
        rates = self.pool_rates(spike_counts)

        candidates = list(self.motor_indices.keys())
        if valid_channels is not None:
            allowed = set(valid_channels)
            candidates = [c for c in candidates if c in allowed]
        if not candidates:
            return IDLE, rates

        scores = np.array([rates[c] for c in candidates], dtype=np.float64)

        if explore and float(np.max(scores)) > 0.0:
            shifted = (scores - scores.max()) / max(1e-3, self.temperature)
            probs = np.exp(shifted)
            probs /= probs.sum()
            winner = candidates[int(np.random.choice(len(candidates), p=probs))]
        else:
            winner = candidates[int(np.argmax(scores))]

        if rates[winner] < self.threshold:
            return IDLE, rates
        return winner, rates

    # ------------------------------------------------------------------ mapping

    def to_soulsgym_action(self, channel: str, rates: Dict[str, float] | None = None) -> int:
        """Translate a winning channel into a SoulsGym discrete action ID.

        A roll is a gait, not a direction: its heading comes from whichever locomotor
        pool is co-active with it, exactly as the walk channels supply their own.
        """
        if channel in SOULSGYM_ATTACK:
            return SOULSGYM_ATTACK[channel]
        if channel in SOULSGYM_WALK:
            return SOULSGYM_WALK[channel]
        if channel == "roll":
            direction = "advance"
            if rates:
                locomotor = {k: rates.get(k, 0.0) for k in SOULSGYM_ROLL}
                best = max(locomotor, key=locomotor.get)
                if locomotor[best] > 0.0:
                    direction = best
            return SOULSGYM_ROLL[direction]
        return SOULSGYM_IDLE

    def to_game_action(
        self,
        channel: str,
        rates: Dict[str, float] | None = None,
        mock: bool = False,
        lock_on: bool = True,
    ) -> Tuple[int, str]:
        """Translate a winning channel into an action for the environment in use.

        Returns ``(action_id, executed_channel)``. The executed channel is what the body
        actually did, which is what the efference copy and the dopamine credit have to be
        based on - not what the circuit asked for.

        Without lock-on the two come apart. Every direction the descending pools encode
        is measured against the boss, but SoulsGym applies movement against the camera,
        and the action space contains no camera control, so an unlocked fly cannot steer
        at all: issuing its locomotor choice makes it sprint off in whatever direction
        the camera was left pointing. Holding position instead keeps it in the arena and
        still lets the environment's per-step routine turn the camera back.
        """
        if mock:
            return self.to_mock_action(channel), channel
        if not lock_on:
            return SOULSGYM_IDLE, IDLE
        return self.to_soulsgym_action(channel, rates), channel

    @staticmethod
    def to_mock_action(channel: str) -> int:
        """Translate a winning channel into a MockIudexEnv action ID."""
        return MOCK_ACTIONS.get(channel, 0)

    @staticmethod
    def channels_for_valid_actions(
        valid_actions: Sequence[int] | None, mock: bool = False
    ) -> set[str] | None:
        """Map an environment's valid-action mask onto the channels it permits.

        The two environments number their actions differently, so the mask has to be
        read in the right dialect. Reading a mock mask as SoulsGym IDs turns "only idle
        is available" (mock 0) into "only walk forward is available" (SoulsGym 0), which
        forces the readout to advance on exactly the steps where the body is locked.

        Returns an empty set when a mask exists but permits nothing, and ``None`` only
        when there is no mask at all. Collapsing the two means that the moment the game
        reports the player is locked mid-animation, the readout becomes *unconstrained*
        and issues a swing the game then throws away.
        """
        if valid_actions is None:
            return None
        valid = set(int(a) for a in valid_actions)
        allowed: set[str] = set()

        if mock:
            for channel, action_id in MOCK_ACTIONS.items():
                if channel != IDLE and action_id in valid:
                    allowed.add(channel)
            return allowed

        for channel, action_id in SOULSGYM_WALK.items():
            if action_id in valid:
                allowed.add(channel)
        for channel, action_id in SOULSGYM_ATTACK.items():
            if action_id in valid:
                allowed.add(channel)
        if any(a in valid for a in SOULSGYM_ROLL.values()):
            allowed.add("roll")
        return allowed
