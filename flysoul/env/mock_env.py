"""Mock Gymnasium environment simulating the Dark Souls III Iudex Gundyr boss fight.

Stands in for SoulsGymIudex-v0 for offline testing. It models the things that actually
decide the fight and that the live environment enforces:

* the player is locked out of input for the duration of whatever animation is running,
  and the set of currently executable actions is published in ``info["valid_actions"]``
* stamina is consumed by rolls and swings and gates them
* the boss telegraphs, and ``boss_animation_duration`` reports how far into the swing it
  is, so the timing information the circuit needs is present here too
* poise breaks open a punish window

It is a test harness, not a substitute for the real fight: the live game is the only
place the agent's behaviour actually counts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action IDs, matching flysoul.motor.decoder.MOCK_ACTIONS
IDLE, ROLL, ATTACK_LIGHT, ATTACK_HEAVY, PARRY, RETREAT, ADVANCE, STRAFE_LEFT, STRAFE_RIGHT = range(9)

# Steps the player is locked out of input after starting each action, and its stamina cost.
_ACTION_LOCK = {
    IDLE: 0, ROLL: 5, ATTACK_LIGHT: 6, ATTACK_HEAVY: 10,
    PARRY: 4, RETREAT: 1, ADVANCE: 1, STRAFE_LEFT: 1, STRAFE_RIGHT: 1,
}
_STAMINA_COST = {
    IDLE: 0.0, ROLL: 0.20, ATTACK_LIGHT: 0.16, ATTACK_HEAVY: 0.30,
    PARRY: 0.10, RETREAT: 0.0, ADVANCE: 0.0, STRAFE_LEFT: 0.0, STRAFE_RIGHT: 0.0,
}
_MELEE_RANGE = 2.8
_BOSS_REACH = 3.8


class MockIudexEnv(gym.Env):
    """Physical and behavioural simulation of the Iudex Gundyr boss fight."""

    metadata = {"render_modes": ["ansi"]}
    step_seconds = 0.1

    def __init__(self, max_steps: int = 1500, seed: int | None = None, skip_steps: bool = True):
        super().__init__()
        self.max_steps = max_steps
        self.skip_steps = skip_steps
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "player_hp": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "player_sp": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "boss_hp": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "boss_distance": spaces.Box(0.0, 30.0, shape=(1,), dtype=np.float32),
            "boss_rel_angle": spaces.Box(-math.pi, math.pi, shape=(1,), dtype=np.float32),
            "boss_attacking": spaces.Discrete(2),
            "boss_staggered": spaces.Discrete(2),
            "boss_animation_duration": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32),
        })

        self.player_max_hp = 454
        self.boss_max_hp = 1037
        self.reset()

    # ------------------------------------------------------------------ helpers

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "player_hp": np.array([self.player_hp / self.player_max_hp], dtype=np.float32),
            "player_sp": np.array([self.player_sp], dtype=np.float32),
            "boss_hp": np.array([self.boss_hp / self.boss_max_hp], dtype=np.float32),
            "boss_distance": np.array([self.distance], dtype=np.float32),
            "boss_rel_angle": np.array([self.angle], dtype=np.float32),
            "boss_attacking": 1 if self.boss_state in ("windup", "strike") else 0,
            "boss_staggered": 1 if self.boss_state == "staggered" else 0,
            "boss_animation_duration": np.array([self.boss_anim_time], dtype=np.float32),
        }

    def valid_actions(self) -> List[int]:
        """Actions the player can execute right now, mirroring SoulsEnv.current_valid_actions."""
        if self.player_lock > 0:
            return [IDLE]
        actions = [IDLE, RETREAT, ADVANCE, STRAFE_LEFT, STRAFE_RIGHT]
        for act in (ROLL, ATTACK_LIGHT, ATTACK_HEAVY, PARRY):
            if self.player_sp >= _STAMINA_COST[act]:
                actions.append(act)
        return sorted(actions)

    def _info(self, extra: dict | None = None) -> dict:
        info = {
            "player_hp_raw": int(self.player_hp),
            "boss_hp_raw": int(self.boss_hp),
            "boss_state": self.boss_state,
            "distance": self.distance,
            "valid_actions": self.valid_actions(),
        }
        if extra:
            info.update(extra)
        return info

    # -------------------------------------------------------------------- gym API

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.player_hp = float(self.player_max_hp)
        self.player_sp = 1.0
        self.boss_hp = float(self.boss_max_hp)
        self.distance = 8.0
        self.angle = 0.0

        self.boss_state = "idle"
        self.boss_timer = 15
        self.boss_anim_time = 0.0
        self.boss_attack_damage = 0
        self.boss_poise = 1.0

        self.player_iframes = 0
        self.player_lock = 0
        # Stamina does not come back during an action or immediately after it.
        self.player_regen_delay = 0
        self.player_blocking = False
        self._strafed = False
        return self._get_obs(), self._info()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        rng = self._rng
        reward = 0.0
        terminated = False
        prev_distance = self.distance

        if self.player_iframes > 0:
            self.player_iframes -= 1
        if self.player_lock > 0:
            self.player_lock -= 1
            action = IDLE  # Input is swallowed by the running animation.
        self.player_blocking = False
        self._strafed = False

        self._regenerate_stamina()

        if action != IDLE and self.player_sp >= _STAMINA_COST[action]:
            self.player_sp = max(0.0, self.player_sp - _STAMINA_COST[action])
            self.player_lock = _ACTION_LOCK[action]
            if _STAMINA_COST[action] > 0.0:
                self.player_regen_delay = self.player_lock + 6
            reward += self._apply_player_action(action, rng)

        reward += self._advance_boss(rng, prev_distance)

        # Mirror SoulsEnv.step: while the player is locked in an animation there is only
        # one executable action, so keep the world running rather than spending the
        # agent's decisions on frames the game would discard anyway.
        if self.skip_steps:
            while self.player_lock > 0 and self.boss_hp > 0 and self.player_hp > 0:
                if self.step_count >= self.max_steps:
                    break
                self.step_count += 1
                self.player_lock -= 1
                if self.player_iframes > 0:
                    self.player_iframes -= 1
                self._regenerate_stamina()
                reward += self._advance_boss(rng, self.distance)

        if self.boss_hp <= 0:
            terminated = True
            reward += 10.0
        elif self.player_hp <= 0:
            terminated = True
            reward -= 5.0

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._info()

    def _regenerate_stamina(self):
        """Stamina only comes back once the action and its recovery are over.

        Regenerating through the animation would refund a heavy swing before it even
        landed, which makes stamina - and the interoceptive pathway that reads it -
        mean nothing.
        """
        if self.player_regen_delay > 0:
            self.player_regen_delay -= 1
            return
        if self.player_lock == 0:
            self.player_sp = min(1.0, self.player_sp + 0.05)

    # --------------------------------------------------------------- mechanics

    def _apply_player_action(self, action: int, rng) -> float:
        reward = 0.0
        if action == ROLL:
            self.player_iframes = 4
            # A roll travels; with lock-on it goes wherever the stick points, and the
            # circuit's locomotor pools decide that. Here it simply opens a little space.
            self.distance = min(18.0, self.distance + 0.8)
        elif action in (ATTACK_LIGHT, ATTACK_HEAVY):
            in_range = self.distance <= _MELEE_RANGE and abs(self.angle) < 0.9
            if in_range:
                dmg = int(rng.integers(40, 56) if action == ATTACK_LIGHT else rng.integers(85, 111))
                self.boss_hp = max(0.0, self.boss_hp - dmg)
                self.boss_poise -= 0.34 if action == ATTACK_HEAVY else 0.18
                reward += dmg / 100.0
                if self.boss_poise <= 0.0:
                    self.boss_state = "staggered"
                    self.boss_timer = 8
                    self.boss_anim_time = 0.0
                    self.boss_poise = 1.0
        elif action == PARRY:
            self.player_blocking = True
        elif action == RETREAT:
            self.distance = min(18.0, self.distance + 0.6)
        elif action == ADVANCE:
            self.distance = max(1.2, self.distance - 1.2)
        elif action in (STRAFE_LEFT, STRAFE_RIGHT):
            self._strafed = True
            # Circling changes the bearing; lock-on pulls it back towards centre.
            delta = 0.35 if action == STRAFE_RIGHT else -0.35
            self.angle = float(np.clip(self.angle + delta, -math.pi, math.pi)) * 0.65
        return reward

    def _advance_boss(self, rng, prev_distance: float) -> float:
        reward = 0.0
        self.boss_anim_time += self.step_seconds
        if self.boss_timer > 0:
            self.boss_timer -= 1
            return reward

        self.boss_anim_time = 0.0
        if self.boss_state == "idle":
            if self.distance > 3.5:
                self.boss_state = "approach"
                self.boss_timer = int(rng.integers(3, 7))
            else:
                self.boss_state = "windup"
                self.boss_timer = int(rng.integers(3, 6))
                self.boss_attack_damage = int(rng.choice([120, 180, 140]))
        elif self.boss_state == "approach":
            self.distance = max(2.2, self.distance - 1.5)
            self.boss_state = "windup"
            self.boss_timer = int(rng.integers(3, 6))
            self.boss_attack_damage = 160
        elif self.boss_state == "windup":
            self.boss_state = "strike"
            self.boss_timer = 1
            reward += self._resolve_strike(rng)
        elif self.boss_state == "strike":
            self.boss_state = "recovery"
            self.boss_timer = int(rng.integers(4, 8))
        elif self.boss_state in ("recovery", "staggered"):
            self.boss_state = "idle"
            self.boss_timer = int(rng.integers(3, 7))
        return reward

    def _resolve_strike(self, rng) -> float:
        if self.distance > _BOSS_REACH:
            return 0.0
        if self.player_iframes > 0:
            return 0.5  # Rolled through it.
        if self._strafed and rng.random() < 0.4:
            return 0.25  # Circled out of the arc.
        if self.player_blocking:
            chip = self.boss_attack_damage * 0.2
            self.player_hp = max(0.0, self.player_hp - chip)
            self.player_sp = max(0.0, self.player_sp - 0.35)
            return -chip / 100.0
        self.player_hp = max(0.0, self.player_hp - self.boss_attack_damage)
        return -(self.boss_attack_damage / 100.0) * 1.5

    def close(self):
        pass
