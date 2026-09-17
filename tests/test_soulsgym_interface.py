"""Exercise the live-game code path without Dark Souls III.

SoulsGym cannot be installed on the development machine (it hooks a running
DarkSoulsIII.exe), so the branch that actually plays the game is the branch that never
gets run locally. This module stands in a fake environment that speaks SoulsGym's exact
dialect -- its observation keys, its 20 discrete actions, its animation ID numbering and
its ``current_valid_actions`` mask -- and drives the real agent through it.
"""

import gymnasium as gym
import numpy as np
import pytest

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.obs import parse_obs
from flysoul.env.souls_wrapper import ValidActionInfo, is_mock
from flysoul.motor.decoder import MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder

# soulsgym/core/data/darksouls3/actions.yaml
WALK_IDS = set(range(0, 8))
ROLL_IDS = set(range(8, 16))
ATTACK_IDS = {16, 17, 18}
IDLE_ID = 19


class FakeSoulsGymIudex(gym.Env):
    """Minimal stand-in speaking the real SoulsGym observation and action dialect."""

    step_size = 0.1  # SoulsEnv.step_size
    ARENA_CENTER = np.array([139.0, 596.0], dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(20)
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, **kwargs):
        self.player_pose = np.array([139.0, 588.0, -68.0, 1.5708], dtype=np.float32)
        self.boss_pose = np.array([139.0, 596.0, -68.0, -1.5708], dtype=np.float32)
        self.player_hp = 454.0
        self.player_sp = 95.0
        self.boss_hp = 1037.0
        self.boss_animation = 23  # IdleBattle
        self.boss_animation_duration = 0.0
        self.player_animation_duration = 0.0
        self.locked_steps = 0
        self.lock_on = True
        self.t = 0
        return self.obs, self.info

    @property
    def obs(self):
        return {
            "phase": 1,
            "player_hp": np.float32(self.player_hp),
            "player_max_hp": 454,
            "player_sp": np.float32(self.player_sp),
            "player_max_sp": 95,
            "boss_hp": np.float32(self.boss_hp),
            "boss_max_hp": 1037,
            "player_pose": self.player_pose.copy(),
            "boss_pose": self.boss_pose.copy(),
            "camera_pose": np.zeros(6, dtype=np.float32),
            "player_animation": 0,
            "player_animation_duration": np.float32(self.player_animation_duration),
            "boss_animation": self.boss_animation,
            "boss_animation_duration": np.float32(self.boss_animation_duration),
            "lock_on": 1 if self.lock_on else 0,
        }

    @property
    def info(self):
        return {}

    def current_valid_actions(self):
        if self.locked_steps > 0:
            return [IDLE_ID]
        actions = sorted(WALK_IDS) + [IDLE_ID]
        if self.player_sp > 0:
            actions = sorted(WALK_IDS | ROLL_IDS | ATTACK_IDS | {IDLE_ID})
        return actions

    def step(self, action):
        assert action in self.current_valid_actions() or action == IDLE_ID, (
            f"agent issued action {action}, which the game cannot execute now"
        )
        self.t += 1
        if self.locked_steps > 0:
            self.locked_steps -= 1
            action = IDLE_ID

        heading = float(self.player_pose[3])
        if action in WALK_IDS or action in ROLL_IDS:
            # Lock-on movement is boss-relative; forward closes distance.
            delta = 0.9 if action in WALK_IDS else 1.6
            to_boss = self.boss_pose[:2] - self.player_pose[:2]
            norm = np.linalg.norm(to_boss) or 1.0
            direction = to_boss / norm
            base = action % 8
            if base == 4:  # backward
                direction = -direction
            elif base in (2, 6):  # strafe
                direction = np.array([-direction[1], direction[0]], dtype=np.float32)
                if base == 6:
                    direction = -direction
            self.player_pose[:2] += direction * delta
            if action in ROLL_IDS:
                self.player_sp = max(0.0, self.player_sp - 20.0)
                self.locked_steps = 4
        elif action in ATTACK_IDS:
            self.player_sp = max(0.0, self.player_sp - 15.0)
            self.locked_steps = 5
            if float(np.linalg.norm(self.boss_pose[:2] - self.player_pose[:2])) <= 3.0:
                self.boss_hp = max(0.0, self.boss_hp - 90.0)
        else:
            self.player_sp = min(95.0, self.player_sp + 5.0)

        # Boss cycles idle -> attack -> idle so both animation categories are exercised.
        if self.t % 7 == 0:
            self.boss_animation = int(self.rng.integers(0, 16))  # an attack ID
            self.boss_animation_duration = 0.0
        elif self.t % 7 == 3:
            self.boss_animation = 23  # IdleBattle
            self.boss_animation_duration = 0.0
        else:
            self.boss_animation_duration += self.step_size

        reward = 0.0
        terminated = self.boss_hp <= 0 or self.player_hp <= 0
        truncated = self.t >= 120
        return self.obs, reward, terminated, truncated, self.info

    def close(self):
        pass


@pytest.fixture
def agent():
    cfg = CircuitConfig(
        num_retina_brightness=32, num_retina_motion=24, num_compass_neurons=8,
        num_ring_inhibitory=8, num_cx_inhibitory=12, num_nociceptors=4,
        num_proprioceptors=8, num_vigor_gate=4, num_approach_brake=4,
        num_kenyon_cells=128, num_mbon_per_compartment=4, num_central_complex=48,
        num_apl=4, num_ppl1_dopamine=8, num_pam_dopamine=8,
        num_dn_per_pool=6, num_dn_inhibitory_per_pool=4,
    )
    bio = BioPhysicsConfig()
    topology = build_fly_circuit(cfg, seed=7)
    return (
        topology,
        ConnectomeEngine(topology.ptr, topology.post, topology.weight, bio),
        SensoryEncoder(topology, bio),
        MotorDecoder(topology, step_ms=100.0),
        DopaminePlasticity(topology),
    )


def test_wrapper_publishes_the_soulsgym_mask():
    env = ValidActionInfo(FakeSoulsGymIudex())
    _, info = env.reset()
    assert IDLE_ID in info["valid_actions"]
    assert not is_mock(env)


def test_full_loop_against_soulsgym_dialect(agent):
    """Every action the readout issues must be one the game would actually accept."""
    topology, engine, encoder, decoder, plasticity = agent
    env = ValidActionInfo(FakeSoulsGymIudex())
    obs, info = env.reset()
    state = parse_obs(obs, info)

    step_ms = float(getattr(env.unwrapped, "step_size", 0.1)) * 1000.0
    decoder.step_s = step_ms / 1000.0
    assert step_ms == 100.0

    issued = set()
    terminated = truncated = False
    while not (terminated or truncated):
        spikes = engine.step(encoder.encode(state), duration_ms=step_ms)
        valid = decoder.channels_for_valid_actions(state.valid_actions, mock=False)
        channel, rates = decoder.decode(spikes, valid_channels=valid)
        action = decoder.to_soulsgym_action(channel, rates)
        issued.add(action)
        # The fake env asserts the action is executable, so an unmasked or
        # mistranslated readout fails here rather than being silently dropped in game.
        obs, reward, terminated, truncated, info = env.step(action)
        state = parse_obs(obs, info)
        encoder.set_efference(channel)
        plasticity.update_traces(spikes, executed_channel=channel)
        plasticity.apply_reinforcement(reward)

    assert issued <= (WALK_IDS | ROLL_IDS | ATTACK_IDS | {IDLE_ID})


def test_boss_attack_state_tracks_the_animation_id():
    """An idling boss must not read as attacking - the bug that pinned the escape reflex on."""
    env = ValidActionInfo(FakeSoulsGymIudex())
    obs, info = env.reset()
    assert parse_obs(obs, info).boss_attacking is False  # IdleBattle (23)

    env.unwrapped.boss_animation = 7  # Attack3007
    assert parse_obs(env.unwrapped.obs, info).boss_attacking is True

    env.unwrapped.boss_animation = 19  # WalkFrontBattle
    assert parse_obs(env.unwrapped.obs, info).boss_attacking is False


def test_masked_readout_never_issues_a_locked_action(agent):
    """While the player is animation-locked, only idle may be issued."""
    topology, engine, encoder, decoder, _ = agent
    env = ValidActionInfo(FakeSoulsGymIudex())
    env.reset()
    env.unwrapped.locked_steps = 3
    info = {"valid_actions": env.unwrapped.current_valid_actions()}
    state = parse_obs(env.unwrapped.obs, info)

    assert state.player_can_act is False
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.motor_indices["attack_heavy"]] = 20
    valid = decoder.channels_for_valid_actions(state.valid_actions, mock=False)
    channel, rates = decoder.decode(spikes, valid_channels=valid)
    assert decoder.to_soulsgym_action(channel, rates) == IDLE_ID


def test_unlocked_camera_holds_position_instead_of_running(agent):
    """Without lock-on the fly must not issue locomotion.

    SoulsGym movement is camera-relative while unlocked and the action space has no
    camera control, so any locomotor command sends the fly sprinting in whatever
    direction the camera was left pointing - away from the fight as often as towards it.
    """
    topology, _, _, decoder, _ = agent
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.motor_indices["advance"]] = 20
    channel, rates = decoder.decode(spikes)
    assert channel == "advance"

    locked_action, locked_channel = decoder.to_game_action(channel, rates, lock_on=True)
    assert locked_action == 0  # walk forward, which with lock-on means towards Iudex
    assert locked_channel == "advance"

    held_action, held_channel = decoder.to_game_action(channel, rates, lock_on=False)
    assert held_action == IDLE_ID
    # The efference copy and the dopamine credit must follow what the body did.
    assert held_channel == "idle"


def test_lock_on_state_is_read_from_the_observation():
    env = ValidActionInfo(FakeSoulsGymIudex())
    obs, info = env.reset()
    assert parse_obs(obs, info).lock_on is True
    env.unwrapped.lock_on = False
    assert parse_obs(env.unwrapped.obs, info).lock_on is False
