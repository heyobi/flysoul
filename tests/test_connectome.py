"""Unit tests for FlySoul connectome, encoder, decoder, and simulation components."""

import numpy as np
import pytest

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import ACTION_CHANNELS, build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.mock_env import MockIudexEnv
from flysoul.env.obs import IUDEX_ATTACK_IDS, CombatState, parse_obs
from flysoul.motor.decoder import IDLE, MOCK_ACTIONS, MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder


@pytest.fixture
def config():
    return CircuitConfig(
        num_retina_brightness=32,
        num_retina_motion=24,
        num_compass_neurons=8,
        num_ring_inhibitory=8,
        num_cx_inhibitory=12,
        num_nociceptors=4,
        num_proprioceptors=8,
        num_vigor_gate=4,
        num_approach_brake=4,
        num_kenyon_cells=128,
        num_mbon_per_compartment=4,
        num_central_complex=48,
        num_apl=4,
        num_ppl1_dopamine=8,
        num_pam_dopamine=8,
        num_dn_per_pool=6,
        num_dn_inhibitory_per_pool=4,
    )


@pytest.fixture
def topology(config):
    return build_fly_circuit(config, seed=123)


def test_topology_generation(topology):
    assert topology.num_neurons > 0
    assert len(topology.ptr) == topology.num_neurons + 1
    assert len(topology.post) == len(topology.weight)
    assert np.sum(topology.plastic_synapse_mask) > 0


def test_circuit_is_sign_constrained(topology):
    """Every neuron must be purely excitatory or purely inhibitory (Dale's law)."""
    assert np.any(topology.weight < 0), "a circuit with no inhibition cannot select actions"
    row_lengths = np.diff(topology.ptr)
    pre_of_edge = np.repeat(np.arange(len(row_lengths)), row_lengths)
    for sign in (1, -1):
        edges = np.sign(topology.weight) == sign
        if not edges.any():
            continue
        assert np.all(topology.neuron_sign[pre_of_edge[edges]] == sign)


def test_every_action_channel_is_reachable(topology):
    """Each descending pool must have afferents, or its action can never be selected."""
    for channel in ACTION_CHANNELS:
        pool = topology.motor_indices[channel]
        incoming = np.isin(topology.post, pool) & (topology.weight > 0)
        assert incoming.sum() > 0, f"{channel} has no excitatory input"


def test_engine_lif_integration(topology):
    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight)
    drive = np.zeros(topology.num_neurons, dtype=np.float32)
    drive[:10] = 30.0
    counts = engine.step(drive, duration_ms=20.0)
    assert counts.shape == (topology.num_neurons,)
    assert np.sum(counts[:10]) > 0  # Driven neurons must spike


def test_sensory_encoder_drives_every_afferent_above_threshold(topology):
    """Sensory drives are useless if they sit below the spike threshold."""
    bio = BioPhysicsConfig()
    encoder = SensoryEncoder(topology, bio)
    gap = bio.v_threshold - bio.v_rest
    state = CombatState(
        distance=2.5, angle=0.3, player_hp=0.6, player_sp=0.9,
        boss_attacking=True, boss_anim_time=0.4,
    )
    drive = encoder.encode(state)
    for name, idx in (
        ("retina", topology.retina_indices),
        ("compass", topology.compass_indices),
        ("telegraph", topology.sensory_subpools["motion_telegraph"]),
        ("object size", topology.sensory_subpools["motion_size"]),
    ):
        assert float(np.max(drive[idx])) > gap, f"{name} never reaches threshold"


def test_encoder_forward_model_cancels_self_generated_looming(topology):
    """Walking forward must not read as an object rushing in.

    The cancellation is learned: the fly builds a forward model of how much closing its
    own locomotor commands produce and subtracts only that.
    """
    bio = BioPhysicsConfig()
    gap = bio.v_threshold - bio.v_rest
    encoder = SensoryEncoder(topology, bio)
    loom = topology.sensory_subpools["motion_looming"]
    encoder.reset(1.0, 8.0)

    def closing_drive(command):
        encoder.prev_distance = 8.0
        encoder.set_efference(command)
        return float(np.max(encoder.encode(CombatState(distance=6.8))[loom]))

    first = closing_drive("advance")
    for _ in range(24):
        closing_drive("advance")
    settled = closing_drive("advance")

    assert settled < first
    assert settled < gap, "self-generated closing must stop driving the escape reflex"

    # An identical closing the fly did not cause is still a threat.
    external = closing_drive(None)
    assert external > gap, "the escape pathway must stay live for the boss's own approach"


def test_motor_decoder_reports_the_winning_pool(topology):
    decoder = MotorDecoder(topology, threshold_hz=1.0, step_ms=100.0)
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.motor_indices["attack_light"]] = 10
    channel, rates = decoder.decode(spikes)
    assert channel == "attack_light"
    assert rates["attack_light"] == pytest.approx(100.0)


def test_motor_decoder_respects_the_valid_action_mask(topology):
    decoder = MotorDecoder(topology, threshold_hz=1.0, step_ms=100.0)
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.motor_indices["attack_light"]] = 10
    spikes[topology.motor_indices["retreat"]] = 5
    channel, _ = decoder.decode(spikes, valid_channels={"retreat"})
    assert channel == "retreat"


def test_mask_dialects_do_not_cross_over(topology):
    """A mock mask must not be read as SoulsGym action IDs, or vice versa."""
    decoder = MotorDecoder(topology)
    # Mock action 0 is "idle"; SoulsGym action 0 is "walk forward".
    assert decoder.channels_for_valid_actions([MOCK_ACTIONS[IDLE]], mock=True) == set()
    assert decoder.channels_for_valid_actions([0], mock=False) == {"advance"}
    # An empty mask means nothing is executable; no mask at all means no constraint.
    assert decoder.channels_for_valid_actions(None) is None


def test_roll_direction_follows_the_locomotor_pools(topology):
    decoder = MotorDecoder(topology)
    rates = {c: 0.0 for c in ACTION_CHANNELS}
    rates["strafe_left"] = 30.0
    assert decoder.to_soulsgym_action("roll", rates) == 14  # left roll
    rates["strafe_left"] = 0.0
    rates["retreat"] = 30.0
    assert decoder.to_soulsgym_action("roll", rates) == 12  # backward roll


def test_boss_attack_animation_ids_are_the_low_ones():
    """Iudex attacks are IDs 0-18; movement and idle are 19+."""
    assert 0 in IUDEX_ATTACK_IDS  # Attack3000
    assert 18 in IUDEX_ATTACK_IDS  # ThrowDef
    assert 23 not in IUDEX_ATTACK_IDS  # IdleBattle
    assert 19 not in IUDEX_ATTACK_IDS  # WalkFrontBattle


@pytest.mark.parametrize(
    "animation, attacking",
    [(0, True), (5, True), (18, True), (19, False), (23, False), (-1, False)],
)
def test_parse_obs_classifies_boss_animation(animation, attacking):
    obs = {
        "player_pose": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "boss_pose": np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "player_hp": 400.0, "player_max_hp": 454.0,
        "player_sp": 90.0, "player_max_sp": 95.0,
        "boss_hp": 1000.0, "boss_max_hp": 1037.0,
        "boss_animation": animation,
    }
    assert parse_obs(obs).boss_attacking is attacking


def test_parse_obs_geometry():
    obs = {
        "player_pose": np.array([100.0, 600.0, -60.0, 0.0], dtype=np.float32),
        "boss_pose": np.array([103.0, 604.0, -60.0, 0.0], dtype=np.float32),
        "player_hp": 454.0, "player_max_hp": 454.0,
        "boss_hp": 1037.0, "boss_max_hp": 1037.0,
        "boss_animation": 23,
    }
    state = parse_obs(obs)
    assert state.distance == pytest.approx(5.0)  # 3-4-5, z is elevation and excluded
    assert state.angle == pytest.approx(np.arctan2(4.0, 3.0))


def test_plasticity_credits_only_the_executed_channel(topology):
    plasticity = DopaminePlasticity(topology, learning_rate=0.5)
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.kenyon_indices[:20]] = 3
    spikes[topology.mbon_indices] = 2
    before = topology.weight[plasticity.edges].copy()

    plasticity.update_traces(spikes, executed_channel="attack_light")
    plasticity.apply_reinforcement(reward=1.0)

    delta = topology.weight[plasticity.edges] - before
    credited = plasticity.channel == ACTION_CHANNELS.index("attack_light")
    assert np.any(delta[credited] > 0), "the executed channel must be reinforced"
    other = plasticity.channel == ACTION_CHANNELS.index("parry")
    assert np.max(np.abs(delta[other])) <= np.max(delta[credited])


def test_plasticity_aversive_signal_depresses(topology):
    plasticity = DopaminePlasticity(topology, learning_rate=0.5)
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.kenyon_indices[:20]] = 3
    spikes[topology.mbon_indices] = 2
    before = topology.weight[plasticity.edges].copy()
    plasticity.update_traces(spikes, executed_channel="roll")
    plasticity.apply_reinforcement(reward=-1.0)
    assert np.any(topology.weight[plasticity.edges] < before)


def test_mock_env_publishes_a_valid_action_mask():
    env = MockIudexEnv(seed=0)
    obs, info = env.reset()
    assert "valid_actions" in info
    assert set(info["valid_actions"]) <= set(MOCK_ACTIONS.values())
    obs, reward, term, trunc, info = env.step(MOCK_ACTIONS["attack_heavy"])
    assert obs["player_sp"][0] < 1.0  # Stamina was spent


def test_mock_env_locks_input_during_animations():
    env = MockIudexEnv(seed=0, skip_steps=False)
    env.reset()
    env.step(MOCK_ACTIONS["attack_heavy"])
    _, _, _, _, info = env.step(MOCK_ACTIONS["advance"])
    assert info["valid_actions"] == [MOCK_ACTIONS[IDLE]]


def test_entrypoint_and_scripts_compile():
    """run.py and the helper scripts are not imported by any other test.

    A syntax error in them therefore passes the whole suite and only shows up when the
    agent is launched - on the machine with the game attached, several minutes into a
    run, in a log nobody is watching.
    """
    import py_compile
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    targets = [root / "run.py", *sorted((root / "scripts").glob("*.py"))]
    for path in targets:
        py_compile.compile(str(path), doraise=True)
