"""Unit tests for FlySoul connectome, encoder, decoder, and simulation components."""

import numpy as np
import pytest

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.mock_env import MockIudexEnv
from flysoul.motor.decoder import MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder


@pytest.fixture
def topology():
    config = CircuitConfig(
        num_retina_brightness=16,
        num_retina_motion=8,
        num_compass_neurons=8,
        num_nociceptors=4,
        num_kenyon_cells=64,
        num_mbon_neurons=8,
        num_central_complex=32,
        num_ppl1_dopamine=2,
        num_pam_dopamine=2,
        num_dn_turn_left=4,
        num_dn_turn_right=4,
        num_dn_attack_light=4,
        num_dn_attack_heavy=4,
        num_dn_dodge_roll=4,
        num_dn_block_parry=4,
        num_dn_step_back=4,
    )
    return build_fly_circuit(config, seed=123)


def test_topology_generation(topology):
    assert topology.num_neurons > 0
    assert len(topology.ptr) == topology.num_neurons + 1
    assert len(topology.post) == len(topology.weight)
    assert np.sum(topology.plastic_synapse_mask) > 0


def test_engine_lif_integration(topology):
    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight)

    # Drive the first 10 neurons with strong current
    drive = np.zeros(topology.num_neurons, dtype=np.float32)
    drive[:10] = 30.0

    counts = engine.step(drive, duration_ms=20.0)
    assert isinstance(counts, np.ndarray)
    assert counts.shape == (topology.num_neurons,)
    assert np.sum(counts[:10]) > 0  # Driven neurons must spike!


def test_sensory_encoder(topology):
    encoder = SensoryEncoder(topology)
    obs = {
        "player_hp": np.array([0.8], dtype=np.float32),
        "player_sp": np.array([0.9], dtype=np.float32),
        "boss_hp": np.array([0.95], dtype=np.float32),
        "boss_distance": np.array([3.2], dtype=np.float32),
        "boss_rel_angle": np.array([0.5], dtype=np.float32),
        "boss_attacking": 1,
    }
    drive = encoder.encode(obs)
    assert drive.shape == (topology.num_neurons,)
    # Retinal and compass neurons should be receiving current
    assert np.sum(drive[topology.retina_indices]) > 0
    assert np.sum(drive[topology.compass_indices]) > 0
    assert np.sum(drive[topology.motion_indices]) > 0


def test_motor_decoder(topology):
    decoder = MotorDecoder(topology, threshold=0.5)

    # Artificial spike counts driving dodge roll
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.motor_indices["dodge_roll"]] = 10

    action_id, action_name, rates = decoder.decode(spikes)
    assert action_id == 1
    assert action_name == "dodge_roll"
    assert rates["dodge_roll"] == 10.0


def test_plasticity_reinforcement(topology):
    plasticity = DopaminePlasticity(topology, learning_rate=0.1)

    initial_weights = topology.weight[topology.plastic_synapse_mask].copy()

    # Pre-synaptic KC spikes and post-synaptic MBON spikes
    spikes = np.zeros(topology.num_neurons, dtype=np.int32)
    spikes[topology.kenyon_indices[:5]] = 3
    spikes[topology.mbon_indices[:2]] = 2

    plasticity.update_traces(spikes)
    # Aversive dopamine (damage taken) -> LTD
    plasticity.apply_reinforcement(reward=-1.0)

    updated_weights = topology.weight[topology.plastic_synapse_mask]
    assert np.any(updated_weights < initial_weights)


def test_mock_iudex_env():
    env = MockIudexEnv()
    obs, info = env.reset()
    assert "player_hp" in obs
    assert "boss_hp" in obs
    assert "boss_distance" in obs

    # Test rolling
    obs, reward, term, trunc, info = env.step(1)
    assert not term
    assert obs["player_sp"][0] < 1.0  # Spent stamina

    env.close()
