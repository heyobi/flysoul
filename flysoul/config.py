"""Configuration and biophysical constants for FlySoul."""

import math
from dataclasses import dataclass
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


@dataclass
class BioPhysicsConfig:
    """Biophysical simulation constants matching MaleCNS reference LIF parameters."""
    dt: float = 0.1  # Integration timestep (ms)
    tau_m: float = 20.0  # Membrane time constant (ms)
    tau_s: float = 5.0  # Synaptic time constant (ms)
    v_rest: float = -52.0  # Resting membrane potential (mV)
    v_threshold: float = -45.0  # Action potential threshold (mV)
    v_reset: float = -52.0  # Post-spike reset potential (mV)
    t_refractory: float = 2.2  # Absolute refractory period (ms)
    synapse_delay: float = 1.8  # Axonal conduction delay (ms)

    @property
    def a_v(self) -> float:
        return math.exp(-self.dt / self.tau_m)

    @property
    def a_g(self) -> float:
        return math.exp(-self.dt / self.tau_s)

    @property
    def coupling(self) -> float:
        return (self.a_v - self.a_g) / 3.0

    @property
    def refractory_steps(self) -> int:
        return int(round(self.t_refractory / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round(self.synapse_delay / self.dt))


@dataclass
class CircuitConfig:
    """Neuron allocation for the circuit sub-networks."""
    # Sensory neuron counts
    num_retina_brightness: int = 128  # Proximal visual sector intensity
    num_retina_motion: int = 64  # Optical flow / looming motion detectors
    num_compass_neurons: int = 32  # Central complex heading/angle tuning (EPG)
    num_nociceptors: int = 16  # Pain / damage afferent neurons

    # Interneuron / Learning populations
    num_kenyon_cells: int = 1024  # Mushroom body Kenyon cells (sparse memory representations)
    num_mbon_neurons: int = 32  # Mushroom body output neurons
    num_central_complex: int = 512  # Central complex steering & navigation loop

    # Dopaminergic populations
    num_ppl1_dopamine: int = 8  # PPL101 aversive / punishment dopamine neurons
    num_pam_dopamine: int = 8  # PAM rewarding dopamine neurons

    # Descending motor neurons (DNs)
    num_dn_turn_left: int = 12  # DNp20 left
    num_dn_turn_right: int = 12  # DNp20 right
    num_dn_attack_light: int = 16  # DNpe017 anterior thrust
    num_dn_attack_heavy: int = 12  # DNpe heavy burst
    num_dn_dodge_roll: int = 16  # DNa02 rapid escape jump/roll
    num_dn_block_parry: int = 12  # DNb defensive stance
    num_dn_step_back: int = 12  # Retraction backward step

    # Synaptic plasticity
    learning_rate: float = 0.05  # Dopamine-gated STDP rate
    aversive_pulse_ms: float = 200.0  # PPL1 burst duration on damage
    reward_pulse_ms: float = 200.0  # PAM burst duration on boss hit
