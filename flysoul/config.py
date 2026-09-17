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
    # Synaptic gain converting connectome weight units into postsynaptic conductance.
    # Calibrated so a Kenyon cell fires when ~3 of its ~5 claw inputs are coincidently
    # active at a typical 40-60 Hz presynaptic rate (MB coincidence-detector regime).
    syn_gain: float = 3.6
    # Background synaptic bombardment, as a fraction of the gap to threshold. Off by
    # default: measured over 200 episodes, adding it makes the agent strictly worse.
    # At 0.30 and again at 0.08 the fly collapsed onto a passive optimum - 34% parry,
    # zero hits, boss at full health - because noise makes a swing unreliable, which
    # lowers the expected value of attacking while blocking still reliably converts a
    # heavy hit into chip damage. The intuition that noise buys exploration does not
    # survive contact with a reward function that has no cost for doing nothing.
    noise_sigma_frac: float = 0.0

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
    num_retina_motion: int = 96  # Lobula cells: looming, attack telegraph, object size
    num_compass_neurons: int = 32  # Central complex heading/angle tuning (EPG)
    num_nociceptors: int = 16  # Pain / damage afferent neurons
    num_proprioceptors: int = 32  # Interoceptive afferents: stamina, action lock, own health
    num_vigor_gate: int = 16  # Inhibitory state population damping costly actions when exhausted
    num_approach_brake: int = 14  # Inhibitory pool terminating approach once the target is reached

    # Interneuron / Learning populations
    num_kenyon_cells: int = 1024  # Mushroom body Kenyon cells (sparse memory representations)
    num_mbon_per_compartment: int = 6  # MBONs per MB compartment (one compartment per action)
    num_central_complex: int = 512  # Central complex steering & navigation loop
    num_apl: int = 8  # APL: GABAergic MB-wide feedback inhibition enforcing sparse KC coding
    num_cx_inhibitory: int = 96  # Local GABAergic interneurons stabilising the CX loop
    num_ring_inhibitory: int = 64  # Delta7-like inhibitory ring neurons in the CX
    num_dn_inhibitory_per_pool: int = 6  # Premotor cross-inhibition pool per action channel

    # Dopaminergic populations
    num_ppl1_dopamine: int = 8  # PPL101 aversive / punishment dopamine neurons
    num_pam_dopamine: int = 8  # PAM rewarding dopamine neurons

    # Descending motor neurons (DNs). One pool per action channel; the winner-take-all
    # between them is resolved by premotor cross-inhibition, not by the decoder.
    num_dn_per_pool: int = 14

    # Input normalisation: every neuron is scaled to the same total excitatory fan-in,
    # with inhibition expressed as a fraction of it.
    exc_fan_in_target: float = 10.0
    kc_inhibition_ratio: float = 1.35  # APL feedback: enforces a sparse KC code
    dn_inhibition_ratio: float = 1.10  # Premotor cross-inhibition: action selection

    # Synaptic plasticity
    learning_rate: float = 0.08  # Dopamine-gated STDP rate
    eligibility_decay: float = 0.92  # Per-step decay of the synaptic eligibility trace
    reward_window_steps: int = 12  # Steps an efference copy stays credit-eligible
    aversive_pulse_ms: float = 200.0  # PPL1 burst duration on damage
    reward_pulse_ms: float = 200.0  # PAM burst duration on boss hit
