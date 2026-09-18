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
    # Spike-frequency adaptation: each spike raises the cell's threshold by this much
    # and the increase decays with the time constant below. A documented property of
    # fly neurons, and in the flybench comparison of whole-brain LIF models the single
    # change that most improved behaviour (0.53 -> 0.68 on the hard tasks): cells stop
    # saturating, the brain returns to rest after a stimulus, and descending pools
    # recruit selectively instead of en masse. Zero disables it.
    #
    # Off by default here, measured: on this calibrated sub-circuit the flybench value of
    # 2 mV stops the homeostatic calibration converging, and even 0.5 mV silences the
    # approach-brake population (10 Hz -> 1 Hz) while the rest still converges. The
    # threshold gap is 7 mV, so a 2 mV jump is a 30% change per spike; the whole-brain
    # models it helped run at a different drive scale. Turning it on needs the
    # calibration targets revisited first, not a flag flip.
    adaptation_jump_mv: float = 0.0
    adaptation_tau_ms: float = 200.0

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
    # Motion-pattern cells: lobula columnar feature detectors, each tuned to one of the
    # boss's attack animations (the way LC cell types each respond to a particular
    # visual motion pattern). Measured offline on 300 fights, the outcome of an
    # action is predicted far better by 'which attack, which phase' (+0.39) than by
    # the Kenyon cell code the mushroom body actually received (+0.26), because the
    # telegraph bank tiles time since the swing began identically for all 25 attacks.
    # Zero keeps the wiring (and its learned weights) exactly as it was; 64 was turned on
    # 2026-09-18 after the offline ceiling measurement, starting learning afresh.
    num_retina_pattern: int = 64
    # Conjunction pattern cells: instead of a hashed subset per animation, one small group
    # of cells per (animation slot, phase bin), driven for any boss animation. Measured
    # offline (980 fights): the outcome ceiling of the raw 'attack id x phase' feature is
    # +0.45 against +0.39 for the Kenyon code built from hashed pattern cells, and that
    # gap is the only sizeable information loss in the encoder. 0 keeps the hashed cells.
    pattern_phase_bins: int = 0
    # >0: exact layout, one group of this many cells per (slot, phase bin).
    # 0: hashed layout - each (animation, phase bin) drives a fixed random eighth of the
    #    num_retina_pattern cells, the same population code the hashed attack cells use.
    pattern_cells_per_conjunction: int = 2
    pattern_slots: int = 32  # boss animation ids seen live are 0..30
    # Kenyon claws sample the pattern population as if it had this many cells, so a
    # larger pattern bank does not crowd the retina and compass out of the calyx.
    # 0 = plain uniform sampling (the live wiring).
    pattern_pool_share: int = 0
    # Conjunction code only while the boss is attacking (as the hashed cells do) or for
    # every animation including idle and walking.
    pattern_attacking_only: bool = True

    @property
    def pattern_cell_count(self) -> int:
        if self.pattern_phase_bins > 0 and self.pattern_cells_per_conjunction > 0:
            return self.pattern_slots * self.pattern_phase_bins * self.pattern_cells_per_conjunction
        return self.num_retina_pattern

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
    # Per-step decay of the synaptic eligibility trace (steps are 100 ms). This is the
    # window over which a KC->MBON synapse can still be blamed or thanked for what just
    # happened, and it sets how sharply timing can be learned. At 0.92 (~1.2 s) a hit
    # taken at 1.0 s into the boss's swing still credited the Kenyon cells active at
    # 0.3 s with 56% of the punishment, so an early roll and a well-timed one were
    # depressed almost alike; measured live over 365 rolls, the early ones stayed at
    # 74-83% failure and never improved although the KC code separates the phases
    # (scripts/kc_phase_code.py). Shortening it to 0.80 did not help either, because the
    # real smear was elsewhere: the trace was accumulating on every compartment every
    # step, so the blame for an early roll landed on whichever cells fired later. With
    # the tag now set only at the moment a command is issued (see
    # DopaminePlasticity.update_traces), a longer trace no longer blurs phases; it
    # just lets an outcome six steps later still find the tag (0.92**6 = 0.61).
    eligibility_decay: float = 0.92
    reward_window_steps: int = 12  # Steps an efference copy stays credit-eligible
    aversive_pulse_ms: float = 200.0  # PPL1 burst duration on damage
    reward_pulse_ms: float = 200.0  # PAM burst duration on boss hit
