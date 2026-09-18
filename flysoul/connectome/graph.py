"""MaleCNS v1.0 graph loader and bio-circuit builder for FlySoul.

Provides topology mapping for MaleCNS v1.0 datasets and generates biophysically
calibrated subcircuits (Central Complex, Mushroom Body, Optic Lobe, Descending Neurons)
with realistic small-world, modular connectivity.

The circuit is **sign-constrained** in the manner of connectome-derived LIF models
(Shiu et al. 2024): every neuron is assigned a neurotransmitter, cholinergic cells
excite all of their targets and GABA/glutamatergic cells inhibit all of theirs
(Dale's law). Without inhibition the descending pathways cannot compete and the
network saturates into a single reflex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from flysoul.config import CircuitConfig

# Action channels. One mushroom body compartment, one descending neuron pool and one
# premotor inhibitory pool exist per channel. The decoder reads these pools out; it
# does not decide between them.
ACTION_CHANNELS: Tuple[str, ...] = (
    "advance",
    "strafe_left",
    "strafe_right",
    "retreat",
    "roll",
    "attack_light",
    "attack_heavy",
    "parry",
)


@dataclass
class CircuitTopology:
    """Complete CSR graph and cell indexing for the fly nervous system."""
    num_neurons: int
    ptr: np.ndarray
    post: np.ndarray
    weight: np.ndarray

    # Index partitions for anatomical populations
    retina_indices: np.ndarray
    motion_indices: np.ndarray
    compass_indices: np.ndarray
    nociceptor_indices: np.ndarray
    proprioceptor_indices: np.ndarray
    kenyon_indices: np.ndarray
    mbon_indices: np.ndarray
    central_complex_indices: np.ndarray
    ring_inhibitory_indices: np.ndarray
    cx_inhibitory_indices: np.ndarray
    apl_indices: np.ndarray
    vigor_gate_indices: np.ndarray
    approach_brake_indices: np.ndarray
    ppl1_indices: np.ndarray
    pam_indices: np.ndarray

    # Functional sub-pools inside the sensory populations
    sensory_subpools: Dict[str, np.ndarray] = field(default_factory=dict)

    # Descending motor neuron indices by action channel
    motor_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    premotor_inhibitory_indices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Mushroom body compartments: one per action channel
    mbon_compartments: Dict[str, np.ndarray] = field(default_factory=dict)
    ppl1_compartments: Dict[str, np.ndarray] = field(default_factory=dict)
    pam_compartments: Dict[str, np.ndarray] = field(default_factory=dict)

    # Plastic synapse mask for KC -> MBON connections
    plastic_synapse_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Action channel index per synapse (-1 for non-plastic synapses)
    plastic_channel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    # Per-neuron sign (+1 cholinergic / -1 GABA-glutamatergic)
    neuron_sign: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    # 3D stereotaxic coordinates [N, 3] in microns (MaleCNS space)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    # Anatomical region labels per neuron
    labels: List[str] = field(default_factory=list)

    @property
    def action_channels(self) -> Tuple[str, ...]:
        return ACTION_CHANNELS


def build_fly_circuit(config: CircuitConfig | None = None, seed: int = 42) -> CircuitTopology:
    """Build a biophysically calibrated subcircuit reflecting Drosophila CNS anatomy.

    Architecture follows connectomic pathways:
      1. Retina / Optic Lobe -> Central Complex (CX steering & heading)
      2. Central Complex + Nociceptors -> Mushroom Body (Kenyon Cells sparse coding)
      3. Kenyon Cells -> compartmentalised MBONs (modulated by PPL1/PAM dopamine)
      4. Central Complex + MBONs -> Descending Neurons (DNs motor execution)
      5. Descending Neurons -> premotor cross-inhibition (action selection)
    """
    cfg = config or CircuitConfig()
    rng = np.random.default_rng(seed)

    # 1. Allocate contiguous blocks of neuron indices
    idx = 0

    def alloc(count: int) -> np.ndarray:
        nonlocal idx
        arr = np.arange(idx, idx + count, dtype=np.int32)
        idx += count
        return arr

    retina_idx = alloc(cfg.num_retina_brightness)
    # Lobula motion cells split by function: half report optic-flow looming, half are
    # time-tuned cells that tile the boss attack animation (the telegraph).
    _mt = max(1, cfg.num_retina_motion // 3)
    motion_loom_idx = alloc(_mt)
    motion_telegraph_idx = alloc(_mt)
    # Object-size cells: monotonic in retinal subtense, i.e. in how large the boss
    # looks. They are what tells the circuit the approach is finished.
    motion_size_idx = alloc(cfg.num_retina_motion - 2 * _mt)
    # Attack-pattern cells (see CircuitConfig.num_retina_pattern); may be empty.
    motion_pattern_idx = alloc(cfg.num_retina_pattern)
    motion_idx = np.concatenate(
        [motion_loom_idx, motion_telegraph_idx, motion_size_idx, motion_pattern_idx]
    ).astype(np.int32)
    compass_idx = alloc(cfg.num_compass_neurons)
    ring_inh_idx = alloc(cfg.num_ring_inhibitory)
    nociceptor_idx = alloc(cfg.num_nociceptors)
    # Interoceptive afferents, split by what they report.
    _pq = max(1, cfg.num_proprioceptors // 4)
    proprio_stamina_low_idx = alloc(_pq)
    proprio_stamina_high_idx = alloc(_pq)
    proprio_health_low_idx = alloc(_pq)
    proprio_action_lock_idx = alloc(cfg.num_proprioceptors - 3 * _pq)
    proprio_idx = np.concatenate([
        proprio_stamina_low_idx, proprio_stamina_high_idx,
        proprio_health_low_idx, proprio_action_lock_idx,
    ]).astype(np.int32)
    vigor_gate_idx = alloc(cfg.num_vigor_gate)
    approach_brake_idx = alloc(cfg.num_approach_brake)
    cx_idx = alloc(cfg.num_central_complex)
    # Local GABAergic interneurons giving the central complex activity-dependent
    # feedback inhibition. Without them the recurrent CX loop is bistable: it is
    # either silent or pinned at the maximum rate, and no gain setting sits between.
    cx_inh_idx = alloc(cfg.num_cx_inhibitory)
    kenyon_idx = alloc(cfg.num_kenyon_cells)
    apl_idx = alloc(cfg.num_apl)

    # Mushroom body compartments: one per action channel, each with its own DANs.
    # This is what makes dopaminergic credit assignment action-specific.
    mbon_compartments: Dict[str, np.ndarray] = {}
    ppl1_compartments: Dict[str, np.ndarray] = {}
    pam_compartments: Dict[str, np.ndarray] = {}
    for channel in ACTION_CHANNELS:
        mbon_compartments[channel] = alloc(cfg.num_mbon_per_compartment)
    mbon_idx = np.concatenate([mbon_compartments[c] for c in ACTION_CHANNELS]).astype(np.int32)

    ppl1_per = max(1, cfg.num_ppl1_dopamine // len(ACTION_CHANNELS))
    pam_per = max(1, cfg.num_pam_dopamine // len(ACTION_CHANNELS))
    for channel in ACTION_CHANNELS:
        ppl1_compartments[channel] = alloc(ppl1_per)
    for channel in ACTION_CHANNELS:
        pam_compartments[channel] = alloc(pam_per)
    ppl1_idx = np.concatenate([ppl1_compartments[c] for c in ACTION_CHANNELS]).astype(np.int32)
    pam_idx = np.concatenate([pam_compartments[c] for c in ACTION_CHANNELS]).astype(np.int32)

    # Descending neurons and their premotor inhibitory partners
    motor_indices: Dict[str, np.ndarray] = {}
    premotor_inh: Dict[str, np.ndarray] = {}
    for channel in ACTION_CHANNELS:
        motor_indices[channel] = alloc(cfg.num_dn_per_pool)
    for channel in ACTION_CHANNELS:
        premotor_inh[channel] = alloc(cfg.num_dn_inhibitory_per_pool)

    num_total = idx

    # 2. Assign neurotransmitter sign per neuron (Dale's law)
    neuron_sign = np.ones(num_total, dtype=np.int8)
    inhibitory_pools = [ring_inh_idx, apl_idx, vigor_gate_idx, cx_inh_idx, approach_brake_idx]
    for pool in premotor_inh.values():
        inhibitory_pools.append(pool)
    # Half of every MB compartment is GABAergic (cf. MBON-gamma1pedc), providing the
    # learned "suppress the alternatives" signal.
    mbon_exc: Dict[str, np.ndarray] = {}
    mbon_inh: Dict[str, np.ndarray] = {}
    for channel, pool in mbon_compartments.items():
        split = len(pool) // 2
        mbon_exc[channel] = pool[:split] if split else pool
        mbon_inh[channel] = pool[split:] if split else pool[:0]
        inhibitory_pools.append(mbon_inh[channel])
    for pool in inhibitory_pools:
        neuron_sign[pool] = -1

    adj: List[List[int]] = [[] for _ in range(num_total)]
    weights: List[List[float]] = [[] for _ in range(num_total)]
    is_plastic: List[List[bool]] = [[] for _ in range(num_total)]
    channel_of: List[List[int]] = [[] for _ in range(num_total)]

    def connect(
        pre_group: np.ndarray,
        post_group: np.ndarray,
        p_conn: float,
        mean_w: float,
        std_w: float,
        plastic: bool = False,
        channel: int = -1,
    ):
        """Wire two populations. The sign always comes from the presynaptic neuron."""
        if len(pre_group) == 0 or len(post_group) == 0:
            return
        for pre in pre_group:
            sign = float(neuron_sign[pre])
            targets = post_group[rng.random(len(post_group)) < p_conn]
            for post in targets:
                if pre == post:
                    continue
                w = max(0.25, rng.normal(mean_w, std_w))
                adj[pre].append(int(post))
                weights[pre].append(sign * float(w))
                is_plastic[pre].append(plastic)
                channel_of[pre].append(channel)

    # ---------------------------------------------------------------- pathways

    # Visual inputs -> Central Complex (visual orientation and navigation)
    connect(retina_idx, cx_idx, p_conn=0.15, mean_w=1.8, std_w=0.4)
    connect(motion_idx, cx_idx, p_conn=0.20, mean_w=2.0, std_w=0.5)

    # Compass neurons (EPG ring attractors) with Delta7-like global inhibition. The
    # inhibitory ring is what makes the compass a bump attractor instead of a blur.
    connect(compass_idx, cx_idx, p_conn=0.30, mean_w=2.2, std_w=0.6)
    connect(cx_idx, compass_idx, p_conn=0.25, mean_w=1.8, std_w=0.4)
    connect(compass_idx, ring_inh_idx, p_conn=0.35, mean_w=1.6, std_w=0.3)
    connect(ring_inh_idx, compass_idx, p_conn=0.40, mean_w=1.4, std_w=0.3)
    connect(ring_inh_idx, cx_idx, p_conn=0.10, mean_w=1.0, std_w=0.2)
    connect(cx_idx, cx_idx, p_conn=0.03, mean_w=1.5, std_w=0.3)
    connect(cx_idx, cx_inh_idx, p_conn=0.25, mean_w=1.8, std_w=0.3)
    connect(cx_inh_idx, cx_idx, p_conn=0.35, mean_w=1.8, std_w=0.3)

    # Proprioceptive / interoceptive state (stamina, action lock, own health)
    connect(proprio_idx, cx_idx, p_conn=0.12, mean_w=1.6, std_w=0.3)

    # Central Complex & Sensory -> Kenyon Cells (sparse random projection in the calyx).
    # Biological fly: ~5-7 claws per KC drawn from a large input pool.
    kc_input_pool = np.concatenate([retina_idx, compass_idx, motion_idx, proprio_idx])
    for kc in kenyon_idx:
        sampled = rng.choice(kc_input_pool, size=rng.integers(5, 8), replace=False)
        for pre in sampled:
            adj[pre].append(int(kc))
            weights[pre].append(float(neuron_sign[pre]) * float(rng.normal(2.5, 0.4)))
            is_plastic[pre].append(False)
            channel_of[pre].append(-1)

    # APL: single large GABAergic neuron giving the mushroom body all-to-all feedback
    # inhibition. This is what keeps the KC code sparse instead of letting it saturate.
    connect(kenyon_idx, apl_idx, p_conn=0.04, mean_w=1.0, std_w=0.2)
    connect(apl_idx, kenyon_idx, p_conn=0.85, mean_w=1.3, std_w=0.25)

    # Kenyon Cells -> compartmentalised MBONs. PLASTIC: these are the synapses that
    # dopamine modulates, and the compartment decides which action gets the credit.
    for ch_id, channel in enumerate(ACTION_CHANNELS):
        connect(
            kenyon_idx,
            mbon_compartments[channel],
            p_conn=0.35,
            mean_w=1.2,
            std_w=0.3,
            plastic=True,
            channel=ch_id,
        )

    # Dopaminergic neurons innervate their own compartment (they gate plasticity there,
    # and also bias its excitability while the burst lasts).
    for channel in ACTION_CHANNELS:
        connect(nociceptor_idx, ppl1_compartments[channel], p_conn=0.80, mean_w=4.0, std_w=0.8)
        connect(ppl1_compartments[channel], mbon_compartments[channel], 0.50, 1.0, 0.2)
        connect(pam_compartments[channel], mbon_compartments[channel], 0.50, 1.2, 0.2)

    # ------------------------------------------------------- descending neurons

    # Central Complex -> Descending Neurons. The CX is the steering system; without this
    # projection the goal-direction signal never reaches the motor output at all. It is
    # deliberately a thin projection: with 512 cells behind it, the central complex will
    # dominate the normalised input budget of every descending pool unless it is kept
    # sparse, and anything it dominates is a pathway learning cannot reach.
    for channel in ACTION_CHANNELS:
        connect(cx_idx, motor_indices[channel], p_conn=0.020, mean_w=1.05, std_w=0.25)

    # Innate visual reflexes, preserved from the original circuit:
    # looming detectors drive escape, retinal hemifields drive body-axis correction.
    connect(motion_loom_idx, motor_indices["roll"], p_conn=0.30, mean_w=2.6, std_w=0.5)
    half_retina = len(retina_idx) // 2
    connect(retina_idx[:half_retina], motor_indices["strafe_left"], 0.22, 1.9, 0.4)
    connect(retina_idx[half_retina:], motor_indices["strafe_right"], 0.22, 1.9, 0.4)
    # Pursuit: the columns looking straight ahead drive forward walking, the same
    # retinotopic logic as the hemifield-to-strafe projections above. This is the
    # small-object pursuit pathway (LC -> CX -> DNp09) and it is what gives the fly any
    # reason at all to close distance. The retina is arranged with azimuth 0 at its
    # centre, so the frontal columns straddle the midpoint.
    frontal = max(1, len(retina_idx) // 8)
    frontal_idx = retina_idx[half_retina - frontal : half_retina + frontal]
    connect(frontal_idx, motor_indices["advance"], 0.18, 1.6, 0.35)

    # Approach termination. An approach drive with nothing to switch it off is not a
    # behaviour, it is a runaway: the fly walks into the boss and keeps pressing
    # forward for the rest of the fight. Object-size cells report that the target now
    # fills the visual field and brake forward locomotion through a local inhibitory
    # pool, which is how approach terminates at contact in every real pursuit circuit.
    # It says when to stop walking, not what to do instead; that stays with the
    # mushroom body and the premotor competition.
    connect(motion_size_idx, approach_brake_idx, p_conn=0.45, mean_w=2.2, std_w=0.4)
    connect(approach_brake_idx, motor_indices["advance"], p_conn=0.60, mean_w=2.4, std_w=0.4)

    # Learned valence: each compartment's cholinergic MBONs promote their own action,
    # its GABAergic MBONs suppress every competing action.
    #
    # These are weighted to give the mushroom body real authority over the descending
    # pools - roughly half of their excitatory budget. Learning can only modify KC->MBON
    # synapses, so whatever share of a pool's drive arrives by another route is a share
    # experience cannot touch. Wired at a tenth of the budget, a compartment could double
    # its output and barely move the pool it is supposed to be steering, and the learning
    # curve comes out flat no matter how good the credit assignment is.
    for channel in ACTION_CHANNELS:
        connect(mbon_exc[channel], motor_indices[channel], p_conn=0.90, mean_w=6.5, std_w=1.0)
        for other in ACTION_CHANNELS:
            if other == channel:
                continue
            connect(mbon_inh[channel], motor_indices[other], p_conn=0.40, mean_w=4.0, std_w=0.7)

    # Premotor cross-inhibition: each DN pool recruits its inhibitory partner, which
    # silences every other pool. This is the winner-take-all that used to live as an
    # argmax with hand-written multipliers in the decoder.
    for channel in ACTION_CHANNELS:
        connect(motor_indices[channel], premotor_inh[channel], p_conn=0.55, mean_w=2.2, std_w=0.4)
        for other in ACTION_CHANNELS:
            if other == channel:
                continue
            connect(premotor_inh[channel], motor_indices[other], p_conn=0.45, mean_w=2.0, std_w=0.4)

    # Vigor gate: an interoceptive inhibitory population reporting exhaustion. Low
    # stamina damps the expensive channels the way octopaminergic state signals do.
    connect(proprio_stamina_low_idx, vigor_gate_idx, p_conn=0.60, mean_w=2.2, std_w=0.4)
    connect(proprio_action_lock_idx, vigor_gate_idx, p_conn=0.45, mean_w=2.0, std_w=0.4)
    for channel in ("roll", "attack_light", "attack_heavy"):
        connect(vigor_gate_idx, motor_indices[channel], p_conn=0.50, mean_w=2.4, std_w=0.4)

    # 3. Flatten into CSR format
    ptr = np.zeros(num_total + 1, dtype=np.int64)
    flat_post: List[int] = []
    flat_weight: List[float] = []
    flat_plastic: List[bool] = []
    flat_channel: List[int] = []

    for i in range(num_total):
        ptr[i] = len(flat_post)
        flat_post.extend(adj[i])
        flat_weight.extend(weights[i])
        flat_plastic.extend(is_plastic[i])
        flat_channel.extend(channel_of[i])
    ptr[num_total] = len(flat_post)

    post_arr = np.asarray(flat_post, dtype=np.int32)
    weight_arr = np.asarray(flat_weight, dtype=np.float32)

    # How hard each population is held in check by its inhibitory input, as a fraction of
    # its excitatory fan-in. The mushroom body and the descending pools need the most:
    # sparse coding and action selection are both inhibition-driven.
    inh_ratio = np.full(num_total, 0.45, dtype=np.float64)
    inh_ratio[kenyon_idx] = cfg.kc_inhibition_ratio
    inh_ratio[compass_idx] = 0.70
    inh_ratio[cx_idx] = 0.85
    inh_ratio[cx_inh_idx] = 0.20
    for pool in motor_indices.values():
        inh_ratio[pool] = cfg.dn_inhibition_ratio
    weight_arr = _normalise_fan_in(
        ptr, post_arr, weight_arr, num_total, cfg.exc_fan_in_target, inh_ratio
    )

    coords, labels = _build_anatomy(
        num_total,
        rng,
        retina_idx=retina_idx,
        motion_idx=motion_idx,
        compass_idx=compass_idx,
        ring_inh_idx=ring_inh_idx,
        cx_inh_idx=cx_inh_idx,
        nociceptor_idx=nociceptor_idx,
        proprio_idx=proprio_idx,
        vigor_gate_idx=vigor_gate_idx,
        approach_brake_idx=approach_brake_idx,
        cx_idx=cx_idx,
        kenyon_idx=kenyon_idx,
        apl_idx=apl_idx,
        mbon_compartments=mbon_compartments,
        ppl1_idx=ppl1_idx,
        pam_idx=pam_idx,
        motor_indices=motor_indices,
        premotor_inh=premotor_inh,
    )

    return CircuitTopology(
        num_neurons=num_total,
        ptr=ptr,
        post=post_arr,
        weight=weight_arr,
        retina_indices=retina_idx,
        motion_indices=motion_idx,
        compass_indices=compass_idx,
        nociceptor_indices=nociceptor_idx,
        proprioceptor_indices=proprio_idx,
        kenyon_indices=kenyon_idx,
        mbon_indices=mbon_idx,
        central_complex_indices=cx_idx,
        ring_inhibitory_indices=ring_inh_idx,
        cx_inhibitory_indices=cx_inh_idx,
        apl_indices=apl_idx,
        vigor_gate_indices=vigor_gate_idx,
        approach_brake_indices=approach_brake_idx,
        ppl1_indices=ppl1_idx,
        pam_indices=pam_idx,
        sensory_subpools={
            "motion_looming": motion_loom_idx,
            "motion_telegraph": motion_telegraph_idx,
            "motion_size": motion_size_idx,
            "motion_pattern": motion_pattern_idx,
            "proprio_stamina_low": proprio_stamina_low_idx,
            "proprio_stamina_high": proprio_stamina_high_idx,
            "proprio_health_low": proprio_health_low_idx,
            "proprio_action_lock": proprio_action_lock_idx,
        },
        motor_indices=motor_indices,
        premotor_inhibitory_indices=premotor_inh,
        mbon_compartments=mbon_compartments,
        ppl1_compartments=ppl1_compartments,
        pam_compartments=pam_compartments,
        plastic_synapse_mask=np.asarray(flat_plastic, dtype=bool),
        plastic_channel=np.asarray(flat_channel, dtype=np.int8),
        neuron_sign=neuron_sign,
        coords=coords,
        labels=labels,
    )


def _normalise_fan_in(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    num_neurons: int,
    exc_target: float,
    inh_ratio: np.ndarray,
) -> np.ndarray:
    """Scale each neuron's incoming weights to a common total excitatory drive.

    Fan-in varies by more than an order of magnitude across this circuit (a compass
    neuron receives from ~128 central complex cells, a Kenyon cell from five claws).
    Without normalisation the high fan-in populations saturate at their maximum rate and
    the recurrent central complex loop runs away, which collapses every situation onto
    the same descending output. Relative weights inside each neuron's input are
    preserved, so the wiring's design intent survives.
    """
    incoming_exc = np.zeros(num_neurons, dtype=np.float64)
    incoming_inh = np.zeros(num_neurons, dtype=np.float64)
    exc_edges = weight > 0
    inh_edges = weight < 0
    np.add.at(incoming_exc, post[exc_edges], weight[exc_edges])
    np.add.at(incoming_inh, post[inh_edges], -weight[inh_edges])

    scale_exc = np.ones(num_neurons, dtype=np.float64)
    np.divide(exc_target, incoming_exc, out=scale_exc, where=incoming_exc > 1e-9)
    inh_target = exc_target * inh_ratio
    scale_inh = np.ones(num_neurons, dtype=np.float64)
    np.divide(inh_target, incoming_inh, out=scale_inh, where=incoming_inh > 1e-9)

    weight = weight.astype(np.float64)
    weight[exc_edges] *= scale_exc[post[exc_edges]]
    weight[inh_edges] *= scale_inh[post[inh_edges]]
    return weight.astype(np.float32)


def _build_anatomy(num_total, rng, **pops) -> Tuple[np.ndarray, List[str]]:
    """Generate 3D stereotaxic coordinates and region labels in MaleCNS standard space."""
    coords = np.zeros((num_total, 3), dtype=np.float32)
    labels = [""] * num_total

    retina_idx = pops["retina_idx"]
    motion_idx = pops["motion_idx"]
    compass_idx = pops["compass_idx"]
    ring_inh_idx = pops["ring_inh_idx"]
    cx_inh_idx = pops["cx_inh_idx"]
    nociceptor_idx = pops["nociceptor_idx"]
    proprio_idx = pops["proprio_idx"]
    vigor_gate_idx = pops["vigor_gate_idx"]
    approach_brake_idx = pops["approach_brake_idx"]
    cx_idx = pops["cx_idx"]
    kenyon_idx = pops["kenyon_idx"]
    apl_idx = pops["apl_idx"]
    mbon_compartments = pops["mbon_compartments"]
    ppl1_idx = pops["ppl1_idx"]
    pam_idx = pops["pam_idx"]
    motor_indices = pops["motor_indices"]
    premotor_inh = pops["premotor_inh"]

    # Optic Lobes - Outer Retina / Lamina & Medulla (Left & Right Lateral Crescents)
    half_r = max(1, len(retina_idx) // 2)
    for i, idx_n in enumerate(retina_idx):
        labels[idx_n] = "Retina_OpticLobe"
        side = -1.0 if i < half_r else 1.0
        u = ((i % half_r) / max(1, half_r - 1)) * math.pi - math.pi / 2.0
        v = (i % 7) / 7.0 * math.pi - math.pi / 2.0
        r_x = 65.0 + float(rng.normal(0, 3.0))
        r_y = 80.0 + float(rng.normal(0, 3.5))
        r_z = 55.0 + float(rng.normal(0, 3.0))
        coords[idx_n] = [
            side * (135.0 + r_x * math.cos(u) * math.cos(v)),
            r_y * math.sin(u) * 0.85 + float(rng.normal(0, 3.0)),
            r_z * math.sin(v) + float(rng.normal(0, 3.0)),
        ]

    # Motion Detectors (Lobula & Lobula Plate LPTC - Posterior Inner Optic Lobe)
    half_m = max(1, len(motion_idx) // 2)
    for i, idx_n in enumerate(motion_idx):
        labels[idx_n] = "Lobula_Motion"
        side = -1.0 if i < half_m else 1.0
        u = ((i % half_m) / max(1, half_m - 1)) * math.pi - math.pi / 2.0
        coords[idx_n] = [
            side * (105.0 + 35.0 * math.cos(u) + float(rng.normal(0, 4.0))),
            60.0 * math.sin(u) + float(rng.normal(-10, 4.0)),
            float(rng.normal(15, 8.0)),
        ]

    # Central Complex - Ellipsoid Body (EB) Ring Attractor (Midline Torus)
    for i, idx_n in enumerate(compass_idx):
        labels[idx_n] = "Compass_EB"
        theta = i / max(1, len(compass_idx)) * 2.0 * math.pi
        r_ring = 28.0 + float(rng.normal(0, 1.5))
        coords[idx_n] = [
            r_ring * math.cos(theta),
            r_ring * math.sin(theta) + 5.0,
            float(rng.normal(8.0, 2.5)),
        ]

    # Delta7-like inhibitory ring neurons, just outside the EB torus
    for i, idx_n in enumerate(ring_inh_idx):
        labels[idx_n] = "Compass_Ring_Inhibitory"
        theta = i / max(1, len(ring_inh_idx)) * 2.0 * math.pi
        r_ring = 38.0 + float(rng.normal(0, 2.0))
        coords[idx_n] = [
            r_ring * math.cos(theta),
            r_ring * math.sin(theta) + 5.0,
            float(rng.normal(6.0, 2.5)),
        ]

    # Central Complex - Fan-shaped Body (FB) and Protocerebral Bridge (PB)
    half_cx = max(1, len(cx_idx) // 2)
    for i, idx_n in enumerate(cx_idx):
        labels[idx_n] = "Central_Complex"
        if i < half_cx:
            t = (i / max(1, half_cx - 1)) * 2.0 - 1.0
            coords[idx_n] = [
                t * 38.0 + float(rng.normal(0, 2.0)),
                24.0 - 10.0 * (t ** 2) + float(rng.normal(0, 3.0)),
                22.0 + float(rng.normal(0, 3.0)),
            ]
        else:
            t = ((i - half_cx) / max(1, half_cx - 1)) * 2.0 - 1.0
            coords[idx_n] = [
                t * 52.0 + float(rng.normal(0, 2.0)),
                40.0 - 8.0 * (t ** 2) + float(rng.normal(0, 2.5)),
                36.0 + float(rng.normal(0, 2.5)),
            ]

    for i, idx_n in enumerate(cx_inh_idx):
        labels[idx_n] = "Central_Complex_Inhibitory"
        t_pos = (i / max(1, len(cx_inh_idx) - 1)) * 2.0 - 1.0
        coords[idx_n] = [
            t_pos * 44.0 + float(rng.normal(0, 2.5)),
            32.0 - 6.0 * (t_pos ** 2) + float(rng.normal(0, 2.5)),
            29.0 + float(rng.normal(0, 2.5)),
        ]

    # Kenyon Cells - Mushroom Body (Calyx and Bifurcating Lobes)
    half_k = max(1, len(kenyon_idx) // 2)
    for i, idx_n in enumerate(kenyon_idx):
        labels[idx_n] = "Kenyon_Cell"
        side = -1.0 if i < half_k else 1.0
        sub_i = i % half_k
        if sub_i < half_k * 0.4:
            coords[idx_n] = [
                side * (62.0 + float(rng.normal(0, 10.0))),
                float(rng.normal(52.0, 10.0)),
                float(rng.normal(38.0, 8.0)),
            ]
        elif sub_i < half_k * 0.7:
            t = (sub_i - half_k * 0.4) / max(1e-6, half_k * 0.3)
            coords[idx_n] = [
                side * (48.0 - t * 15.0 + float(rng.normal(0, 3.0))),
                20.0 + t * 25.0 + float(rng.normal(0, 4.0)),
                25.0 + t * 30.0 + float(rng.normal(0, 3.5)),
            ]
        else:
            t = (sub_i - half_k * 0.7) / max(1e-6, half_k * 0.3)
            coords[idx_n] = [
                side * (48.0 - t * 35.0 + float(rng.normal(0, 3.0))),
                15.0 - t * 10.0 + float(rng.normal(0, 3.5)),
                12.0 - t * 8.0 + float(rng.normal(0, 3.0)),
            ]

    # APL: one wide-field inhibitory neuron per hemisphere, sitting across the lobes
    for i, idx_n in enumerate(apl_idx):
        labels[idx_n] = "APL_Inhibitory"
        side = -1.0 if i % 2 == 0 else 1.0
        coords[idx_n] = [
            side * (55.0 + float(rng.normal(0, 5.0))),
            float(rng.normal(34.0, 8.0)),
            float(rng.normal(26.0, 6.0)),
        ]

    # MBONs, arranged compartment by compartment along the MB lobes
    for c_i, (channel, pool) in enumerate(mbon_compartments.items()):
        theta = c_i / max(1, len(mbon_compartments)) * 2.0 * math.pi
        for idx_n in pool:
            labels[idx_n] = f"MBON_{channel}"
            coords[idx_n] = [
                22.0 * math.cos(theta) + float(rng.normal(0, 3.0)),
                12.0 + 10.0 * math.sin(theta) + float(rng.normal(0, 3.0)),
                float(rng.normal(10.0, 4.0)),
            ]

    # Dopaminergic Neurons - PPL1 (Aversive / Punishment) & PAM (Reward)
    for idx_n in ppl1_idx:
        labels[idx_n] = "Dopamine_PPL1"
        side = -1.0 if rng.random() < 0.5 else 1.0
        coords[idx_n] = [
            side * (42.0 + float(rng.normal(0, 6.0))),
            float(rng.normal(-18.0, 6.0)),
            float(rng.normal(45.0, 5.0)),
        ]
    for idx_n in pam_idx:
        labels[idx_n] = "Dopamine_PAM"
        coords[idx_n] = [
            float(rng.normal(0, 14.0)),
            float(rng.normal(16.0, 5.0)),
            float(rng.normal(-8.0, 5.0)),
        ]

    # Nociceptors (Sensory Pain Reflex)
    for idx_n in nociceptor_idx:
        labels[idx_n] = "Nociceptor"
        coords[idx_n] = [
            float(rng.normal(0, 35.0)),
            float(rng.normal(-45.0, 8.0)),
            float(rng.normal(32.0, 8.0)),
        ]

    # Proprioceptive / interoceptive afferents (stamina, action lock, own health)
    for idx_n in proprio_idx:
        labels[idx_n] = "Proprioceptor"
        coords[idx_n] = [
            float(rng.normal(0, 30.0)),
            float(rng.normal(-58.0, 7.0)),
            float(rng.normal(14.0, 8.0)),
        ]

    for idx_n in approach_brake_idx:
        labels[idx_n] = "Approach_Brake_Inhibitory"
        coords[idx_n] = [
            float(rng.normal(0, 16.0)),
            float(rng.normal(-34.0, 6.0)),
            float(rng.normal(-6.0, 6.0)),
        ]

    for idx_n in vigor_gate_idx:
        labels[idx_n] = "Vigor_Gate_Inhibitory"
        coords[idx_n] = [
            float(rng.normal(0, 18.0)),
            float(rng.normal(-40.0, 6.0)),
            float(rng.normal(-16.0, 6.0)),
        ]

    # Descending Motor Neurons (Ventral Nerve Cord - tract to T1-T3 leg neuropils)
    for c_i, (act_name, pool) in enumerate(motor_indices.items()):
        x_center = (c_i - (len(motor_indices) - 1) / 2.0) * 13.0
        for idx_n in pool:
            labels[idx_n] = f"Motor_{act_name}"
            coords[idx_n] = [
                x_center + float(rng.normal(0, 4.0)),
                float(rng.normal(-25.0, 8.0)),
                -30.0 - float(rng.uniform(0, 110.0)),
            ]

    for c_i, (act_name, pool) in enumerate(premotor_inh.items()):
        x_center = (c_i - (len(premotor_inh) - 1) / 2.0) * 13.0
        for idx_n in pool:
            labels[idx_n] = f"Premotor_Inhibitory_{act_name}"
            coords[idx_n] = [
                x_center + float(rng.normal(0, 4.0)),
                float(rng.normal(-42.0, 6.0)),
                -25.0 - float(rng.uniform(0, 40.0)),
            ]

    return coords, labels
