"""Homeostatic calibration of the connectome's operating point.

A connectome gives connectivity, not synaptic strength. Scaling every neuron to the
same total fan-in does not work here: a central complex cell draws from ~87 presynaptic
partners of which only two or three are ever inside the visual bump at once, so its
realised input is a small fraction of its nominal fan-in and it never reaches threshold,
while a Kenyon cell with five claws saturates.

This module measures what each population actually does on a batch of representative
combat states and rescales that population's incoming excitatory and inhibitory weights
until it sits at a target rate. It is the developmental homeostasis step that connectome
models normally perform by hand, done automatically and reproducibly.

Nothing here is task knowledge: the targets are firing rates and a sparsity level taken
from Drosophila physiology, not preferences over actions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from flysoul.config import CHECKPOINT_DIR, BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import CircuitTopology
from flysoul.env.obs import CombatState

# Target firing rates in Hz. Expressing these per simulation step made the operating
# point depend on how long a step happened to be: the circuit was calibrated against a
# 16.67 ms frame and then run against 100 ms game steps, six times over-driven.
DEFAULT_RATE_TARGETS_HZ: Dict[str, float] = {
    'central_complex': 18.0,
    'ring_inhibitory': 27.0,
    'cx_inhibitory': 30.0,
    'apl': 54.0,
    'mbon': 33.0,
    'premotor': 24.0,
    'motor': 33.0,
}
# vigor_gate and the nociceptors are state-gated: they are silent in a healthy, rested
# fly by design, so they are reported but not driven towards a target rate.

# Fraction of Kenyon cells firing at least once within a step.
DEFAULT_KC_SPARSITY = 0.10


@dataclass
class CalibrationReport:
    iterations: int
    rates: Dict[str, float]
    kc_sparsity: float
    converged: bool
    step_s: float = 0.1

    def format(self) -> str:
        parts = [f"KC sparsity {self.kc_sparsity * 100:.1f}%"]
        parts += [
            f"{k} {v / max(1e-9, self.step_s):.0f}Hz" for k, v in sorted(self.rates.items())
        ]
        status = "converged" if self.converged else "did not converge"
        return f"[{self.iterations} iters, {status}] " + ", ".join(parts)


def default_probe_states() -> List[CombatState]:
    """A spread of combat situations covering the states the fly must tell apart."""
    return [
        CombatState(distance=12.0, angle=0.0, player_sp=1.0),
        CombatState(distance=6.0, angle=0.6, player_sp=1.0),
        CombatState(distance=2.5, angle=0.0, player_sp=1.0),
        CombatState(distance=2.5, angle=0.0, boss_attacking=True, boss_anim_time=0.25),
        CombatState(distance=2.5, angle=0.0, boss_attacking=True, boss_anim_time=0.80),
        CombatState(distance=3.5, angle=-1.2, boss_attacking=True, boss_anim_time=0.50),
        CombatState(distance=2.5, angle=0.0, boss_staggered=True),
        CombatState(distance=2.5, angle=0.0, player_sp=0.05),
        CombatState(distance=4.0, angle=2.8, player_sp=0.6, player_hp=0.4),
    ]


def _population_groups(topology: CircuitTopology) -> Dict[str, np.ndarray]:
    groups: Dict[str, np.ndarray] = {
        "central_complex": topology.central_complex_indices,
        "ring_inhibitory": topology.ring_inhibitory_indices,
        "cx_inhibitory": topology.cx_inhibitory_indices,
        "apl": topology.apl_indices,
        "mbon": topology.mbon_indices,
        "vigor_gate": topology.vigor_gate_indices,
        "approach_brake": topology.approach_brake_indices,
    }
    motor = [idx for idx in topology.motor_indices.values() if len(idx)]
    if motor:
        groups["motor"] = np.concatenate(motor).astype(np.int32)
    premotor = [idx for idx in topology.premotor_inhibitory_indices.values() if len(idx)]
    if premotor:
        groups["premotor"] = np.concatenate(premotor).astype(np.int32)
    return groups


def _scale_incoming(
    topology: CircuitTopology,
    targets: np.ndarray,
    factor: float,
    excitatory: bool,
):
    """Multiply the incoming excitatory (or inhibitory) weights of `targets` by `factor`."""
    if factor == 1.0 or len(targets) == 0:
        return
    is_target = np.zeros(topology.num_neurons, dtype=bool)
    is_target[targets] = True
    sel = is_target[topology.post]
    sel &= (topology.weight > 0) if excitatory else (topology.weight < 0)
    topology.weight[sel] *= np.float32(factor)


def calibrate_circuit(
    topology: CircuitTopology,
    biophysics: BioPhysicsConfig | None = None,
    encoder_factory: Callable[[CircuitTopology, BioPhysicsConfig], object] | None = None,
    probe_states: Sequence[CombatState] | None = None,
    rate_targets_hz: Dict[str, float] | None = None,
    kc_sparsity_target: float = DEFAULT_KC_SPARSITY,
    step_ms: float = 100.0,
    max_iterations: int = 40,
    settle_steps: int = 22,
    measure_steps: int = 8,
    tolerance: float = 0.28,
    progress: Callable[[int, CalibrationReport], None] | None = None,
) -> CalibrationReport:
    """Rescale population input gains until the circuit sits at physiological rates.

    Mutates ``topology.weight`` in place and returns what it settled on.
    """
    bio = biophysics or BioPhysicsConfig()
    step_s = step_ms / 1000.0
    targets = {k: v * step_s for k, v in (rate_targets_hz or DEFAULT_RATE_TARGETS_HZ).items()}
    states = list(probe_states or default_probe_states())
    groups = _population_groups(topology)

    if encoder_factory is None:
        from flysoul.sensory.encoder import SensoryEncoder

        def encoder_factory(topo, cfg):  # type: ignore[misc]
            return SensoryEncoder(topo, cfg)

    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, bio)
    # Calibration must be reproducible, so the background noise is seeded here even
    # though it is free-running during play.
    engine.seed(20260917)
    encoder = encoder_factory(topology, bio)

    report = CalibrationReport(0, {}, 0.0, False, step_s)
    for iteration in range(1, max_iterations + 1):
        per_state = {name: [] for name in groups}
        per_state_kc = []

        for state in states:
            engine.reset_state()
            encoder.reset(state.player_hp, state.distance)
            for _ in range(settle_steps):
                engine.step(encoder.encode(state), step_ms)
            sums = {name: 0.0 for name in groups}
            kc_active = 0.0
            for _ in range(measure_steps):
                counts = engine.step(encoder.encode(state), step_ms)
                for name, idx in groups.items():
                    sums[name] += float(np.mean(counts[idx])) if len(idx) else 0.0
                kc_active += float((counts[topology.kenyon_indices] > 0).mean())
            for name in groups:
                per_state[name].append(sums[name] / measure_steps)
            per_state_kc.append(kc_active / measure_steps)

        # Blend the mean with the worst state. Optimising the mean alone lets a
        # population run hot in melee and stone dead at spawn distance and still report
        # itself on target; measured that way, every step on which the motor system fell
        # silent was a step spent standing at the distance the episode starts from.
        rates = {
            name: 0.5 * float(np.mean(v)) + 0.5 * float(np.min(v))
            for name, v in per_state.items()
        }
        sparsity = 0.5 * float(np.mean(per_state_kc)) + 0.5 * float(np.min(per_state_kc))
        report = CalibrationReport(iteration, rates, sparsity, False, step_s)
        if progress is not None:
            progress(iteration, report)

        # Are we there yet?
        errors = [
            abs(rates[name] - targets[name]) / max(1e-6, targets[name])
            for name in groups
            if name in targets
        ]
        errors.append(abs(sparsity - kc_sparsity_target) / kc_sparsity_target)
        if max(errors) <= tolerance:
            report.converged = True
            break

        # Kenyon cells are tuned on population sparsity, everything else on rate.
        _apply_correction(
            topology,
            topology.kenyon_indices,
            measured=sparsity,
            target=kc_sparsity_target,
        )
        for name, idx in groups.items():
            if name not in targets:
                continue
            _apply_correction(topology, idx, measured=rates[name], target=targets[name])

        engine.weight[:] = topology.weight

    return report


def _apply_correction(
    topology: CircuitTopology,
    targets: np.ndarray,
    measured: float,
    target: float,
    damping: float = 0.35,
    clamp: float = 1.28,
):
    """Nudge a population's excitatory input gain towards its target activity."""
    if len(targets) == 0:
        return
    if measured <= 1e-9:
        factor = clamp  # Silent population: open the gain up.
    else:
        factor = float((target / measured) ** damping)
        factor = float(np.clip(factor, 1.0 / clamp, clamp))
    _scale_incoming(topology, targets, factor, excitatory=True)


def _topology_fingerprint(topology: CircuitTopology) -> str:
    """Fingerprint the wiring itself, not just the configuration that produced it.

    Most of the circuit's structure lives in build_fly_circuit as connection
    probabilities and weights, not in CircuitConfig. Keying the cache on the config
    alone means rewiring a pathway silently reloads the calibration for the *old*
    circuit, and the new one runs at an operating point that was never measured for it.
    """
    h = hashlib.sha1()
    h.update(topology.ptr.tobytes())
    h.update(topology.post.tobytes())
    h.update(np.round(topology.weight.astype(np.float64), 5).tobytes())
    h.update(topology.neuron_sign.tobytes())
    return h.hexdigest()[:16]


def _cache_key(circuit: CircuitConfig, bio: BioPhysicsConfig, seed: int, wiring: str) -> str:
    payload = json.dumps(
        {
            "wiring": wiring,
            "circuit": asdict(circuit),
            "syn_gain": bio.syn_gain,
            "noise_sigma_frac": bio.noise_sigma_frac,
            "v_gap": bio.v_threshold - bio.v_rest,
            "tau_m": bio.tau_m,
            "tau_s": bio.tau_s,
            "adaptation": [bio.adaptation_jump_mv, bio.adaptation_tau_ms],
            "seed": seed,
            "rate_targets_hz": DEFAULT_RATE_TARGETS_HZ,
            "step_ms": 100.0,
            "kc_sparsity": DEFAULT_KC_SPARSITY,
            "version": 6,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def calibrate_or_load(
    topology: CircuitTopology,
    circuit: CircuitConfig,
    biophysics: BioPhysicsConfig,
    seed: int,
    cache_dir: Path | None = None,
    force: bool = False,
    progress: Callable[[int, CalibrationReport], None] | None = None,
) -> tuple[CalibrationReport, bool]:
    """Calibrate the circuit, reusing a cached result for an identical configuration.

    Returns ``(report, loaded_from_cache)``. Calibration is deterministic for a given
    configuration and seed, so caching changes nothing about the resulting circuit.
    """
    cache_dir = Path(cache_dir or CHECKPOINT_DIR)
    key = _cache_key(circuit, biophysics, seed, _topology_fingerprint(topology))
    path = cache_dir / f"calibration_{key}.npz"

    if not force and path.exists():
        try:
            blob = np.load(path, allow_pickle=False)
            weights = blob["weight"]
            if weights.shape == topology.weight.shape:
                topology.weight[:] = weights
                report = CalibrationReport(
                    iterations=int(blob["iterations"]),
                    rates={str(k): float(v) for k, v in zip(blob["rate_names"], blob["rate_values"])},
                    kc_sparsity=float(blob["kc_sparsity"]),
                    converged=bool(blob["converged"]),
                    step_s=float(blob["step_s"]) if "step_s" in blob else 0.1,
                )
                return report, True
        except Exception:
            pass  # A damaged cache is not worth failing over; recalibrate instead.

    report = calibrate_circuit(topology, biophysics, progress=progress)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weight=topology.weight,
            iterations=report.iterations,
            rate_names=np.array(list(report.rates.keys())),
            rate_values=np.array(list(report.rates.values()), dtype=np.float64),
            kc_sparsity=report.kc_sparsity,
            converged=report.converged,
            step_s=report.step_s,
        )
    except Exception:
        pass
    return report, False
