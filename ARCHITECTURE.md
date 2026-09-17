# FlySoul System Architecture & AI Agent Guide

Welcome to **FlySoul**! This document provides an exhaustive overview of the system architecture, directory layout, biophysical modeling, and deployment instructions so that any AI agent or researcher can immediately understand and extend this project.

---

## 1. Scientific & Engineering Context

* **Biological Ground Truth:** Google Research & HHMI Janelia *MaleCNS v1.0* (published in *Cell*, September 2026: *"Sexual dimorphism in the complete connectome of the Drosophila male central nervous system"*). Contains **166,000+ neurons** and **125 million synapses**, annotated into 11,691 cell types with predicted neurotransmitters.
* **Target Task:** Dark Souls III tutorial boss fight (**Iudex Gundyr**) via **SoulsGym** (`amacati/SoulsGym`).
* **Why Dark Souls III?** Unlike simple toy games, Dark Souls demands strict timing, stamina conservation, telegraph recognition (windup detection), and directional evasion.

### Design rule: the circuit decides

The single hard constraint on this codebase is that **no module outside the connectome may choose an action.** The encoder states facts about the world in spikes; the decoder reports which descending pool won; the training loop moves reward around. Everything between those is the circuit.

This rule is not decoration — it is the thing that broke. An earlier version scored actions in the motor decoder with hand-written constants (`advance_drive = 3.0` when far, a forced `attack_light` rate in melee, a 1.8x bonus on dodging). Measured on the mock environment, **every descending pool except `dodge_roll` fired exactly zero spikes**, and 0 of 1024 Kenyon cells were ever active. The agent's entire observable behaviour — run forward, roll backward on contact, repeat, never land a hit — was produced by those constants over a silent brain. If you find yourself adding a rule to the decoder, the circuit is missing a pathway; add the pathway.

---

## 2. Neural Circuit Architecture

The agent does **not** use deep artificial neural network weights trained with backpropagation. Instead, it simulates a **Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN)** whose topology maps biological Drosophila brain centers.

```
  Sensory                      Brain Centers                      Motor
  ---------------------------------------------------------------------------------
  bearing map      ----+
  looming (LPTC)   ----+---> [ Optic lobe ] --+
  telegraph bank   ----+                      |
  object size (LC) ----+                      v
                                     [ Central Complex ] --------------+
  compass (EPG) <---> [ Delta7 ring inhibition ]                       |
                                     [ CX inhibitory interneurons ]    |
                                              |                        |
                                              v                        v
  proprioception ----------------> [ Mushroom Body ]         [ Descending Neurons ]
  (stamina, lock, HP)              - Kenyon cells (sparse)    - one pool per action
                                   - APL feedback inhibition  - premotor cross-inhibition
                                   - 8 compartments, one         resolves the winner
                                     per action channel
  nociceptors ---> PPL1 (aversive) ----+  |
  boss damage ---> PAM  (reward)   ----+--+ dopamine gates KC->MBON plasticity
                                          |
  low stamina ---> [ vigor gate ] --------+--> damps the expensive channels
  object size  ---> [ approach brake ] -------> terminates forward locomotion
```

### Action channels

Eight channels, each owning one MB compartment, one descending pool and one premotor
inhibitory pool: `advance`, `strafe_left`, `strafe_right`, `retreat`, `roll`,
`attack_light`, `attack_heavy`, `parry`.

### Sign constraint (Dale's law)

Every neuron is cholinergic (excites all its targets) or GABA/glutamatergic (inhibits all
of them), following the sign-constrained convention of connectome-derived LIF models.
Inhibitory populations: the Delta7-like compass ring, the central complex interneurons,
APL, the GABAergic half of every MB compartment, the premotor pools, the vigor gate and
the approach brake. Without inhibition there is no competition between descending
pathways, and the "winner" is decided by whatever constants the readout happens to carry.

### Biophysical Dynamics

* **Membrane Potential:** $\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R I_{\text{syn}} + R I_{\text{ext}}$
* **Parameters:** $dt = 0.1$ms, $\tau_m = 20$ms, $\tau_s = 5$ms, delay $= 1.8$ms, $V_{\text{rest}} = -52$mV, $V_{\text{thresh}} = -45$mV.
* **Plasticity:** Three-factor STDP. Eligibility traces $e_{ij}(t)$ updated on pre-post spike coincidence, gated by dopamine reward ($\text{PAM}$) / punishment ($\text{PPL101}$), and **credited to the compartment of the action the fly actually executed** via an efference copy.

### Homeostatic calibration

A connectome supplies connectivity, not synaptic strength, and the two are not
interchangeable. `flysoul/connectome/calibration.py` measures each population on a batch
of representative combat states and rescales its input gain until it sits at a target
firing rate (in **Hz**, not per-step, so the operating point does not move when the step
length changes) with the mushroom body at ~10% population sparsity. The result is cached
under `checkpoints/` keyed by a hash of the configuration; `--recalibrate` forces a
rebuild. Calibration is deterministic, so the cache changes nothing about the circuit.

### Encoding principles

* **Bearing, size and motion are separate populations.** The retinal map has constant
  amplitude and width at every range. Letting distance dim it puts a threshold cliff in
  front of every downstream population — the circuit simply switches off past ~5 m.
* **The telegraph bank tiles time,** one cell per phase of the boss's swing, so the
  circuit can learn *when* in an attack to act without being told what to do.
* **Efference copy cancels self-generated optic flow.** A fly walking forward sees the
  target expand; without cancellation the looming pathway reads its own approach as an
  incoming object and fires the escape reflex, so the agent walks, frightens itself,
  rolls away, and repeats forever.

---

## 3. Distributed Hardware Topology

The project is structured to run across two connected machines:

1. **Host Machine (`ibox@ziverbey` - HP Pavilion 15-cb0xx):**
   * **GPU:** NVIDIA GeForce GTX 1050 Mobile 2GB (Driver `580.178.04`, CUDA 13.0).
   * **OS:** Ubuntu 24.04 LTS (Kernel 6.8.0).
   * **Role:** Runs *Dark Souls III* under Proton Experimental + Sunshine NVENC streaming server + SoulsGym memory agent (`DISPLAY=:0`).
2. **Client Machine (Dell Latitude 5490 - Windows 11):**
   * **Role:** Moonlight HEVC client viewer (zero input lag, 60 FPS) + local Python development and offline simulation.

---

## 4. Codebase Organization

```
flysoul/
├── run.py                  # Main CLI entry point (--game, --episodes, --explore, --recalibrate)
├── flysoul/
│   ├── config.py           # Biophysical constants & anatomical population sizes
│   ├── connectome/
│   │   ├── engine.py       # Numba JIT LIF kernel with circular delay queue
│   │   ├── graph.py        # Sign-constrained CSR connectome constructor
│   │   ├── calibration.py  # Homeostatic tuning of population operating points
│   │   └── plasticity.py   # Compartment-specific dopamine-gated STDP
│   ├── sensory/encoder.py  # Combat state -> sensory spike drives (threshold-relative)
│   ├── motor/decoder.py    # Descending pool readout + action-space mapping. No policy.
│   ├── env/
│   │   ├── obs.py          # Single source of truth for parsing observations
│   │   ├── mock_env.py     # Offline Iudex simulator with action locks and stamina
│   │   └── souls_wrapper.py# SoulsGym loader; publishes the valid-action mask
│   ├── telemetry/dashboard.py
│   └── visualizer/web_server.py
└── tests/test_connectome.py
```

### SoulsGym interface facts worth knowing

* **Action IDs** (`soulsgym/core/data/darksouls3/actions.yaml`): 0-7 walk in eight
  directions, 8-15 roll in the same eight, 16 light attack, 17 heavy, 18 parry, 19 idle.
* **Iudex animation IDs** are assigned in category order, so **attacks are 0-18**,
  movement and idle are 19-29, and stagger/parry-break are 30-31. A test written as
  `boss_animation > 0` is therefore inverted: a standing boss reads as attacking and
  `Attack3000` reads as idle.
* **`step_size` is 0.1 s.** The connectome must be advanced by the same amount of
  biological time, or the brain runs behind the fight it is reacting to.
* **`current_valid_actions()`** is the mask of what the player can execute right now.
  SoulsGym silently discards anything else, so an unmasked readout spends decisions on
  frames the game throws away. `skip_steps=True` advances through them.
* **Lock-on decides what "forward" means.** With lock-on, action 0 walks towards the boss
  and attacks track it. Without it, movement is camera-relative and the action space
  contains no camera control, so the agent cannot steer at all - it can only run in
  whatever direction the camera was left pointing. `SoulsEnv._lock_on` does not simply
  press a button: it walks the camera towards the boss with `cameraleft`/`cameraright`/
  `cameraup`/`cameradown` and presses lock only once the camera is within roughly 37
  degrees, with each press queued for the following step. Converging takes seconds of
  wall time, so a short timeout around `_camera_reset` hands back an unlocked camera and
  the episode opens with the agent sprinting the wrong way while Iudex closes on it.
  `run.py` waits for the lock before the fight and holds position whenever it is lost.

---

## 5. How to Run

### Mode A: Standalone Simulation (Offline Mock)

```bash
python run.py --episodes 10 --explore --no-web
```

### Mode B: Live Game with Dark Souls III (On `ziverbey`)

1. Ensure Dark Souls III is running in Windowed mode (800x450) and Offline mode.
2. Launch:

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority ./venv/bin/python3 run.py --game --continuous --explore
```

The first launch runs the homeostatic calibration (about 90 s) and caches it; later runs
start immediately.
