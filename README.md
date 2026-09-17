# 🪰 FlySoul: Fruit Fly Connectome plays Dark Souls III

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MaleCNS v1.0](https://img.shields.io/badge/connectome-MaleCNS%20v1.0-orange.svg)](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
[![SoulsGym](https://img.shields.io/badge/environment-SoulsGym%20(DS3)-red.svg)](https://github.com/amacati/SoulsGym)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Can a fruit fly beat Iudex Gundyr?*

**An embodied bio-connectomic AI agent controlling combat in Dark Souls III using Google Research & HHMI Janelia's complete MaleCNS v1.0 fruit fly connectome.**

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [How it Works](#-how-it-works) • [Scientific References](#-scientific-references)

</div>

---

## 🎮 The Premise

In September 2026, Google Research and HHMI Janelia published the complete cellular wiring map of the male fruit fly (*Drosophila melanogaster*) central nervous system (**MaleCNS v1.0**, *Cell* 2026) — spanning **166,700 neurons** and **125 million synaptic connections**.

While other researchers gave the fly a shotgun in *Doom* or let it trade Bitcoin, we decided to test its reflexes in the ultimate proving ground of human patience and pain: **Dark Souls III**.

**FlySoul** bridges the biological neural network of a fruit fly with **SoulsGym**, driving real-time melee combat against the tutorial boss, **Iudex Gundyr**.

```
                           THE BIO-COMBAT LOOP
  +-------------------------------------------------------------------+
  |                                                                   |
  |   👹 IUDEX GUNDYR (Dark Souls III / SoulsGym)                     |
  |      [Boss HP, Distance, Rel. Angle, Attack Phase, Player HP]     |
  |                                |                                  |
  |                                v                                  |
  |   👁️ SENSORY ENCODER                                             |
  |      - Compound Eye (bearing map, constant across range)          |
  |      - Lobula: looming, object size, attack-telegraph time cells  |
  |      - Central Complex (EPG Heading Compass + Delta7 inhibition)  |
  |      - Interoception (stamina, animation lock, own health)        |
  |      - Nociceptive Afferents (Damage / Pain Receptors)            |
  |                                |                                  |
  |                                v                                  |
  |   🧠 MaleCNS v1.0 BIOLOGICAL LIF ENGINE (Numba JIT)               |
  |      - 1.8 ms axonal transmission delay                           |
  |      - Subthreshold integration: tau_m = 20ms, tau_s = 5ms       |
  |      - Three-Factor Plasticity:                                   |
  |          * Damage taken -> PPL101 aversive dopamine burst -> LTD  |
  |          * Boss struck  -> PAM reward dopamine burst -> LTP       |
  |                                |                                  |
  |                                v                                  |
  |   🦾 MOTOR READOUT (Descending Neurons / DNs)                     |
  |      - one pool per action channel, eight in all                  |
  |      - premotor cross-inhibition picks the winner                 |
  |      - the readout reports that winner; it does not choose it     |
  |                                |                                  |
  |                                v                                  |
  |   ⚔️ GAME EXECUTION -> Advance, Strafe, Retreat, Roll, Attack,    |
  |                        Heavy, Parry                               |
  |                                                                   |
  +-------------------------------------------------------------------+
```

---

## ⚡ Quickstart

### 1. Installation

Requires **Python 3.11+**:

```bash
git clone https://github.com/flysoul/flysoul.git
cd flysoul
pip install -r requirements.txt
```

### 2. Run the Autonomous Arena (Mock Mode)

You can run FlySoul right now in standalone simulation mode with zero setup — no copy of Dark Souls 3 required! It uses a high-fidelity Gymnasium simulation of Iudex Gundyr:

```bash
python run.py --episodes 10 --explore
```

The first launch runs a homeostatic calibration pass (about 90 s) that sets each
population's operating point, then caches it under `checkpoints/`. Later launches start
immediately. `--recalibrate` forces a rebuild, `--no-dashboard --quiet` gives one line
per episode, and `--no-learning` freezes the KC->MBON synapses so you can see what the
innate circuit does on its own.

### 3. Connect to Live Dark Souls III (SoulsGym)

When running Dark Souls 3 on your gaming machine:

```bash
python run.py --game --continuous --explore
```

---

## 🔬 How It Works

### 1. Biophysical Leaky Integrate-and-Fire (LIF) Kernel
Neurons are modeled as Leaky Integrate-and-Fire units with an analytic membrane decay ($\tau_m = 20.0 \text{ ms}$) and synaptic conductance decay ($\tau_s = 5.0 \text{ ms}$). Synaptic transmission features an exact **1.8 ms axonal delay queue** pre-compiled with Numba JIT:

$$\tau_m \frac{dV_i}{dt} = -(V_i - V_{\text{rest}}) + R \cdot I_i(t) + g_i(t)$$

### 2. Sensory Transduction (Visual & Spatial Orientation)
* **Compound Eye:** 128 retinal columns with Gaussian tuning over azimuth $[-\pi, +\pi]$, at
  constant amplitude and width. Bearing, object size and optic flow are carried by
  *separate* populations: every downstream cell is a threshold element, so a retina that
  dims with distance does not fade gracefully, it switches the whole circuit off.
* **Telegraph time cells:** a bank tiling time-since-attack-onset, so the circuit can
  learn *when* in Iudex's swing to act rather than being told what to do about it.
* **Central Complex Compass:** Ellipsoid Body / Protocerebral Bridge (EPG) neurons with
  Delta7-like ring inhibition, forming a heading vector toward the boss.
* **Efference copy:** locomotor commands suppress the looming pathway. Without it the fly
  reads its own approach as an object rushing in, rolls away from nothing, and never
  closes the distance.
* **Interoception & nociception:** stamina, animation lock and health; HP drops fire the
  nociceptors that drive PPL1.

### 2b. Sign constraint and homeostatic calibration
Every neuron is cholinergic or GABA/glutamatergic and keeps that sign for all of its
outputs (Dale's law), as in connectome-derived LIF models. Inhibition is not a detail
here: without it the descending pools cannot compete and whatever reads them out ends up
making the decision.

A connectome gives connectivity, not synaptic strength. `flysoul/connectome/calibration.py`
measures each population across a batch of combat states and rescales its input gain until
it sits at a target rate in Hz, with the mushroom body at ~10% sparsity. The objective
blends the mean with the *worst* state, because matching the mean alone lets a population
run hot in melee and stone dead at spawn distance.

### 3. Dopaminergic Three-Factor Plasticity (PPL101 vs. PAM)
In *Drosophila*, learning occurs primarily at Kenyon Cell (KC) to Mushroom Body Output Neuron (MBON) synapses:
* **Punishment (Taking Damage):** Activates **PPL101** dopaminergic neurons, inducing
  **Long-Term Depression (LTD)** on synapses active prior to being hit.
* **Reward (Hitting the Boss):** Activates **PAM** cluster dopaminergic neurons,
  reinforcing aggressive strike timing.
* **Compartments:** the mushroom body is split into one compartment per action channel,
  each with its own dopaminergic neurons. An **efference copy** of the action the fly
  actually executed keeps that compartment eligible for a short window. A single scalar
  reward applied to every plastic synapse can only make the whole mushroom body louder or
  quieter — never prefer one action over another.
* **Prediction error:** dopamine carries $r - \bar{r}$, not $r$. A hit from Iudex is worth
  about $-2.8$ and lands three times as often as the $+0.5$ the fly earns for its own, so
  the raw signal is net negative on every event; that depresses whatever the fly was
  doing until it does nothing, which is the one behaviour that owns no compartment and so
  can never be punished.

$$\Delta W_{ij} = \eta \cdot \tanh\left(\frac{r - \bar{r}}{\sigma}\right) \cdot c_{k(ij)}(t) \cdot \text{Trace}_{ij}(t)$$

where $c_k$ is the responsibility of compartment $k$ for what just happened.

### 4. What the readout is not allowed to do
`flysoul/motor/decoder.py` contains no policy. It converts descending spike counts into
firing rates, drops the channels the game cannot currently execute, and reports the
winner. An earlier version scored the actions itself with hand-written constants; with
those in place **every descending pool except `dodge_roll` fired exactly zero spikes and
0 of 1024 Kenyon cells were ever active.** The agent ran forward on a constant, rolled
backward when the constant expired, and never landed a hit — behaviour that came entirely
from the readout, over a brain that was not participating.

---

## 📊 Live Telemetry Dashboard

FlySoul includes a terminal user interface powered by `rich`:

```
┌─────────────────────────── Arena Status ───────────────────────────┐┌─────────────────────── Connectome Telemetry ───────────────────────┐
│ 👹 IUDEX GUNDYR                                                    ││ 🧠 Total Neurons: 1,916                                              │
│ HP [██████████████████░░░░░░░░░░░░] 60.0%                          ││ ⚡ Active Spiking: 247 (12.9%)                                       │
│ State: WINDUP | Distance: 3.2m                                     ││                                                                     │
│                                                                    ││ Dopamine Plasticity:                                                │
│ 🪰 FRUIT FLY (MaleCNS)                                             ││ 🔴 PPL101 Aversive Burst (-1.40) -> LTD                             │
│ HP [██████████████████████████░░░░] 85.0%                          ││                                                                     │
│ SP [████████████████░░░░] 80.0%                                    ││ Descending Neuron (DN) Firing Pools:                                │
│                                                                    ││   roll        : 61.5 Hz ▓▓▓▓▓▓▓▓▓                                   │
│ ⚡ Current Action: DODGE_ROLL                                       ││   attack_light: 38.1 Hz ▓                                           │
└────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

All unit tests verify:
- LIF subthreshold dynamics and delay queues
- Dale's law: every neuron's outputs share one sign, and inhibition exists
- Every action channel has afferents, so every action is reachable
- Sensory drives actually cross the spike threshold
- The efference copy suppresses self-generated looming
- Descending readout, the valid-action mask, and roll direction composition
- Iudex animation IDs: attacks are 0-18, movement and idle are 19+
- Compartment-specific dopamine credit assignment
- Mock environment action locks, stamina and reward mechanics

`tests/test_soulsgym_interface.py` drives the agent through a stand-in that speaks
SoulsGym's exact dialect — its observation keys, its 20 discrete actions, its animation
ID numbering and its `current_valid_actions` mask. SoulsGym itself cannot be installed on
a development machine (it hooks a running `DarkSoulsIII.exe`), so without this the branch
that plays the game is the one branch never exercised locally.

---

## 📜 Scientific References

1. **MaleCNS v1.0 Connectome:**
   * Januszewski, M., Jain, V., et al. (2026). *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*. **Cell**, 189, 1–24.
   * Google Research: [A connectomics milestone: Mapping the complete male fruit fly brain](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).
2. **Drosophila Mushroom Body & Dopaminergic Plasticity:**
   * Aso, Y., et al. (2014). *The neuronal architecture of the mushroom body provides a logic for associative learning*. **Cell**, 159(4), 892–909.
   * Hige, T., et al. (2015). *Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila*. **Nature**, 526(7571), 149–152.
3. **SoulsGym:**
   * Amacati, et al. (2023). *SoulsGym: A Gymnasium Environment for Dark Souls III and Elden Ring*. [GitHub Repository](https://github.com/amacati/SoulsGym).

---

## ⚖️ License

MIT License. Designed with curiosity and perseverance for science and gaming.
