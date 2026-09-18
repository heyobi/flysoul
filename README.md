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
* **Prediction error:** dopamine carries $\delta = r + \gamma V(s') - V(s)$, not $r$, with
  $V$ read linearly off the sparse Kenyon cell population. This is what makes dodging
  learnable at all: SoulsGym pays for damage dealt and damage taken and for nothing else,
  so a successful dodge scores exactly zero and is indistinguishable from standing still.
  Against a mock whose reward matches the game's, the plain signal collapses to zero hits
  by episode 150 with `parry` the most-reinforced channel; the TD signal improves
  monotonically over 200 episodes.

$$\Delta W_{ij} = \eta \cdot \tanh\left(\frac{\delta}{\sigma}\right) \cdot c_{k(ij)}(t) \cdot \text{Trace}_{ij}(t)$$

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

## 📈 Results so far, honestly (as of 2026-09-18)

Everything below is from the live game (SoulsGym, Iudex Gundyr, ~2.3 episodes per
minute including the death reload). All learning happens inside the circuit: KC→MBON
three-factor plasticity with a temporal-difference dopamine signal, replay of remembered
fights during the loading screen ("sleep"), and uniform homeostatic scaling. There is no
trained readout and no scripted policy; `scripts/ablation.py` is the check.

**The fly has beaten the boss twice** in roughly a thousand live episodes:

| when | run configuration | episode | fly HP left | steps |
|---|---|---|---|---|
| 2026-09-17 23:26 | `--aggression 8`, sleep, uniform homeostasis | 16 | 13% | 82 |
| 2026-09-18 03:10 | + `--impatience 0.02`, credit decay 0.90 | 283 | 6% | 74 |

That is a win rate on the order of 1 in 300. The average fight ends with the boss at
71–78% health and the fly dead after ~23 decisions; it has not moved in 500 episodes of
the last configuration. For scale, the SoulsGym author's reference agent (dueling double
DQN, replay buffer, 3x game speed, several machines) reached a 45% win rate after about
five million environment samples; this run has seen about twelve thousand.

**What the learned synapses say** (KC→MBON weight per compartment, relative to the
innate circuit, after 511 episodes): attack light +115%, attack heavy +32%, parry +11%,
roll −4%, advance −46%, retreat −51%, strafes −27%. Compared with the first victory the
fly has stopped fleeing (retreat was +98%), recovered rolling (was −76%) and is slowly
recovering approach (was −75%). The critic's value estimates span −0.6 to +0.9.

**What the fights say.** Sleep memory is also used to measure how far each action sits
from the reinforcement that follows it. Approaching precedes a landed hit by a median of
4–6 steps and precedes a hit taken by 2–7 steps: no credit-trace decay can separate the
two. Over the night the ratio of punishments to rewards fell from about 6:1 to 3:1. The
bottleneck is defence — the fly gets hit several times per hit it lands — not offence.

**Lesion controls** (`scripts/ablation.py`, mock arena, innate circuit, 10 episodes each):

| condition | hits/ep | boss HP | idle | action mix |
|---|---|---|---|---|
| intact | 2.7 | 86% | 0% | advance 39, retreat 24, attack light 15 |
| MBON output cut | 3.3 | 72% | 42% | advance 43, idle 42 |
| descending pools deafened | 0.0 | 100% | 100% | idle 100 |
| sensory afferents cut | 0.0 | 100% | 100% | idle 100 |
| weights shuffled | 3.1 | 71% | 0% | strafe right 55, advance 23 |

The behaviour depends on the circuit: without a path from brain to motor output, or
without sensory input, the fly does nothing, and shuffling the wiring produces a
different animal. This demonstrates dependence on the circuit, **not** that the
biological wiring is better at the game — the shuffled circuit scores higher here.

**Things that did not work, and what they cost to find out.**
- Three bugs masked everything for the first ~400 episodes: the player heading convention
  was 146° off (the retina saw the boss in the wrong place), the patched camera reset
  rotated the character's body instead of the camera, and a per-compartment homeostatic
  cap pinned five of eight compartments at exactly +53% so no experience could change
  behaviour. Any result measured before those fixes is void, including this project's
  earlier "reward shaping does not help" conclusions.
- Reweighting damage dealt (`--aggression 8`) alone: no effect while the cap bug was
  live; after the fix it produced the first victory within 16 episodes.
- Sleep replay alone (with the cap bug live): worse. More updates pressed harder against
  the same wall.
- Lengthening the credit trace (0.82 → 0.90) turned the fly into a parry turtle within 40
  fights; measuring the action-to-outcome delays afterwards showed why it could not help.
- Spike-frequency adaptation, the single change that most improved flybench's whole-brain
  LIF models, stops this sub-circuit's calibration converging even at a quarter of the
  published value. Implemented, tested, left off.

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
