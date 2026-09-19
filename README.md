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

## 📈 Results so far, honestly (as of 2026-09-18, evening)

Everything below is from the live game (SoulsGym, Iudex Gundyr). All learning happens
inside the circuit: KC→MBON three-factor plasticity with a temporal-difference dopamine
signal, replay of remembered fights between episodes ("sleep"), uniform homeostatic
scaling. There is no trained readout and no scripted policy; `scripts/ablation.py` is
the check, and `scripts/trend.py` is how progress is read.

**The fly has beaten the boss eight times** in roughly 3,550 live episodes:

| when | circuit | configuration | episode of run | fly HP left | steps |
|---|---|---|---|---|---|
| 09-17 23:26 | original | `--aggression 8`, sleep, uniform homeostasis | 16 | 13% | 82 |
| 09-18 03:10 | original | + `--impatience 0.02`, credit decay 0.90 | 283 | 6% | 74 |
| 09-18 06:07 | original | eligibility 0.80 (later reverted) | 7 | 6% | 68 |
| 09-18 08:08 | original | exploration temperature 1.5 | 158 | 6% | 65 |
| 09-18 16:4x | **attack-pattern cells**, fresh weights | γ 0.90, temperature 1.5 | ~65th fight of the new circuit | 6% | 65 |
| 09-18 19:50 | attack-pattern cells | same, 3x game speed | 565 (fight ~940 of the circuit) | 8% | 59 |
| 09-18 21:1x | attack-pattern cells | learning rate 0.02, sleep memory 50 | 263 (fight ~1,220 of the circuit) | 9% | 72 |
| 09-18 21:3x | attack-pattern cells | same | 312 | 17% | 66 |

Frames of the fifth to eighth victories were recorded (`checkpoints/runs/<run>/clip_ep*_victory/`).
The win rate is on the order of 1 in 300; the average fight ends with the boss at ~72%
and the fly dead after ~25 decisions. The SoulsGym author's reference agent (dueling
double DQN, replay buffer, 3x game speed, several machines) reached a 45% win rate after
about five million environment samples; this project has seen about 70 thousand.

**How progress is read.** Single fights scatter by ±18 points of boss HP, so a 50-fight
block mean has a standard error of 2.6 and two blocks differ by chance up to ~7 points.
`scripts/trend.py` (and the "every fight this circuit has had" chart in the visualizer)
report the least-squares slope with its 95% interval and a first-half/second-half
Mann-Whitney test; a change counts as progress only when the interval excludes zero or
p < 0.05. Detecting a 5-point improvement between two configurations needs ~200 fights
per arm; 3 points needs ~560.

**The circuits.** The original circuit plateaued at 71–78% boss HP over ~2,000 fights
regardless of nine single-parameter interventions. Its learned KC→MBON weights did move
sensibly (retreat +98% → −51%, roll −76% → −4%, advance −75% → −30%; punishments per
reward 6:1 → 3:1) but the score did not follow. The least-squares ceiling explained
why: on 300 archived fights, the best linear readout of the Kenyon cell code predicted
the outcome of an action at Spearman +0.26, while "which attack the boss is performing,
at which phase" predicted it at +0.39 — information the telegraph bank, which tiles time
since the swing began identically for all 25 attacks, never gave the mushroom body.
Sixty-four attack-pattern cells (each animation drives a fixed sparse subset, the way a
lobula columnar type answers one visual motion pattern) raise the simulated KC ceiling to
+0.38 (`scripts/kc_ceiling_sim.py`, which re-encodes archived fights through the real
circuit). Learning was restarted from the innate circuit on 2026-09-18 16:03 with 2,326
neurons. Its first 283 fights: 71.7% boss HP and 5.8 hits per fight against the original
circuit's 80.6% and 3.8 over its own first 283 (p < 0.0001), with its first victory at
fight ~65 instead of ~800 — confounded by the other fixes that were already in place,
but the direction is not in doubt. Within the new circuit, 956 fights show no trend
(slope 0.0 ± 0.4 per 100 fights, halves 72.3% vs 72.4%).

**Offline measurement, before touching the live run.** Every step is archived
(`checkpoints/runs/<run>/archive.npz`: spikes, action, reinforcement, raw reward, boss
animation and phase, distance, motor rates, critic values). Tools:

- `scripts/offline_replay.py` — replay a candidate plasticity rule over real fights;
- `scripts/offline_search.py` — score rule settings by whether, on held-out fights,
  they raised the drive of actions that worked and lowered it where they failed;
- `scripts/offline_ceiling.py` — the same mapping fitted by least squares: the ceiling
  no local rule can beat, plus a one-shot LSTD critic;
- `scripts/kc_ceiling_sim.py` — the ceiling of a *different encoder*, by re-simulating
  the circuit on archived contexts;
- `scripts/offline_shaping.py` — the same score for every reward shaping;
- `scripts/diagnose_learning.py` — where the chain from experience to behaviour breaks:
  weight trajectory (drift vs random walk), knowledge in the weights vs the ceiling,
  whether the learned preference reaches the motor pools, and the teaching signal;
- `scripts/offline_stability.py` — replays archived fights *sequentially* (online pass,
  then a night of sleep) for a learning configuration and tracks held-out knowledge,
  the cosine between successive weight changes and travel vs random walk;
- `scripts/offline_rule_limit.py` — the ceiling's learning curve against training-set
  size, and credit by executed vs intended compartment.

Findings: the value horizon γ 0.95 → 0.90 took credit assignment from ~0.00 to +0.16 on
both splits (deployed); eligibility, credit and learning-rate settings made no consistent
difference; the raw SoulsGym reward assigns credit *against* real outcomes (−0.10) because
damage taken dominates, `--aggression` 8 sits at the top of a plateau that begins at 4,
and `--impatience` changes nothing (0.00–0.05 identical). Higher aggression trades dodge
timing for attacking (roll late/early ratio 1.14 → 1.0).

**Why the score is flat: the diagnosis (2026-09-18 evening, 580 archived fights of the
pattern-cell circuit).** The signal is there: a least-squares readout of the Kenyon code
predicts the outcome of an action at +0.41 held-out, and reaches +0.20 from only 15
training fights. The learned weights do carry knowledge and it does reach behaviour:
the change in MBON drive correlates with pool rates at +0.52 per step, and when the
executed action is the learned favourite the outcome is better (−0.31 vs −0.43). But the
weights hold only about a third of the ceiling (+0.14), and they do not accumulate:
successive 50-fight weight changes point in opposite directions (cosine −0.4 to −0.6),
the total distance travelled over 400 fights is half of a random walk's, and the mean
weight of the attack compartment swings 1.0 → 1.6 → 1.1 → 1.5 → 1.2. The same
oscillation appears when the archived fights are replayed sequentially through the rule
on a fixed dataset, so it is the rule, not the policy chasing its own consequences. In
effect the fly remembers its last ~20 fights. Twenty-two configurations were replayed
over 290 fights and scored on the other 290 (both ways round): a learning rate of 0.02
(+0.13/+0.14 vs live +0.10/+0.13) and a sleep memory of 50 fights (weights then keep a
direction) are the only consistent gains; no sleep, no critic, fixed or linear dopamine
scaling, per-compartment or global dopamine baselines, γ-matched traces, sharper credit
and wider clip bounds were all worse or equal. A separate finding: 41% of executed
actions differ from the motor winner, and Boltzmann exploration explains only 2.5% of
that; the rest is SoulsGym's valid-action mask (attacks and rolls are locked during
animations, so the circuit's competition runs among walking actions, and "attack won →
retreat executed" happened 792 times). The remaining gap to the ceiling looks like a
property of the local three-factor family on this data — a Hebbian sum against a
perfectly centred outcome reaches +0.22 on the same fights — rather than of any
parameter. Two of these findings went live on 2026-09-18 at 20:20, at a lossless
restart: `--learning-rate 0.02` and `--sleep-memory 50` (and `--impatience` dropped,
having measured as irrelevant). Replaying only the best fights was tested offline and
is harmful (knowledge ~0 on both splits): the rule learns from the contrast between
good and bad outcomes, and a memory of wins alone removes it.

**A diagnostic probe, not a result (2026-09-18 22:45–23:30).** To learn whether the
+0.39 least-squares ceiling would even *win*, the ceiling's own KC→MBON weights (fitted
outside the brain on 980 archived fights, `scripts/fit_probe_weights.py`) were loaded
into the same circuit with learning, sleep and saving off, under a separate fingerprint
(`run.py --probe-weights`). Over 124 fights the fly survived longer (39 vs 25 decisions)
but hit less (3.3 vs 5.9 per fight) and left the boss at 79–86% against the learned
fly's 72%: it parried and backed off. The outcome label the whole offline toolkit scores
against (hit within three steps = +1, damage = −1) is dominated by damage avoidance, so a
perfect readout of it is a cautious fly, not a winning one. Two consequences: the
offline "knowledge" score ranks rules by how well they predict that label, which is a
proxy and not the objective, and the gap that matters is not rule vs ceiling but label
vs winning. A second probe fitted to the aggression-weighted learning reward instead
(held-out +0.46, the signal the fly actually chases) did the same over 120 fights:
76–82% boss HP, 3.5–4.7 hits, no victory. Neither readout of this Kenyon code, fitted
on 980 fights of data, plays better than the synapses the fly learned itself. The probe
weights were discarded and the learned checkpoint restored. Two more in-brain levers were
then measured and found empty: a conjunction code for attack × phase in the pattern cells
(re-simulated KC ceiling +0.32–0.34 against +0.33 live; exact cell groups are worse), a
wider mushroom body (1,536 / 2,048 Kenyon cells: +0.33 / +0.34), and a motor decoder
that holds still when the winning pool is not executable instead of running the best
executable one (live A/B, 315 vs 381 fights: 72.4% vs 72.2%, p = 0.87).

**Throughput, measured rather than assumed.** SoulsGym resets by teleporting and
rewriting health, not through the game's death reload: a reset is 1.6 s. The fight is
~20 s of wall clock for ~25 decisions because the environment advances the game through
animation-locked steps in game time, so `--game-speed` scales most of an episode. At 3x
(the SoulsGym author's setting; the brain simulates a 100 ms step in ~33 ms) and with
the between-fight replay paced to 2.5 s instead of 7, an episode takes 11.5 s instead of
28 (`scripts/reset_timing.py`), about 300 episodes an hour.

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

**What is innate and what is learned.** Innate, designed by us from MaleCNS cell types:
the retina map, looming and object-size cells, telegraph time cells and attack-pattern
cells, the compass ring, the looming→roll and hemifield→strafe reflexes, premotor
cross-inhibition, the vigor gate and approach brake. Learned, and the only thing that
is: the KC→MBON synapses, i.e. which action each sensory situation favours, through
dopamine alone. The 3D view draws the model's neurons on the somata and skeletons of real
MaleCNS neurons of the same cell types (`scripts/fetch_malecns.py`, CC-BY); that mapping
is by type and is presentation, not a claim that the model neuron is that cell.

**Things that did not work, and what they cost to find out.**
- Three bugs masked everything for the first ~400 episodes: the player heading convention
  was 146° off (the retina saw the boss in the wrong place), the patched camera reset
  rotated the character's body instead of the camera, and a per-compartment homeostatic
  cap pinned five of eight compartments at exactly +53% so no experience could change
  behaviour. Any result measured before those fixes is void.
- Reweighting damage dealt (`--aggression 8`) alone: no effect while the cap bug was
  live; after the fix it produced the first victory within 16 episodes.
- Sleep replay alone (with the cap bug live): worse. More updates pressed harder against
  the same wall.
- Lengthening the credit trace (0.82 → 0.90) turned the fly into a parry turtle within 40
  fights; shortening the eligibility trace (0.92 → 0.80) changed nothing; both were later
  shown offline to make no difference.
- Setting the synaptic tag only at decision time (the rule before accumulated it on every
  compartment every step, blaming the cells active *after* an early roll) is right by the
  unit test, improved dodge statistics for ~40 fights, and did not move the score.
- Spike-frequency adaptation, the single change that most improved flybench's whole-brain
  LIF models, stops this sub-circuit's calibration converging even at a quarter of the
  published value. Implemented, tested, left off.
- A day of per-step data was lost because the archive lived only in memory and each
  restart overwrote the file; the run recorder (`flysoul/telemetry/recorder.py`) now
  writes per-run folders, sleep memory persists across restarts, and restarts are
  lossless.

## 🧪 3b — a synthetic readout, kept apart (from 2026-09-19)

After every measurable lever inside the brain came back empty, the project added one
thing that is **not** the fly's learning and is labelled as such everywhere: a linear Q
function over the fly's own Kenyon cell code, fitted *outside the brain* by least-squares
Q-iteration on the archived fights (`scripts/fit_q_readout.py`, `flysoul/synthetic/`),
and run live with `run.py --synthetic-q`. When it is on, it chooses the action; the
biological circuit still produces the sensory code, its learned synapses are loaded only
to report how often the two agree, and are never updated or saved. Such runs carry the
fingerprint `3b`, have their own learning curve, and show a magenta "3b · SYNTHETIC
READOUT" card in the visualizer. Results from 3b runs are reported separately from the
fly's, and never in the victory table above. First readout: fitted on 1,670 fights,
γ 0.90, Bellman RMSE 0.26 held-out, agrees with the fly's own choice on 31% of steps.
It is refitted every ~150 fights on all data including its own. Rounds so far (boss HP
left): 72%, 70%, 70%, 67.5%; the fourth round beats the fly's own plateau for the first
time (67.5% vs 72.2% over 157 vs 381 fights, p = 0.005, 6.7 vs 5.7 hits). Two victories
under the synthetic readout (episode 12 of round 1b, 115 steps, 3% HP; episode 22 of
round 4, 80 steps, a mutual kill) are listed here, not above. The Kenyon-only linear
readout converged at ~68% over rounds 4–6. Round 7 replaced the Kenyon code with the
binned raw game state (`flysoul/synthetic/features.py`, 366 one-hot features: attack ×
phase, distance, angle, health, previous action): 65.8% over 150 fights, 6.9 hits, twice
the survival (51 decisions), a third synthetic victory; +6.3 points over the fly
(p = 0.0002) and +2.6 over the Kenyon-only readout (p = 0.20, not yet significant).
Round 8, refitted with round 7's own fights: 54.6% boss HP, 9.1 hits, 52 decisions, one
more synthetic victory; +17.5 points over the fly and +11.4 over round 7 (both p < 10⁻⁴).
The synthetic readout learns from the same data volume the fly had; what differs is the
objective (Q-iteration), the batch fit, and the raw state. Rounds 9 and 10 (59.6%, 55.6%;
8–9 hits; five more synthetic victories, about one per 60 fights) suggest the linear
raw-state readout plateaus around 55–60% boss HP. Rounds 11–12: 56.0% and 53.8%, 9 hits,
four more synthetic victories (now one in about 50 fights; the best left the fly at 46% HP).
Round 15 added the player's stamina to the state and cut exploration to ε = 0.02: 50.6% boss
HP, 9.8 hits and nine victories in 154 fights (one in 17; the cleanest left the fly at 73%
HP). A two-layer network on the same features (round 14) was not better than the linear
readout, and the training horizon (γ 0.90 vs 0.95) made no difference.

## 🛠 Operating the live run

- `scripts/supervise.sh` keeps the agent running; `touch STOP` stops it. Kill the agent
  with a pattern anchored to the process path (`pkill -f "^.../venv/bin/python run.py"`);
  an unanchored pattern matches the shell running the command.
- Live flags since 20:20: `--game --continuous --explore --explore-temperature 1.5 --game-speed 3
  --aggression 8 --learning-rate 0.02 --sleep-memory 50` (run.json in each run folder
  records them).
- Between-fight work: sleep replay runs in a thread while the game resets; the archive is
  written every 10 episodes, weights every 50, clips on victories and near misses
  (boss ≤ 10%).
- Read progress with `scripts/progress.py --since-restart` (blocks) and
  `scripts/trend.py <log> --from-line N` (statistics); `scripts/reset_timing.py` shows
  where wall time goes; `scripts/press_key.py Escape` sends a key to the game host's
  display (a stray Steam Big Picture press once stalled the agent for 8 minutes; the
  reset path now closes it automatically).
- The visualizer is served from `flysoul/visualizer/index.html` on every request, so page
  changes need no restart; server or agent changes do. A Cloudflare quick tunnel
  (`cloudflared tunnel --url http://localhost:8080`, run with an isolated `HOME` if the
  host has its own tunnel config) gives a read-only public link.

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
