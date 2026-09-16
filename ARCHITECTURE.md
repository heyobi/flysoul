# FlySoul System Architecture & AI Agent Guide

Welcome to **FlySoul**! This document provides an exhaustive overview of the system architecture, directory layout, biophysical modeling, and deployment instructions so that any AI agent or researcher can immediately understand and extend this project.

---

## 1. Scientific & Engineering Context

* **Biological Ground Truth:** Google Research & HHMI Janelia *MaleCNS v1.0* (published in *Cell*, September 2026: *"Sexual dimorphism in the complete connectome of the Drosophila male central nervous system"*). Contains **166,000+ neurons** and **125 million synapses**.
* **Target Task:** Dark Souls III tutorial boss fight (**Iudex Gundyr**) via **SoulsGym** (`amacati/SoulsGym`).
* **Why Dark Souls III?** Unlike simple toy games, Dark Souls demands strict timing, stamina conservation, telegraph recognition (windup detection), and directional evasion.

---

## 2. Neural Circuit Architecture

The agent does **not** use deep artificial neural network weights trained with backpropagation. Instead, it simulates a **Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN)** whose topology maps biological Drosophila brain centers:

```
                                  [ FlySoul Bio-Connectome Circuit ]
                                  
  Sensory Inputs                          Brain Centers                         Motor Outputs
  (SoulsGym Memory)                      (Connectome SNN)                       (Game Actions)
  
  [ Boss Distance ] ──────┐
  [ Boss Rel Angle] ──────┼───> [ Visual Retina (Compound Eye) ]
  [ Boss Windup   ] ──────┤     - 16 azimuthal receptive fields
                          │     - Looming motion detectors (LC11/LPTC)
                          │                       │
                          │                       ▼
                          └───> [ Central Complex (CX) ] ─────────────────┐
                                - Protocerebral bridge                    │
                                - Ellipsoid Body (EPG heading compass)    │
                                                  │                       │
                                                  ▼                       ▼
  [ Player Damage ] ──────────> [ Mushroom Body (MB) ] ───────────> [ Descending Neurons ]
  (Nociceptive Burst)           - Kenyon Cells (sparse encoding)    - DNp20: Dodge Roll
                                - Plastic KC -> MBON synapses       - DNpe017: Light Attack
                                - PPL1 (Pain/LTD) & PAM (LTP)       - DNa02: Heavy Attack
                                                                    - DNb01: Parry/Block
                                                                          │
                                                                          ▼
                                                                  [ SoulsGym Execution ]
```

### Biophysical Dynamics
* **Membrane Potential:** $\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R I_{\text{syn}} + R I_{\text{ext}}$
* **Parameters:** $dt = 0.1$ms, $\tau_m = 20$ms, $\tau_s = 5$ms, delay $= 1.8$ms, $V_{\text{rest}} = -52$mV, $V_{\text{thresh}} = -45$mV.
* **Plasticity:** Three-factor STDP. Eligibility traces $e_{ij}(t)$ updated on pre-post spike coincidence, gated by dopamine reward ($\text{PAM}$) / punishment ($\text{PPL101}$).

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
├── .gitignore              # Ignores venv, binaries, temp files, moonlight
├── README.md               # GitHub / Show HN presentation and benchmark results
├── ARCHITECTURE.md         # Detailed technical manual for developers & AI agents
├── requirements.txt        # numpy, numba, scipy, gymnasium, rich, pytest, etc.
├── run.py                  # Main CLI entry point (--mock, --game, --episodes, --live-web)
├── flysoul/
│   ├── __init__.py
│   ├── config.py           # Biophysical constants & anatomical neuron population sizes
│   ├── connectome/
│   │   ├── engine.py       # Numba JIT Leaky Integrate-and-Fire kernel with circular delay queue
│   │   ├── graph.py        # CSR connectome constructor (MaleCNS topology)
│   │   └── plasticity.py   # Three-factor dopamine-modulated STDP rule
│   ├── sensory/
│   │   └── encoder.py      # Translates SoulsGym memory variables into sensory spike drives
│   ├── motor/
│   │   └── decoder.py      # Decodes Descending Neuron firing rates to discrete SoulsGym actions
│   ├── env/
│   │   ├── mock_env.py     # High-fidelity standalone Iudex Gundyr physics & hitbox simulator
│   │   └── souls_wrapper.py# Dynamic environment loader (Mock fallback <-> SoulsGym 1.2.0)
│   ├── telemetry/
│   │   └── dashboard.py    # Terminal TUI dashboard built with Rich
│   └── visualizer/
│       └── web_server.py   # Real-time WebGL/Canvas 3D live brain monitor (flashing spikes)
├── scripts/
│   ├── check_env.py        # Validates SoulsGym connection to running DarkSoulsIII.exe
│   └── setup_ds3.py        # Installs speedhack proxy DLLs and sets Wine DllOverrides
└── tests/
    └── test_connectome.py  # Unit tests covering LIF dynamics, synapses, and plasticity
```

---

## 5. How to Run

### Mode A: Standalone Simulation (Offline Mock)
Runs the complete fruit fly connectome fighting a simulated Iudex Gundyr in under 5 seconds:
```bash
python run.py --episodes 3
```

### Mode B: Live Game with Dark Souls III (On `ziverbey`)
1. Ensure Dark Souls III is running in Windowed mode (800x450) and Offline mode.
2. Launch:
```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority ./venv/bin/python3 run.py --game
```
