"""Live WebGL 3D Fruit Fly Brain Visualizer Server for FlySoul.

Serves an interactive 3D WebGL visualization of the MaleCNS fruit fly connectome
with real-time synaptic spiking, dopamine cascades, and Dark Souls III combat HUD.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

HTML_PATH = Path(__file__).parent / "index.html"

from flysoul.connectome.graph import ACTION_CHANNELS, CircuitTopology

# Global event broadcast queue for Server-Sent Events (SSE)
_subscriber_queues: List[queue.Queue] = []
_subscriber_lock = threading.Lock()
_topology_data: Optional[Dict[str, Any]] = None

# Video feed JPEG frame buffer
_latest_frame_jpeg: Optional[bytes] = None
_frame_lock = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes):
    """Cache the latest JPEG frame from Dark Souls III for the web stream."""
    global _latest_frame_jpeg
    with _frame_lock:
        _latest_frame_jpeg = jpeg_bytes


def get_latest_frame() -> Optional[bytes]:
    """Retrieve the latest JPEG frame."""
    with _frame_lock:
        return _latest_frame_jpeg


MALECNS_PATH = Path(__file__).parent / "malecns.json"
SCENE_UNITS_PER_UM = 0.45  # brain extent ~1000 um -> fits the viewer's +-220 frame

_FAMILY_KEYS = (
    "Retina_OpticLobe", "Lobula_Motion", "Compass_EB", "Compass_Ring_Inhibitory",
    "Central_Complex_Inhibitory", "Central_Complex", "Kenyon_Cell", "APL_Inhibitory",
    "MBON_", "Dopamine_PPL1", "Dopamine_PAM", "Nociceptor", "Proprioceptor",
    "Vigor_Gate_Inhibitory", "Approach_Brake_Inhibitory", "Premotor_Inhibitory_", "Motor_",
)


def _family_of(label: str) -> Optional[str]:
    best = None
    for key in _FAMILY_KEYS:
        if label.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return best


def _map_to_malecns(topology: CircuitTopology) -> Optional[Dict[str, Any]]:
    """Place every model neuron on a real MaleCNS neuron of the same cell type.

    The circuit is derived from the connectome's cell types and wiring statistics, not
    copied neuron by neuron, so the mapping is by family: each modelled Kenyon cell is
    drawn at the soma (and along the skeleton, where fetched) of a real Kenyon cell,
    assigned round-robin. What is drawn is real anatomy; which neuron is which is a
    presentation choice, and the file says so.
    """
    if not MALECNS_PATH.exists():
        return None
    try:
        data = json.loads(MALECNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    fams = data.get("families") or {}
    centre = np.asarray(data.get("centre", [0, 0, 0]), dtype=np.float64)
    um = float(data.get("voxel_um", 0.008))

    def to_scene(v):
        p = (np.asarray(v, dtype=np.float64) - centre) * um * SCENE_UNITS_PER_UM
        # Dataset axes: x lateral, y dorsal-ventral, z anterior-posterior along the
        # CNS. Show the CNS upright: brain on top, nerve cord below, viewed from the front.
        return [float(p[0]), float(-p[2]), float(-p[1])]

    # Which way is the brain? Kenyon cell somata are in the brain, ascending neurons
    # come up from the nerve cord; flip the vertical axis so the brain ends up on top.
    kc = fams.get("Kenyon_Cell") or []
    an = fams.get("Nociceptor") or []
    flip = 1.0
    if kc and an:
        if np.mean([r["soma"][2] for r in kc]) > np.mean([r["soma"][2] for r in an]):
            flip = -1.0

    cursor: Dict[str, int] = {}
    coords, skeletons, types = [], [], []
    n_skel = 0
    for label, model_xyz in zip(topology.labels, topology.coords.tolist()):
        fam = _family_of(label or "")
        rows = fams.get(fam) if fam else None
        if not rows:
            coords.append(model_xyz); skeletons.append(None); types.append(None)
            continue
        i = cursor.get(fam, 0); cursor[fam] = i + 1
        row = rows[i % len(rows)]
        p = to_scene(row["soma"]); p[1] *= flip
        coords.append([round(p[0], 2), round(p[1], 2), round(p[2], 2)])
        types.append(row.get("type"))
        segs = row.get("segments")
        if segs and i < len(rows):  # a skeleton is drawn once, for its first assignee
            flat: List[float] = []
            for a, b in segs:
                pa, pb = to_scene(a), to_scene(b)
                flat += [round(pa[0], 1), round(pa[1] * flip, 1), round(pa[2], 1),
                         round(pb[0], 1), round(pb[1] * flip, 1), round(pb[2], 1)]
            skeletons.append(flat); n_skel += 1
        else:
            skeletons.append(None)
    return {"coords": coords, "skeletons": skeletons, "types": types,
            "skeleton_count": n_skel, "dataset": data.get("dataset")}


def set_topology(topology: CircuitTopology):
    """Cache the circuit topology for the 3D client."""
    global _topology_data
    # Downsample connections for efficient WebGL rendering (top strongest synapses)
    edge_pairs = []
    num_n = topology.num_neurons
    for pre in range(num_n):
        start = topology.ptr[pre]
        end = topology.ptr[pre + 1]
        for idx in range(start, end):
            post = int(topology.post[idx])
            weight = float(topology.weight[idx])
            # Only send significant connections to keep WebGL smooth
            if weight > 1.2 or (idx % 10 == 0):
                edge_pairs.append([pre, post, round(weight, 2)])

    # Population sizes, so the client can label the anatomy from the real circuit
    # instead of a hard-coded list that silently goes stale when the wiring changes.
    populations: Dict[str, int] = {}
    for label in topology.labels:
        populations[label] = populations.get(label, 0) + 1

    real = _map_to_malecns(topology)
    _topology_data = {
        "num_neurons": num_n,
        "coords": real["coords"] if real else topology.coords.tolist(),
        "real": bool(real),
        "dataset": real["dataset"] if real else None,
        "skeletons": real["skeletons"] if real else None,
        "real_types": real["types"] if real else None,
        "skeleton_count": real["skeleton_count"] if real else 0,
        "labels": topology.labels,
        "edges": edge_pairs[:4000],  # Top 4,000 synaptic pathways
        # Sign per neuron: the circuit is sign-constrained, and which cells inhibit is
        # the single most informative thing to see in the anatomy.
        "inhibitory": [int(s < 0) for s in topology.neuron_sign.tolist()],
        "action_channels": list(ACTION_CHANNELS),
        "populations": populations,
        "num_inhibitory": int((topology.neuron_sign < 0).sum()),
        "num_synapses": int(len(topology.post)),
        "num_plastic": int(topology.plastic_synapse_mask.sum()),
    }


def broadcast_event(data: Dict[str, Any]):
    """Broadcast real-time simulation frame to all connected browsers."""
    json_str = f"data: {json.dumps(data)}\n\n"
    with _subscriber_lock:
        dead = []
        for q in _subscriber_queues:
            try:
                q.put_nowait(json_str)
            except queue.Full:
                dead.append(q)
        for d in dead:
            _subscriber_queues.remove(d)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FlySoul: MaleCNS Connectome vs Dark Souls III</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    :root {
      --bg-dark: #06090e;
      --panel-bg: rgba(12, 17, 26, 0.82);
      --border: rgba(56, 189, 248, 0.25);
      --cyan: #00f0ff;
      --violet: #b300ff;
      --gold: #ffaa00;
      --crimson: #ff0055;
      --lime: #00ff66;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg-dark); color: #e2e8f0; overflow: hidden; }
    #canvas-container { width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; z-index: 1; }

    /* HUD Layout */
    .hud-panel {
      position: absolute;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    #header-panel {
      top: 16px; left: 16px; max-width: 420px;
    }
    #combat-panel {
      top: 16px; right: 16px; width: 360px;
    }
    #brain-panel {
      bottom: 16px; left: 16px; width: 440px;
    }
    #action-badge-panel {
      bottom: 24px; left: 50%; transform: translateX(-50%);
      text-align: center; padding: 12px 36px;
    }

    h1 { font-size: 18px; font-weight: 700; color: #f8fafc; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; }
    .badge {
      font-size: 10px; text-transform: uppercase; padding: 3px 8px; border-radius: 4px;
      background: rgba(56, 189, 248, 0.15); border: 1px solid var(--cyan); color: var(--cyan);
    }
    .subtext { font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.4; }

    /* Bars */
    .bar-group { margin-top: 12px; }
    .bar-label { display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; margin-bottom: 4px; }
    .bar-track { width: 100%; height: 10px; background: rgba(255, 255, 255, 0.08); border-radius: 6px; overflow: hidden; }
    .bar-fill { height: 100%; transition: width 0.15s ease-out; border-radius: 6px; }
    .hp-player { background: linear-gradient(90deg, #ef4444, #f87171); width: 100%; }
    .sp-player { background: linear-gradient(90deg, #10b981, #34d399); width: 100%; }
    .hp-boss { background: linear-gradient(90deg, #dc2626, #b91c1c); width: 100%; box-shadow: 0 0 10px rgba(220,38,38,0.5); }

    /* Action Badge */
    #action-title { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    #action-name {
      font-size: 24px; font-weight: 800; color: var(--lime); letter-spacing: 1.5px;
      text-shadow: 0 0 16px rgba(0, 255, 102, 0.6); margin-top: 2px;
    }

    /* Dopamine meter */
    .dopamine-container { display: flex; align-items: center; gap: 10px; margin-top: 12px; font-size: 12px; }
    .dopamine-meter { flex: 1; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; position: relative; overflow: hidden; }
    #dopamine-marker { width: 50%; height: 100%; background: var(--gold); transition: width 0.2s; }

    /* Legend */
    .legend { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px; margin-top: 10px; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

    /* Camera Instructions */
    #controls-hint {
      position: absolute; top: 80px; left: 50%; transform: translateX(-50%);
      font-size: 11px; color: rgba(255,255,255,0.4); pointer-events: none; z-index: 10;
    }
  </style>
</head>
<body>
  <div id="controls-hint">🖱️ Drag to rotate 3D fly brain • Scroll to zoom • Right-click to pan</div>

  <div id="header-panel" class="hud-panel">
    <h1>🪰 FlySoul: Connectome AI <span class="badge">MaleCNS v1.0</span></h1>
    <div class="subtext">
      Simulated <em>Drosophila melanogaster</em> brain (Google Research / HHMI Janelia) playing <strong>Dark Souls III</strong>.
    </div>
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background: var(--cyan)"></span> Retina (Compound Eye)</div>
      <div class="legend-item"><span class="dot" style="background: var(--violet)"></span> Central Complex / EPG</div>
      <div class="legend-item"><span class="dot" style="background: var(--gold)"></span> Kenyon Cells (MB)</div>
      <div class="legend-item"><span class="dot" style="background: var(--crimson)"></span> Dopamine / Nociceptors</div>
      <div class="legend-item"><span class="dot" style="background: var(--lime)"></span> Descending Motor Neurons</div>
    </div>
  </div>

  <div id="combat-panel" class="hud-panel">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span style="font-weight:700; font-size:13px; color:var(--cyan);">⚔️ COMBAT TELEMETRY</span>
      <span id="episode-badge" class="badge">EPISODE #1</span>
    </div>

    <!-- Boss Bar -->
    <div class="bar-group">
      <div class="bar-label">
        <span style="color:#f87171;">IUDEX GUNDYR</span>
        <span id="boss-hp-val">100%</span>
      </div>
      <div class="bar-track"><div id="boss-hp-fill" class="bar-fill hp-boss"></div></div>
    </div>

    <!-- Player Bar -->
    <div class="bar-group">
      <div class="bar-label">
        <span style="color:#38bdf8;">FLY SOUL (KNIGHT)</span>
        <span id="player-hp-val">100%</span>
      </div>
      <div class="bar-track"><div id="player-hp-fill" class="bar-fill hp-player"></div></div>
    </div>

    <!-- Stamina Bar -->
    <div class="bar-group">
      <div class="bar-label">
        <span style="color:#34d399;">STAMINA</span>
        <span id="player-sp-val">100%</span>
      </div>
      <div class="bar-track"><div id="player-sp-fill" class="bar-fill sp-player"></div></div>
    </div>

    <!-- Distance & Windup -->
    <div style="display:flex; justify-content:space-between; margin-top:14px; font-size:12px;">
      <div>Distance: <strong id="boss-dist-val" style="color:var(--cyan);">5.0m</strong></div>
      <div>Attack Windup: <strong id="boss-attack-val" style="color:#34d399;">IDLE</strong></div>
    </div>

    <!-- Dopamine -->
    <div class="dopamine-container">
      <span>Dopamine:</span>
      <div class="dopamine-meter"><div id="dopamine-marker"></div></div>
      <span id="dopamine-val" style="font-weight:700; color:var(--gold);">1.00x</span>
    </div>
  </div>

  <div id="brain-panel" class="hud-panel">
    <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:700;">
      <span>🧠 NEURAL CONNECTOME ACTIVITY</span>
      <span id="step-badge" style="color:var(--cyan);">Step: 0</span>
    </div>
    <div style="display:flex; gap:16px; margin-top:8px; font-size:12px;">
      <div>Active Neurons: <strong id="active-neurons-val" style="color:var(--cyan);">0</strong> / <span id="total-neurons-val">1,916</span></div>
      <div>Reward: <strong id="reward-val" style="color:var(--gold);">+0.00</strong></div>
    </div>
  </div>

  <div id="action-badge-panel" class="hud-panel">
    <div id="action-title">DECISION (MOTOR DECODER)</div>
    <div id="action-name">INITIALIZING...</div>
  </div>

  <div id="canvas-container"></div>

  <script>
    // 1. Initialize Three.js Scene
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x06090e, 0.0018);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 3000);
    camera.position.set(0, -220, 360);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;

    // Ambient & directional lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0x00f0ff, 1.2);
    dirLight.position.set(100, 200, 150);
    scene.add(dirLight);

    // Neuron particle representation
    let neuronParticles = null;
    let neuronColors = null;
    let defaultColors = [];
    let numNeurons = 0;

    // Fetch Topology & build 3D brain
    fetch('/api/topology')
      .then(res => res.json())
      .then(data => {
        numNeurons = data.num_neurons;
        document.getElementById('total-neurons-val').innerText = numNeurons.toLocaleString();
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(numNeurons * 3);
        const colors = new Float32Array(numNeurons * 3);

        for (let i = 0; i < numNeurons; i++) {
          positions[i * 3 + 0] = data.coords[i][0];
          positions[i * 3 + 1] = data.coords[i][1];
          positions[i * 3 + 2] = data.coords[i][2];

          const lbl = data.labels[i] || '';
          let c = new THREE.Color(0x00f0ff); // default retina cyan
          // Inhibitory populations are drawn cold so the excitation/inhibition balance
          // of the sign-constrained circuit is visible at a glance.
          if (lbl.includes('Inhibitory') || lbl.includes('APL')) c = new THREE.Color(0x2b6cff);
          else if (lbl.includes('Compass') || lbl.includes('Central')) c = new THREE.Color(0xb300ff);
          else if (lbl.includes('Kenyon') || lbl.includes('MBON')) c = new THREE.Color(0xffaa00);
          else if (lbl.includes('Dopamine') || lbl.includes('Nociceptor')) c = new THREE.Color(0xff0055);
          else if (lbl.includes('Proprioceptor')) c = new THREE.Color(0xff8adf);
          else if (lbl.includes('Motor')) c = new THREE.Color(0x00ff66);

          colors[i * 3 + 0] = c.r;
          colors[i * 3 + 1] = c.g;
          colors[i * 3 + 2] = c.b;

          defaultColors.push([c.r, c.g, c.b]);
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Glow particle texture
        const canvas = document.createElement('canvas');
        canvas.width = 64; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        grad.addColorStop(0, 'rgba(255,255,255,1)');
        grad.addColorStop(0.3, 'rgba(0,240,255,0.8)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);
        const pTexture = new THREE.CanvasTexture(canvas);

        const material = new THREE.PointsMaterial({
          size: 7.5,
          vertexColors: true,
          map: pTexture,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });

        neuronParticles = new THREE.Points(geometry, material);
        scene.add(neuronParticles);
        neuronColors = neuronParticles.geometry.attributes.color;

        // Draw Synaptic Filaments
        const lineGeo = new THREE.BufferGeometry();
        const linePos = [];
        const edges = data.edges || [];
        for (let e of edges) {
          const u = e[0], v = e[1];
          linePos.push(
            data.coords[u][0], data.coords[u][1], data.coords[u][2],
            data.coords[v][0], data.coords[v][1], data.coords[v][2]
          );
        }
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x1e3a8a,
          transparent: true,
          opacity: 0.15,
          blending: THREE.AdditiveBlending,
        });
        const filaments = new THREE.LineSegments(lineGeo, lineMat);
        scene.add(filaments);
      });

    // 2. Real-Time Telemetry Stream via Server-Sent Events (SSE)
    const evtSource = new EventSource('/api/stream');
    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);

      // Update Combat HUD
      document.getElementById('episode-badge').innerText = `EPISODE #${data.episode}`;
      document.getElementById('step-badge').innerText = `Step: ${data.step}`;
      document.getElementById('active-neurons-val').innerText = (data.active_neurons || 0).toLocaleString();
      document.getElementById('reward-val').innerText = (data.cumulative_reward >= 0 ? '+' : '') + data.cumulative_reward.toFixed(2);

      const p_hp = Math.round(data.player_hp * 100);
      const b_hp = Math.round(data.boss_hp * 100);
      const p_sp = Math.round(data.player_sp * 100);

      document.getElementById('player-hp-val').innerText = `${p_hp}%`;
      document.getElementById('player-hp-fill').style.width = `${p_hp}%`;
      document.getElementById('boss-hp-val').innerText = `${b_hp}%`;
      document.getElementById('boss-hp-fill').style.width = `${b_hp}%`;
      document.getElementById('player-sp-val').innerText = `${p_sp}%`;
      document.getElementById('player-sp-fill').style.width = `${p_sp}%`;

      document.getElementById('boss-dist-val').innerText = `${data.boss_distance.toFixed(1)}m`;
      const atkElem = document.getElementById('boss-attack-val');
      if (data.boss_attacking) {
        atkElem.innerText = '⚠️ ATTACKING!';
        atkElem.style.color = '#ef4444';
      } else {
        atkElem.innerText = 'IDLE';
        atkElem.style.color = '#34d399';
      }

      // Dopamine
      const dop = data.dopamine || 1.0;
      document.getElementById('dopamine-val').innerText = `${dop.toFixed(2)}x`;
      document.getElementById('dopamine-marker').style.width = `${Math.min(100, dop * 50)}%`;

      // Action Badge
      const actionName = data.action_name || 'IDLE';
      const actionElem = document.getElementById('action-name');
      actionElem.innerText = actionName.toUpperCase().replace('_', ' ');
      if (actionName.includes('roll')) actionElem.style.color = '#00f0ff';
      else if (actionName.includes('attack')) actionElem.style.color = '#ffaa00';
      else if (actionName.includes('parry')) actionElem.style.color = '#b300ff';
      else actionElem.style.color = '#00ff66';

      // Flash Spiking Neurons in 3D
      if (neuronParticles && neuronColors && data.spikes && data.spikes.length > 0) {
        const spikeSet = new Set(data.spikes);
        for (let i = 0; i < numNeurons; i++) {
          if (spikeSet.has(i)) {
            // Flash pure white/bright bloom
            neuronColors.array[i * 3 + 0] = 1.0;
            neuronColors.array[i * 3 + 1] = 1.0;
            neuronColors.array[i * 3 + 2] = 1.0;
          } else {
            // Slowly decay back to base color
            neuronColors.array[i * 3 + 0] = defaultColors[i][0] * 0.85;
            neuronColors.array[i * 3 + 1] = defaultColors[i][1] * 0.85;
            neuronColors.array[i * 3 + 2] = defaultColors[i][2] * 0.85;
          }
        }
        neuronColors.needsUpdate = true;
      }
    };

    // Render loop
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  </script>
</body>
</html>
"""


class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            # A viewer closed the tab mid-response. Nothing to do and nothing to log:
            # with a public link this happens many times an hour.
            return

    def _route(self):
        # Cache-busting query strings (frame.jpg?t=...) are not part of the route.
        self.path = self.path.split("?", 1)[0]
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if HTML_PATH.exists():
                try:
                    content = HTML_PATH.read_text(encoding="utf-8")
                except Exception:
                    content = INDEX_HTML
            else:
                content = INDEX_HTML
            self.wfile.write(content.encode("utf-8"))

        elif self.path == "/api/topology":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps(_topology_data or {})
            self.wfile.write(resp.encode("utf-8"))

        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q: queue.Queue = queue.Queue(maxsize=100)
            with _subscriber_lock:
                _subscriber_queues.append(q)

            try:
                while True:
                    msg = q.get()
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                with _subscriber_lock:
                    if q in _subscriber_queues:
                        _subscriber_queues.remove(q)

        elif self.path == "/api/frame.jpg":
            frame = get_latest_frame()
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(204)
                self.end_headers()

        elif self.path == "/api/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    frame = get_latest_frame()
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.066)  # ~15 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


class _QuietServer(ThreadingHTTPServer):
    """A viewer dropping a connection is not an error worth a traceback in the fight log."""

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def start_visualizer(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """Start the 3D fruit fly brain visualizer server in a background daemon thread."""
    server = _QuietServer((host, port), VisualizerHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server
