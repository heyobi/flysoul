"""Live WebGL 3D Fruit Fly Brain Visualizer Server for FlySoul.

Serves an interactive 3D WebGL visualization of the MaleCNS fruit fly connectome
with real-time synaptic spiking, dopamine cascades, and Dark Souls III combat HUD.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

HTML_PATH = Path(__file__).parent / "index.html"

from flysoul.connectome.graph import CircuitTopology

# Global event broadcast queue for Server-Sent Events (SSE)
_subscriber_queues: List[queue.Queue] = []
_subscriber_lock = threading.Lock()
_topology_data: Optional[Dict[str, Any]] = None


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

    _topology_data = {
        "num_neurons": num_n,
        "coords": topology.coords.tolist(),
        "labels": topology.labels,
        "edges": edge_pairs[:4000],  # Top 4,000 synaptic pathways
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
          if (lbl.includes('Compass') || lbl.includes('Central')) c = new THREE.Color(0xb300ff);
          else if (lbl.includes('Kenyon') || lbl.includes('MBON')) c = new THREE.Color(0xffaa00);
          else if (lbl.includes('Dopamine') || lbl.includes('Nociceptor')) c = new THREE.Color(0xff0055);
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

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


def start_visualizer(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """Start the 3D fruit fly brain visualizer server in a background daemon thread."""
    server = ThreadingHTTPServer((host, port), VisualizerHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server
