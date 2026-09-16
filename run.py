"""FlySoul: Fruit Fly Connectome (MaleCNS v1.0) × Dark Souls 3 (SoulsGym).

Main execution loop.
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.live import Live

import cv2

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.souls_wrapper import make_souls_env
from flysoul.motor.decoder import MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder
from flysoul.telemetry.dashboard import FlySoulDashboard
from flysoul.visualizer import (
    start_visualizer,
    set_topology,
    broadcast_event,
    set_latest_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlySoul: MaleCNS Fruit Fly Connectome plays Dark Souls III."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use standalone Mock Iudex Gundyr simulation (default: True).",
    )
    parser.add_argument(
        "--game",
        action="store_true",
        help="Connect to live Dark Souls III via SoulsGym (requires game running).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of combat episodes to run.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run combat indefinitely in a continuous loop until stopped.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable interactive terminal dashboard and use standard console output.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable Boltzmann exploratory sampling for action selection.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the real-time 3D WebGL fruit fly brain visualizer.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port for the 3D connectome visualizer (default: 8080).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for circuit construction.",
    )
    return parser.parse_args()


def extract_combat_metrics(obs, info):
    """Extract normalized telemetry values from SoulsGym or Mock environment observations."""
    def _safe_float(v, default=0.0) -> float:
        if v is None:
            return default
        if isinstance(v, (np.ndarray, list)):
            return float(v[0]) if len(v) > 0 else default
        return float(v)

    if isinstance(obs, dict) and "player_pose" in obs and "boss_pose" in obs:
        p_pose = np.asarray(obs["player_pose"])
        b_pose = np.asarray(obs["boss_pose"])
        distance = float(np.hypot(b_pose[0] - p_pose[0], b_pose[1] - p_pose[1]))
        p_max = max(1.0, _safe_float(obs.get("player_max_hp"), 1000.0))
        b_max = max(1.0, _safe_float(obs.get("boss_max_hp"), 1037.0))
        player_hp_pct = float(np.clip(_safe_float(obs.get("player_hp")) / p_max, 0.0, 1.0))
        player_sp_pct = float(np.clip(_safe_float(obs.get("player_sp"), 100.0) / 100.0, 0.0, 1.0))
        boss_hp_pct = float(np.clip(_safe_float(obs.get("boss_hp")) / b_max, 0.0, 1.0))
        boss_attacking = bool(obs.get("boss_animation", -1) > 0 or obs.get("boss_attacking", False))
    elif isinstance(obs, dict):
        player_hp_pct = float(np.clip(_safe_float(obs.get("player_hp"), 1.0), 0.0, 1.0))
        player_sp_pct = float(np.clip(_safe_float(obs.get("player_sp"), 1.0), 0.0, 1.0))
        boss_hp_pct = float(np.clip(_safe_float(obs.get("boss_hp"), 1.0), 0.0, 1.0))
        distance = _safe_float(obs.get("boss_distance"), 5.0)
        boss_attacking = bool(obs.get("boss_attacking", False))
    else:
        player_hp_pct, player_sp_pct, boss_hp_pct, distance, boss_attacking = 1.0, 1.0, 1.0, 5.0, False

    boss_state = info.get("boss_state", "attacking" if boss_attacking else "active")
    return player_hp_pct, player_sp_pct, boss_hp_pct, distance, boss_attacking, boss_state


def main():
    args = parse_args()
    console = Console(legacy_windows=False)

    console.print(
        "[bold cyan]🪰 Initializing FlySoul: MaleCNS v1.0 Fruit Fly Connectome Engine...[/bold cyan]"
    )

    # 1. Build connectome topology & neural engine
    circuit_cfg = CircuitConfig()
    biophys_cfg = BioPhysicsConfig()

    t0 = time.perf_counter()
    topology = build_fly_circuit(circuit_cfg, seed=args.seed)
    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, biophys_cfg)
    plasticity = DopaminePlasticity(topology, learning_rate=circuit_cfg.learning_rate)
    encoder = SensoryEncoder(topology)
    decoder = MotorDecoder(topology, threshold=1.0)
    t_build = (time.perf_counter() - t0) * 1000.0

    console.print(
        f"[green]✓ Connectome built in {t_build:.1f}ms:[/green] "
        f"[bold]{topology.num_neurons:,}[/bold] neurons, "
        f"[bold]{len(topology.post):,}[/bold] synapses, "
        f"[bold]{int(np.sum(topology.plastic_synapse_mask)):,}[/bold] plastic KC->MBON connections."
    )

    # 2. Start Live 3D Fruit Fly Brain Visualizer
    enable_web = not args.no_web
    if enable_web:
        set_topology(topology)
        try:
            start_visualizer(host="0.0.0.0", port=args.port)
            console.print(
                f"[bold green]🌐 Live 3D Fruit Fly Brain Visualizer active at: [/bold green]"
                f"[bold yellow underline]http://localhost:{args.port}[/bold yellow underline]"
            )
        except Exception as e:
            console.print(f"[yellow]Could not bind web visualizer port {args.port}: {e}[/yellow]")

    # 3. Initialize environment
    use_mock = not args.game
    env = make_souls_env(use_mock=use_mock)
    dashboard = FlySoulDashboard(console)

    victories = 0
    total_episodes = 999999 if args.continuous else args.episodes
    if args.continuous:
        console.print("[bold cyan]🔄 Running in continuous combat loop. Press Ctrl+C to stop.[/bold cyan]")

    episode_history = []
    global_action_counts = {}
    ep = 0
    while ep < total_episodes:
        ep += 1
        obs, _ = env.reset()
        engine.reset_state()
        encoder.reset()
        plasticity.reset()

        ep_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        if not args.no_dashboard:
            with Live(console=console, refresh_per_second=20, transient=True) as live:
                while not (terminated or truncated):
                    step += 1

                    # 1. Encode combat state into sensory current
                    drive = encoder.encode(obs)

                    # 2. Advance connectome dynamics (1 frame = 16.67 ms)
                    spike_counts = engine.step(drive, duration_ms=16.67)

                    # Extract pre-step combat state for context-aware motor decoding
                    player_hp_pct, player_sp_pct, boss_hp_pct, distance, boss_attacking, boss_state = extract_combat_metrics(obs, info if 'info' in locals() else {})

                    # 3. Decode descending motor spikes into game action
                    action_id, action_name, motor_rates = decoder.decode(
                        spike_counts, explore=args.explore, distance=distance, boss_attacking=boss_attacking
                    )

                    # 4. Step game environment (translate to SoulsGym discrete actions if live)
                    step_action = decoder.to_soulsgym_action(action_id) if args.game else action_id
                    obs, reward, terminated, truncated, info = env.step(step_action)
                    ep_reward += reward

                    # 5. Dopaminergic plasticity update (Three-factor learning)
                    plasticity.update_traces(spike_counts)
                    plasticity.apply_reinforcement(reward)

                    # 6. Extract telemetry & broadcast to 3D Web Visualizer
                    active_count = int(np.count_nonzero(spike_counts))
                    player_hp_pct, player_sp_pct, boss_hp_pct, distance, boss_attacking, boss_state = extract_combat_metrics(obs, info)
                    global_action_counts[action_name] = global_action_counts.get(action_name, 0) + 1

                    if enable_web:
                        # Grab and stream video frame from SoulsGym
                        if step % 3 == 0:
                            try:
                                raw = env.unwrapped.game._game_window.raw_img
                                if raw is not None and raw.size > 0:
                                    bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                                    resized = cv2.resize(bgr, (360, 202), interpolation=cv2.INTER_LINEAR)
                                    ret, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
                                    if ret:
                                        set_latest_frame(encoded.tobytes())
                            except Exception:
                                pass

                        spikes = np.where(spike_counts > 0)[0].tolist()
                        broadcast_event({
                            "episode": ep,
                            "step": step,
                            "player_hp": player_hp_pct,
                            "player_sp": player_sp_pct,
                            "boss_hp": boss_hp_pct,
                            "boss_distance": distance,
                            "boss_attacking": boss_attacking,
                            "action_name": action_name,
                            "dopamine": float(plasticity.dopamine_level),
                            "active_neurons": active_count,
                            "cumulative_reward": ep_reward,
                            "spikes": spikes,
                            "mean_weight": round(float(plasticity.mean_plastic_weight), 3),
                            "total_ltp": plasticity.total_ltp_events,
                            "total_ltd": plasticity.total_ltd_events,
                            "action_counts": global_action_counts,
                            "history": episode_history[-20:],
                        })

                    layout = dashboard.create_layout(
                        episode=ep,
                        step=step,
                        player_hp_pct=player_hp_pct,
                        player_sp_pct=player_sp_pct,
                        boss_hp_pct=boss_hp_pct,
                        distance=distance,
                        boss_state=boss_state,
                        active_neurons=active_count,
                        total_neurons=topology.num_neurons,
                        dopamine=plasticity.dopamine_level,
                        action_name=action_name,
                        motor_rates=motor_rates,
                        cumulative_reward=ep_reward,
                    )
                    live.update(layout)
                    time.sleep(0.015)  # Real-time pacing

        else:
            # Headless logging mode
            while not (terminated or truncated):
                step += 1
                drive = encoder.encode(obs)
                spike_counts = engine.step(drive, duration_ms=16.67)

                pre_hp_pct, pre_sp_pct, pre_boss_hp_pct, distance, boss_attacking, boss_state = extract_combat_metrics(obs, info if 'info' in locals() else {})
                action_id, action_name, motor_rates = decoder.decode(
                    spike_counts, explore=args.explore, distance=distance, boss_attacking=boss_attacking
                )
                global_action_counts[action_name] = global_action_counts.get(action_name, 0) + 1
                step_action = decoder.to_soulsgym_action(action_id) if args.game else action_id
                obs, reward, terminated, truncated, info = env.step(step_action)
                ep_reward += reward
                plasticity.update_traces(spike_counts)
                plasticity.apply_reinforcement(reward)

                active_count = int(np.count_nonzero(spike_counts))
                player_hp_pct, player_sp_pct, boss_hp_pct, distance, boss_attacking, boss_state = extract_combat_metrics(obs, info)

                if enable_web:
                    # Stream video frame from SoulsGym
                    if step % 3 == 0:
                        try:
                            raw = env.unwrapped.game._game_window.raw_img
                            if raw is not None and raw.size > 0:
                                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                                resized = cv2.resize(bgr, (360, 202), interpolation=cv2.INTER_LINEAR)
                                ret, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
                                if ret:
                                    set_latest_frame(encoded.tobytes())
                        except Exception:
                            pass

                    spikes = np.where(spike_counts > 0)[0].tolist()
                    broadcast_event({
                        "episode": ep,
                        "step": step,
                        "player_hp": player_hp_pct,
                        "player_sp": player_sp_pct,
                        "boss_hp": boss_hp_pct,
                        "boss_distance": distance,
                        "boss_attacking": boss_attacking,
                        "action_name": action_name,
                        "dopamine": float(plasticity.dopamine_level),
                        "active_neurons": active_count,
                        "cumulative_reward": ep_reward,
                        "spikes": spikes,
                        "mean_weight": round(float(plasticity.mean_plastic_weight), 3),
                        "total_ltp": plasticity.total_ltp_events,
                        "total_ltd": plasticity.total_ltd_events,
                        "action_counts": global_action_counts,
                        "history": episode_history[-20:],
                    })
                if step % 50 == 0:
                    console.print(
                        f"  [Fly] Step {step:4d} | Action: {action_name:<14} | Dist: {distance:.1f}m | "
                        f"HP: {int(player_hp_pct*100)}% | Boss: {int(boss_hp_pct*100)}% | DA: {plasticity.dopamine_level:.2f}"
                    )
        # Episode summary
        boss_hp_raw = info.get("boss_hp_raw", int(boss_hp_pct * 1037) if 'boss_hp_pct' in locals() else 0)
        player_hp_raw = info.get("player_hp_raw", int(player_hp_pct * 1000) if 'player_hp_pct' in locals() else 0)

        ep_summary = {
            "episode": ep,
            "steps": step,
            "reward": round(ep_reward, 2),
            "boss_hp_pct": round(boss_hp_pct, 3),
            "player_hp_pct": round(player_hp_pct, 3),
            "victory": boss_hp_raw <= 0,
            "mean_weight": round(float(plasticity.mean_plastic_weight), 3),
        }
        episode_history.append(ep_summary)

        if enable_web:
            broadcast_event({
                "type": "episode_summary",
                "episode": ep,
                "steps": step,
                "reward": ep_reward,
                "history": episode_history[-25:],
                "victories": victories,
                "mean_weight": float(plasticity.mean_plastic_weight),
                "total_ltp": plasticity.total_ltp_events,
                "total_ltd": plasticity.total_ltd_events,
                "action_counts": global_action_counts,
            })

        if boss_hp_raw <= 0:
            victories += 1
            console.print(
                f"[bold gold1]🏆 EPISODE #{ep} VICTORY! HEIR OF FIRE DESTROYED![/bold gold1] "
                f"Surviving Fly HP: {player_hp_raw} | Steps: {step} | Reward: {ep_reward:+.2f}"
            )
        else:
            console.print(
                f"[bold red]💀 EPISODE #{ep} YOU DIED.[/bold red] "
                f"Remaining Boss HP: {boss_hp_raw} | Steps: {step} | Reward: {ep_reward:+.2f}"
            )

    console.print(
        f"\n[bold cyan]🏁 Simulation Finished.[/bold cyan] "
        f"Victories: [bold green]{victories}/{total_episodes}[/bold green] "
        f"({(victories/total_episodes)*100:.1f}%)"
    )
    env.close()


if __name__ == "__main__":
    main()
