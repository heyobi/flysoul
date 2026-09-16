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

from flysoul.config import BioPhysicsConfig, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.souls_wrapper import make_souls_env
from flysoul.motor.decoder import MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder
from flysoul.telemetry.dashboard import FlySoulDashboard


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for circuit construction.",
    )
    return parser.parse_args()


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

    # 2. Initialize environment
    use_mock = not args.game
    env = make_souls_env(use_mock=use_mock)
    dashboard = FlySoulDashboard(console)

    victories = 0
    total_episodes = args.episodes

    for ep in range(1, total_episodes + 1):
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

                    # 3. Decode descending motor spikes into game action
                    action_id, action_name, motor_rates = decoder.decode(
                        spike_counts, explore=args.explore
                    )

                    # 4. Step game environment (translate to SoulsGym discrete actions if live)
                    step_action = decoder.to_soulsgym_action(action_id) if args.game else action_id
                    obs, reward, terminated, truncated, info = env.step(step_action)
                    ep_reward += reward

                    # 5. Dopaminergic plasticity update (Three-factor learning)
                    plasticity.update_traces(spike_counts)
                    plasticity.apply_reinforcement(reward)

                    # 6. Update visual telemetry
                    active_count = int(np.count_nonzero(spike_counts))

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
                        player_hp_pct = _safe_float(obs.get("player_hp")) / p_max
                        player_sp_pct = _safe_float(obs.get("player_sp"), 100.0) / 100.0
                        boss_hp_pct = _safe_float(obs.get("boss_hp")) / b_max
                    elif isinstance(obs, dict):
                        player_hp_pct = _safe_float(obs.get("player_hp"), 1.0)
                        player_sp_pct = _safe_float(obs.get("player_sp"), 1.0)
                        boss_hp_pct = _safe_float(obs.get("boss_hp"), 1.0)
                        distance = _safe_float(obs.get("boss_distance"), 5.0)
                    else:
                        player_hp_pct, player_sp_pct, boss_hp_pct, distance = 1.0, 1.0, 1.0, 5.0

                    boss_state = info.get("boss_state", "active")

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
                action_id, action_name, motor_rates = decoder.decode(
                    spike_counts, explore=args.explore
                )
                step_action = decoder.to_soulsgym_action(action_id) if args.game else action_id
                obs, reward, terminated, truncated, info = env.step(step_action)
                ep_reward += reward
                plasticity.update_traces(spike_counts)
                plasticity.apply_reinforcement(reward)

        # Episode summary
        boss_hp_raw = info.get("boss_hp_raw", int(boss_hp_pct * 1037) if 'boss_hp_pct' in locals() else 0)
        player_hp_raw = info.get("player_hp_raw", int(player_hp_pct * 1000) if 'player_hp_pct' in locals() else 0)

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
