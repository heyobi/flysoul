"""FlySoul: Fruit Fly Connectome (MaleCNS v1.0) x Dark Souls 3 (SoulsGym).

Main execution loop.

The loop's job is to move information, not to decide anything: encode the combat state
into sensory drive, advance the connectome, read out whichever descending pool won, map
that to a game action, and deliver the reward as dopamine. Every behavioural choice is
made inside the circuit.
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

from flysoul.config import CHECKPOINT_DIR, BioPhysicsConfig, CircuitConfig
from flysoul.connectome.calibration import _topology_fingerprint, calibrate_or_load
from flysoul.connectome.engine import ConnectomeEngine
from flysoul.connectome.graph import build_fly_circuit
from flysoul.connectome.plasticity import DopaminePlasticity
from flysoul.env.obs import parse_obs
from flysoul.env.souls_wrapper import (
    BINDING_HELP,
    LiveEnvUnavailable,
    check_input_bindings,
    is_mock,
    make_souls_env,
)
from flysoul.motor.decoder import SOULSGYM_IDLE, MotorDecoder
from flysoul.sensory.encoder import SensoryEncoder
from flysoul.telemetry.dashboard import FlySoulDashboard
from flysoul.visualizer import (
    broadcast_event,
    set_latest_frame,
    set_topology,
    start_visualizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlySoul: MaleCNS Fruit Fly Connectome plays Dark Souls III."
    )
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use standalone Mock Iudex Gundyr simulation (default: True).")
    parser.add_argument("--game", action="store_true",
                        help="Connect to live Dark Souls III via SoulsGym (requires game running).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of combat episodes to run.")
    parser.add_argument("--continuous", action="store_true",
                        help="Run combat indefinitely in a continuous loop until stopped.")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable interactive terminal dashboard and use standard console output.")
    parser.add_argument("--explore", action="store_true",
                        help="Enable Boltzmann exploratory sampling over the descending pools.")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable the real-time 3D WebGL fruit fly brain visualizer.")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port for the 3D connectome visualizer (default: 8080).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for circuit construction.")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Ignore the cached calibration and re-run it from scratch.")
    parser.add_argument("--no-learning", action="store_true",
                        help="Freeze the KC->MBON synapses (evaluate the innate circuit only).")
    parser.add_argument("--no-memory", action="store_true",
                        help="Do not load or save learned synapses; start from the innate circuit.")
    parser.add_argument("--skip-input-check", action="store_true",
                        help="Skip the pre-flight check that the attack key reaches the game.")
    parser.add_argument("--allow-mock-fallback", action="store_true",
                        help="Allow --game to silently fall back to the offline simulator.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the per-episode summary lines.")
    return parser.parse_args()


def release_all_keys():
    """Release every movement and combat key in X11 so the player never gets stuck."""
    try:
        import ctypes
        x11 = ctypes.CDLL("libX11.so.6")
        xtest = ctypes.CDLL("libXtst.so.6")
        disp = x11.XOpenDisplay(None)
        if disp:
            key_names = ["w", "a", "s", "d", "q", "e", "r", "f", "c", "v", "x", "z", "g",
                         "space", "Shift_L", "Control_L", "Alt_L"]
            for k in key_names:
                sym = x11.XStringToKeysym(k.encode("ascii"))
                if sym != 0:
                    kc = x11.XKeysymToKeycode(disp, sym)
                    if kc != 0:
                        xtest.XTestFakeKeyEvent(disp, kc, False, 0)
            x11.XFlush(disp)
            x11.XCloseDisplay(disp)
    except Exception:
        pass


def grab_video_frame(env):
    """Pull the latest game frame out of SoulsGym for the web visualizer, if available."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        raw = env.unwrapped.game._game_window.raw_img
        if raw is None or raw.size == 0:
            return None
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (360, 202), interpolation=cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return encoded.tobytes() if ok else None
    except Exception:
        return None


def build_agent(args, console):
    """Construct the connectome, calibrate it, and wire up the sensory/motor interface."""
    circuit_cfg = CircuitConfig()
    biophys_cfg = BioPhysicsConfig()

    t0 = time.perf_counter()
    topology = build_fly_circuit(circuit_cfg, seed=args.seed)
    t_build = (time.perf_counter() - t0) * 1000.0

    console.print(
        f"[green]OK Connectome built in {t_build:.1f}ms:[/green] "
        f"[bold]{topology.num_neurons:,}[/bold] neurons, "
        f"[bold]{len(topology.post):,}[/bold] synapses "
        f"([bold]{int(np.sum(topology.weight < 0)):,}[/bold] inhibitory), "
        f"[bold]{int(np.sum(topology.plastic_synapse_mask)):,}[/bold] plastic KC->MBON connections."
    )

    t0 = time.perf_counter()
    report, cached = calibrate_or_load(
        topology, circuit_cfg, biophys_cfg, seed=args.seed, force=args.recalibrate
    )
    source = "cached" if cached else f"{time.perf_counter() - t0:.1f}s"
    console.print(f"[green]OK Homeostatic calibration ({source}):[/green] {report.format()}")
    if not report.converged and not cached:
        console.print(
            "[yellow]WARNING calibration did not reach all targets; the circuit may be "
            "over- or under-excitable.[/yellow]"
        )

    engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, biophys_cfg)
    encoder = SensoryEncoder(topology, biophys_cfg)
    decoder = MotorDecoder(topology, step_ms=100.0)
    plasticity = DopaminePlasticity(
        topology,
        learning_rate=circuit_cfg.learning_rate,
        eligibility_decay=circuit_cfg.eligibility_decay,
    )

    # Learned synapses persist across runs, keyed by the wiring they were learned on.
    memory_path = CHECKPOINT_DIR / f"learned_{_topology_fingerprint(topology)}.npz"
    if args.no_memory:
        console.print("[yellow]Starting from the innate circuit (--no-memory).[/yellow]")
    elif plasticity.load(memory_path):
        console.print(
            f"[green]OK Restored learned synapses[/green] from {memory_path.name} "
            f"(LTP {plasticity.total_ltp_events:,} / LTD {plasticity.total_ltd_events:,}, "
            f"reward baseline {plasticity.reward_baseline:+.2f})"
        )
    else:
        console.print("No saved synapses yet; the fly starts from its innate circuit.")
    return topology, engine, encoder, decoder, plasticity, memory_path


def main():
    args = parse_args()
    console = Console(legacy_windows=False)
    console.print("[bold cyan]Initializing FlySoul: MaleCNS v1.0 Fruit Fly Connectome Engine...[/bold cyan]")

    topology, engine, encoder, decoder, plasticity, memory_path = build_agent(args, console)

    enable_web = not args.no_web
    if enable_web:
        set_topology(topology)
        try:
            start_visualizer(host="0.0.0.0", port=args.port)
            console.print(
                f"[bold green]Live 3D Fruit Fly Brain Visualizer active at: [/bold green]"
                f"[bold yellow underline]http://localhost:{args.port}[/bold yellow underline]"
            )
        except Exception as e:
            console.print(f"[yellow]Could not bind web visualizer port {args.port}: {e}[/yellow]")

    use_mock = not args.game
    try:
        env = make_souls_env(use_mock=use_mock, allow_fallback=args.allow_mock_fallback)
    except LiveEnvUnavailable as e:
        console.print(f"[bold red]{e}[/bold red]")
        console.print(
            "[yellow]Refusing to run the offline simulator under --game. Check that "
            "Dark Souls III is running and that soulsgym imports cleanly:[/yellow]\n"
            "  python -c 'import soulsgym'\n"
            "[yellow]Pass --allow-mock-fallback if you really want the simulator.[/yellow]"
        )
        return
    mock_mode = is_mock(env)
    if args.game and mock_mode:
        console.print("[bold red]Refusing to report simulator results as live play.[/bold red]")
        return

    if not mock_mode and not args.skip_input_check:
        console.print("Checking that the game responds to the attack key ...")
        if check_input_bindings(env):
            console.print("[green]OK attack input reaches the game.[/green]")
        else:
            console.print("[bold red]The attack key does nothing in game.[/bold red]")
            console.print(BINDING_HELP)
            env.close()
            return
    dashboard = FlySoulDashboard(console)

    victories = 0
    total_episodes = 999999 if args.continuous else args.episodes
    if args.continuous:
        console.print("[bold cyan]Running in continuous combat loop. Press Ctrl+C to stop.[/bold cyan]")

    episode_history = []
    global_action_counts: dict[str, int] = {}

    try:
        for ep in range(1, total_episodes + 1):
            release_all_keys()
            obs, info, env = reset_episode(env, console, use_mock)
            if not use_mock and not ensure_lock_on(env, console):
                console.print("[yellow]Starting the episode without lock-on.[/yellow]")
            engine.reset_state()
            encoder.reset()
            plasticity.reset()
            settle_connectome(engine, encoder, obs, info)

            summary = run_episode(
                ep=ep,
                env=env,
                mock_mode=mock_mode,
                engine=engine,
                encoder=encoder,
                decoder=decoder,
                plasticity=plasticity,
                topology=topology,
                obs=obs,
                info=info,
                args=args,
                console=console,
                dashboard=dashboard,
                enable_web=enable_web,
                episode_history=episode_history,
                global_action_counts=global_action_counts,
            )
            episode_history.append(summary)
            if summary["victory"]:
                victories += 1
            if not args.no_memory:
                plasticity.save(memory_path)

            if enable_web:
                broadcast_event({
                    "type": "episode_summary", "episode": ep, "steps": summary["steps"],
                    "reward": summary["reward"], "history": episode_history[-25:],
                    "victories": victories, "mean_weight": float(plasticity.mean_plastic_weight),
                    "total_ltp": plasticity.total_ltp_events, "total_ltd": plasticity.total_ltd_events,
                    "action_counts": global_action_counts,
                })

            if summary["victory"]:
                console.print(
                    f"[bold gold1]EPISODE #{ep} VICTORY - HEIR OF FIRE DESTROYED![/bold gold1] "
                    f"Fly HP left: {summary['player_hp_pct']*100:.0f}% | "
                    f"Steps: {summary['steps']} | Reward: {summary['reward']:+.2f}"
                )
            else:
                total = max(1, sum(summary["actions"].values()))
                mix = " ".join(
                    f"{k[:5]}:{100 * v // total}%"
                    for k, v in sorted(summary["actions"].items(), key=lambda kv: -kv[1])[:4]
                )
                console.print(
                    f"[bold red]EPISODE #{ep} YOU DIED.[/bold red] "
                    f"Boss HP left: {summary['boss_hp_pct']*100:.0f}% | "
                    f"Steps: {summary['steps']} | Reward: {summary['reward']:+.2f} | "
                    f"Hits landed: {summary['hits']} | {mix}"
                )
            release_all_keys()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    finally:
        release_all_keys()
        try:
            env.close()
        except Exception:
            pass

    episodes_run = max(1, len(episode_history))
    console.print(
        f"\n[bold cyan]Simulation Finished.[/bold cyan] "
        f"Victories: [bold green]{victories}/{episodes_run}[/bold green] "
        f"({(victories / episodes_run) * 100:.1f}%)"
    )


def ensure_lock_on(env, console, seconds: float = 10.0):
    """Do not let the fly move until the camera is locked on to Iudex.

    SoulsGym's movement actions are frame-relative: **with** lock-on, action 0 walks
    towards the boss and attacks track it; **without** it, action 0 walks wherever the
    camera happens to be pointing. The action space has no camera control at all, so an
    unlocked fly cannot steer - it can only sprint off in whatever direction the camera
    was left in, which is exactly what it looks like from the outside.

    Establishing the lock takes several steps by design: SoulsEnv._lock_on first nudges
    the camera towards the boss with cameraleft/right/up/down and only presses the lock
    button once the camera is within about 37 degrees, and those presses are queued for
    the following step. A short timeout around that loop gives up mid-way and hands back
    an unlocked camera.

    Returns True if the camera is locked on when this returns.
    """
    unwrapped = env.unwrapped
    game = getattr(unwrapped, "game", None)
    if game is None:
        return True  # Mock environment: no camera to align.

    camera_reset = getattr(unwrapped, "_camera_reset", None)
    deadline = time.time() + seconds
    attempts = 0
    while time.time() < deadline:
        try:
            if bool(game.lock_on):
                return True
        except Exception:
            return True  # Cannot read the flag; let the episode proceed.
        attempts += 1
        if camera_reset is not None:
            try:
                camera_reset()
                continue
            except Exception:
                camera_reset = None
        # Fallback: idle steps still let the environment's own routine turn the camera.
        try:
            _, _, terminated, truncated, _ = env.step(SOULSGYM_IDLE)
            if terminated or truncated:
                return bool(game.lock_on)
        except Exception:
            break

    locked = False
    try:
        locked = bool(game.lock_on)
    except Exception:
        pass
    if not locked:
        console.print(
            f"[bold yellow]WARNING could not establish lock-on after {attempts} attempts "
            f"({seconds:.0f}s).[/bold yellow] The fly will hold position rather than run "
            "camera-relative. If this keeps happening, raise the camera timeout patched "
            "into soulsgym's _camera_reset."
        )
    return locked


def settle_connectome(engine, encoder, obs, info, seconds: float = 2.0):
    """Let the circuit reach its steady state before the fight is scored.

    Membrane potentials start at rest and the recurrent loops need on the order of a
    second of biological time to fill in. Stepping straight from reset into combat means
    the first two seconds of every episode are decided by a silent brain.
    """
    state = parse_obs(obs, info)
    steps = max(1, int(seconds * 1000.0 / 100.0))
    for _ in range(steps):
        engine.step(encoder.encode(state), duration_ms=100.0)
    encoder.reset(state.player_hp, state.distance)


def reset_episode(env, console, use_mock):
    """Reset the environment, rebuilding it if the live game refuses to come back."""
    for attempt in range(5):
        try:
            obs, info = env.reset()
            return obs, info, env
        except Exception as e:
            console.print(f"[yellow]WARNING env.reset() retry {attempt + 1}/5: {e}[/yellow]")
            release_all_keys()
            time.sleep(1.2)
    console.print("[bold yellow]WARNING Re-initializing environment after reset failure...[/bold yellow]")
    try:
        env.close()
    except Exception:
        pass
    env = make_souls_env(use_mock=use_mock)
    obs, info = env.reset()
    return obs, info, env


def run_episode(*, ep, env, mock_mode, engine, encoder, decoder, plasticity, topology,
                obs, info, args, console, dashboard, enable_web, episode_history,
                global_action_counts):
    """Run one fight. Returns the episode summary."""
    ep_reward = 0.0
    hits = 0
    step = 0
    episode_actions: dict[str, int] = {}
    unlocked_steps = 0
    lock_warned = False
    step_ms = float(
        getattr(env.unwrapped, "step_seconds", None)
        or getattr(env.unwrapped, "step_size", 0.1)
    ) * 1000.0
    decoder.step_s = step_ms / 1000.0
    terminated = truncated = False
    state = parse_obs(obs, info)

    live = None
    if not args.no_dashboard:
        live = Live(console=console, refresh_per_second=20, transient=True)
        live.__enter__()

    try:
        while not (terminated or truncated):
            step += 1

            # 1. Encode the combat state into sensory drive current.
            drive = encoder.encode(state)

            # 2. Advance the connectome by as much biological time as the game is
            #    about to advance. Running the brain for one render frame per 0.1 s
            #    game step left the circuit permanently in transient, six times
            #    behind the fight it was supposed to be reacting to.
            spike_counts = engine.step(drive, duration_ms=step_ms)

            # 3. Read out the winning descending pool, restricted to what the body can
            #    currently execute.
            valid_channels = decoder.channels_for_valid_actions(
                state.valid_actions, mock=mock_mode
            )
            channel, motor_rates = decoder.decode(
                spike_counts, explore=args.explore, valid_channels=valid_channels
            )

            # 4. Translate that into a game action and step the world.
            action, channel = decoder.to_game_action(
                channel, motor_rates, mock=mock_mode, lock_on=state.lock_on
            )
            prev_boss_hp = state.boss_hp
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            state = parse_obs(obs, info)
            if state.boss_hp < prev_boss_hp - 1e-6:
                hits += 1

            # 5. Dopaminergic plasticity. The efference copy of the executed action is
            #    what makes the credit land in the right mushroom body compartment.
            encoder.set_efference(channel)
            plasticity.update_traces(spike_counts, executed_channel=channel)
            if not args.no_learning:
                plasticity.apply_reinforcement(reward)

            global_action_counts[channel] = global_action_counts.get(channel, 0) + 1
            episode_actions[channel] = episode_actions.get(channel, 0) + 1
            active_count = int(np.count_nonzero(spike_counts))

            # Every locomotor mapping assumes the camera is locked on: with lock-on,
            # SoulsGym action 0 walks towards the boss and attacks track it. Without it,
            # 0 walks wherever the camera happens to point and the agent runs in a
            # straight line past the fight. SoulsGym re-locks on its own each step, so a
            # short gap is normal and a persistent one is not.
            if not mock_mode:
                unlocked_steps = 0 if state.lock_on else unlocked_steps + 1
                if unlocked_steps >= 8 and not lock_warned:
                    lock_warned = True
                    console.print(
                        "[bold yellow]WARNING lock-on has been lost for "
                        f"{unlocked_steps} steps.[/bold yellow] Movement is camera-relative "
                        "while unlocked, so the fly cannot steer towards Iudex. Check the "
                        "camera timeout patched into soulsgym's _camera_reset."
                    )

            if enable_web:
                if step % 3 == 0 and not mock_mode:
                    frame = grab_video_frame(env)
                    if frame is not None:
                        set_latest_frame(frame)
                broadcast_event({
                    "episode": ep, "step": step,
                    "player_hp": state.player_hp, "player_sp": state.player_sp,
                    "boss_hp": state.boss_hp, "boss_distance": state.distance,
                    "boss_attacking": state.boss_attacking, "action_name": channel,
                    "lock_on": state.lock_on,
                    "dopamine": float(plasticity.dopamine_level),
                    "active_neurons": active_count, "cumulative_reward": ep_reward,
                    "spikes": np.flatnonzero(spike_counts).tolist(),
                    "mean_weight": round(float(plasticity.mean_plastic_weight), 4),
                    "channel_weights": plasticity.channel_weights(),
                    "total_ltp": plasticity.total_ltp_events,
                    "total_ltd": plasticity.total_ltd_events,
                    "action_counts": global_action_counts,
                    "history": episode_history[-20:],
                })

            if live is not None:
                live.update(dashboard.create_layout(
                    episode=ep, step=step,
                    player_hp_pct=state.player_hp, player_sp_pct=state.player_sp,
                    boss_hp_pct=state.boss_hp, distance=state.distance,
                    boss_state=state.boss_state_name, active_neurons=active_count,
                    total_neurons=topology.num_neurons, dopamine=plasticity.dopamine_level,
                    action_name=channel, motor_rates=motor_rates, cumulative_reward=ep_reward,
                ))
                time.sleep(0.015)
            elif not args.quiet and step % 50 == 0:
                console.print(
                    f"  [Fly] Step {step:4d} | {channel:<13} | Dist: {state.distance:5.1f}m | "
                    f"HP: {state.player_hp*100:3.0f}% | SP: {state.player_sp*100:3.0f}% | "
                    f"Boss: {state.boss_hp*100:3.0f}% | Hits: {hits} | DA: {plasticity.dopamine_level:+.2f}"
                )
    finally:
        if live is not None:
            live.__exit__(None, None, None)

    return {
        "episode": ep, "steps": step, "reward": round(ep_reward, 2), "hits": hits,
        "actions": episode_actions,
        "boss_hp_pct": round(state.boss_hp, 3), "player_hp_pct": round(state.player_hp, 3),
        "victory": state.boss_hp <= 1e-6,
        "mean_weight": round(float(plasticity.mean_plastic_weight), 4),
    }


if __name__ == "__main__":
    main()
