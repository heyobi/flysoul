"""Does the camera actually turn onto Iudex and lock, from a bad start?

The patched `_camera_reset` is meant to do what a player does by hand when the fight
opens with the camera pointing the wrong way: turn it until the boss is in view, then
lock on. This reproduces the bad start deliberately - drop the lock, spin the camera
away - and times the recovery, several times, so the fix is measured rather than
assumed. Run it with the agent stopped.

    python scripts/check_camera_reset.py [--trials 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.env.mouse_input import enable_mouse_bindings

enable_mouse_bindings()

import gymnasium as gym  # noqa: E402
import soulsgym  # noqa: E402,F401


def camera_dot(game) -> float:
    game.clear_cache()
    target = game.iudex_pose[:3] - game.player_pose[:3]
    norm = float(np.linalg.norm(target))
    if norm < 1e-6:
        return float("nan")
    return float(np.dot(game.camera_pose[3:], target / norm))


def spoil(game, spin: int) -> None:
    """Drop the lock and turn the camera away, as a bad episode start does."""
    game.game_speed = 1.0
    game.clear_cache()
    if game.lock_on:
        game._game_input.single_action("lock_on", 0.06)
        time.sleep(0.5)
    for _ in range(spin):
        game._game_input.single_action("cameraright", 0.02)
        time.sleep(0.02)
    time.sleep(0.4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--spin", type=int, default=45, help="camera ticks to turn away")
    args = ap.parse_args()

    env = gym.make("SoulsGymIudex-v0")
    unwrapped = env.unwrapped
    game = unwrapped.game
    env.reset()
    reset = getattr(unwrapped, "_camera_reset")
    print(f"[*] lock_on after reset: {game.lock_on}")

    results = []
    for i in range(1, args.trials + 1):
        spoil(game, args.spin)
        before = camera_dot(game)
        locked_before = bool(game.lock_on)
        t0 = time.time()
        try:
            locked = reset(timeout=12.0)
        except TypeError:
            reset()
            game.clear_cache()
            locked = bool(game.lock_on)
        took = time.time() - t0
        game.pause()
        after = camera_dot(game)
        results.append((locked, took))
        print(f"  trial {i}: camera {before:+.2f} (lock {locked_before}) -> "
              f"{after:+.2f}  lock {locked}  in {took:.1f}s")

    ok = sum(1 for locked, _ in results if locked)
    print()
    if ok == len(results):
        mean = sum(t for _, t in results) / len(results)
        print(f"[+] {ok}/{len(results)} recovered, mean {mean:.1f}s: the camera turns onto "
              "Iudex and locks from a bad start.")
    else:
        print(f"[!] only {ok}/{len(results)} recovered; the camera path is still broken.")
    env.close()
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
