"""Can the agent aim the camera and establish lock-on at all?

SoulsGym aims the camera at the boss and then presses lock on. Both halves are worth
checking independently on this setup, because `camera_pose` is a *setter that presses
the camera keys* rather than writing memory — so if those keys are unbound in the game,
the entire aiming path silently does nothing and lock-on only ever happens when the boss
wanders into view by itself.

    python scripts/check_lockon.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.env.mouse_input import enable_mouse_bindings

enable_mouse_bindings()

import gymnasium as gym  # noqa: E402
import soulsgym  # noqa: E402,F401
from soulsgym.core.static import keybindings  # noqa: E402


def toggles(game, label: str, action: str) -> bool:
    """Press `action` and report whether the lock-on state changed."""
    game.clear_cache()
    before = game.lock_on
    game._game_input.single_action(action, 0.06)
    time.sleep(0.7)
    game.clear_cache()
    after = game.lock_on
    changed = after != before
    verdict = "TOGGLED" if changed else "no effect"
    print(f"  {label:24} {before} -> {after}   {verdict}")
    return changed


def main():
    env = gym.make("SoulsGymIudex-v0")
    game = env.unwrapped.game
    obs, info = env.reset()
    # The environment keeps the game paused between steps, and the camera cannot move
    # while it is paused.
    game.game_speed = 1.0
    time.sleep(0.3)
    game.clear_cache()
    print(f"[*] lock_on after reset: {game.lock_on}")

    print("\n[*] Which control actually toggles lock-on?")
    working = None
    for label, key in (("key 'q' (soulsgym default)", "q"), ("middle click", "mouse2")):
        keybindings["DarkSoulsIII"]["lock_on"] = key
        if toggles(game, label, "lock_on"):
            working = key
            # Put it back the way it was so the next probe starts from the same state.
            toggles(game, f"{label} (restore)", "lock_on")
            break
    keybindings["DarkSoulsIII"]["lock_on"] = working or "q"

    print("\n[*] Can the camera be aimed at the boss?")
    target = game.iudex_pose[:3] - game.player_pose[:3]
    normal = target / np.linalg.norm(target)

    # Turn the camera away first. Without this the test passes without the camera ever
    # having to move, which is not the situation being reproduced: the complaint is that
    # an episode sometimes starts facing the wrong way and never recovers.
    if game.lock_on:
        game._game_input.single_action("lock_on", 0.06)
        time.sleep(0.5)
    for _ in range(45):
        game._game_input.single_action("cameraright", 0.02)
        time.sleep(0.02)
    time.sleep(0.5)
    game.clear_cache()
    turned_dot = float(np.dot(game.camera_pose[3:], normal))
    print(f"  turned away: dot(camera, boss) = {turned_dot:+.3f}")

    before_dot = turned_dot
    started = time.time()
    try:
        game.camera_pose = target
    except Exception as exc:
        print(f"  camera_pose setter failed: {exc}")
    game.clear_cache()
    after_dot = float(np.dot(game.camera_pose[3:], normal))
    print(f"  dot(camera, boss): {before_dot:+.3f} -> {after_dot:+.3f} "
          f"in {time.time() - started:.1f}s   (soulsgym needs > 0.8 before it presses lock)")

    print("\n[*] And does lock-on follow?")
    game.clear_cache()
    locked = game.lock_on
    presses = 0
    while not locked and presses < 12:
        game._game_input.single_action("lock_on", 0.06)
        presses += 1
        time.sleep(0.3)
        game.clear_cache()
        locked = game.lock_on
    print(f"  lock re-established: {locked} (after {presses} presses)")

    print("\n[*] Verdict:")
    if working is None:
        print("    No control toggles lock-on. The agent can never lock on by itself;")
        print("    it only ever inherits a lock the game happened to have.")
    else:
        print(f"    lock-on responds to {working!r}")
    if after_dot > 0.8:
        print("    The camera can be aimed at the boss.")
    else:
        print("    The camera could NOT be brought onto the boss - aiming is still broken.")
    env.close()


if __name__ == "__main__":
    main()
