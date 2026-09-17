"""Does each key SoulsGym sends actually reach Dark Souls III?

SoulsGym drives the game with keyboard events and expects a specific binding:

    forward/back/left/right  w a s d      roll  space
    lock_on  q               interact  e
    lightattack  l           heavyattack  h      parry  m

Movement, roll, lock-on and interact happen to be Dark Souls III's own keyboard
defaults. The three attack keys are not - by default the game binds attacking to the
mouse buttons, which this input path cannot send. If the game has not been rebound, the
agent walks and rolls perfectly while every swing it asks for silently does nothing, and
the boss finishes the fight at full health no matter how good the policy is.

This presses each key on its own and reports whether the player animation changed.

    python scripts/check_input.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soulsgym.games.darksouls3 import DarkSoulsIII

# (action name, seconds to hold, what a working binding should produce)
PROBES = [
    ("forward", 0.35, "a walking/running animation"),
    ("backward", 0.35, "a walking animation"),
    ("left", 0.35, "a walking animation"),
    ("right", 0.35, "a walking animation"),
    ("roll", 0.10, "a roll"),
    ("lightattack", 0.15, "a weapon swing"),
    ("heavyattack", 0.15, "a heavy swing"),
    ("parry", 0.15, "a parry or off-hand attack"),
    ("lock_on", 0.05, "lock-on toggling"),
    ("interact", 0.10, "an interaction"),
]
# The camera keys move the camera, not the body, so they are checked by watching the
# camera pose instead of the player animation. _lock_on steers with these before it
# presses lock, so if they do nothing the camera can never be aimed at the boss.
CAMERA_PROBES = ["cameraleft", "cameraright", "cameraup", "cameradown"]


def _binding(game):
    """Read the action->key table from whichever attribute this soulsgym exposes."""
    for owner in (getattr(game, "data", None), game):
        for attr in ("keys", "keybindings", "binding", "key_bindings"):
            table = getattr(owner, attr, None)
            if isinstance(table, dict):
                return table.get("binding", table)
    from soulsgym.core.static import keybindings
    table = keybindings["DarkSoulsIII"]
    return table.get("binding", table)


def main():
    print("[*] Attaching to DarkSoulsIII ...")
    game = DarkSoulsIII()
    game.game_speed = 1.0
    game.allow_attacks = True
    game.allow_moves = True

    binding = _binding(game)
    print("[*] soulsgym expects these bindings:")
    for action, _, _ in PROBES:
        print(f"      {action:12} -> {binding.get(action)!r}")
    print("[*] Make sure Dark Souls III has focus. Probing in 2 s ...")
    time.sleep(2.0)

    results = []
    for action, hold, expected in PROBES:
        game._game_input.reset()
        time.sleep(0.4)
        before = game.player_animation
        game._game_input.single_action(action, hold)
        seen = set()
        deadline = time.time() + 1.6
        while time.time() < deadline:
            game.clear_cache()
            seen.add(game.player_animation)
            time.sleep(0.05)
        seen.discard(before)
        reached = bool(seen)
        results.append((action, reached, before, sorted(seen)[:4]))
        status = "REACHED" if reached else "NO EFFECT"
        print(f"[{'+' if reached else '!'}] {action:12} key={binding.get(action)!r:5} "
              f"{status:9} | from {before!r} -> {sorted(seen)[:4]} (expected {expected})")

    # Camera keys: watch the camera pose rather than the player animation.
    import numpy as np

    for action in CAMERA_PROBES:
        game._game_input.reset()
        time.sleep(0.3)
        game.clear_cache()
        before = np.array(game.camera_pose, dtype=float)
        game._game_input.single_action(action, 0.25)
        time.sleep(0.6)
        game.clear_cache()  # memory reads are cached; a stale read reads as 'dead key'
        after = np.array(game.camera_pose, dtype=float)
        moved = float(np.max(np.abs(after - before)))
        reached = moved > 1e-3
        results.append((action, reached, "camera", [f"delta {moved:.4f}"]))
        print(f"[{'+' if reached else '!'}] {action:12} key={binding.get(action)!r:5} "
              f"{'REACHED' if reached else 'NO EFFECT':9} | camera delta {moved:.4f}")

    game._game_input.reset()
    dead = [a for a, ok, _, _ in results if not ok]
    print()
    if not dead:
        print("[+] Every probed key reaches the game.")
    else:
        print(f"[!] These keys do nothing in game: {', '.join(dead)}")
        print("    Rebind them in Dark Souls III -> Settings -> Key Bindings to match the")
        print("    table above. The game binds attacking to the mouse by default, and this")
        print("    input path only sends keyboard events, so an unbound attack key means")
        print("    the agent can never land a hit however well it plays.")


if __name__ == "__main__":
    main()
