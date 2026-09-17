"""Work out what player_pose[3] actually means, by watching where the player walks.

None of the obvious sign/axis combinations turned player_pose[3] into a bearing that
put the boss in front of a player who was locked onto it. So stop guessing the formula
and measure the quantity directly: walk, and compare the direction actually travelled
against the value in the pose.

Run it with the agent stopped, in the arena.

    python scripts/fit_heading2.py
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.env.mouse_input import enable_mouse_bindings

enable_mouse_bindings()

import gymnasium as gym  # noqa: E402
import soulsgym  # noqa: E402,F401

WALK = {"forward": 0, "backward": 4, "right": 2, "left": 6}


def circular_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def circular_spread(angles, mean):
    return math.degrees(
        math.sqrt(sum(((a - mean + math.pi) % (2 * math.pi) - math.pi) ** 2 for a in angles)
                  / max(1, len(angles)))
    )


def main():
    env = gym.make("SoulsGymIudex-v0")
    game = env.unwrapped.game
    obs, info = env.reset()

    # Strafe around the boss with lock-on held. The character stays facing the boss the
    # whole way, so the bearing computed from the two positions and the value in pose[3]
    # must rotate together. Walking in a straight line cannot separate them, and walking
    # into arena geometry just adds noise.
    game.game_speed = 1.0
    time.sleep(0.2)
    game.clear_cache()
    print(f"[*] lock_on = {game.lock_on} (want True: the body then provably faces the boss)")

    a_values = []
    offsets = []
    for i in range(60):
        p0 = np.asarray(obs["player_pose"], dtype=np.float64)
        b0 = np.asarray(obs["boss_pose"], dtype=np.float64)
        dx, dy = float(b0[0] - p0[0]), float(b0[1] - p0[1])
        if math.hypot(dx, dy) > 1.0 and bool(np.ravel(obs["lock_on"])[0]):
            bearing = math.atan2(dy, dx)
            a_values.append(float(p0[3]))
            offsets.append((bearing - float(p0[3]) + math.pi) % (2 * math.pi) - math.pi)
        # Circle the boss so the bearing sweeps through a wide range.
        action = WALK["right"] if (i // 12) % 2 == 0 else WALK["left"]
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if len(offsets) < 8:
        print(f"[!] only {len(offsets)} usable samples")
        env.close()
        return

    mean = circular_mean(offsets)
    spread = circular_spread(offsets, mean)
    print(f"[*] {len(offsets)} walking steps")
    print(f"    player_pose[3] range observed: {min(a_values):+.2f} .. {max(a_values):+.2f}")
    print(f"    bearing_to_boss - pose[3]: mean {math.degrees(mean):+.1f} deg, "
          f"spread {spread:.1f} deg")
    print(f"    pose[3] swept {math.degrees(max(a_values) - min(a_values)):.0f} deg "
          f"during the circle")
    print()
    if spread < 35:
        print(f"[+] pose[3] IS the heading, offset by {math.degrees(mean):+.1f} deg.")
        print(f"    bearing_to_boss = atan2(dy, dx) - (pose[3] + {math.radians(round(math.degrees(mean))):+.4f})")
    else:
        print("[!] No consistent offset - pose[3] does not track the direction of travel,")
        print("    so it is not a yaw angle and the bearing has to come from somewhere else.")
    env.close()


if __name__ == "__main__":
    main()
