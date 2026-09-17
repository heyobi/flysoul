"""Is the player heading convention right?

`parse_obs` computes the boss's bearing relative to the player as

    angle = atan2(by - py, bx - px) - player_pose[3]

and the sensory encoder feeds that straight into the retinal bearing map. If the heading
convention is off — a different zero, a different sign, a swapped axis — then the fly has
been seeing the boss in the wrong direction this whole time, and every steering decision
it makes is built on it.

Dark Souls III keeps the character facing its target while locked on, so with lock-on
established the measured bearing should sit near zero. Anything else is the convention
being wrong, not the fly being turned around.

    python scripts/check_heading.py
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

from flysoul.env.obs import parse_obs  # noqa: E402

FORWARD, IDLE = 0, 19


def main():
    env = gym.make("SoulsGymIudex-v0")
    game = env.unwrapped.game
    obs, info = env.reset()

    samples = []
    for step in range(60):
        state = parse_obs(obs, info)
        game.clear_cache()
        if game.lock_on:
            samples.append(state.angle)
        action = FORWARD if state.distance > 3.0 else IDLE
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if not samples:
        print("[!] never locked on; cannot judge the convention")
        env.close()
        return

    degrees = [math.degrees(a) for a in samples]
    median = statistics.median(degrees)
    spread = statistics.pstdev(degrees) if len(degrees) > 1 else 0.0
    within_30 = sum(abs(d) < 30 for d in degrees) / len(degrees)

    print(f"[*] {len(samples)} samples taken while locked on")
    print(f"    median bearing to boss: {median:+.1f} deg")
    print(f"    spread (sd):            {spread:.1f} deg")
    print(f"    fraction within +/-30:  {within_30:.0%}")
    print()
    if abs(median) < 25:
        print("[+] The convention is right: locked on, the boss sits ahead of the player.")
    else:
        off = round(median / 90) * 90
        print(f"[!] Systematically {median:+.0f} deg off - close to {off:+d}.")
        print("    The heading zero or sign is wrong, which means the retinal bearing the")
        print("    circuit has been steering by points in the wrong direction.")
    env.close()


if __name__ == "__main__":
    main()
