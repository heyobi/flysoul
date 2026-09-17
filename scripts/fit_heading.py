"""Find the pose convention that actually makes the boss appear where it is.

`parse_obs` assumed the bearing to the boss is

    atan2(by - py, bx - px) - player_pose[3]

Measured against the game with lock-on established — where the character provably faces
the boss, so the true bearing is ~0 — that expression reads about -150 degrees with a
spread of only 15. A tight, systematic offset like that is a wrong convention, not a
turned-around fly, and the same expression feeds the retinal bearing map the circuit
steers by.

Rather than guess which axis or sign is off, this tries every combination and reports
which one puts the boss in front of a player who is facing it.

    python scripts/fit_heading.py
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flysoul.env.mouse_input import enable_mouse_bindings

enable_mouse_bindings()

import gymnasium as gym  # noqa: E402
import soulsgym  # noqa: E402,F401

FORWARD, IDLE = 0, 19


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# Candidate ways of turning (dx, dy, heading) into a bearing relative to the player.
CANDIDATES = {
    "atan2(dy,dx) - h": lambda dx, dy, h: wrap(math.atan2(dy, dx) - h),
    "atan2(dy,dx) + h": lambda dx, dy, h: wrap(math.atan2(dy, dx) + h),
    "atan2(dx,dy) - h": lambda dx, dy, h: wrap(math.atan2(dx, dy) - h),
    "atan2(dx,dy) + h": lambda dx, dy, h: wrap(math.atan2(dx, dy) + h),
    "-atan2(dy,dx) - h": lambda dx, dy, h: wrap(-math.atan2(dy, dx) - h),
    "-atan2(dy,dx) + h": lambda dx, dy, h: wrap(-math.atan2(dy, dx) + h),
    "-atan2(dx,dy) - h": lambda dx, dy, h: wrap(-math.atan2(dx, dy) - h),
    "-atan2(dx,dy) + h": lambda dx, dy, h: wrap(-math.atan2(dx, dy) + h),
}


def main():
    env = gym.make("SoulsGymIudex-v0")
    game = env.unwrapped.game
    obs, info = env.reset()

    samples = []
    for _ in range(70):
        game.clear_cache()
        locked = bool(game.lock_on)
        p = np.asarray(obs["player_pose"], dtype=np.float64)
        b = np.asarray(obs["boss_pose"], dtype=np.float64)
        dist = float(np.hypot(b[0] - p[0], b[1] - p[1]))
        if locked and dist > 1.0:
            samples.append((float(b[0] - p[0]), float(b[1] - p[1]), float(p[3])))
        obs, _, terminated, truncated, info = env.step(FORWARD if dist > 3.0 else IDLE)
        if terminated or truncated:
            break

    if len(samples) < 10:
        print(f"[!] only {len(samples)} usable samples; run again during a fight")
        env.close()
        return

    print(f"[*] {len(samples)} samples with lock-on established.")
    print("    Locked on, the character faces the boss, so the right formula reads ~0.\n")
    print(f"    {'formula':22} {'median':>9} {'sd':>7} {'|err|<30 deg':>13}")
    print("    " + "-" * 56)

    best = None
    for name, fn in CANDIDATES.items():
        vals = [math.degrees(fn(dx, dy, h)) for dx, dy, h in samples]
        med = statistics.median(vals)
        sd = statistics.pstdev(vals)
        frac = sum(abs(v) < 30 for v in vals) / len(vals)
        print(f"    {name:22} {med:+8.1f} {sd:7.1f} {frac:12.0%}")
        score = abs(med) + sd
        if best is None or score < best[0]:
            best = (score, name, med, sd, frac)

    print()
    _, name, med, sd, frac = best
    if abs(med) < 25 and frac > 0.6:
        print(f"[+] Use: {name}   (median {med:+.1f} deg, {frac:.0%} within 30 deg)")
    else:
        print(f"[!] Best candidate {name} is still {med:+.1f} deg off - none of these fit.")
        print("    The heading may not be a plain yaw angle in this pose field.")
    env.close()


if __name__ == "__main__":
    main()
