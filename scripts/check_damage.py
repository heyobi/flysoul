"""Does a swing at melee range actually damage Iudex?

Before blaming the circuit for not landing hits, establish that a hit is landable at all
through the path the agent uses. This script teleports the player next to the boss and
issues light attacks with ``env.step(16)`` - the same call the agent makes - reporting
the boss's health, the player's animation and the player's equipment around each swing.

If the boss takes no damage here, the problem is on the game side (weapon, reach, or
input delivery) and no amount of work on the connectome will land a hit.

    python scripts/check_damage.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soulsgym  # noqa: F401  (registers the environments)

LIGHT_ATTACK = 16
HEAVY_ATTACK = 17
FORWARD = 0
IDLE = 19


def describe_equipment(game):
    """Report whatever the installed soulsgym exposes about what the player is holding."""
    out = {}
    for attr in ("player_stats", "weapon", "right_weapon", "left_weapon", "equipment"):
        if hasattr(game, attr):
            try:
                out[attr] = getattr(game, attr)
            except Exception as exc:
                out[attr] = f"<unreadable: {exc}>"
    return out


def main():
    print("[*] Creating SoulsGymIudex-v0 ...")
    env = gym.make("SoulsGymIudex-v0")
    game = env.unwrapped.game
    obs, info = env.reset()

    print(f"[*] allow_attacks={getattr(game, 'allow_attacks', '?')} "
          f"allow_hits={getattr(game, 'allow_hits', '?')} "
          f"allow_moves={getattr(game, 'allow_moves', '?')}")
    for k, v in describe_equipment(game).items():
        print(f"[*] {k}: {v}")

    player = np.asarray(obs["player_pose"], dtype=np.float32)
    boss = np.asarray(obs["boss_pose"], dtype=np.float32)
    print(f"[*] start: player {player[:3].round(2)}  boss {boss[:3].round(2)}  "
          f"dist {np.linalg.norm(boss[:2] - player[:2]):.2f} m  lock_on {obs['lock_on']}")

    # Close the distance with the agent's own forward action rather than teleporting, so
    # this measures the situation the agent is actually in.
    dist = float(np.linalg.norm(boss[:2] - player[:2]))
    for i in range(120):
        obs, _, terminated, truncated, info = env.step(FORWARD)
        player = np.asarray(obs["player_pose"], dtype=np.float32)
        boss = np.asarray(obs["boss_pose"], dtype=np.float32)
        dist = float(np.linalg.norm(boss[:2] - player[:2]))
        if i % 20 == 0:
            print(f"    step {i:3d}: dist {dist:5.2f} m  lock_on {int(np.ravel(obs['lock_on'])[0])}")
        if dist <= 2.4 or terminated or truncated:
            break
    print(f"[*] closed to {dist:.2f} m  lock_on={obs['lock_on']}  "
          f"boss_hp={float(np.ravel(obs['boss_hp'])[0]):.0f}")
    print(f"[*] arena_init={getattr(env.unwrapped, '_arena_init', None)} "
          f"phase_init={getattr(env.unwrapped, '_phase_init', None)} "
          f"iudex_flags={game.iudex_flags} "
          f"boss animation now: {game.iudex_animation!r}")

    # A minimal scripted attacker: walk in when out of reach, swing when in reach and
    # the body is free. This is a diagnostic, not the agent - the point is to establish
    # that a hit is landable at all through env.step().
    landed = 0
    swings = 0
    min_dist = dist
    for step in range(220):
        player = np.asarray(obs["player_pose"], dtype=np.float32)
        boss = np.asarray(obs["boss_pose"], dtype=np.float32)
        dist = float(np.linalg.norm(boss[:2] - player[:2]))
        min_dist = min(min_dist, dist)
        valid = env.unwrapped.current_valid_actions()

        if dist > 2.3:
            action = FORWARD if FORWARD in valid else IDLE
        elif LIGHT_ATTACK in valid:
            action = LIGHT_ATTACK
            swings += 1
        else:
            action = IDLE

        before = float(np.ravel(obs["boss_hp"])[0])
        obs, reward, terminated, truncated, info = env.step(action)
        after = float(np.ravel(obs["boss_hp"])[0])
        if after < before - 1e-6:
            landed += 1
            print(f"[+] HIT at step {step}: dist {dist:.2f} m  "
                  f"boss hp {before:.0f} -> {after:.0f}  ({before - after:.0f} damage)")
        if terminated or truncated:
            print(f"[*] episode ended at step {step} "
                  f"(boss {after:.0f}, player {float(np.ravel(obs['player_hp'])[0]):.0f})")
            break

    print(f"\n[*] {swings} swings issued in range, {landed} landed, "
          f"closest approach {min_dist:.2f} m, boss hp "
          f"{float(np.ravel(obs['boss_hp'])[0]):.0f}/1037")

    print("\n[*] Verdict:")
    print("    boss hp changed  -> hits are landable; the circuit has to choose to swing in range")
    print("    boss hp constant -> the problem is game-side, not in the connectome")
    env.close()


if __name__ == "__main__":
    main()
