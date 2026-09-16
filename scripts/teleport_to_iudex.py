"""Teleport player right into the arena facing Iudex."""
import numpy as np
from soulsgym.games.darksouls3 import DarkSoulsIII

g = DarkSoulsIII()
# Iudex is around [142.8, 594.6, -68.88]
# Place player 3 meters in front facing him
target_pose = np.array([142.8, 590.0, -68.88, 0.0])
g.player_pose = target_pose
print(f"[+] Player teleported to: {g.player_pose}")
