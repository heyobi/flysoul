import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Stop any key press
g._game_input.reset()

# Teleport inside the arena, facing Gundyr
target_pose = np.array([138.0, 582.0, -68.5, 1.25], dtype=np.float32)
print("Setting player pose to:", target_pose)
g.player_pose = target_pose
time.sleep(0.5)

print("Player actual pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_teleport.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_teleport.jpg")

env.close()
