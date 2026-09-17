import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

print("[*] Placing character standing up in arena...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

# Arena entrance facing Gundyr
arena_pos = np.array([132.176, 575.064, -68.066, 1.074], dtype=np.float32)
g.player_pose = arena_pos
time.sleep(0.5)

# Camera facing Gundyr
g.camera_pose = (0.42, 0.90, 0.0)
time.sleep(0.5)

print("Player animation:", g.player_animation)
print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_standing_arena.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_standing_arena.jpg")

env.close()
