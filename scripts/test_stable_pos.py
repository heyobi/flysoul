import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Clear any inputs
g._game_input.reset()

# Official stable ground position: [132.176, 575.064, -68.066], heading = 1.074 rad (facing Gundyr)
target_pose = np.array([132.176, 575.064, -68.066, 1.074], dtype=np.float32)
print("Teleporting to official arena start facing Gundyr:", target_pose)
g.player_pose = target_pose
time.sleep(1.0)

print("After 1s - Player pose:", g.player_pose)
print("Player HP:", g.player_hp, "/", g.player_max_hp)
print("Player animation:", g.player_animation)
print("Boss pose:", g.iudex_pose)
print("Boss HP:", g.iudex_hp)
print("Boss anim:", g.iudex_animation)

# Wait another 2 seconds to make sure not falling
time.sleep(2.0)
print("After 3s - Player pose:", g.player_pose)
print("Player HP:", g.player_hp)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_stable.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_stable.jpg")

env.close()
