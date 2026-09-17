import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

# Gundyr position is [141.3, 591.9, -68.8]
# Place player directly in front of Gundyr (2.0m away) facing Gundyr (+y direction)
player_pos = np.array([141.3, 589.5, -68.8, 1.5708], dtype=np.float32)
print("Placing player directly in front of Gundyr:", player_pos)
g.player_pose = player_pos
time.sleep(0.5)

# Camera normal pointing in +y direction (towards Gundyr)
print("[*] Setting camera normal to face Gundyr...")
g.camera_pose = (0.0, 1.0, 0.0)
time.sleep(0.5)

print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Boss HP:", g.iudex_hp)
print("Boss animation:", g.iudex_animation)
print("Boss flags:", g.iudex_flags)
print("Lock on:", g.lock_on)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_gundyr_face.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_gundyr_face.jpg")

env.close()
