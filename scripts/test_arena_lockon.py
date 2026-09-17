import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)

# Face Gundyr
# From [126.49, 560.53] to [141.3, 591.9]: angle = atan2(31.37, 14.81) = 1.13 rad
# Turn camera towards Gundyr
g.camera_pose = (0.42, 0.90, 0.0)
time.sleep(0.5)

# Try locking on
for i in range(5):
    if g.lock_on:
        print("[+] LOCKED ON TO GUNDYR!")
        break
    g._game_input.single_action("lock_on", 0.05)
    time.sleep(0.2)

print("Lock on status:", g.lock_on)
print("Boss anim:", repr(g.iudex_animation))
print("Boss HP:", g.iudex_hp)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_lockon_test.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_lockon_test.jpg")

env.close()
