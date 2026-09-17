import time
import math
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Step forward towards Gundyr
print("[+] Walking forward towards Gundyr...")
g._game_input.single_action("forward", 1.2)
time.sleep(0.3)

# Lock on
for _ in range(5):
    if g.lock_on:
        break
    g._game_input.single_action("lock_on", 0.1)
    time.sleep(0.2)

print("[+] Lock on status:", g.lock_on)
print("[+] Player pose:", g.player_pose)
print("[+] Boss pose:", g.iudex_pose)
print("[+] Boss HP:", g.iudex_hp)
print("[+] Boss animation:", g.iudex_animation)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_walk.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_walk.jpg")
