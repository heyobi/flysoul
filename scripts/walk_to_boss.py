import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Walk forward towards Gundyr
print("[*] Walking towards boss...")
g._game_input.single_action("forward", 2.5)
time.sleep(2.8)

# Check state
print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Boss HP:", g.iudex_hp)
print("Boss anim:", g.iudex_animation)

# Try lock-on
for _ in range(5):
    if g.lock_on:
        print("[+] Locked on!")
        break
    g._game_input.single_action("lock_on", 0.05)
    time.sleep(0.15)

print("Lock on:", g.lock_on)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_near_boss.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_near_boss.jpg")

env.close()
