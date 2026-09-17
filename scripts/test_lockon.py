import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Turn camera by pressing lock-on or turning right
print("[*] Pressing lock_on...")
for i in range(10):
    if g.lock_on:
        print("[+] Successfully locked on!")
        break
    g._game_input.single_action("lock_on", 0.05)
    time.sleep(0.2)

print("Lock on:", g.lock_on)
print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Camera pose:", g.camera_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_turn.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_turn.jpg")

env.close()
