import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

print("[*] Fixing stuck sitting animation...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

print("Current anim:", g.player_animation)
print("Current pose:", g.player_pose)

# Lift player 1.2 meters into the air so fall animation resets pose
p = g.player_pose.copy()
p[2] += 1.2  # Increase height (z)
print("Lifting player to:", p)
g.player_pose = p
time.sleep(0.5)

# Wait for land animation to complete
for i in range(15):
    anim = g.player_animation
    print(f"  t={i*0.2:.1f}s | anim={anim}")
    if anim == "Idle":
        print("[+] Player successfully stood up in Idle!")
        break
    time.sleep(0.2)

print("Final player pose:", g.player_pose)
print("Final player anim:", g.player_animation)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_fixed_sitting.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_fixed_sitting.jpg")

env.close()
