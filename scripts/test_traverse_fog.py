import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

print("[*] Traversing fog gate...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Press interact to enter fog gate
for _ in range(3):
    g._game_input.single_action("interact", 0.1)
    time.sleep(0.15)

print("[*] Waiting for traverse animation...")
for i in range(20):
    anim = g.player_animation
    print(f"  t={i*0.2:.1f}s | anim={anim}")
    if anim == "Idle" and i > 5:
        break
    time.sleep(0.2)

# Set post-fog pose
g.player_pose = g.data.coordinates["iudex"]["post_fog_wall"]
time.sleep(0.5)

print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Boss anim:", repr(g.iudex_animation))
print("Boss HP:", g.iudex_hp)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_inside_fog.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_inside_fog.jpg")

env.close()
