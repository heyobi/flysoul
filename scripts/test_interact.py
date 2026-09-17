import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

# Gundyr is at [141.3, 591.9, -68.8]
# Test standing 1 meter in front of Gundyr
g.player_pose = np.array([141.3, 590.8, -68.8, 1.57], dtype=np.float32)
time.sleep(0.5)

# Press interact (E)
print("[*] Pressing interact (E)...")
for _ in range(3):
    g._game_input.single_action("interact", 0.15)
    time.sleep(0.3)

time.sleep(1.0)
print("Player anim:", g.player_animation)
print("Boss anim:", repr(g.iudex_animation))
print("Boss HP:", g.iudex_hp)
print("Lock on:", g.lock_on)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_interact.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_interact.jpg")

env.close()
