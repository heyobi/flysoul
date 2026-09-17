import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

print("[*] Teleporting back to bonfire to stand up...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

# Bonfire coordinates: [104, 507, -49.5, 0.0]
bonfire_pos = np.array([104.0, 507.0, -49.5, 0.0], dtype=np.float32)
g.player_pose = bonfire_pos
time.sleep(1.0)

print("Player anim at bonfire:", g.player_animation)
print("Player pose:", g.player_pose)

# Press interact / roll / back
for _ in range(5):
    g._game_input.single_action("interact", 0.1)
    time.sleep(0.3)
    g._game_input.single_action("roll", 0.1)
    time.sleep(0.3)

time.sleep(1.0)
print("Player anim after buttons:", g.player_animation)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_bonfire_stand.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_bonfire_stand.jpg")

env.close()
