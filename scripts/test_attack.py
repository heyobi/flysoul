import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

print("[*] Current boss HP:", g.iudex_hp)
print("[*] Current boss anim:", repr(g.iudex_animation))

# Move close to Gundyr
g.player_pose = np.array([141.3, 591.0, -68.8, 1.5708], dtype=np.float32)
time.sleep(0.3)

# Swing sword (light attack)
print("[*] Swinging sword at Gundyr...")
for _ in range(3):
    g._game_input.single_action("lightattack", 0.1)
    time.sleep(0.5)

time.sleep(1.0)
print("[*] After attack - Boss HP:", g.iudex_hp)
print("[*] After attack - Boss anim:", repr(g.iudex_animation))
print("[*] After attack - Lock on:", g.lock_on)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_attack_gundyr.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_attack_gundyr.jpg")

env.close()
