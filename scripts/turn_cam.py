import time
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Turn camera right to center on Gundyr
print("[+] Rotating camera right towards Gundyr...")
g._game_input.single_action("cameraright", 1.2)
time.sleep(0.3)

# Try lock-on
print("[+] Pressing lock-on...")
for _ in range(5):
    if g.lock_on:
        break
    g._game_input.single_action("lock_on", 0.1)
    time.sleep(0.15)

print("[+] Lock-on status:", g.lock_on)
print("[+] Boss animation:", g.iudex_animation)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_turn.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_turn.jpg")
