import time
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

print("[*] Pressing interact (e) to Leave bonfire...")
g._game_input.single_action("interact", 0.1)
time.sleep(1.5)

print("Player anim:", g.player_animation)
print("Player pose:", g.player_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_leave_bonfire.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_leave_bonfire.jpg")

env.close()
