import time
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
print("Player pose:", g.player_pose)
print("Player HP:", g.player_hp, "/", g.player_max_hp)
print("Player anim:", g.player_animation)
print("Boss pose:", g.iudex_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_died.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_died.jpg")

env.close()
