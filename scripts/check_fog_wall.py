import time
import numpy as np
import gymnasium as gym
import soulsgym
import cv2

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

fog_pos = g.data.coordinates["iudex"]["fog_wall"]
print("Teleporting to fog wall:", fog_pos)
g.player_pose = fog_pos
time.sleep(0.5)

print("Player pose:", g.player_pose)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_fog_wall.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_fog_wall.jpg")

env.close()
