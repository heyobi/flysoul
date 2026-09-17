import gymnasium as gym
import soulsgym
import numpy as np

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

print("Player pose:", g.player_pose)
print("Boss pose:", g.boss_pose)
print("Camera pose:", g.camera_pose)

# Calculate vector from camera to boss and player to boss
p = g.player_pose[:3]
b = g.boss_pose[:3]
cam = g.camera_pose[:3]
cam_dir = g.camera_pose[3:6]

print("Player -> Boss vector:", b - p)
print("Camera pos:", cam)
print("Camera normal:", cam_dir)
env.close()
