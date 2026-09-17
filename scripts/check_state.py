import gymnasium as gym
import soulsgym
import numpy as np

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

print("=== GAME STATE ===")
print("Player pose:", g.player_pose)
print("Iudex pose:", g.iudex_pose)
print("Iudex HP:", g.iudex_hp, "/", g.iudex_max_hp)
print("Iudex flags:", g.iudex_flags)
print("Iudex animation:", g.iudex_animation)
print("Player animation:", g.player_animation)
print("Lock on:", g.lock_on)
print("Arena init:", getattr(env.unwrapped, "_arena_init", None))
print("Phase init:", getattr(env.unwrapped, "_phase_init", None))
