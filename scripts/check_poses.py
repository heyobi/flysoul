import gymnasium as gym
import soulsgym

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
print("Player pose:", g.player_pose)
print("Player HP:", g.player_hp)
print("Boss pose:", g.iudex_pose)
print("Boss HP:", g.iudex_hp)
print("Boss anim:", g.iudex_animation)
print("Lock on:", g.lock_on)
env.close()
