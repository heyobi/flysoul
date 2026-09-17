import gymnasium as gym
import soulsgym

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

print("[+] Resetting game input...")
g._game_input.reset()
print("[+] Player animation after input reset:", g.player_animation)
print("[+] Player pose:", g.player_pose)
