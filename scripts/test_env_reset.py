import time
import gymnasium as gym
import soulsgym

print("[*] Testing env.reset() with active boss fight...")
env = gym.make("SoulsGymIudex-v0")
obs, info = env.reset()
print("[+] Reset succeeded!")
print("Player HP:", env.unwrapped.game.player_hp)
print("Boss HP:", env.unwrapped.game.iudex_hp)
print("Lock on:", env.unwrapped.game.lock_on)
print("Player anim:", env.unwrapped.game.player_animation)
print("Boss anim:", env.unwrapped.game.iudex_animation)

# Take 1 step to test action
obs, reward, terminated, truncated, info = env.step(16)  # light attack
print(f"[+] Took 1 step (attack): reward={reward:.2f}")

env.close()
