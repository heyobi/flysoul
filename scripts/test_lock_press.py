import time
import gymnasium as gym
import soulsgym

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

print("Before lock-on status:", g.lock_on)
g._game_input.single_action("lock_on", 0.1)
time.sleep(0.3)
print("After 0.1s press lock-on status:", g.lock_on)

env.close()
