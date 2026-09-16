"""Test script to test making the SoulsGym environment."""

import gymnasium as gym
import soulsgym

print("[+] Creating SoulsGymIudex-v0 environment...")
try:
    env = gym.make("SoulsGymIudex-v0")
    print(f"[+] Environment created successfully: {env}")
    print("[+] Calling env.reset()...")
    obs, info = env.reset()
    print("[+] Reset SUCCEEDED!")
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  Player HP: {obs.get('player_hp')}, Boss HP: {obs.get('boss_hp')}")
except Exception as e:
    print(f"[-] Environment test failed: {e}")
