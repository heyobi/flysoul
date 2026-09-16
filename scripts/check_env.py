"""Test script to test making the SoulsGym environment."""

import gymnasium as gym
import soulsgym

print("[+] Creating SoulsGymIudex-v0 environment...")
try:
    env = gym.make("SoulsGymIudex-v0")
    print(f"[+] Environment created successfully: {env}")
except Exception as e:
    print(f"[-] Environment creation failed (expected if game not launched/in-game): {e}")
