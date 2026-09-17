import time
import numpy as np
import gymnasium as gym
import soulsgym
from flysoul.connectome.graph import build_fly_circuit, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine, BioPhysicsConfig
from flysoul.sensory.encoder import SensoryEncoder
from flysoul.motor.decoder import MotorDecoder

# Build circuit
topology = build_fly_circuit(CircuitConfig())
engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, BioPhysicsConfig())
encoder = SensoryEncoder(topology)
decoder = MotorDecoder(topology)

env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

# Set safe spawn point (9m in front of Gundyr)
safe_pose = np.array([140.0, 583.0, -68.8, 1.25], dtype=np.float32)
env.unwrapped.game.data.coordinates["iudex"]["player_init_pose"] = safe_pose

obs, info = env.reset()
print("[+] Reset with safe 9m spawn point:")
print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Lock on:", g.lock_on)
print("Boss anim:", g.iudex_animation)

print("[+] Executing 10 combat steps:")
for step in range(1, 11):
    drive = encoder.encode(obs)
    spikes = engine.step(drive, duration_ms=16.67)
    
    p_pose = obs["player_pose"]
    b_pose = obs["boss_pose"]
    dist = float(((b_pose[0]-p_pose[0])**2 + (b_pose[1]-p_pose[1])**2)**0.5)
    
    action_id, action_name, rates = decoder.decode(spikes, explore=False, distance=dist)
    step_action = decoder.to_soulsgym_action(action_id)
    
    obs, reward, terminated, truncated, info = env.step(step_action)
    print(f"  Step {step:2d} | Action: {action_name:<14} | Dist: {dist:.1f}m | HP: {obs['player_hp'][0]:.0f} | Lock: {obs['lock_on']}")

env.close()
