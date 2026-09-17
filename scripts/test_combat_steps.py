import time
import gymnasium as gym
import soulsgym
from flysoul.connectome.graph import CircuitTopology, build_fly_circuit, CircuitConfig
from flysoul.connectome.engine import ConnectomeEngine, BioPhysicsConfig
from flysoul.sensory.encoder import SensoryEncoder
from flysoul.motor.decoder import MotorDecoder

print("[*] Building test circuit...")
topology = build_fly_circuit(CircuitConfig())
engine = ConnectomeEngine(topology.ptr, topology.post, topology.weight, BioPhysicsConfig())
encoder = SensoryEncoder(topology)
decoder = MotorDecoder(topology)

print("[*] Initializing SoulsGymIudex...")
env = gym.make("SoulsGymIudex-v0")
obs, info = env.reset()

print("[+] Starting 15 combat test steps:")
for step in range(1, 16):
    drive = encoder.encode(obs)
    spikes = engine.step(drive, duration_ms=16.67)
    
    # Calculate distance
    p_pose = obs["player_pose"]
    b_pose = obs["boss_pose"]
    dist = float(((b_pose[0]-p_pose[0])**2 + (b_pose[1]-p_pose[1])**2)**0.5)
    
    action_id, action_name, rates = decoder.decode(spikes, explore=False, distance=dist)
    step_action = decoder.to_soulsgym_action(action_id)
    
    obs, reward, terminated, truncated, info = env.step(step_action)
    print(f"  Step {step:2d} | Action: {action_name:<14} (id={step_action:2d}) | Dist: {dist:.1f}m | Player HP: {obs['player_hp'][0]:.0f} | LockOn: {obs['lock_on']}")
    if terminated:
        print("[-] Terminated early!")
        break

env.close()
