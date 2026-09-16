"""Script to inspect and fix Iudex Gundyr flags and state."""
import time
from soulsgym.games.darksouls3 import DarkSoulsIII

print("[+] Attaching to DarkSoulsIII...")
g = DarkSoulsIII()

raw_flag = g.mem.read_record(g.data.addresses["IudexFlags"])
print(f"Current IudexFlags: {raw_flag.hex()} (binary: {bin(raw_flag[0])})")
print(f"Iudex Animation: {g.iudex_animation}")
print(f"Player Animation: {g.player_animation}")
print(f"Player Pose: {g.player_pose}")
print(f"Iudex Pose: {g.iudex_pose}")

print("\nOptions:")
print("1. Set IudexFlags to 0x00 (sword not pulled, not encountered)")
print("2. Set IudexFlags to 0x60 (sword pulled, encountered)")
print("3. Teleport player in front of fog wall")
print("4. Allow hits/moves/attacks")
