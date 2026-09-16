"""Script to reset Iudex flags to 0x00 and reload the area so the coiled sword appears."""
import time
from soulsgym.games.darksouls3 import DarkSoulsIII

print("[+] Attaching to DarkSoulsIII...")
g = DarkSoulsIII()

print("[+] Setting IudexFlags to 0x00 (pristine unencountered state)...")
g.mem.write_record(g.data.addresses["IudexFlags"], b"\x00")
g.mem.write_record(g.data.addresses["UntendedGravesFlag"], b"\x00")

print("[+] Triggering game reload so the engine re-initializes Iudex with the coiled sword...")
g.reload()

time.sleep(2.0)
print(f"[+] Reload complete. Current IudexFlags: {g.mem.read_record(g.data.addresses['IudexFlags']).hex()}")
print(f"[+] Player Pose: {g.player_pose}")
print(f"[+] Iudex Animation: {g.iudex_animation}")
