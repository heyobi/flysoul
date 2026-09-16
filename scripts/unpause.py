"""Ensure game is unpaused, speed is 1.0, and player can move."""
from soulsgym.games.darksouls3 import DarkSoulsIII

g = DarkSoulsIII()
g.game_speed = 1.0
g.resume()
g.allow_moves = True
g.allow_attacks = True
g.allow_hits = True
print("[+] Game unpaused, speed set to 1.0, moves/attacks/hits enabled.")
print(f"Current Player Pose: {g.player_pose}")
print(f"Current Flags: {g.mem.read_record(g.data.addresses['IudexFlags']).hex()}")
