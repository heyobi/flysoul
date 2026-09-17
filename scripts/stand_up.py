import time
import gymnasium as gym
import soulsgym
import cv2

print("[*] Inspecting sitting/bonfire state...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game

print("Player animation:", g.player_animation)
print("Allow moves:", g.allow_moves)
print("Allow attacks:", g.allow_attacks)
print("Game speed:", g.game_speed)
print("Player pose:", g.player_pose)

# Clear any held keys
g._game_input.reset()
g.allow_moves = True
g.allow_attacks = True
g.allow_hits = True

# In Dark Souls III, pressing cancel (ESC / 'q' / roll / Space / 'e' on Leave)
# If bonfire menu is open, pressing 'e' or ' ' closes it
print("[*] Sending inputs to stand up...")

# Try pressing 'e' (Leave) then Space (Roll/Backstep/Cancel) then 'w' (move forward)
g._game_input.single_action("interact", 0.1)
time.sleep(0.5)
g._game_input.single_action("roll", 0.1)
time.sleep(0.5)
g._game_input.single_action("backward", 0.1)
time.sleep(0.5)

print("Player animation after stand up attempts:", g.player_animation)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_standup.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_standup.jpg")

env.close()
