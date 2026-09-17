import time
import gymnasium as gym
import soulsgym
import cv2

print("[*] Connecting to SoulsGym...")
env = gym.make("SoulsGymIudex-v0")
g = env.unwrapped.game
g._game_input.reset()

print("[*] Setting iudex_flags = True and calling reload()...")
g.iudex_flags = True
g.reload()

print("[*] Reload finished! Checking state:")
print("Player pose:", g.player_pose)
print("Boss pose:", g.iudex_pose)
print("Boss HP:", g.iudex_hp)
print("Boss anim:", repr(g.iudex_animation))
print("Lock on:", g.lock_on)

raw = g._game_window.raw_img
if raw is not None and raw.size > 0:
    cv2.imwrite("/tmp/snap_after_reload.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print("[+] Saved /tmp/snap_after_reload.jpg")

env.close()
