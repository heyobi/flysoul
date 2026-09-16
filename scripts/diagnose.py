"""Diagnostic script to print current Dark Souls III game memory variables."""

from soulsgym.games.darksouls3 import DarkSoulsIII

print("[+] Attaching to DarkSoulsIII...")
g = DarkSoulsIII()

print(f"Player Pose: {g.player_pose}")
print(f"Player Animation: {g.player_animation}")
print(f"Player HP: {g.player_hp} / {g.player_max_hp}")
print(f"Iudex HP: {g.iudex_hp} / {g.iudex_max_hp}")
print(f"Iudex Animation: {g.iudex_animation}")
print(f"Iudex Pose: {g.iudex_pose}")
print(f"Lock on: {g.lock_on}")
import ctypes
try:
    x11 = ctypes.CDLL("libX11.so.6")
    disp = x11.XOpenDisplay(None)
    for k in ["e", "w", "s", "a", "d", " "]:
        kc = x11.XKeysymToKeycode(disp, ord(k))
        print(f"Key '{k}' -> keycode: {kc}")
except Exception as e:
    print(f"X11 error: {e}")
