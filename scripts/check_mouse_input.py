"""Can the attack be driven with the mouse instead of rebinding the game?

Dark Souls III binds attacking to the mouse by default, and the Linux input layer in
this soulsgym build only sends XTestFakeKeyEvent - keyboard events. XTest can also send
button events, and the library is already loaded, so if the game responds to a synthetic
left click there is no need to touch the game's key bindings at all.

This presses each mouse button directly and reports whether the player animated.

    python scripts/check_mouse_input.py
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soulsgym.games.darksouls3 import DarkSoulsIII

# X11 button numbers. 1 left, 2 middle, 3 right.
BUTTONS = {
    1: "left click  (Dark Souls III default: Attack)",
    3: "right click (Dark Souls III default: left-hand attack / parry)",
    2: "middle click (Dark Souls III default: Lock On)",
}


class XTestMouse:
    """Send synthetic mouse buttons through the same XTest path soulsgym uses for keys."""

    def __init__(self):
        self._x11 = ctypes.CDLL("libX11.so.6")
        self._xtst = ctypes.CDLL("libXtst.so.6")
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._display = ctypes.c_void_p(self._x11.XOpenDisplay(None))
        if not self._display:
            raise RuntimeError("cannot open X display")
        self._xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]

    def click(self, button: int, hold: float = 0.08):
        self._xtst.XTestFakeButtonEvent(self._display, button, 1, 0)
        self._x11.XFlush(self._display)
        time.sleep(hold)
        self._xtst.XTestFakeButtonEvent(self._display, button, 0, 0)
        self._x11.XFlush(self._display)

    def shift_click(self, button: int, hold: float = 0.08):
        shift = self._x11.XKeysymToKeycode(
            self._display, self._x11.XStringToKeysym(b"Shift_L")
        )
        self._xtst.XTestFakeKeyEvent(self._display, shift, 1, 0)
        self._x11.XFlush(self._display)
        time.sleep(0.03)
        self.click(button, hold)
        self._xtst.XTestFakeKeyEvent(self._display, shift, 0, 0)
        self._x11.XFlush(self._display)


def watch(game, action, timeout_s: float = 1.8):
    """Run `action` and report which player animations appeared."""
    game.clear_cache()
    before = game.player_animation
    action()
    seen = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        game.clear_cache()
        seen.add(game.player_animation)
        time.sleep(0.05)
    seen.discard(before)
    return before, sorted(seen)[:4]


def main():
    print("[*] Attaching to DarkSoulsIII ...")
    game = DarkSoulsIII()
    game.game_speed = 1.0
    game.allow_attacks = True
    game.allow_moves = True
    mouse = XTestMouse()

    print("[*] Probing in 2 s. Dark Souls III must have focus.")
    time.sleep(2.0)

    results = {}
    for button, label in BUTTONS.items():
        before, seen = watch(game, lambda b=button: mouse.click(b))
        results[f"button{button}"] = bool(seen)
        print(f"[{'+' if seen else '!'}] {label:58} {before!r} -> {seen}")

    before, seen = watch(game, lambda: mouse.shift_click(1))
    results["shift+button1"] = bool(seen)
    print(f"[{'+' if seen else '!'}] {'shift + left click (default: Strong Attack)':58} "
          f"{before!r} -> {seen}")

    print()
    if results.get("button1"):
        print("[+] The game responds to a synthetic left click.")
        print("    Attacks can be driven through the mouse, so the game's key bindings")
        print("    can be left exactly as they are.")
    else:
        print("[!] Synthetic mouse clicks do nothing either.")
        print("    The bindings have to be changed inside the game after all.")


if __name__ == "__main__":
    main()
