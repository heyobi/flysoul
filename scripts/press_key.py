"""Press one key in the game host's X display, e.g. to close a menu opened by accident.

The agent drives the game through X11 XTest, so a stray keypress from a remote viewer
(Moonlight, VNC) that opens the pause menu freezes the fight and blocks the reset. This
sends the key the same way the agent does, without stopping it.

    DISPLAY=:0 python scripts/press_key.py Escape
    python scripts/press_key.py Escape --hold 0.08
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time


def press(name: str, hold: float = 0.06) -> bool:
    x11 = ctypes.CDLL("libX11.so.6")
    xtst = ctypes.CDLL("libXtst.so.6")
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    x11.XStringToKeysym.restype = ctypes.c_ulong
    x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XKeysymToKeycode.restype = ctypes.c_ubyte
    xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

    disp = x11.XOpenDisplay(None)
    if not disp:
        print("[!] cannot open X display; set DISPLAY and XAUTHORITY")
        return False
    keysym = x11.XStringToKeysym(name.encode())
    if keysym == 0:
        print(f"[!] unknown key name {name!r}")
        x11.XCloseDisplay(disp)
        return False
    code = x11.XKeysymToKeycode(disp, keysym)
    xtst.XTestFakeKeyEvent(disp, code, 1, 0)
    x11.XFlush(disp)
    time.sleep(hold)
    xtst.XTestFakeKeyEvent(disp, code, 0, 0)
    x11.XFlush(disp)
    x11.XCloseDisplay(disp)
    print(f"[+] pressed {name} (keycode {code})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", help="X keysym name, e.g. Escape, q, space")
    ap.add_argument("--hold", type=float, default=0.06)
    args = ap.parse_args()
    return 0 if press(args.key, args.hold) else 1


if __name__ == "__main__":
    sys.exit(main())
