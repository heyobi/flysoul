"""Script to release all keys in X11 so player stops moving."""
import ctypes
import time

x11 = ctypes.CDLL("libX11.so.6")
xtest = ctypes.CDLL("libXtst.so.6")

disp = x11.XOpenDisplay(None)
if not disp:
    print("Failed to open X display")
    exit(1)

# All standard Dark Souls III keys: w, a, s, d, q, e, r, f, space, shift, ctrl, alt, tab, etc.
key_names = [
    "w", "a", "s", "d", "q", "e", "r", "f", "c", "v", "x", "z", "g",
    "space", "Shift_L", "Control_L", "Alt_L", "Tab", "Escape"
]

print("[+] Releasing all X11 keys...")
for k in key_names:
    keysym = x11.XStringToKeysym(k.encode("ascii"))
    if keysym == 0:
        continue
    kc = x11.XKeysymToKeycode(disp, keysym)
    if kc != 0:
        # False means key up
        xtest.XTestFakeKeyEvent(disp, kc, False, 0)

x11.XFlush(disp)
x11.XCloseDisplay(disp)
print("[+] All X11 keys released.")
