"""Test screen capture speed and resolution from Dark Souls III."""
import time
from soulsgym.games.darksouls3 import DarkSoulsIII

print("[+] Initializing DarkSoulsIII...")
g = DarkSoulsIII()

print(f"Current img resolution: {g.img_resolution}")
print(f"Current window resolution: {g.window_resolution}")

# Test default 90x160 capture
t0 = time.time()
img1 = g.img
t1 = time.time()
print(f"Default capture: shape={img1.shape}, time={(t1-t0)*1000:.2f}ms")

# Test higher resolution like 225x400 or 450x800 for the web box
g.img_resolution = (225, 400)
t0 = time.time()
img2 = g.img
t2 = time.time()
print(f"400x225 capture: shape={img2.shape}, time={(t2-t0)*1000:.2f}ms")
