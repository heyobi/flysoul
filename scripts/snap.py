import cv2
from soulsgym.core.game_window import GameWindow

try:
    gw = GameWindow("DarkSoulsIII")
    raw = gw.raw_img
    if raw is not None and raw.size > 0:
        cv2.imwrite("/tmp/snap.jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        print("[+] Saved /tmp/snap.jpg successfully")
    else:
        print("[-] raw_img was None or empty")
except Exception as e:
    print(f"Error: {e}")
