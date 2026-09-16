"""Patch soulsgym on Linux to handle Gundyr's Event30002 animation gracefully."""

from pathlib import Path

def patch():
    # 1. Patch iudex.py
    p1 = Path("/home/ibox/flysoul/venv/lib/python3.12/site-packages/soulsgym/envs/darksouls3/iudex.py")
    if p1.exists():
        text = p1.read_text(encoding="utf-8")
        if '("Walk", "Idle")' in text:
            text = text.replace('("Walk", "Idle")', '("Walk", "Idle", "Event")')
            p1.write_text(text, encoding="utf-8")
            print("[+] Patched soulsgym iudex.py: allowed Event in boss animations")
        else:
            print("[*] iudex.py already patched")

    # 2. Patch darksouls3.py
    p2 = Path("/home/ibox/flysoul/venv/lib/python3.12/site-packages/soulsgym/games/darksouls3.py")
    if p2.exists():
        text2 = p2.read_text(encoding="utf-8")
        target = 'if "SABlend" in animation or "Attack" in animation or "Part" in animation:'
        replacement = 'if "Event3000" in animation:\n                return "IdleBattle"\n            if "SABlend" in animation or "Attack" in animation or "Part" in animation:'
        if target in text2 and 'if "Event3000" in animation:' not in text2:
            text2 = text2.replace(target, replacement, 1)
            p2.write_text(text2, encoding="utf-8")
            print("[+] Patched soulsgym darksouls3.py: mapped Event3000 to IdleBattle")
        else:
            print("[*] darksouls3.py already patched")

if __name__ == "__main__":
    patch()
