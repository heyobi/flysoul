"""Patch soulsgym on Linux to handle Gundyr's Event30002 animation, fog wall, and camera reset gracefully."""

import time
from pathlib import Path

def patch():
    # 1. Patch iudex.py
    p1 = Path("/home/ibox/flysoul/venv/lib/python3.12/site-packages/soulsgym/envs/darksouls3/iudex.py")
    if p1.exists():
        text = p1.read_text(encoding="utf-8")
        if '("Walk", "Idle")' in text:
            text = text.replace('("Walk", "Idle")', '("Walk", "Idle", "Event")')
            print("[+] Patched soulsgym iudex.py: allowed Event in boss animations")

        target_fog = 'post_fog_wall_pos = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"][:3]'
        replacement_fog = 'self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"]\n        post_fog_wall_pos = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"][:3]'
        if target_fog in text and 'self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"]' not in text:
            text = text.replace(target_fog, replacement_fog, 1)
            print("[+] Patched soulsgym iudex.py: auto-teleport post_fog_wall")

        target_cam = 'while not self.game.lock_on:'
        replacement_cam = 't_cam = time.time()\n        while not self.game.lock_on and (time.time() - t_cam < 1.5):'
        if target_cam in text:
            text = text.replace(target_cam, replacement_cam, 1)
            print("[+] Patched soulsgym iudex.py: added 1.5s timeout to camera lock_on")

        p1.write_text(text, encoding="utf-8")

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
