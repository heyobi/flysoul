"""Patch the installed soulsgym on the Linux host so the Iudex fight runs unattended.

1. Make SpeedHackConnector optional, so the game can be hooked without the dinput8 proxy
2. Accept the Event3000x boss animations instead of treating them as unknown
3. Restore the bonfire reload in _arena_setup, which is what lets the fight start at all
4. Give _camera_reset enough time to actually establish lock on

Every step is idempotent and safe to re-run. An earlier version of this script was not:
re-running it produced a SyntaxError in speedhack.py, which made `import soulsgym` fail,
which made the agent fall back to its offline simulator while still printing episode
results - so what looked like training against Iudex was training against a simulation.

    python scripts/patch_soulsgym.py
"""

import re
from pathlib import Path

VENV = Path("/home/ibox/flysoul/venv/lib/python3.12/site-packages/soulsgym")
CAMERA_TIMEOUT = 12.0


def patch_speedhack():
    """Let the connector run without the speed hack DLL instead of raising."""
    path = VENV / "core/speedhack/speedhack.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    before = text

    if "raise InjectionFailure" in text:
        text = text.replace(
            "raise InjectionFailure(\n"
            '                "Speed hack control port is not open. The DLL must be loaded as a '
            'dinput8 proxy at "\n'
            "                'game launch. Ensure the launch option "
            'WINEDLLOVERRIDES="dinput8=n,b" is used, then \'\n'
            '                "start the game."\n'
            "            )",
            'logger.warning("Speed hack disabled; running at normal 1.0x game speed.")\n'
            "            return",
        )

    # Guard update_game_speed against a missing socket. Adding the check unconditionally
    # nests it inside itself on a second run, which is a SyntaxError.
    if "if self.sock is not None:" not in text:
        text = text.replace(
            'self.sock.sendall(struct.pack("<f", value))',
            "if self.sock is not None:\n            "
            'self.sock.sendall(struct.pack("<f", value))',
        )
    # Repair an install already damaged by an earlier, unguarded run of this script.
    text, collapsed = re.subn(
        r"( +)if self\.sock is not None:\n(?: +if self\.sock is not None:\n)+",
        r"\1if self.sock is not None:\n",
        text,
    )
    if collapsed:
        print("[+] speedhack.py: collapsed duplicated sock guard")

    if text != before:
        path.write_text(text, encoding="utf-8")
        print("[+] speedhack.py: SpeedHackConnector is optional")
    else:
        print("[*] speedhack.py: already patched")


def patch_iudex():
    """Allow the Event animations, restore the arena reload, widen the camera timeout."""
    path = VENV / "envs/darksouls3/iudex.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    before = text

    if '("Walk", "Idle")' in text:
        text = text.replace('("Walk", "Idle")', '("Walk", "Idle", "Event")')
        print("[+] iudex.py: Event animations accepted")

    # --- restore the bonfire reload --------------------------------------------------
    #
    # An earlier version of this script replaced the reload with `pass`, meaning to stop
    # it throwing the player out of the arena. It does the opposite.
    #
    # _arena_setup runs: reload to the bonfire, teleport to the fog wall, verify the
    # player is standing there, then _enter_fog_gate() - and entering the fog gate is
    # what wakes Iudex up. The reload is what gets the player to a known position first.
    # Without it a player who ends up outside the arena is never brought back, the
    # fog-wall check raises ResetError, _arena_setup burns its five retries, and the
    # fight never starts. The boss then reads with an empty animation and takes no
    # damage at all, so every swing the agent lands passes straight through it.
    #
    # _arena_setup only runs when the arena needs setting up, not on every episode.
    disabled = (
        "# Disabled save reload to keep player inside arena with boss fight active\n"
        "        pass"
    )
    original = (
        'if np.linalg.norm(d_pos) > 10 or self.game.player_animation != "Idle":\n'
        "            self.game.reload()"
    )
    if disabled in text:
        text = text.replace(disabled, original, 1)
        print("[+] iudex.py: RESTORED the bonfire reload (the fight can start again)")
    elif original in text:
        print("[*] iudex.py: bonfire reload already present")

    # --- camera timeout ---------------------------------------------------------------
    #
    # _camera_reset loops until lock on is established, and that is not one button press:
    # SoulsEnv._lock_on walks the camera towards the boss with cameraleft/right/up/down
    # and only presses lock once the camera is within about 37 degrees, with each press
    # queued for the following iteration. Converging takes seconds of wall time.
    #
    # Cutting it short hands back an unlocked camera, and while unlocked SoulsGym
    # movement is camera-relative with no camera control in the action space - so the
    # agent runs off in whatever direction the camera was left pointing. The timeout is
    # only here to stop the loop hanging, so it should be generous.
    unpatched = "while not self.game.lock_on:"
    if unpatched in text:
        text = text.replace(
            unpatched,
            "t_cam = time.time()\n        while not self.game.lock_on and "
            f"(time.time() - t_cam < {CAMERA_TIMEOUT}):",
            1,
        )
        print(f"[+] iudex.py: camera timeout set to {CAMERA_TIMEOUT}s")
    else:
        text, bumped = re.subn(
            r"while not self\.game\.lock_on and \(time\.time\(\) - t_cam < [0-9.]+\)",
            f"while not self.game.lock_on and (time.time() - t_cam < {CAMERA_TIMEOUT})",
            text,
        )
        if bumped:
            print(f"[+] iudex.py: camera timeout raised to {CAMERA_TIMEOUT}s")

    if text != before:
        path.write_text(text, encoding="utf-8")


def patch_darksouls3():
    """Map the Event3000x animations onto IdleBattle instead of dropping them."""
    path = VENV / "games/darksouls3.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    target = 'if "SABlend" in animation or "Attack" in animation or "Part" in animation:'
    if target in text and 'if "Event3000" in animation:' not in text:
        replacement = (
            'if "Event3000" in animation:\n'
            '                return "IdleBattle"\n'
            "            " + target
        )
        text = text.replace(target, replacement, 1)
        path.write_text(text, encoding="utf-8")
        print("[+] darksouls3.py: Event3000x mapped to IdleBattle")
    else:
        print("[*] darksouls3.py: already patched")


def verify():
    """Refuse to report success if anything we touched no longer compiles."""
    import py_compile

    ok = True
    for rel in ("core/speedhack/speedhack.py", "envs/darksouls3/iudex.py", "games/darksouls3.py"):
        path = VENV / rel
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            ok = False
            print(f"[!] {rel} DOES NOT COMPILE: {exc}")
    print("[+] all patched modules compile" if ok else "[!] PATCH LEFT soulsgym BROKEN")
    return ok


def patch():
    patch_speedhack()
    patch_iudex()
    patch_darksouls3()
    return verify()


if __name__ == "__main__":
    raise SystemExit(0 if patch() else 1)
