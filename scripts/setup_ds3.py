"""Setup and verification script for Dark Souls III and SoulsGym on Linux host."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


def setup():
    steam_common = pathlib.Path.home() / ".local/share/Steam/steamapps/common"
    game_dir = steam_common / "DARK SOULS III" / "Game"
    compat_dir = pathlib.Path.home() / ".local/share/Steam/steamapps/compatdata/374320"
    user_reg = compat_dir / "pfx" / "user.reg"

    print("[1] Verifying game directory...")
    if not game_dir.exists():
        print(f"Error: Game directory not found at {game_dir}")
        return False
    print(f"  Found {game_dir}")

    # 1. Install speedhack proxy DLLs
    print("[2] Installing speedhack proxy DLLs...")
    try:
        import soulsgym.core.speedhack as sh
        proxy_dll = pathlib.Path(sh.__file__).parent / "_bin" / "speedhack.dll"
    except Exception as e:
        print(f"Error locating soulsgym speedhack: {e}")
        return False

    proton_dir = steam_common / "Proton - Experimental"
    wine_dinput8 = proton_dir / "files/lib/wine/x86_64-windows/dinput8.dll"
    if not wine_dinput8.exists():
        wine_dinput8 = proton_dir / "dist/lib64/wine/x86_64-windows/dinput8.dll"

    if not proxy_dll.exists():
        print(f"Error: Speedhack DLL not found at {proxy_dll}")
        return False
    if not wine_dinput8.exists():
        print(f"Error: Proton builtin dinput8.dll not found at {wine_dinput8}")
        return False

    target_dinput8 = game_dir / "dinput8.dll"
    target_hook = game_dir / "dinput8_hook.dll"

    shutil.copy(proxy_dll, target_dinput8)
    shutil.copy(wine_dinput8, target_hook)
    print(f"  Installed {target_dinput8.name} and {target_hook.name}")

    # 2. Configure Wine DllOverrides in user.reg
    print("[3] Configuring Wine DllOverrides in user.reg...")
    if user_reg.exists():
        text = user_reg.read_text(encoding="utf-8", errors="replace")
        section = "[Software\\\\Wine\\\\DllOverrides]"
        override_line = '"dinput8"="native,builtin"'

        if override_line not in text:
            pos = text.find(section)
            if pos != -1:
                end_pos = text.find("\n", pos)
                text = text[:end_pos + 1] + f"{override_line}\n" + text[end_pos + 1:]
                user_reg.write_text(text, encoding="utf-8")
                print("  Added 'dinput8'='native,builtin' to user.reg")
            else:
                print("  Warning: Section [Software\\Wine\\DllOverrides] not found in user.reg")
        else:
            print("  'dinput8' override already present in user.reg")
    else:
        print(f"  Note: {user_reg} does not exist yet (Proton will create on first run)")

    # 3. Check ptrace_scope
    print("[4] Checking ptrace_scope...")
    try:
        ptrace_val = pathlib.Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip()
        print(f"  kernel.yama.ptrace_scope = {ptrace_val}")
    except Exception as e:
        print(f"  Could not read ptrace_scope: {e}")

    # 4. Check NVIDIA GPU
    print("[5] Checking NVIDIA offload environment...")
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        print(f"  GPU detected: {res.stdout.strip()}")
    else:
        print(f"  nvidia-smi check failed: {res.stderr}")

    print("\n[SUCCESS] Setup and verification completed!")
    return True


if __name__ == "__main__":
    setup()
