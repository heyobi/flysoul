"""Patch SoulsGym iudex.py for robust arena entry and detection."""

import pathlib

iudex_file = pathlib.Path.home() / "flysoul/venv/lib/python3.12/site-packages/soulsgym/envs/darksouls3/iudex.py"
text = iudex_file.read_text(encoding="utf-8")

# 1. Add arena check to _arena_setup so it doesn't kill player if already in arena
old_arena_setup = '''    def _arena_setup(self):
        """Set up the arena."""
        self.game.game_speed = 3'''

new_arena_setup = '''    def _arena_setup(self):
        """Set up the arena."""
        # If player is already inside the arena, mark initialized and proceed
        p_pos = self.game.player_pose[:3]
        if (self.ARENA_LIM_LOW[0] <= p_pos[0] <= self.ARENA_LIM_HIGH[0] and
            self.ARENA_LIM_LOW[1] <= p_pos[1] <= self.ARENA_LIM_HIGH[1]):
            self._arena_init = True
            return
        self.game.game_speed = 3'''

if old_arena_setup in text:
    text = text.replace(old_arena_setup, new_arena_setup, 1)
    print("[+] Patched _arena_setup successfully.")
else:
    print("[!] _arena_setup pattern not found or already patched.")

# 2. Make _enter_fog_gate more forgiving and retry interact
old_fog_gate = '''    def _enter_fog_gate(self):
        """Enter the fog gate."""
        self.game.camera_pose = self.CAM_SETUP_POSE
        self._game_input.single_action("interact")'''

new_fog_gate = '''    def _enter_fog_gate(self):
        """Enter the fog gate."""
        p_pos = self.game.player_pose[:3]
        if (self.ARENA_LIM_LOW[0] <= p_pos[0] <= self.ARENA_LIM_HIGH[0] and
            self.ARENA_LIM_LOW[1] <= p_pos[1] <= self.ARENA_LIM_HIGH[1]):
            return
        self.game.camera_pose = self.CAM_SETUP_POSE
        self.game.sleep(0.2)
        for _ in range(3):
            self._game_input.single_action("interact", 0.1)
            self.game.sleep(0.15)'''

if old_fog_gate in text:
    text = text.replace(old_fog_gate, new_fog_gate, 1)
    print("[+] Patched _enter_fog_gate successfully.")
else:
    print("[!] _enter_fog_gate pattern not found or already patched.")

# 3. Allow player animation to be Idle or Move during entity reset check
old_check = 'if self.game.player_animation != "Idle":'
new_check = 'if self.game.player_animation not in ("Idle", "Move"):'
if old_check in text:
    text = text.replace(old_check, new_check, 1)
    print("[+] Patched _entity_reset_check successfully.")

# 4. Ensure keys are released at the start of _entity_reset
old_ent_reset = '    def _entity_reset(self):\n        """Reset the player and boss HP and reset their poses."""'
new_ent_reset = '    def _entity_reset(self):\n        """Reset the player and boss HP and reset their poses."""\n        self._game_input.reset()'
if old_ent_reset in text:
    text = text.replace(old_ent_reset, new_ent_reset, 1)
    print("[+] Patched _entity_reset key release successfully.")

iudex_file.write_text(text, encoding="utf-8")
print("[+] Patching complete.")
