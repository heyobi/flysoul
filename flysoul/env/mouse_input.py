"""Let SoulsGym drive Dark Souls III's mouse bindings, not just its keyboard ones.

SoulsGym presses keys: ``l`` for light attack, ``h`` for heavy, ``m`` for parry. Dark
Souls III binds all three to the mouse out of the box, and its Linux input backend only
sends ``XTestFakeKeyEvent``. The result is an agent whose movement works perfectly -
WASD and space happen to match the game's defaults - while every attack it asks for is
delivered to a key the game has never heard of.

Measured on the live game, that is exactly what happened: 83 swings issued from inside
melee range, closest approach 1.22 m, boss at 1037/1037 at the end of it. No policy can
recover from an input that never arrives, and from the outside it looks indistinguishable
from an agent that simply never learned to attack.

XTest can send button events as well as key events, and the library is already loaded by
the backend. So rather than asking for the game's bindings to be changed, this module
teaches the backend to understand mouse pseudo-keys and points the attack actions at the
buttons the game already listens to:

    left click          -> AttackRightLight1
    shift + left click  -> AttackRightHeavy1Start
    right click         -> GuardStart (left-hand weapon / shield)

Applied at runtime against the imported classes, so the soulsgym install on disk is
left untouched.
"""

from __future__ import annotations

import ctypes
import logging
import time

logger = logging.getLogger(__name__)

# X11 button numbers.
MOUSE_BUTTONS = {"mouse1": 1, "mouse2": 2, "mouse3": 3, "mouse4": 4, "mouse5": 5}

# Dark Souls III's own defaults, which is the point: nothing in the game changes.
DEFAULT_MOUSE_BINDINGS = {
    "lightattack": "mouse1",
    "heavyattack": "shift+mouse1",
    "parry": "mouse3",
}

_MODIFIER_KEYSYMS = {"shift": b"Shift_L", "ctrl": b"Control_L", "alt": b"Alt_L"}

_PATCH_FLAG = "_flysoul_mouse_patched"


def _parse(key: str):
    """Split a binding into ``(modifier or None, button number)``, or None if it is a key."""
    if not isinstance(key, str):
        return None
    modifier = None
    token = key
    if "+" in key:
        modifier, token = key.split("+", 1)
        modifier = modifier.strip().lower()
        if modifier not in _MODIFIER_KEYSYMS:
            return None
    button = MOUSE_BUTTONS.get(token.strip().lower())
    if button is None:
        return None
    return modifier, button


def _install_backend_patch(x11_keyboard_cls) -> bool:
    """Teach an _X11Keyboard class to press mouse buttons as well as keys."""
    if getattr(x11_keyboard_cls, _PATCH_FLAG, False):
        return False

    original_press = x11_keyboard_cls.press
    original_release = x11_keyboard_cls.release

    def _ensure_mouse(self):
        """Bind XTestFakeButtonEvent on the display the backend already opened."""
        if getattr(self, "_flysoul_mouse_ready", False):
            return
        self._xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._flysoul_mouse_ready = True

    def _modifier_keycode(self, modifier: str) -> int:
        return self._x11.XKeysymToKeycode(
            self._display, self._x11.XStringToKeysym(_MODIFIER_KEYSYMS[modifier])
        )

    def press(self, key: str):
        parsed = _parse(key)
        if parsed is None:
            return original_press(self, key)
        modifier, button = parsed
        _ensure_mouse(self)
        if modifier:
            self._xtst.XTestFakeKeyEvent(self._display, _modifier_keycode(self, modifier), 1, 0)
            self._x11.XFlush(self._display)
            # The game samples the modifier when the click arrives, not before it.
            time.sleep(0.02)
        self._xtst.XTestFakeButtonEvent(self._display, button, 1, 0)
        self._x11.XFlush(self._display)

    def release(self, key: str):
        parsed = _parse(key)
        if parsed is None:
            return original_release(self, key)
        modifier, button = parsed
        _ensure_mouse(self)
        self._xtst.XTestFakeButtonEvent(self._display, button, 0, 0)
        if modifier:
            self._xtst.XTestFakeKeyEvent(self._display, _modifier_keycode(self, modifier), 0, 0)
        self._x11.XFlush(self._display)

    x11_keyboard_cls.press = press
    x11_keyboard_cls.release = release
    setattr(x11_keyboard_cls, _PATCH_FLAG, True)
    return True


def enable_mouse_bindings(bindings: dict[str, str] | None = None, game_id: str = "DarkSoulsIII"):
    """Point the attack actions at the mouse buttons the game already responds to.

    Safe to call more than once, and a no-op on Windows or on a soulsgym build without
    the X11 backend. Returns the bindings that ended up in effect, or an empty dict if
    nothing was changed.
    """
    try:
        from soulsgym.core import game_input as gi
        from soulsgym.core.static import keybindings
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.debug(f"soulsgym not importable, leaving input alone: {exc}")
        return {}

    x11_cls = getattr(gi, "_X11Keyboard", None)
    if x11_cls is None:
        logger.debug("No X11 backend in this soulsgym build; leaving input alone.")
        return {}
    _install_backend_patch(x11_cls)

    table = keybindings.get(game_id)
    if table is None:
        return {}
    applied = {}
    for action, key in (bindings or DEFAULT_MOUSE_BINDINGS).items():
        if action in table:
            table[action] = key
            applied[action] = key
    if applied:
        logger.info(f"Attack actions routed to the mouse: {applied}")
    return applied
