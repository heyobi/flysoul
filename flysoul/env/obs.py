"""Single source of truth for turning a SoulsGym / Mock observation into combat state.

The encoder, the motor decoder, the telemetry dashboard and the training loop all used
to parse observations independently and disagreed with each other. In particular the
boss attack test was written as ``boss_animation > 0``, which is backwards: SoulsGym
encodes Iudex animations as integer IDs where **attacks occupy the low IDs and idle /
walking occupies the high ones**, so a standing boss read as "attacking" and
``Attack3000`` read as "idle".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Iudex animation IDs, assigned by soulsgym.core.static in category order:
#   attacks   -> 0 .. 18   (Attack3000-3015, Attack3029, ThrowAtk, ThrowDef)
#   movement  -> 19 .. 29  (WalkFront/Left/Right/BackBattle, IdleBattle, TurnBattle*, Fall, Land)
#   misc      -> 30 .. 31  (SABreak, DamageParryEnemy1)
#   unknown   -> -1
_IUDEX_ATTACK_ID_MAX = 18
_IUDEX_STAGGER_IDS = frozenset({30, 31})

# Offset between player_pose[3] and the world bearing convention used here.
#
# pose[3] is a yaw - it sweeps with the character, measured at 233 degrees over a
# circling strafe - but its zero is not the +x axis. Measured against the game with
# lock-on held, where the character provably faces the boss and the bearing must
# therefore read zero, `atan2(dy, dx) - pose[3]` came out at -146 degrees with a
# spread of 20 across 60 samples.
#
# Left uncorrected this is not a cosmetic error: the bearing feeds the retinal map,
# so the pursuit pathway and both strafe pathways were driven by a boss that appeared
# nearly behind the fly. Re-measure with scripts/fit_heading2.py if the game or the
# soulsgym pose reader changes.
PLAYER_HEADING_OFFSET = math.radians(146.0)


def _iudex_attack_ids() -> frozenset[int]:
    """Read the attack ID set from the installed soulsgym, falling back to the known range."""
    try:
        from soulsgym.core.static import boss_animations

        table = boss_animations["DarkSoulsIII"]["iudex"]["all"]
        ids = {meta["ID"] for meta in table.values() if meta.get("type") == "attacks"}
        if ids:
            return frozenset(ids)
    except Exception:
        pass
    return frozenset(range(0, _IUDEX_ATTACK_ID_MAX + 1))


IUDEX_ATTACK_IDS = _iudex_attack_ids()


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce a scalar, 0-d array or length-1 array observation field into a float."""
    if value is None:
        return default
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value).ravel()
        return float(arr[0]) if arr.size else default
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CombatState:
    """Normalised combat state shared by every FlySoul component."""

    player_hp: float = 1.0          # [0, 1]
    player_sp: float = 1.0          # [0, 1]
    boss_hp: float = 1.0            # [0, 1]
    distance: float = 8.0           # metres, horizontal
    angle: float = 0.0              # radians, boss bearing relative to player heading
    boss_attacking: bool = False    # boss is inside an attack animation
    boss_staggered: bool = False    # poise break / parried: the punish window
    boss_anim_time: float = 0.0     # seconds since the boss animation started
    boss_anim_id: int = -1          # raw SoulsGym animation ID, -1 if unknown
    player_anim_time: float = 0.0   # seconds since the player animation started
    player_can_act: bool = True     # player is not locked inside an animation
    lock_on: bool = True            # camera is locked on to the boss
    valid_actions: tuple[int, ...] = field(default_factory=tuple)  # SoulsGym action mask

    @property
    def boss_state_name(self) -> str:
        if self.boss_staggered:
            return "staggered"
        return "attacking" if self.boss_attacking else "active"


def parse_obs(obs: Any, info: dict | None = None) -> CombatState:
    """Translate a SoulsGym, Mock or vector observation into a :class:`CombatState`."""
    info = info or {}
    st = CombatState()

    if isinstance(obs, dict) and "player_pose" in obs and "boss_pose" in obs:
        # Official SoulsGym observation. Pose is (x, y, z, heading); x/y are the ground
        # plane and z is elevation, so horizontal distance uses x/y only.
        p_pose = np.asarray(obs["player_pose"], dtype=np.float32).ravel()
        b_pose = np.asarray(obs["boss_pose"], dtype=np.float32).ravel()
        dx = float(b_pose[0] - p_pose[0])
        dy = float(b_pose[1] - p_pose[1])
        st.distance = float(math.hypot(dx, dy))
        world_angle = math.atan2(dy, dx)
        heading = float(p_pose[3]) if p_pose.size > 3 else 0.0
        st.angle = float(
            (world_angle - heading + PLAYER_HEADING_OFFSET + math.pi) % (2.0 * math.pi)
            - math.pi
        )

        p_max_hp = max(1.0, _f(obs.get("player_max_hp"), 454.0))
        b_max_hp = max(1.0, _f(obs.get("boss_max_hp"), 1037.0))
        p_max_sp = max(1.0, _f(obs.get("player_max_sp"), 95.0))
        st.player_hp = float(np.clip(_f(obs.get("player_hp"), p_max_hp) / p_max_hp, 0.0, 1.0))
        st.player_sp = float(np.clip(_f(obs.get("player_sp"), p_max_sp) / p_max_sp, 0.0, 1.0))
        st.boss_hp = float(np.clip(_f(obs.get("boss_hp"), b_max_hp) / b_max_hp, 0.0, 1.0))

        st.boss_attacking, st.boss_staggered = _classify_boss_animation(obs.get("boss_animation"))
        st.boss_anim_id = int(_f(obs.get("boss_animation"), -1.0))
        st.boss_anim_time = _f(obs.get("boss_animation_duration"), 0.0)
        st.player_anim_time = _f(obs.get("player_animation_duration"), 0.0)
        st.lock_on = bool(_f(obs.get("lock_on"), 1.0) > 0.5)

    elif isinstance(obs, dict):
        # Mock environment / flat dict observation
        st.player_hp = float(np.clip(_f(obs.get("player_hp"), 1.0), 0.0, 1.0))
        st.player_sp = float(np.clip(_f(obs.get("player_sp"), 1.0), 0.0, 1.0))
        st.boss_hp = float(np.clip(_f(obs.get("boss_hp"), 1.0), 0.0, 1.0))
        st.distance = _f(obs.get("boss_distance"), 8.0)
        st.angle = _f(obs.get("boss_rel_angle"), 0.0)
        st.boss_attacking = _f(obs.get("boss_attacking"), 0.0) > 0.5
        st.boss_staggered = _f(obs.get("boss_staggered"), 0.0) > 0.5
        st.boss_anim_time = _f(obs.get("boss_animation_duration"), 0.0)

    elif isinstance(obs, (np.ndarray, list, tuple)):
        v = np.asarray(obs, dtype=np.float32).ravel()

        def at(i: int, default: float) -> float:
            return float(v[i]) if v.size > i else default

        st.player_hp = at(0, 1.0)
        st.player_sp = at(1, 1.0)
        st.boss_hp = at(2, 1.0)
        st.distance = at(3, 8.0)
        st.angle = at(4, 0.0)
        st.boss_attacking = at(5, 0.0) > 0.5

    # The environment knows better than we do which actions are currently executable.
    valid = info.get("valid_actions")
    if valid is not None:
        st.valid_actions = tuple(int(a) for a in valid)
        # SoulsGym drops any action outside this set; if only "do nothing" survives, the
        # player is mid-animation and genuinely cannot act.
        st.player_can_act = len(st.valid_actions) > 1
    return st


def _classify_boss_animation(animation: Any) -> tuple[bool, bool]:
    """Return ``(is_attacking, is_staggered)`` for a boss animation ID or name."""
    if animation is None:
        return False, False
    if isinstance(animation, str):
        low = animation.lower()
        if "sabreak" in low or "parry" in low:
            return False, True
        return ("attack" in low or "throw" in low), False
    anim_id = int(_f(animation, -1.0))
    if anim_id < 0:
        return False, False
    if anim_id in _IUDEX_STAGGER_IDS:
        return False, True
    return anim_id in IUDEX_ATTACK_IDS, False


def summarise(state: CombatState) -> dict:
    """Flatten a :class:`CombatState` for telemetry broadcast."""
    return {
        "player_hp": state.player_hp,
        "player_sp": state.player_sp,
        "boss_hp": state.boss_hp,
        "boss_distance": state.distance,
        "boss_angle": state.angle,
        "boss_attacking": state.boss_attacking,
        "boss_state": state.boss_state_name,
        "player_can_act": state.player_can_act,
        "lock_on": state.lock_on,
    }
