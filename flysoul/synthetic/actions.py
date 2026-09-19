"""3b - SYNTHETIC: the synthetic readout's action set, with directional rolls.

The fly's descending pools carry one "roll" channel whose direction comes from whichever
locomotor pool is co-active (flysoul.motor.decoder.to_soulsgym_action). The synthetic
readout may instead choose the roll direction itself, as SoulsAI's agent could: the
action set here splits "roll" into four directional rolls (SoulsGym ids 8/10/12/14).

Archived fights recorded the channel ("roll") and the motor rates, so their rolls can
be relabelled with the direction that was actually executed, by the same rule the
decoder used; from 2026-09-19 the executed SoulsGym action id is also archived
(`action_id` in EXTRA_NAMES) and takes precedence.
"""

from __future__ import annotations

import numpy as np

from flysoul.connectome.graph import ACTION_CHANNELS
from flysoul.motor.decoder import SOULSGYM_ATTACK, SOULSGYM_IDLE, SOULSGYM_ROLL, SOULSGYM_WALK

ROLL_DIRS = ["advance", "strafe_right", "retreat", "strafe_left"]
SYNTH_ACTIONS = [c for c in ACTION_CHANNELS if c != "roll"] + [f"roll_{d}" for d in ROLL_DIRS] + ["idle"]
CHANNEL_ACTIONS = list(ACTION_CHANNELS) + ["idle"]
_ID_TO_DIR = {v: k for k, v in SOULSGYM_ROLL.items()}


def to_channel(name: str) -> str:
    """The descending channel a synthetic action is an instance of (for the efference copy)."""
    return "roll" if name.startswith("roll_") else name


def channel_index(name: str) -> int:
    c = to_channel(name)
    return ACTION_CHANNELS.index(c) if c in ACTION_CHANNELS else len(ACTION_CHANNELS)


def game_action_id(name: str) -> int | None:
    """SoulsGym action id for a synthetic action; None when the decoder should decide (idle/walk/attacks)."""
    if name.startswith("roll_"):
        return SOULSGYM_ROLL[name[5:]]
    if name in SOULSGYM_ATTACK:
        return SOULSGYM_ATTACK[name]
    if name in SOULSGYM_WALK:
        return SOULSGYM_WALK[name]
    if name == "idle":
        return SOULSGYM_IDLE
    return None


def relabel_archive(z, actions: list[str]) -> np.ndarray:
    """Index into `actions` for every archived step. With the directional set, rolls are
    resolved from the archived action id when present, else from the motor rates."""
    ch = z["channel"].astype(int)
    names = list(z["extra_names"])
    ex = z["extra"]
    n = len(ch)
    out = np.full(n, actions.index("idle"), dtype=np.int64)
    directional = any(a.startswith("roll_") for a in actions)
    rate_cols = {d: ex[:, names.index(f"rate_{d}")] for d in ROLL_DIRS if f"rate_{d}" in names}
    aid = ex[:, names.index("action_id")] if "action_id" in names else None
    for i in range(n):
        if ch[i] < 0:
            continue
        c = ACTION_CHANNELS[ch[i]]
        if c == "roll" and directional:
            d = None
            if aid is not None and aid[i] >= 0 and int(aid[i]) in _ID_TO_DIR:
                d = _ID_TO_DIR[int(aid[i])]
            if d is None:
                if rate_cols:
                    best = max(ROLL_DIRS, key=lambda k: rate_cols[k][i] if k in rate_cols else -1.0)
                    d = best if rate_cols.get(best, np.zeros(n))[i] > 0.0 else "advance"
                else:
                    d = "advance"
            out[i] = actions.index(f"roll_{d}")
        elif c in actions:
            out[i] = actions.index(c)
    return out
