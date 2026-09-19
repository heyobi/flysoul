"""3b - SYNTHETIC: raw game-state features for the synthetic Q readout.

The fly's Kenyon code is a lossy random projection of the game state (measured: the
outcome ceiling of the raw state is +0.47 against +0.39 for the code). A synthetic
readout is outside the brain anyway, so it may also see the raw state, binned one-hot
the way SoulsAI's network saw it. This module is the single definition used both when
fitting on archives and when acting live, so the two can never drift apart.
"""

from __future__ import annotations

import numpy as np

from flysoul.connectome.graph import ACTION_CHANNELS

DIST_EDGES = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10], dtype=np.float64)
ANGLE_EDGES = np.linspace(-150, 150, 11)
PHASE_EDGES = np.array([0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2], dtype=np.float64)
HP_EDGES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
BHP_EDGES = np.array([0.25, 0.5, 0.75], dtype=np.float64)
SP_EDGES = np.array([0.15, 0.35, 0.6, 0.85], dtype=np.float64)  # stamina; archives before 2026-09-19 lack it (assumed full)
ANIM_SLOTS = 32
N_ACTIONS = len(ACTION_CHANNELS) + 1  # + idle

RAW_DIM = (ANIM_SLOTS * (len(PHASE_EDGES) + 1) + (len(DIST_EDGES) + 1) + (len(ANGLE_EDGES) + 1)
           + (len(HP_EDGES) + 1) + (len(BHP_EDGES) + 1) + (len(SP_EDGES) + 1) + 3 + N_ACTIONS)
CONT_DIM = 6  # distance/8, angle/180, anim_t/2, player_hp, boss_hp, player_sp as smooth inputs


def cont_features(distance, angle, player_hp, boss_hp, anim_t, player_sp) -> np.ndarray:
    return np.array([min(distance, 12.0) / 8.0, angle / 180.0, min(anim_t, 3.0) / 2.0,
                     player_hp, boss_hp, player_sp], dtype=np.float32)


def cont_features_from_archive(z) -> np.ndarray:
    names = list(z["extra_names"])
    ex = z["extra"]
    col = lambda k: ex[:, names.index(k)]
    n = len(z["fight"])
    sp = col("player_sp") if "player_sp" in names else np.ones(n, dtype=np.float32)
    out = np.zeros((n, CONT_DIM), dtype=np.float32)
    out[:, 0] = np.minimum(col("distance"), 12.0) / 8.0
    out[:, 1] = col("angle") / 180.0
    out[:, 2] = np.minimum(z["anim_t"], 3.0) / 2.0
    out[:, 3] = col("player_hp")
    out[:, 4] = col("boss_hp")
    out[:, 5] = sp
    return out


def _onehot(v: float, edges: np.ndarray) -> np.ndarray:
    out = np.zeros(len(edges) + 1, dtype=np.float32)
    out[int(np.digitize(v, edges))] = 1.0
    return out


def raw_features(distance: float, angle: float, player_hp: float, boss_hp: float, staggered: bool,
                 can_act: bool, anim_id: int, anim_t: float, attacking: bool, prev_action: int,
                 player_sp: float = 1.0) -> np.ndarray:
    """One step's raw state as a fixed-length one-hot vector (RAW_DIM)."""
    nb = len(PHASE_EDGES) + 1
    idph = np.zeros(ANIM_SLOTS * nb, dtype=np.float32)
    if anim_id >= 0:
        idph[(int(anim_id) % ANIM_SLOTS) * nb + int(np.digitize(anim_t, PHASE_EDGES))] = 1.0
    prev = np.zeros(N_ACTIONS, dtype=np.float32)
    if 0 <= prev_action < N_ACTIONS:
        prev[prev_action] = 1.0
    return np.concatenate([
        idph, _onehot(distance, DIST_EDGES), _onehot(angle, ANGLE_EDGES), _onehot(player_hp, HP_EDGES),
        _onehot(boss_hp, BHP_EDGES), _onehot(player_sp, SP_EDGES),
        np.array([float(attacking), float(staggered), float(can_act)], dtype=np.float32), prev,
    ])


def raw_features_from_archive(z) -> np.ndarray:
    names = list(z["extra_names"])
    ex = z["extra"]
    col = lambda k: ex[:, names.index(k)]
    n = len(z["fight"])
    ch = z["channel"].astype(int)
    sp = col("player_sp") if "player_sp" in names else np.ones(n, dtype=np.float32)
    out = np.zeros((n, RAW_DIM), dtype=np.float32)
    for i in range(n):
        prev = -1
        if i > 0 and z["fight"][i - 1] == z["fight"][i]:
            prev = ch[i - 1] if ch[i - 1] >= 0 else N_ACTIONS - 1
        out[i] = raw_features(float(col("distance")[i]), float(col("angle")[i]), float(col("player_hp")[i]),
                              float(col("boss_hp")[i]), bool(col("boss_staggered")[i] > 0.5),
                              bool(col("player_can_act")[i] > 0.5), int(col("boss_anim_id")[i]),
                              float(z["anim_t"][i]), bool(z["attacking"][i]), prev, float(sp[i]))
    return out
