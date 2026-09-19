"""Concatenate per-run archives into one dataset, offsetting fight ids and padding the
`extra` columns when archives were written with different EXTRA_NAMES (older archives
lack `player_sp`; it is filled with 1.0, full stamina, for them).

    python scripts/merge_archives.py out.npz a.npz b.npz ...
"""

from __future__ import annotations

import sys

import numpy as np

KEYS = ["spikes", "channel", "reward", "terminal", "anim_t", "attacking", "damage_taken", "hit", "fight", "extra"]
DEFAULTS = {"player_sp": 1.0}


def merge(paths):
    parts = [dict(np.load(p).items()) for p in paths]
    names = []
    for z in parts:
        for n in list(z["extra_names"]):
            if n not in names:
                names.append(n)
    out = {k: [] for k in KEYS}
    off = 0
    for z in parts:
        zn = list(z["extra_names"])
        ex = np.zeros((len(z["fight"]), len(names)), dtype=np.float32)
        for j, n in enumerate(names):
            ex[:, j] = z["extra"][:, zn.index(n)] if n in zn else DEFAULTS.get(n, 0.0)
        z = dict(z)
        z["extra"] = ex
        z["fight"] = z["fight"] + off
        off = int(z["fight"].max()) + 1
        for k in KEYS:
            out[k].append(z[k])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["channels"] = parts[0]["channels"]
    out["extra_names"] = np.array(names)
    return out


def main() -> int:
    dst, srcs = sys.argv[1], sys.argv[2:]
    out = merge(srcs)
    np.savez_compressed(dst, **out)
    print(f"{dst}: {len(np.unique(out['fight']))} fights, {len(out['fight'])} steps, extras {list(out['extra_names'])[-3:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
