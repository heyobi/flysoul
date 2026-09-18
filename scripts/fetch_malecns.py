"""Fetch real MaleCNS v1.0 neurons for the visualizer: somata and skeletons by cell type.

The circuit in `flysoul/connectome/graph.py` is built from the cell types and wiring
statistics of the MaleCNS connectome, not from its individual neurons, and until now
the 3D view placed the model's neurons on a hand-drawn layout. This pulls the public
MaleCNS release (CC-BY, Google Research / HHMI Janelia) so every modelled population
can be shown on the somata and skeletons of real neurons of the same type:

    body annotations  -> bodyId, type, soma position (8 nm voxels)
    skeletons (SWC)   -> one polyline tree per neuron, downsampled here

Output: flysoul/visualizer/malecns.json, a few MB, read by the web server at start-up.
Nothing about the fight or the learning changes; this is what is drawn.

    python scripts/fetch_malecns.py                    # ~5 min, ~400 skeletons
    python scripts/fetch_malecns.py --skeletons 40     # fewer skeletons per family

Sources: https://male-cns.janelia.org/download/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
ANNOTATIONS_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                   "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather")
SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"
VOXEL_UM = 0.008  # 8 nm voxels

# Model population family -> (type regex on the annotation table, how many somata to
# take, how many of those to also fetch skeletons for). The families are the ones the
# visualizer already knows (see FAMILIES in index.html).
FAMILIES = {
    "Retina_OpticLobe":           (r"^(Mi1|Tm3)$",                      128, 40),
    "Lobula_Motion":              (r"^(LC4|LPLC2)$",                     96, 48),
    "Compass_EB":                 (r"^EPG$",                             32, 32),
    "Compass_Ring_Inhibitory":    (r"^Delta7$",                          42, 42),
    "Central_Complex":            (r"^(PFN|hDelta|vDelta|PFL|PEN|FB)",  512, 60),
    "Central_Complex_Inhibitory": (r"^(EL|ER)\w*",                       96, 24),
    "Kenyon_Cell":                (r"^KC",                             1024, 160),
    "APL_Inhibitory":             (r"^APL$",                              2, 2),
    "MBON_":                      (r"^MBON",                             97, 97),
    "Dopamine_PPL1":              (r"^PPL1",                             16, 16),
    "Dopamine_PAM":               (r"^PAM",                              64, 40),
    "Nociceptor":                 (r"^AN\d",                             16, 8),
    "Proprioceptor":              (r"^AN\d",                             32, 8),
    "Vigor_Gate_Inhibitory":      (r"^(oviIN|SIFa|SAG)",                 16, 8),
    "Approach_Brake_Inhibitory":  (r"^(oviIN|SIFa|SAG)",                 16, 8),
    "Premotor_Inhibitory_":       (r"^DN(b|g)",                          48, 24),
    "Motor_":                     (r"^DN(a|p)",                         112, 80),
}


def load_annotations():
    import pandas as pd
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "body-annotations.feather"
    if not path.exists():
        print(f"downloading annotations (14 MB) ...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, path)
    df = pd.read_feather(path)
    df = df[df["somaLocation"].notna() & df["type"].notna()].copy()
    df["type"] = df["type"].astype(str)
    return df


def parse_swc(text: str, max_points: int) -> list | None:
    """SWC -> list of segments [[x,y,z],[x,y,z]] in voxel units, downsampled to about
    max_points nodes while keeping the tree connected."""
    ids, xyz, parent = [], [], []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        ids.append(int(parts[0])); xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
        parent.append(int(parts[6]))
    n = len(ids)
    if n < 2:
        return None
    index = {i: k for k, i in enumerate(ids)}
    stride = max(1, n // max_points)
    keep = set(range(0, n, stride)) | {k for k, p in enumerate(parent) if p == -1}
    segs = []
    for k in keep:
        # walk up to the nearest kept ancestor
        p = parent[k]
        hops = 0
        while p != -1 and index.get(p) is not None and index[p] not in keep and hops < 10000:
            p = parent[index[p]]; hops += 1
        if p != -1 and index.get(p) is not None:
            a, b = xyz[k], xyz[index[p]]
            segs.append([[round(a[0]), round(a[1]), round(a[2])], [round(b[0]), round(b[1]), round(b[2])]])
    return segs or None


def fetch_swc(body_id: int, max_points: int):
    try:
        with urllib.request.urlopen(SWC_URL.format(body_id), timeout=60) as r:
            return body_id, parse_swc(r.read().decode("utf-8", "replace"), max_points)
    except Exception:
        return body_id, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", type=int, default=None, help="cap skeletons per family")
    ap.add_argument("--points", type=int, default=90, help="nodes kept per skeleton")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "flysoul" / "visualizer" / "malecns.json"))
    args = ap.parse_args()

    df = load_annotations()
    rng = np.random.default_rng(args.seed)
    picked: dict[str, list] = {}
    to_fetch: list[tuple[str, int]] = []
    for fam, (pattern, n_soma, n_skel) in FAMILIES.items():
        sub = df[df["type"].str.match(pattern)]
        if len(sub) == 0:
            print(f"  {fam:28} no match for {pattern}")
            continue
        take = sub.sample(n=min(n_soma, len(sub)), random_state=int(rng.integers(1 << 30)))
        rows = []
        for _, r in take.iterrows():
            loc = np.asarray(r["somaLocation"], dtype=float).tolist()
            rows.append({"bodyId": int(r["bodyId"]), "type": r["type"],
                         "side": None if r.get("somaSide") is None else str(r.get("somaSide")),
                         "soma": [round(v) for v in loc], "segments": None})
        picked[fam] = rows
        n_skel = min(n_skel, len(rows)) if args.skeletons is None else min(args.skeletons, len(rows))
        to_fetch += [(fam, rows[i]["bodyId"]) for i in range(n_skel)]
        print(f"  {fam:28} {len(rows):5d} somata from {len(sub):5d} cells of {pattern}; {n_skel} skeletons")

    print(f"fetching {len(to_fetch)} skeletons ...")
    t0 = time.perf_counter()
    got = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for body_id, segs in ex.map(lambda t: fetch_swc(t[1], args.points), to_fetch):
            got[body_id] = segs
    ok = sum(1 for v in got.values() if v)
    print(f"  {ok}/{len(to_fetch)} skeletons in {time.perf_counter() - t0:.0f}s")
    for fam, rows in picked.items():
        for r in rows:
            r["segments"] = got.get(r["bodyId"])

    # Scene frame: centre on the mean soma, voxels -> micrometres.
    all_soma = np.array([r["soma"] for rows in picked.values() for r in rows], dtype=float)
    centre = all_soma.mean(axis=0)
    extent = (all_soma.max(axis=0) - all_soma.min(axis=0)) * VOXEL_UM
    out = {
        "dataset": "MaleCNS v1.0 (Google Research / HHMI Janelia, CC-BY)",
        "source": "https://male-cns.janelia.org/download/",
        "units": "voxels of 8 nm; centre subtracted in the viewer",
        "voxel_um": VOXEL_UM,
        "centre": [round(v) for v in centre.tolist()],
        "extent_um": [round(v) for v in extent.tolist()],
        "families": picked,
    }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB); brain extent {extent.round()} um")
    return 0


if __name__ == "__main__":
    sys.exit(main())
