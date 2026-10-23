"""Download the synchronized radar+Vicon CSVs and convert them to a compact npz.

Each recording becomes a flat point array plus per-frame offsets, labels and poses.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DATASET = "sahilrajpurkar/mmwave-dataset"
SYNC_DIR = "01_SYNCRONIZED_RADAR-VICON_DATASET"
DYNAMIC = [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2), (4, 1)]
STATIC = {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4, 5], 4: [1, 2, 4, 5]}
RECORDINGS = [f"DYNAMIC_OBJECT{o}_MOVEMENT{m}" for o, m in DYNAMIC] + [
    f"STATIC_OBJECT{o}_ORIENTATION{r}" for o in STATIC for r in STATIC[o]
]
CLASSES = ["Forklift + KLT", "Forklift", "Workstation + Robot", "Robot"]
POINT_COLS = ["x", "y", "v", "snr", "range"]


def download(cache):
    os.environ.setdefault("KAGGLEHUB_CACHE", cache)
    import kagglehub

    paths = [
        kagglehub.dataset_download(DATASET, path=f"{SYNC_DIR}/RADAR+VICON_{name}.csv")
        for name in RECORDINGS
    ]
    return os.path.dirname(paths[0])


def parse(path):
    df = pd.read_csv(path)
    cols = {c: [np.asarray(json.loads(s), np.float32) for s in df[c]] for c in POINT_COLS}
    counts = np.array([len(a) for a in cols["x"]], np.int32)
    points = np.stack([np.concatenate(cols[c]) for c in POINT_COLS], axis=1)
    profile = np.stack([np.asarray(json.loads(s), np.float32) for s in df.rangeProfile])
    pose = df[["pos_mm_x", "pos_mm_y", "angle_y"]].to_numpy(np.float32)
    return points, counts, profile, pose, df.timestamp.to_numpy(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="directory with RADAR+VICON_*.csv (downloads if omitted)")
    ap.add_argument("--out", default="data/proc.npz")
    args = ap.parse_args()
    src = args.src or download(os.path.abspath("data/.kh"))

    arrays = {}
    for i, name in enumerate(RECORDINGS):
        points, counts, profile, pose, ts = parse(os.path.join(src, f"RADAR+VICON_{name}.csv"))
        obj = int(name.split("OBJECT")[1][0]) - 1
        arrays |= {
            f"{name}/points": points,
            f"{name}/counts": counts,
            f"{name}/profile": profile,
            f"{name}/pose": pose,
            f"{name}/ts": ts,
            f"{name}/label": np.int32(obj),
        }
        print(f"{name:32s} frames={len(counts):5d} pts/frame={counts.mean():5.1f} class={CLASSES[obj]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    allp = np.concatenate([arrays[f"{n}/points"] for n in RECORDINGS])
    for j, c in enumerate(POINT_COLS):
        q = np.percentile(allp[:, j], [0, 1, 50, 99, 100])
        print(f"{c:6s} " + " ".join(f"{v:8.2f}" for v in q))


if __name__ == "__main__":
    main()
