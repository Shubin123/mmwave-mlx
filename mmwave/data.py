"""Frame stacking and feature normalization shared by training and export.

docs/js/features.js mirrors this file; keep the two in sync.
"""
import numpy as np

from .prep import RECORDINGS

HISTORY = 3  # current frame + 2 previous (0.6 s at 5 Hz)
MAX_POINTS = 256
FEATURES = 6
POS_CENTER = np.array([-5.5, 0.0], np.float32)
POS_SCALE = 3.0


def featurize(p, age):
    """p: [n, 5] raw (x, y, v, snr, range) -> [n, 6] normalized features."""
    return np.stack(
        [
            p[:, 0] / 4.0,
            (p[:, 1] - 3.0) / 3.0,
            np.clip(p[:, 2], -6.0, 6.0) / 3.0,
            (p[:, 3] - 20.0) / 15.0,
            (p[:, 4] - 3.0) / 3.0,
            np.full(len(p), age - 1.0, np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def window(points, offsets, t):
    """Stack frames t-HISTORY+1..t, keep the MAX_POINTS strongest returns."""
    parts = []
    for age in range(HISTORY):
        f = t - age
        if f < 0:
            break
        parts.append(featurize(points[offsets[f] : offsets[f + 1]], age))
    x = np.concatenate(parts)
    snr = np.concatenate([points[offsets[t - a] : offsets[t - a + 1], 3] for a in range(len(parts))])
    if len(x) > MAX_POINTS:
        x = x[np.argsort(-snr, kind="stable")[:MAX_POINTS]]
    return x


def load(path="data/proc.npz"):
    z = np.load(path)
    recs = []
    for name in RECORDINGS:
        counts = z[f"{name}/counts"]
        recs.append(
            dict(
                name=name,
                points=z[f"{name}/points"],
                counts=counts,
                offsets=np.concatenate([[0], np.cumsum(counts)]),
                profile=z[f"{name}/profile"],
                pose=z[f"{name}/pose"],
                label=int(z[f"{name}/label"]),
            )
        )
    return recs


def split_indices(n, val=0.15, test=0.15):
    """Contiguous blocks per recording; a HISTORY-frame gap stops windows leaking across."""
    a = int(n * (1 - val - test))
    b = int(n * (1 - test))
    return (
        np.arange(0, a),
        np.arange(a + HISTORY, b),
        np.arange(b + HISTORY, n),
    )


def build(recs, which):
    xs, ns, ys, targets, rec_ids, frame_ids = [], [], [], [], [], []
    for r_i, r in enumerate(recs):
        idx = split_indices(len(r["counts"]))[which]
        for t in idx:
            w = window(r["points"], r["offsets"], t)
            pad = np.zeros((MAX_POINTS, FEATURES), np.float32)
            pad[: len(w)] = w
            xs.append(pad)
            ns.append(len(w))
            x, y, yaw = r["pose"][t]
            targets.append(
                [(x - POS_CENTER[0]) / POS_SCALE, (y - POS_CENTER[1]) / POS_SCALE, np.sin(yaw), np.cos(yaw)]
            )
            ys.append(r["label"])
            rec_ids.append(r_i)
            frame_ids.append(t)
    return dict(
        x=np.stack(xs),
        n=np.array(ns, np.int32),
        y=np.array(ys, np.int32),
        target=np.array(targets, np.float32),
        rec=np.array(rec_ids, np.int32),
        frame=np.array(frame_ids, np.int32),
    )
