"""Export weights, metrics and quantized recordings for the browser app in docs/.

Recording binary layout (little-endian), one file per recording:
  pose    float32[frames, 3]   Vicon x, y (m) and yaw (rad)
  points  int16[total, 5]      x, y, range in mm; v, snr x100
  counts  uint16[frames]
  profile uint8[frames, 256]   range profile, globally min/max scaled
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import data
from .model import ENC, HEAD, PointNet
from .prep import CLASSES, RECORDINGS

QUANT = np.array([1000, 1000, 100, 100, 1000], np.float32)
PARITY_FRAMES = 256


def quantize(points):
    return np.round(points * QUANT).astype(np.int16)


def main(art="artifacts", out="docs"):
    model = PointNet()
    model.load_weights(os.path.join(art, "pointnet.safetensors"))
    model.eval()
    params = dict(nn.utils.tree_flatten(model.parameters()))

    os.makedirs(f"{out}/model", exist_ok=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    layers, blobs, off = [], [], 0
    for group, dims in (("enc", ENC), ("head", HEAD)):
        for i in range(len(dims) - 1):
            w = np.array(params[f"{group}.{i}.weight"], np.float32)  # [out, in]
            b = np.array(params[f"{group}.{i}.bias"], np.float32)
            layers.append(dict(name=f"{group}{i}", inputs=dims[i], outputs=dims[i + 1], w=off, b=off + w.size))
            blobs += [w.ravel(), b]
            off += w.size + b.size
    np.concatenate(blobs).astype("<f4").tofile(f"{out}/model/weights.bin")

    recs = data.load()
    prof_all = np.concatenate([r["profile"] for r in recs])
    prof_lo, prof_hi = float(prof_all.min()), float(prof_all.max())
    manifest = []
    for r in recs:
        q = quantize(r["points"])
        prof = np.round((r["profile"] - prof_lo) / (prof_hi - prof_lo) * 255).astype(np.uint8)
        with open(f"{out}/data/{r['name']}.bin", "wb") as f:
            for arr in (r["pose"].astype("<f4"), q.astype("<i2"), r["counts"].astype("<u2"), prof):
                f.write(arr.tobytes())
        tr, va, te = data.split_indices(len(r["counts"]))
        manifest.append(
            dict(
                name=r["name"],
                label=r["label"],
                frames=int(len(r["counts"])),
                points=int(len(q)),
                split=dict(val=int(va[0]), test=int(te[0])),
            )
        )

    # MLX reference outputs on dequantized inputs, so the browser can check parity exactly.
    r = recs[0]
    deq = quantize(r["points"]).astype(np.float32) / QUANT
    frames = np.arange(PARITY_FRAMES)
    x = np.zeros((len(frames), data.MAX_POINTS, data.FEATURES), np.float32)
    mask = np.zeros((len(frames), data.MAX_POINTS), np.float32)
    for i, t in enumerate(frames):
        w = data.window(deq, r["offsets"], t)
        x[i, : len(w)] = w
        mask[i, : len(w)] = 1
    ref = np.array(model(mx.array(x), mx.array(mask)))

    with open(f"{art}/metrics.json") as f:
        metrics = json.load(f)
    model_json = dict(
        layers=layers,
        history=data.HISTORY,
        max_points=data.MAX_POINTS,
        features=data.FEATURES,
        pos_center=data.POS_CENTER.tolist(),
        pos_scale=data.POS_SCALE,
        classes=CLASSES,
        quant=QUANT.tolist(),
        profile_range=[prof_lo, prof_hi],
        recordings=manifest,
        parity=dict(recording=r["name"], frames=frames.tolist(), outputs=np.round(ref, 6).tolist()),
        metrics=metrics,
    )
    with open(f"{out}/model/model.json", "w") as f:
        json.dump(model_json, f, separators=(",", ":"))
    total = sum(os.path.getsize(f"{out}/data/{m['name']}.bin") for m in manifest)
    print(f"weights {off} floats, {len(manifest)} recordings, {total / 1e6:.1f} MB data")


if __name__ == "__main__":
    main()
