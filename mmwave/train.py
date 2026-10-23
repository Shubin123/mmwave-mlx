"""Train the PointNet on Apple silicon with MLX and write metrics for the site."""
import argparse
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import data
from .model import PointNet
from .prep import CLASSES, RECORDINGS


def cached_splits(path="data/windows.npz"):
    if os.path.exists(path):
        z = np.load(path)
        return [{k[len(s) + 1 :]: z[k] for k in z.files if k.startswith(s + "/")} for s in ("train", "val", "test")]
    recs = data.load()
    splits = [data.build(recs, i) for i in range(3)]
    np.savez(path, **{f"{s}/{k}": v for s, d in zip(("train", "val", "test"), splits) for k, v in d.items()})
    return splits


def to_mx(d, idx, augment=False, rng=None):
    x = d["x"][idx]
    n = d["n"][idx]
    mask = (np.arange(data.MAX_POINTS)[None] < n[:, None]).astype(np.float32)
    if augment:
        # Randomly drop up to 30% of returns and jitter SNR to mimic sparser scans.
        keep = rng.random(mask.shape) > rng.uniform(0, 0.3, (len(idx), 1))
        keep[:, 0] = True
        mask = mask * keep
        x = x.copy()
        x[..., 3] += rng.normal(0, 0.05, x[..., 3].shape).astype(np.float32)
    return mx.array(x), mx.array(mask), mx.array(d["y"][idx]), mx.array(d["target"][idx])


def loss_fn(model, x, mask, y, target):
    out = model(x, mask)
    ce = nn.losses.cross_entropy(out[:, :4], y).mean()
    pos = nn.losses.smooth_l1_loss(out[:, 4:6], target[:, :2], beta=0.1)
    yaw = ((out[:, 6:8] - target[:, 2:]) ** 2).mean()
    return ce + 2.0 * pos + 0.5 * yaw


def predict(model, d, bs=2048):
    model.eval()
    outs = []
    for i in range(0, len(d["n"]), bs):
        idx = np.arange(i, min(i + bs, len(d["n"])))
        x, mask, _, _ = to_mx(d, idx)
        outs.append(np.array(model(x, mask)))
    model.train()
    return np.concatenate(outs)


def evaluate(out, d):
    pred = out[:, :4].argmax(1)
    pos = out[:, 4:6] * data.POS_SCALE + data.POS_CENTER
    gt = d["target"][:, :2] * data.POS_SCALE + data.POS_CENTER
    err = np.linalg.norm(pos - gt, axis=1)
    yaw_p = np.arctan2(out[:, 6], out[:, 7])
    yaw_t = np.arctan2(d["target"][:, 2], d["target"][:, 3])
    yaw_err = np.degrees(np.abs((yaw_p - yaw_t + np.pi) % (2 * np.pi) - np.pi))
    conf = np.zeros((4, 4), int)
    np.add.at(conf, (d["y"], pred), 1)
    dyn = np.isin(d["rec"], [i for i, r in enumerate(RECORDINGS) if r.startswith("DYNAMIC")])
    per_rec = {}
    for i, r in enumerate(RECORDINGS):
        m = d["rec"] == i
        per_rec[r] = dict(
            n=int(m.sum()),
            acc=float((pred[m] == d["y"][m]).mean()),
            pos_err_m=float(np.median(err[m])),
            yaw_err_deg=float(np.median(yaw_err[m])),
        )
    return dict(
        acc=float((pred == d["y"]).mean()),
        acc_dynamic=float((pred[dyn] == d["y"][dyn]).mean()),
        pos_err_median_m=float(np.median(err)),
        pos_err_dynamic_median_m=float(np.median(err[dyn])),
        pos_err_dynamic_p90_m=float(np.percentile(err[dyn], 90)),
        yaw_err_median_deg=float(np.median(yaw_err)),
        yaw_err_dynamic_median_deg=float(np.median(yaw_err[dyn])),
        confusion=conf.tolist(),
        per_recording=per_rec,
    )


def benchmark(model, d, batches=(1, 64, 1024, 4096)):
    model.eval()
    res = {}
    for bs in batches:
        idx = np.arange(bs) % len(d["n"])
        x, mask, _, _ = to_mx(d, idx)
        for _ in range(3):
            mx.eval(model(x, mask))
        reps = max(5, 20000 // bs)
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(model(x, mask))
        res[bs] = bs * reps / (time.perf_counter() - t)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    mx.random.seed(0)
    rng = np.random.default_rng(0)

    train, val, test = cached_splits()
    print(f"windows train={len(train['n'])} val={len(val['n'])} test={len(test['n'])} device={mx.default_device()}")

    model = PointNet()
    mx.eval(model.parameters())
    steps = args.epochs * math.ceil(len(train["n"]) / args.bs)
    warm = min(300, steps // 4)
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-5, args.lr, warm), optim.cosine_decay(args.lr, steps - warm, 1e-5)], [warm]
    )
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    step = nn.value_and_grad(model, loss_fn)


    history = []
    best, best_state = -1e9, None
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(len(train["n"]))
        losses = []
        for i in range(0, len(perm), args.bs):
            x, mask, y, target = to_mx(train, perm[i : i + args.bs], augment=True, rng=rng)
            loss, grads = step(model, x, mask, y, target)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            losses.append(loss.item())
        v = evaluate(predict(model, val), val)
        score = v["acc"] - v["pos_err_dynamic_median_m"]
        history.append(
            dict(
                epoch=epoch + 1,
                loss=float(np.mean(losses)),
                val_acc=v["acc"],
                val_pos_err_dynamic_m=v["pos_err_dynamic_median_m"],
                val_yaw_err_deg=v["yaw_err_median_deg"],
                seconds=time.perf_counter() - t0,
            )
        )
        print(
            f"epoch {epoch + 1:3d} loss {history[-1]['loss']:.4f} val acc {v['acc']:.3f} "
            f"dyn pos {v['pos_err_dynamic_median_m']:.3f} m yaw {v['yaw_err_median_deg']:.1f} deg "
            f"({history[-1]['seconds']:.1f}s)"
        )
        if score > best:
            best = score
            best_state = {k: mx.array(v) for k, v in nn.utils.tree_flatten(model.parameters())}

    model.load_weights(list(best_state.items()))
    os.makedirs(args.out, exist_ok=True)
    model.save_weights(os.path.join(args.out, "pointnet.safetensors"))
    test_out = predict(model, test)
    np.save(os.path.join(args.out, "test_out.npy"), test_out)
    metrics = dict(
        classes=CLASSES,
        sizes={s: int(len(d["n"])) for s, d in zip(("train", "val", "test"), (train, val, test))},
        params=int(sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))),
        history=history,
        val=evaluate(predict(model, val), val),
        test=evaluate(test_out, test),
        mlx_throughput=benchmark(model, test),
        device=str(mx.default_device()),
    )
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    t = metrics["test"]
    print(
        f"TEST acc {t['acc']:.3f} (dynamic {t['acc_dynamic']:.3f}) pos {t['pos_err_median_m']:.3f} m "
        f"(dynamic {t['pos_err_dynamic_median_m']:.3f} m, p90 {t['pos_err_dynamic_p90_m']:.3f}) "
        f"yaw {t['yaw_err_median_deg']:.1f} deg"
    )
    print("MLX throughput (frames/s):", {k: round(v) for k, v in metrics["mlx_throughput"].items()})


if __name__ == "__main__":
    main()
