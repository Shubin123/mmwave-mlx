# mmWave radar PointNet: MLX training + WebGPU inference

**Live demo: https://shubin123.github.io/mmwave-mlx/**

This repo shows what the [mmWave-Dataset](https://www.kaggle.com/datasets/sahilrajpurkar/mmwave-dataset) can do. The dataset is an indoor-logistics radar dataset with 60 GHz mmWave point clouds synchronized to Vicon motion capture, covering four objects (forklift + KLT, forklift, workstation + robot, robot) in moving and static scenes.

A small PointNet (83k parameters) is trained with [MLX](https://github.com/ml-explore/mlx) on Apple silicon. From the radar returns alone it predicts:

- **which object** is in the scene (4 classes)
- **where** it is in the room (x, y in the Vicon frame)
- **which way** it faces (yaw)

The trained model runs in the browser as a single fused **WebGPU** compute shader. If WebGPU is missing or fails, it falls back to a pool of **CPU Web Workers**. Both backends check themselves against MLX reference outputs on page load (max |Δ| ≈ 5e-6).

## Results (held-out test segments)

| | all recordings | moving objects only |
|---|---|---|
| classification accuracy | 99.96% | 99.88% |
| median position error | 4 cm | 29 cm (p90: 61 cm) |
| median heading error | 2° | 26° |

Each recording is split in time: the first 70% is for training, then 15% validation, then 15% test, with a gap between splits. In static recordings the pose never changes, so their scores are optimistic. Treat the moving-object numbers as the real localization result.

Training: 60 epochs, about 4 minutes on an M-series GPU.

## Layout

```
mmwave/prep.py     download the 23 synchronized radar+Vicon CSVs, pack into data/proc.npz
mmwave/data.py     frame stacking (3 frames, <=256 strongest returns) and normalization
mmwave/model.py    PointNet in MLX
mmwave/train.py    training, evaluation, MLX throughput benchmark -> artifacts/
mmwave/export.py   weights, metrics and quantized recordings -> docs/
docs/              static site (GitHub Pages)
  js/features.js     JS port of data.py
  js/engine-webgpu.js fused WGSL kernel
  js/engine-cpu.js   worker-pool fallback (js/cpu-kernel.js)
tests/parity.mjs   JS pipeline vs MLX outputs
```

## Reproduce

```sh
uv venv -p 3.12 && uv pip install -r requirements.txt
python -m mmwave.prep            # ~215 MB from Kaggle (only the synchronized subset)
python -m mmwave.train
python -m mmwave.export
node tests/parity.mjs
python -m http.server -d docs    # http://localhost:8000
```

## How the WebGPU kernel works

Each workgroup (256 threads) processes one frame window. Points go through workgroup memory 16 at a time, and each tile runs all three shared encoder layers. The last encoder layer is fused with the max-pool, so per-point activations never reach global memory. Inputs are packed as a ragged batch (a flat point list plus per-frame offsets), so padding costs nothing. After the pool, the same workgroup runs the 3-layer head and writes 8 floats.

## Data credit

Dataset by Sahil Rajpurkar, published on Kaggle as [mmWave-Dataset](https://www.kaggle.com/datasets/sahilrajpurkar/mmwave-dataset). `docs/data/` holds a quantized copy of the synchronized radar/Vicon subset, used only for the demo.
