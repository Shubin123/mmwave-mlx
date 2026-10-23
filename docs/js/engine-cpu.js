// CPU fallback: splits each batch across a pool of module workers.
import { FEATURES as F } from "./features.js";

export class CPUEngine {
  constructor(model, weights) {
    this.name = "CPU";
    this.layers = model.layers;
    this.nOut = model.layers[5].outputs;
    const n = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 16) - 1);
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./cpu-worker.js", import.meta.url), { type: "module" });
      w.postMessage({ init: true, weights: weights.slice(0), layers: model.layers });
      return w;
    });
    this.detail = `${n} Web Worker${n > 1 ? "s" : ""}`;
    this.seq = 0;
  }

  run(feats, offsets, batch) {
    const total = offsets[batch];
    const jobs = [];
    let b0 = 0;
    // Balance by point count, not frame count.
    for (let k = 0; k < this.workers.length && b0 < batch; k++) {
      const goal = (total * (k + 1)) / this.workers.length;
      let b1 = b0 + 1;
      while (b1 < batch && offsets[b1] < goal) b1++;
      if (k === this.workers.length - 1) b1 = batch;
      jobs.push([b0, b1]);
      b0 = b1;
    }
    const out = new Float32Array(batch * this.nOut);
    return Promise.all(
      jobs.map(([s, e], k) => new Promise((resolve) => {
        const id = ++this.seq, w = this.workers[k];
        const off = offsets.slice(s, e + 1).map((v) => v - offsets[s]);
        const f = feats.slice(offsets[s] * F, offsets[e] * F);
        const onmsg = ({ data }) => {
          if (data.id !== id) return;
          w.removeEventListener("message", onmsg);
          out.set(data.out, s * this.nOut);
          resolve();
        };
        w.addEventListener("message", onmsg);
        w.postMessage({ id, feats: f, offsets: off, batch: e - s }, [f.buffer, off.buffer]);
      }))
    ).then(() => out);
  }

  async bench(feats, offsets, batch, reps) {
    for (let i = 0; i < reps; i++) await this.run(feats, offsets, batch);
  }

  destroy() { this.workers.forEach((w) => w.terminate()); }
}
