// Mirrors mmwave/data.py: stack HISTORY frames, keep the MAX_POINTS strongest returns.
export const FEATURES = 6;

export function decodeRecording(buf, rec, model) {
  const F = rec.frames, P = rec.points;
  let o = 0;
  const pose = new Float32Array(buf, o, F * 3); o += F * 12;
  const q = new Int16Array(buf, o, P * 5); o += P * 10;
  const counts = new Uint16Array(buf, o, F); o += F * 2;
  const profile = new Uint8Array(buf, o, F * 256);
  const points = new Float32Array(P * 5);
  for (let i = 0; i < P * 5; i++) points[i] = q[i] / model.quant[i % 5];
  const offsets = new Uint32Array(F + 1);
  for (let f = 0; f < F; f++) offsets[f + 1] = offsets[f] + counts[f];
  return { ...rec, pose, points, counts, offsets, profile };
}

function featurize(points, i, age, out, j) {
  const k = i * 5, d = j * FEATURES;
  out[d] = points[k] / 4;
  out[d + 1] = (points[k + 1] - 3) / 3;
  out[d + 2] = Math.min(Math.max(points[k + 2], -6), 6) / 3;
  out[d + 3] = (points[k + 3] - 20) / 15;
  out[d + 4] = (points[k + 4] - 3) / 3;
  out[d + 5] = age - 1;
}

// Packs windows for `frames` into a ragged batch: feats[totalPoints*6], offsets[B+1].
export function makeBatch(rec, frames, model) {
  const H = model.history, N = model.max_points, { points, offsets } = rec;
  const sel = frames.map((t) => {
    const idx = [], age = [];
    for (let a = 0; a < H && t - a >= 0; a++)
      for (let i = offsets[t - a]; i < offsets[t - a + 1]; i++) { idx.push(i); age.push(a); }
    let order = idx.map((_, j) => j);
    if (order.length > N) order = order.sort((x, y) => points[idx[y] * 5 + 3] - points[idx[x] * 5 + 3]).slice(0, N);
    return order.map((j) => [idx[j], age[j]]);
  });
  const boff = new Uint32Array(frames.length + 1);
  sel.forEach((s, b) => (boff[b + 1] = boff[b] + s.length));
  const feats = new Float32Array(boff[frames.length] * FEATURES);
  let j = 0;
  for (const s of sel) for (const [i, a] of s) featurize(points, i, a, feats, j++);
  return { feats, offsets: boff, batch: frames.length };
}
