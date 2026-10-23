// Reference forward pass in plain JS; used by the CPU workers and for tests.
export function forward(W, layers, feats, offsets, start, end, out) {
  const [e0, e1, e2, h0, h1, h2] = layers;
  const a1 = new Float32Array(e0.outputs), a2 = new Float32Array(e1.outputs);
  const pooled = new Float32Array(e2.outputs), g1 = new Float32Array(h0.outputs), g2 = new Float32Array(h1.outputs);
  for (let b = start; b < end; b++) {
    pooled.fill(0);
    for (let p = offsets[b]; p < offsets[b + 1]; p++) {
      dense(W, e0, feats, p * e0.inputs, a1, true);
      dense(W, e1, a1, 0, a2, true);
      const n = e2.inputs;
      for (let c = 0; c < e2.outputs; c++) {
        let s = W[e2.b + c];
        const w = e2.w + c * n;
        for (let i = 0; i < n; i++) s += W[w + i] * a2[i];
        if (s > pooled[c]) pooled[c] = s;
      }
    }
    dense(W, h0, pooled, 0, g1, true);
    dense(W, h1, g1, 0, g2, true);
    const o = new Float32Array(out.buffer, out.byteOffset + (b - start) * h2.outputs * 4, h2.outputs);
    dense(W, h2, g2, 0, o, false);
  }
}

function dense(W, L, x, xo, y, relu) {
  const n = L.inputs;
  for (let c = 0; c < L.outputs; c++) {
    let s = W[L.b + c];
    const w = L.w + c * n;
    for (let i = 0; i < n; i++) s += W[w + i] * x[xo + i];
    y[c] = relu && s < 0 ? 0 : s;
  }
}
