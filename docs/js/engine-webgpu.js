// WebGPU backend: the whole network runs as one fused compute pass.
// One workgroup per frame window; 256 threads = one per encoder output channel.
// Points stream through workgroup memory in tiles of 16, so per-point activations
// never touch global memory and padding is never computed.

const TILE = 16;

function shader(L) {
  const [e0, e1, e2, h0, h1, h2] = L;
  return /* wgsl */ `
const TILE = ${TILE}u;
struct Params { batch: u32, row: u32 }
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> OFF: array<u32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

var<workgroup> xin: array<f32, ${TILE * e0.inputs}>;
var<workgroup> a1: array<f32, ${TILE * e0.outputs}>;
var<workgroup> a2: array<f32, ${TILE * e1.outputs}>;
var<workgroup> pooled: array<f32, ${e2.outputs}>;
var<workgroup> g1: array<f32, ${h0.outputs}>;
var<workgroup> g2: array<f32, ${h1.outputs}>;
var<workgroup> span: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = wg.x + wg.y * P.row;
  if (b >= P.batch) { return; }
  if (t == 0u) { span = vec2<u32>(OFF[b], OFF[b + 1u]); }
  let r = workgroupUniformLoad(&span);
  var best = 0.0;

  for (var p0 = r.x; p0 < r.y; p0 += TILE) {
    let live = min(TILE, r.y - p0);
    if (t < TILE * ${e0.inputs}u) {
      xin[t] = select(0.0, X[p0 * ${e0.inputs}u + t], t / ${e0.inputs}u < live);
    }
    workgroupBarrier();

    // enc0: ${e0.inputs} -> ${e0.outputs}; thread = (channel, group of ${(TILE * e0.outputs) / 256} points)
    {
      let c = t % ${e0.outputs}u;
      let pg = (t / ${e0.outputs}u) * ${(TILE * e0.outputs) / 256}u;
      for (var k = 0u; k < ${(TILE * e0.outputs) / 256}u; k++) {
        var s = W[${e0.b}u + c];
        for (var i = 0u; i < ${e0.inputs}u; i++) { s += W[${e0.w}u + c * ${e0.inputs}u + i] * xin[(pg + k) * ${e0.inputs}u + i]; }
        a1[(pg + k) * ${e0.outputs}u + c] = max(s, 0.0);
      }
    }
    workgroupBarrier();

    // enc1: ${e1.inputs} -> ${e1.outputs}
    {
      let c = t % ${e1.outputs}u;
      let pg = (t / ${e1.outputs}u) * ${(TILE * e1.outputs) / 256}u;
      var s: array<f32, ${(TILE * e1.outputs) / 256}>;
      for (var k = 0u; k < ${(TILE * e1.outputs) / 256}u; k++) { s[k] = W[${e1.b}u + c]; }
      for (var i = 0u; i < ${e1.inputs}u; i++) {
        let w = W[${e1.w}u + c * ${e1.inputs}u + i];
        for (var k = 0u; k < ${(TILE * e1.outputs) / 256}u; k++) { s[k] += w * a1[(pg + k) * ${e1.inputs}u + i]; }
      }
      for (var k = 0u; k < ${(TILE * e1.outputs) / 256}u; k++) { a2[(pg + k) * ${e1.outputs}u + c] = max(s[k], 0.0); }
    }
    workgroupBarrier();

    // enc2: ${e2.inputs} -> ${e2.outputs}, fused with the max-pool over points
    {
      var s: array<f32, ${TILE}>;
      for (var k = 0u; k < TILE; k++) { s[k] = W[${e2.b}u + t]; }
      for (var i = 0u; i < ${e2.inputs}u; i++) {
        let w = W[${e2.w}u + t * ${e2.inputs}u + i];
        for (var k = 0u; k < TILE; k++) { s[k] += w * a2[k * ${e2.inputs}u + i]; }
      }
      for (var k = 0u; k < live; k++) { best = max(best, s[k]); }
    }
    workgroupBarrier();
  }

  pooled[t] = best;
  workgroupBarrier();
  if (t < ${h0.outputs}u) {
    var s = W[${h0.b}u + t];
    for (var i = 0u; i < ${h0.inputs}u; i++) { s += W[${h0.w}u + t * ${h0.inputs}u + i] * pooled[i]; }
    g1[t] = max(s, 0.0);
  }
  workgroupBarrier();
  if (t < ${h1.outputs}u) {
    var s = W[${h1.b}u + t];
    for (var i = 0u; i < ${h1.inputs}u; i++) { s += W[${h1.w}u + t * ${h1.inputs}u + i] * g1[i]; }
    g2[t] = max(s, 0.0);
  }
  workgroupBarrier();
  if (t < ${h2.outputs}u) {
    var s = W[${h2.b}u + t];
    for (var i = 0u; i < ${h2.inputs}u; i++) { s += W[${h2.w}u + t * ${h2.inputs}u + i] * g2[i]; }
    OUT[b * ${h2.outputs}u + t] = s;
  }
}`;
}

export class WebGPUEngine {
  static async create(model, weights) {
    if (!navigator.gpu) throw new Error("navigator.gpu unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const e = new WebGPUEngine();
    e.device = device;
    e.nOut = model.layers[5].outputs;
    e.feat = model.layers[0].inputs;
    const info = adapter.info || {};
    e.name = "WebGPU";
    e.detail = [info.vendor, info.architecture || info.description].filter(Boolean).join(" · ") || "GPU";
    const module = device.createShaderModule({ code: shader(model.layers) });
    const msgs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (msgs.length) throw new Error(msgs.map((m) => `${m.lineNum}: ${m.message}`).join("\n"));
    e.pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    e.W = device.createBuffer({ size: weights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(e.W, 0, weights);
    e.params = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    e.cap = { x: 0, off: 0, out: 0 };
    device.lost.then((info) => { e.lost = info.message; });
    return e;
  }

  ensure(xBytes, batch) {
    const d = this.device, c = this.cap;
    const grow = (want, have) => Math.max(want, have * 2, 1024);
    if (xBytes > c.x) {
      this.X?.destroy();
      c.x = grow(xBytes, c.x);
      this.X = d.createBuffer({ size: c.x, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.bind = null;
    }
    const offBytes = (batch + 1) * 4, outBytes = batch * this.nOut * 4;
    if (offBytes > c.off) {
      this.OFF?.destroy();
      c.off = grow(offBytes, c.off);
      this.OFF = d.createBuffer({ size: c.off, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.bind = null;
    }
    if (outBytes > c.out) {
      this.OUT?.destroy();
      this.READ?.destroy();
      c.out = grow(outBytes, c.out);
      this.OUT = d.createBuffer({ size: c.out, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.READ = d.createBuffer({ size: c.out, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.bind = null;
    }
    if (!this.bind) {
      this.bind = d.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [this.W, this.X, this.OFF, this.OUT, this.params].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
  }

  upload(feats, offsets, batch) {
    const xBytes = Math.max(feats.byteLength, 16);
    this.ensure(xBytes, batch);
    const q = this.device.queue;
    if (feats.byteLength) q.writeBuffer(this.X, 0, feats);
    q.writeBuffer(this.OFF, 0, offsets, 0, batch + 1);
    const row = Math.min(batch, 65535);
    q.writeBuffer(this.params, 0, new Uint32Array([batch, row]));
    return [row, Math.ceil(batch / row)];
  }

  encode(enc, grid) {
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.dispatchWorkgroups(grid[0], grid[1]);
    pass.end();
  }

  async run(feats, offsets, batch) {
    if (this.lost) throw new Error(`GPU device lost: ${this.lost}`);
    const grid = this.upload(feats, offsets, batch);
    const bytes = batch * this.nOut * 4;
    const enc = this.device.createCommandEncoder();
    this.encode(enc, grid);
    enc.copyBufferToBuffer(this.OUT, 0, this.READ, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await this.READ.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.READ.getMappedRange(0, bytes).slice(0));
    this.READ.unmap();
    return out;
  }

  // Throughput mode: upload once, queue `reps` dispatches, read back once.
  async bench(feats, offsets, batch, reps) {
    const grid = this.upload(feats, offsets, batch);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < reps; i++) this.encode(enc, grid);
    enc.copyBufferToBuffer(this.OUT, 0, this.READ, 0, 16);
    this.device.queue.submit([enc.finish()]);
    await this.READ.mapAsync(GPUMapMode.READ, 0, 16);
    this.READ.unmap();
  }

  destroy() { this.device.destroy(); }
}
