// Checks the JS feature pipeline + CPU kernel against MLX reference outputs.
import { readFileSync } from "node:fs";
import { decodeRecording, makeBatch } from "../docs/js/features.js";
import { forward } from "../docs/js/cpu-kernel.js";

const model = JSON.parse(readFileSync("docs/model/model.json"));
const W = new Float32Array(readFileSync("docs/model/weights.bin").buffer.slice(0));
const meta = model.recordings.find((r) => r.name === model.parity.recording);
const buf = readFileSync(`docs/data/${meta.name}.bin`);
const rec = decodeRecording(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), meta, model);
const { feats, offsets, batch } = makeBatch(rec, model.parity.frames, model);
const out = new Float32Array(batch * 8);
const t = performance.now();
forward(W, model.layers, feats, offsets, 0, batch, out);
const ms = performance.now() - t;
let maxDiff = 0;
model.parity.outputs.flat().forEach((v, i) => (maxDiff = Math.max(maxDiff, Math.abs(v - out[i]))));
console.log(`frames=${batch} maxAbsDiff=${maxDiff.toExponential(2)} cpu1thread=${((batch / ms) * 1000).toFixed(0)} fps`);
process.exit(maxDiff < 1e-3 ? 0 : 1);
