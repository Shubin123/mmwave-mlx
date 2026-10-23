import { WebGPUEngine } from "./engine-webgpu.js";
import { CPUEngine } from "./engine-cpu.js";

// Prefer the GPU for throughput; fall back to CPU workers when WebGPU is missing or fails.
export async function createEngine(model, weights, prefer = "auto") {
  const errors = [];
  if (prefer !== "cpu") {
    try {
      const gpu = await WebGPUEngine.create(model, weights);
      return { engine: gpu, errors };
    } catch (e) {
      errors.push(`WebGPU: ${e.message}`);
    }
  }
  return { engine: new CPUEngine(model, weights), errors };
}
