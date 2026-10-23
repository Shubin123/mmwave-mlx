import { forward } from "./cpu-kernel.js";

let W, layers;
onmessage = ({ data }) => {
  if (data.init) {
    W = new Float32Array(data.weights);
    layers = data.layers;
    return;
  }
  const { id, feats, offsets, batch } = data;
  const out = new Float32Array(batch * layers[5].outputs);
  forward(W, layers, feats, offsets, 0, batch, out);
  postMessage({ id, out }, [out.buffer]);
};
