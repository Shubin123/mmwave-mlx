import { decodeRecording, makeBatch } from "./features.js";
import { createEngine } from "./engine.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const CLASS_VARS = ["--c1", "--c2", "--c3", "--c4"];
const fmt = (v, d = 0) => v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const pretty = (name) => name.replace("DYNAMIC_OBJECT", "Moving · object ").replace("STATIC_OBJECT", "Static · object ")
  .replace("_MOVEMENT", " · run ").replace("_ORIENTATION", " · orientation ");

const state = { model: null, weights: null, engine: null, rec: null, out: null, frame: 0, playing: false, cache: new Map(), bench: {} };

// ---------- tooltip ----------
const tip = $("tip");
function showTip(e, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.hidden = true);

// ---------- boot ----------
async function main() {
  const [model, wbuf] = await Promise.all([
    fetch("model/model.json").then((r) => r.json()),
    fetch("model/weights.bin").then((r) => r.arrayBuffer()),
  ]);
  state.model = model;
  state.weights = new Float32Array(wbuf);
  renderTiles();
  renderTraining();
  renderConfusion();
  renderPerRec();
  fillRecordings();
  await switchEngine("auto");
  await loadRecording(model.recordings[0].name);
  wire();
  new ResizeObserver(() => draw()).observe($("bev"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { draw(); renderTraining(); renderBench(); });
}

async function switchEngine(prefer) {
  state.engine?.destroy();
  $("engineBadge").textContent = "starting…";
  const { engine, errors } = await createEngine(state.model, state.weights, prefer);
  state.engine = engine;
  const fell = errors.length && prefer !== "cpu";
  $("engineBadge").innerHTML = `${engine.name} · ${engine.detail}${fell ? ` <span class="bad" title="${errors.join("; ")}">(WebGPU unavailable, using fallback)</span>` : ""}`;
  await checkParity();
  if (state.rec) await infer();
}

async function checkParity() {
  const m = state.model, p = m.parity;
  const rec = await getRecording(p.recording);
  const { feats, offsets, batch } = makeBatch(rec, p.frames, m);
  const out = await state.engine.run(feats, offsets, batch);
  let d = 0;
  p.outputs.flat().forEach((v, i) => (d = Math.max(d, Math.abs(v - out[i]))));
  const ok = d < 1e-3;
  $("parity").innerHTML = `Parity vs MLX on ${batch} frames: max |Δ| = ${d.toExponential(1)} <span class="${ok ? "ok" : "bad"}">${ok ? "✓ match" : "✗ mismatch"}</span>`;
}

async function getRecording(name) {
  if (!state.cache.has(name)) {
    const meta = state.model.recordings.find((r) => r.name === name);
    state.cache.set(name, fetch(`data/${name}.bin`).then((r) => r.arrayBuffer()).then((b) => decodeRecording(b, meta, state.model)));
  }
  return state.cache.get(name);
}

async function loadRecording(name) {
  $("runInfo").textContent = "Loading…";
  state.rec = await getRecording(name);
  $("frame").max = state.rec.frames - 1;
  state.frame = state.rec.split.test;
  $("frame").value = state.frame;
  state.profileImg = null;
  await infer();
}

async function infer() {
  const rec = state.rec;
  const frames = Array.from({ length: rec.frames }, (_, i) => i);
  const t0 = performance.now();
  const { feats, offsets, batch } = makeBatch(rec, frames, state.model);
  const t1 = performance.now();
  state.out = await state.engine.run(feats, offsets, batch);
  const t2 = performance.now();
  $("runInfo").innerHTML = `Scored all <b>${fmt(batch)}</b> frames (${fmt(offsets[batch])} points) in <b>${fmt(t2 - t1, 1)} ms</b> on ${state.engine.name} ` +
    `(${fmt((batch / (t2 - t1)) * 1000)} frames/s end to end, including upload and readback). Feature packing took ${fmt(t1 - t0, 1)} ms.`;
  draw();
}

// ---------- explorer ----------
function fillRecordings() {
  const sel = $("rec");
  const groups = {};
  for (const r of state.model.recordings) {
    const g = state.model.classes[r.label];
    (groups[g] ||= []).push(r);
  }
  for (const [g, rs] of Object.entries(groups)) {
    const og = document.createElement("optgroup");
    og.label = g;
    for (const r of rs) og.append(new Option(pretty(r.name), r.name));
    sel.append(og);
  }
}

function wire() {
  $("rec").onchange = (e) => loadRecording(e.target.value);
  $("frame").oninput = (e) => { state.frame = +e.target.value; draw(); };
  $("play").onclick = () => {
    state.playing = !state.playing;
    $("play").textContent = state.playing ? "❚❚ Pause" : "▶ Play";
    if (state.playing) tick();
  };
  $("backend").onchange = (e) => switchEngine(e.target.value);
  $("benchBtn").onclick = runBench;
  for (const id of ["bev", "world"]) $(id).addEventListener("mouseleave", hideTip);
}

let last = 0;
function tick(ts = 0) {
  if (!state.playing) return;
  if (ts - last > 100) {  // 2x real time (data is 5 Hz)
    last = ts;
    state.frame = (state.frame + 1) % state.rec.frames;
    $("frame").value = state.frame;
    draw();
  }
  requestAnimationFrame(tick);
}

function setupCanvas(c) {
  const dpr = devicePixelRatio || 1, w = c.clientWidth, h = +c.getAttribute("height");
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function decodeOut(i) {
  const m = state.model, o = state.out.subarray(i * 8, i * 8 + 8);
  const mx = Math.max(...o.slice(0, 4));
  const e = Array.from(o.slice(0, 4), (v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b);
  return {
    probs: e.map((v) => v / s),
    x: o[4] * m.pos_scale + m.pos_center[0],
    y: o[5] * m.pos_scale + m.pos_center[1],
    yaw: Math.atan2(o[6], o[7]),
  };
}

function draw() {
  if (!state.rec || !state.out) return;
  drawBEV();
  drawWorld();
  drawProfile();
  drawReadout();
  const r = state.rec, f = state.frame;
  const split = f >= r.split.test ? "test" : f >= r.split.val ? "val" : "train";
  $("frameLabel").textContent = `${f + 1}/${r.frames} · t=${fmt(f / 5, 1)} s · ${split} split`;
}

function lerp(a, b, t) {
  const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16)), pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
}
function dopplerColor(v) {
  const t = Math.max(-1, Math.min(1, v / 2));
  return t < 0 ? lerp(css("--mid"), css("--neg"), -t) : lerp(css("--mid"), css("--pos"), t);
}

function drawBEV() {
  const c = $("bev"), [g, w, h] = setupCanvas(c), r = state.rec, f = state.frame;
  const R = 9.5, s = Math.min(w / (2 * R), (h - 20) / R), cx = w / 2, cy = h - 10;
  g.strokeStyle = css("--grid");
  g.fillStyle = css("--muted");
  g.font = "11px system-ui";
  g.lineWidth = 1;
  for (let m = 2; m <= 8; m += 2) {
    g.beginPath(); g.arc(cx, cy, m * s, Math.PI, 2 * Math.PI); g.stroke();
    g.fillText(`${m} m`, cx + m * s * 0.71 + 3, cy - m * s * 0.71);
  }
  for (const a of [-60, -30, 0, 30, 60]) {
    const t = (a * Math.PI) / 180;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.sin(t) * R * s, cy - Math.cos(t) * R * s); g.stroke();
  }
  g.fillStyle = css("--ink");
  g.beginPath(); g.moveTo(cx - 6, cy + 4); g.lineTo(cx + 6, cy + 4); g.lineTo(cx, cy - 6); g.fill();
  const pts = [];
  for (let a = 2; a >= 0; a--) {
    const t = f - a;
    if (t < 0) continue;
    g.globalAlpha = a === 0 ? 1 : 0.35;
    for (let i = r.offsets[t]; i < r.offsets[t + 1]; i++) {
      const p = r.points.subarray(i * 5, i * 5 + 5);
      const px = cx + p[0] * s, py = cy - p[1] * s, rad = 2.5 + Math.max(0, p[3] - 10) / 12;
      g.fillStyle = dopplerColor(p[2]);
      g.beginPath(); g.arc(px, py, rad, 0, 2 * Math.PI); g.fill();
      if (a === 0) pts.push([px, py, p]);
    }
  }
  g.globalAlpha = 1;
  c.onmousemove = (e) => {
    const b = c.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
    let best = null, bd = 100;
    for (const q of pts) { const d = (q[0] - mx) ** 2 + (q[1] - my) ** 2; if (d < bd) { bd = d; best = q; } }
    if (!best) return hideTip();
    const p = best[2];
    showTip(e, `x ${fmt(p[0], 2)} m · y ${fmt(p[1], 2)} m<br>range ${fmt(p[4], 2)} m<br>doppler ${fmt(p[2], 2)} m/s<br>SNR ${fmt(p[3], 1)} dB`);
  };
}

function drawWorld() {
  const c = $("world"), [g, w, h] = setupCanvas(c), r = state.rec, f = state.frame;
  const X = [-9.6, -0.6], Y = [-3.8, 4.2], pad = 28;
  const s = Math.min((w - 2 * pad) / (X[1] - X[0]), (h - 2 * pad) / (Y[1] - Y[0]));
  const ox = (w - s * (X[1] - X[0])) / 2, oy = (h - s * (Y[1] - Y[0])) / 2;
  const tx = (x) => ox + (x - X[0]) * s, ty = (y) => h - oy - (y - Y[0]) * s;
  g.strokeStyle = css("--grid");
  g.fillStyle = css("--muted");
  g.font = "11px system-ui";
  for (let x = -9; x <= -1; x++) { g.beginPath(); g.moveTo(tx(x), ty(Y[0])); g.lineTo(tx(x), ty(Y[1])); g.stroke(); if (x % 2 === 1 || x === -1) g.fillText(`${x}`, tx(x) - 6, ty(Y[0]) + 14); }
  for (let y = -3; y <= 4; y++) { g.beginPath(); g.moveTo(tx(X[0]), ty(y)); g.lineTo(tx(X[1]), ty(y)); g.stroke(); if (y % 2 === 0) g.fillText(`${y}`, tx(X[0]) - 18, ty(y) + 4); }
  // ground-truth path, prediction trail
  g.strokeStyle = css("--axis"); g.lineWidth = 2;
  g.beginPath();
  for (let t = 0; t < r.frames; t++) { const x = tx(r.pose[t * 3]), y = ty(r.pose[t * 3 + 1]); t ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.stroke();
  g.fillStyle = css("--c1");
  for (let t = Math.max(0, f - 25); t <= f; t++) {
    const o = decodeOut(t);
    g.globalAlpha = 0.15 + (0.6 * (t - f + 25)) / 25;
    g.beginPath(); g.arc(tx(o.x), ty(o.y), 2.5, 0, 2 * Math.PI); g.fill();
  }
  g.globalAlpha = 1;
  const arrow = (x, y, yaw, color, fill) => {
    const px = tx(x), py = ty(y);
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
    g.beginPath(); g.arc(px, py, 9, 0, 2 * Math.PI); fill ? g.fill() : g.stroke();
    g.beginPath(); g.moveTo(px, py); g.lineTo(px + Math.cos(yaw) * 22, py - Math.sin(yaw) * 22); g.stroke();
  };
  const gt = [r.pose[f * 3], r.pose[f * 3 + 1], r.pose[f * 3 + 2]], o = decodeOut(f);
  arrow(o.x, o.y, o.yaw, css("--c1"), true);
  g.strokeStyle = css("--surface"); g.lineWidth = 2; g.beginPath(); g.arc(tx(o.x), ty(o.y), 9, 0, 2 * Math.PI); g.stroke();
  arrow(gt[0], gt[1], gt[2], css("--ink"), false);
  g.fillStyle = css("--muted");
  g.fillText("x (m)", w - pad - 24, h - 4);
  c.onmousemove = (e) => {
    const b = c.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
    const near = (x, y) => (tx(x) - mx) ** 2 + (ty(y) - my) ** 2 < 200;
    if (near(o.x, o.y)) showTip(e, `Model: (${fmt(o.x, 2)}, ${fmt(o.y, 2)}) m<br>heading ${fmt((o.yaw * 180) / Math.PI)}°`);
    else if (near(gt[0], gt[1])) showTip(e, `Vicon: (${fmt(gt[0], 2)}, ${fmt(gt[1], 2)}) m<br>heading ${fmt((gt[2] * 180) / Math.PI)}°`);
    else hideTip();
  };
}

function drawProfile() {
  const c = $("profile"), [g, w, h] = setupCanvas(c), r = state.rec;
  if (!state.profileImg || state.profileTheme !== css("--seq-hi")) {
    const img = new ImageData(r.frames, 256);
    const lo = css("--seq-lo").match(/\w\w/g).map((v) => parseInt(v, 16)), hi = css("--seq-hi").match(/\w\w/g).map((v) => parseInt(v, 16));
    for (let t = 0; t < r.frames; t++)
      for (let b = 0; b < 256; b++) {
        const v = Math.pow(r.profile[t * 256 + b] / 255, 0.8), k = ((255 - b) * r.frames + t) * 4;
        for (let ch = 0; ch < 3; ch++) img.data[k + ch] = lo[ch] + (hi[ch] - lo[ch]) * v;
        img.data[k + 3] = 255;
      }
    state.profileImg = img;
    state.profileTheme = css("--seq-hi");
    state.profileBitmap = null;
    createImageBitmap(img).then((bm) => { state.profileBitmap = bm; drawProfile(); });
  }
  const pl = 34, pb = 18;
  if (state.profileBitmap) g.drawImage(state.profileBitmap, pl, 0, w - pl, h - pb);
  g.fillStyle = css("--muted");
  g.font = "11px system-ui";
  g.fillText("far", 4, 12);
  g.fillText("near", 4, h - pb - 2);
  g.fillText("0 s", pl, h - 4);
  g.fillText(`${fmt(r.frames / 5)} s`, w - 34, h - 4);
  const x = pl + ((state.frame + 0.5) / r.frames) * (w - pl);
  g.strokeStyle = css("--ink");
  g.lineWidth = 2;
  g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h - pb); g.stroke();
  for (const [k, lab] of [["val", "val"], ["test", "test"]]) {
    const sx = pl + (r.split[k] / r.frames) * (w - pl);
    g.setLineDash([3, 3]); g.strokeStyle = css("--muted"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, h - pb); g.stroke(); g.setLineDash([]);
    g.fillText(lab, sx + 3, h - 4);
  }
  c.onclick = (e) => {
    const b = c.getBoundingClientRect(), t = Math.floor(((e.clientX - b.left - pl) / (w - pl)) * r.frames);
    if (t >= 0 && t < r.frames) { state.frame = t; $("frame").value = t; draw(); }
  };
}

function drawReadout() {
  const r = state.rec, f = state.frame, o = decodeOut(f), m = state.model;
  const gx = r.pose[f * 3], gy = r.pose[f * 3 + 1], gyaw = r.pose[f * 3 + 2];
  const err = Math.hypot(o.x - gx, o.y - gy);
  let dy = Math.abs(((o.yaw - gyaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI) * (180 / Math.PI);
  const top = o.probs.indexOf(Math.max(...o.probs));
  $("readout").innerHTML = `
    <h3>What the model sees in frame ${f + 1}</h3>
    ${m.classes.map((c, i) => `<div class="prob"><span class="lab"><i class="sw" style="background:var(${CLASS_VARS[i]})"></i>${c}${i === r.label ? " ✓" : ""}</span>
      <span class="bar"><i style="width:${(o.probs[i] * 100).toFixed(1)}%;background:var(${CLASS_VARS[i]})"></i></span><span class="mono">${fmt(o.probs[i] * 100, 1)}%</span></div>`).join("")}
    <div class="kv">
      <div><b class="${top === r.label ? "ok" : "bad"}">${top === r.label ? "Correct" : "Wrong"}</b><span>class (✓ = truth)</span></div>
      <div><b>${fmt(err, 2)} m</b><span>position error</span></div>
      <div><b>${fmt(dy)}°</b><span>heading error</span></div>
      <div><b>${r.counts[f]}</b><span>radar points this frame</span></div>
    </div>`;
}

// ---------- model section ----------
function renderTiles() {
  const mt = state.model.metrics, t = mt.test, secs = mt.history.reduce((a, h) => a + h.seconds, 0);
  const tiles = [
    [`${fmt(t.acc * 100, 1)}%`, "test accuracy across 4 objects"],
    [`${fmt(t.pos_err_dynamic_median_m * 100)} cm`, "median position error, moving objects"],
    [`${fmt(mt.params / 1000)}k`, "parameters"],
    [`${fmt(secs / 60, 1)} min`, `MLX training on Apple ${mt.device.includes("gpu") ? "GPU" : "CPU"}`],
    [fmt(mt.sizes.train + mt.sizes.val + mt.sizes.test), "radar frame windows"],
  ];
  $("tiles").innerHTML = tiles.map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join("");
}

function lineChart(el, xs, ys, { fmtY = (v) => fmt(v, 2), label }) {
  const W = 520, H = 220, L = 44, R = 12, T = 10, B = 26;
  const yMax = Math.max(...ys) * 1.05, yMin = Math.min(0, Math.min(...ys));
  const sx = (i) => L + (i / (xs.length - 1)) * (W - L - R), sy = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((k) => yMin + k * (yMax - yMin));
  const path = ys.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">
    ${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${fmtY(v)}</text>`).join("")}
    <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"/>
    ${[0, 0.5, 1].map((k) => { const i = Math.round(k * (xs.length - 1)); return `<text x="${sx(i)}" y="${H - 8}" text-anchor="middle">epoch ${xs[i]}</text>`; }).join("")}
    <path d="${path}" fill="none" stroke="var(--c1)" stroke-width="2" stroke-linejoin="round"/>
    <line class="xh" y1="${T}" y2="${H - B}" stroke="var(--axis)" visibility="hidden"/>
    <circle class="dot" r="4" fill="var(--c1)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
    <rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>
  </svg>`;
  const svg = el.querySelector("svg"), xh = svg.querySelector(".xh"), dot = svg.querySelector(".dot");
  svg.onmousemove = (e) => {
    const b = svg.getBoundingClientRect(), x = ((e.clientX - b.left) / b.width) * W;
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((x - L) / (W - L - R)) * (xs.length - 1))));
    xh.setAttribute("x1", sx(i)); xh.setAttribute("x2", sx(i)); xh.setAttribute("visibility", "visible");
    dot.setAttribute("cx", sx(i)); dot.setAttribute("cy", sy(ys[i])); dot.setAttribute("visibility", "visible");
    showTip(e, `epoch ${xs[i]}<br><b>${fmtY(ys[i])}</b> ${label}`);
  };
  svg.onmouseleave = () => { xh.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); };
}

function renderTraining() {
  const h = state.model.metrics.history, xs = h.map((d) => d.epoch);
  lineChart($("lossChart"), xs, h.map((d) => d.loss), { label: "loss" });
  lineChart($("posChart"), xs, h.map((d) => d.val_pos_err_dynamic_m), { label: "m", fmtY: (v) => fmt(v, 2) });
}

function renderConfusion() {
  const m = state.model, cm = m.metrics.test.confusion, mx = Math.max(...cm.flat());
  $("confusion").innerHTML = `<table class="cm"><tr><th></th>${m.classes.map((c) => `<th style="text-align:center">${c}</th>`).join("")}</tr>
    ${cm.map((row, i) => `<tr><th>${m.classes[i]}</th>${row.map((v) => {
      const t = Math.sqrt(v / mx);
      return `<td style="background:color-mix(in oklab, var(--seq-hi) ${Math.round(t * 85)}%, var(--seq-lo));color:${t > 0.5 ? "#fff" : "var(--ink)"}">${v}</td>`;
    }).join("")}</tr>`).join("")}</table>`;
}

function renderPerRec() {
  const pr = state.model.metrics.test.per_recording;
  $("perRec").innerHTML = `<table><tr><th>Recording</th><th>frames</th><th>acc</th><th>pos err</th><th>yaw err</th></tr>
    ${Object.entries(pr).map(([k, v]) => `<tr><td>${pretty(k)}</td><td>${v.n}</td><td>${fmt(v.acc * 100, 1)}%</td><td>${fmt(v.pos_err_m, 2)} m</td><td>${fmt(v.yaw_err_deg, 1)}°</td></tr>`).join("")}</table>`;
}

// ---------- benchmark ----------
const SIZES = [1, 16, 256, 1024, 4096, 16384];

async function runBench() {
  const btn = $("benchBtn");
  btn.disabled = true;
  const eng = state.engine, res = {};
  const rec = state.rec, m = state.model;
  try {
    for (const B of SIZES) {
      btn.textContent = `Batch ${fmt(B)}…`;
      const frames = Array.from({ length: B }, (_, i) => (i * 997) % rec.frames);
      const { feats, offsets } = makeBatch(rec, frames, m);
      await eng.bench(feats, offsets, B, 1);
      let t = performance.now();
      await eng.bench(feats, offsets, B, 1);
      const one = performance.now() - t;
      const reps = Math.max(1, Math.min(500, Math.ceil(700 / Math.max(one, 0.05))));
      t = performance.now();
      await eng.bench(feats, offsets, B, reps);
      res[B] = (B * reps) / ((performance.now() - t) / 1000);
      state.bench[`${eng.name} (${eng.detail})`] = { ...res };
      renderBench();
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    $("parity").innerHTML += ` <span class="bad">Benchmark failed: ${e.message}</span>`;
  }
  btn.disabled = false;
  btn.textContent = "Run benchmark";
}

function renderBench() {
  const series = { ...state.bench };
  const mlx = state.model.metrics.mlx_throughput;
  series["MLX (Apple GPU, Python) reference"] = Object.fromEntries(Object.entries(mlx).map(([k, v]) => [k, v]));
  const names = Object.keys(series);
  const colors = ["--c1", "--c2", "--c3", "--c4"];
  const color = (n) => (n.startsWith("MLX") ? "--muted" : colors[Object.keys(state.bench).indexOf(n) % 4]);
  const all = names.flatMap((n) => Object.values(series[n]));
  const max = Math.max(...all, 1);
  const rowH = 14, gap = 2, groupGap = 14, L = 96, W = 620, R = 90;
  let y = 0;
  const rows = [];
  for (const B of [...new Set([...SIZES, ...Object.keys(mlx).map(Number)])].sort((a, b) => a - b)) {
    const present = names.filter((n) => series[n][B] != null);
    if (!present.length) continue;
    rows.push(`<text x="${L - 8}" y="${y + (present.length * (rowH + gap)) / 2 + 3}" text-anchor="end">batch ${fmt(B)}</text>`);
    for (const n of present) {
      const v = series[n][B], w = Math.max(2, (Math.log10(v + 1) / Math.log10(max + 1)) * (W - L - R));
      rows.push(`<rect x="${L}" y="${y}" width="${w}" height="${rowH}" rx="4" fill="var(${color(n)})" data-tip="${n}<br>batch ${fmt(B)}: <b>${fmt(v)}</b> frames/s"/>
        <text x="${L + w + 6}" y="${y + rowH - 3}">${fmt(v)}</text>`);
      y += rowH + gap;
    }
    y += groupGap;
  }
  const legend = names.map((n) => `<span><i class="sw" style="background:var(${color(n)});border-radius:3px"></i> ${n}</span>`).join(" ");
  $("benchChart").innerHTML = `<div class="legend" style="margin-bottom:8px">${legend}</div>
    <svg viewBox="0 0 ${W} ${y}" style="width:100%;max-width:${W}px;height:auto">${rows.join("")}</svg>
    <p class="note">Log-scale bars. Browser numbers are measured on your device (throughput mode: inputs uploaded once, kernels queued back to back). The MLX row was measured at training time on the author's Mac.${state.bench && Object.keys(state.bench).length ? "" : " Press <b>Run benchmark</b> to add your device."}</p>`;
  $("benchChart").querySelectorAll("rect").forEach((r) => {
    r.onmousemove = (e) => showTip(e, r.dataset.tip);
    r.onmouseleave = hideTip;
  });
}

main().then(renderBench).catch((e) => {
  console.error(e);
  $("runInfo").innerHTML = `<span class="bad">Failed to start: ${e.message}</span>`;
});
