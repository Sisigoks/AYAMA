// CPU benchmark for the Phase 4 viewer. Emits JSON on stdout.
//
//   node scripts/bench_viewer.mjs [tileset-dir]
//
// This measures the real web/src/renderer.js, not a reimplementation, because
// the point is the code a browser actually runs. Everything here is the CPU work that
// happens BEFORE the GPU is involved: turning PNG bytes back into metres,
// building vertex buffers, applying colour ramps.
//
// What is deliberately not measured: rasterisation, shader compilation, frame
// rate. Those need a real GL context and a real browser, and a number invented
// for them here would be worth less than nothing.
//
// Timings are best-of-N. The floor is the honest number; a slow repeat measures
// whatever else the machine was doing.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tilesetDir = process.argv[2] || path.join(ROOT, 'web/data');

// colourize calls `new ImageData(...)`. Node has no DOM, so a minimal stand-in
// keeps the measurement on the LUT loop where it belongs.
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) {
      this.data = data; this.width = width; this.height = height;
    }
  };
}

// pathToFileURL, not a bare path: on Windows an absolute path starts with a
// drive letter, which the ESM loader reads as an unknown URL scheme.
const TRAKSHA = await import(
  pathToFileURL(path.join(ROOT, 'web/src/renderer.js')).href);

function best(label, fn, repeats, units) {
  let t = Infinity;
  let out;
  // Warm up first. V8 tiers a function up over its first few calls, so without
  // this the op that happens to run first is measured un-optimised and looks
  // slower than an identical one later in the file.
  for (let i = 0; i < 3; i++) out = fn();
  for (let i = 0; i < repeats; i++) {
    const t0 = process.hrtime.bigint();
    out = fn();
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;   // ms
    if (dt < t) t = dt;
  }
  return { op: label, ms: Math.round(t * 1000) / 1000, units: units || null, result: out };
}

const report = { node: process.version, platform: process.platform, ops: [] };

// ── the manifest the page would load ────────────────────────────────────────
const manifestPath = path.join(tilesetDir, 'tileset.json');
if (!fs.existsSync(manifestPath)) {
  console.log(JSON.stringify({ skipped: 'no tileset at ' + manifestPath }));
  process.exit(0);
}
const manifestText = fs.readFileSync(manifestPath, 'utf8');
const manifest = JSON.parse(manifestText);
const lod0 = manifest.lods[0];
const px = lod0.width * lod0.height;
const mpix = px / 1e6;

report.scene = { width: lod0.width, height: lod0.height, megapixels: +mpix.toFixed(3),
                 tiles: lod0.tiles.length, lods: manifest.lods.length };

const R = 5;
const add = (o) => { delete o.result; report.ops.push(o); };

// ── JSON parse: the very first thing the page does ──────────────────────────
add(best('parse tileset.json', () => JSON.parse(manifestText), R));

// ── decode: PNG bytes -> metres ─────────────────────────────────────────────
// A full-scene RGBA buffer, which is what canvas.getImageData hands back.
const rgba = new Uint8ClampedArray(px * 4);
for (let i = 0; i < px; i++) {
  rgba[i * 4] = i & 255; rgba[i * 4 + 1] = (i >> 8) & 255;
  rgba[i * 4 + 2] = (i >> 16) & 255; rgba[i * 4 + 3] = 255;
}
const heights = TRAKSHA.decodeTerrainRGBA(rgba);
add(best('decode terrain-rgb, whole scene', () => TRAKSHA.decodeTerrainRGBA(rgba), R));
add(best('decode linear, whole scene', () => TRAKSHA.decodeLinearRGBA(rgba, 0, 42), R));

// Reused output buffer: the allocation is a real part of the cost, and the page
// could avoid it. Measuring both says how much that optimisation is worth.
const scratch = new Float32Array(px);
add(best('decode terrain-rgb, reusing the buffer',
         () => TRAKSHA.decodeTerrainRGBA(rgba, scratch), R));

// ── geometry: what goes into the GL buffers ─────────────────────────────────
const t = lod0.tiles[0];
const tw = t.width, th = t.height;
const tileHeights = heights.subarray(0, tw * th);
add(best('tileGeometry, one tile',
         () => TRAKSHA.tileGeometry(tileHeights, tw, th, lod0.gsd_m, t.x0, t.y0,
                                  lod0.height, [0, 0]), R));
add(best('gridIndices, one tile', () => TRAKSHA.gridIndices(tw, th, true), R));

// ── colour: what a layer switch costs ───────────────────────────────────────
if (TRAKSHA.colourize) {
  add(best('colourize, one tile',
           () => TRAKSHA.colourize(tileHeights, tw, th, 'viridis', 0, 1), R));
} else {
  report.ops.push({ op: 'colourize, one tile', ms: null,
                    note: 'not exported from web/src/renderer.js' });
}
add(best('build a 256-entry LUT', () => TRAKSHA.lut('magma'), R));

// ── per-scene totals: what the page really pays at LOD 0 ────────────────────
const nTiles = lod0.tiles.length;
const ms = (name) => (report.ops.find(o => o.op === name) || {}).ms || 0;
const dataLayers = ['dsm', 'ndsm', 'sigma', 'error']
  .filter(k => manifest.layers && manifest.layers[k]).length;

report.totals_ms = {
  decode_all_layers: +(ms('decode linear, whole scene') * dataLayers).toFixed(2),
  geometry_all_tiles: +((ms('tileGeometry, one tile') +
                         ms('gridIndices, one tile')) * nTiles).toFixed(2),
  colourize_layer_switch: +(ms('colourize, one tile') * nTiles).toFixed(2),
};
report.totals_ms.first_paint_cpu = +(
  report.totals_ms.decode_all_layers +
  report.totals_ms.geometry_all_tiles).toFixed(2);
report.throughput_mpix_per_s = {
  decode_terrain_rgb: +(mpix / (ms('decode terrain-rgb, whole scene') / 1000)).toFixed(1),
  decode_linear: +(mpix / (ms('decode linear, whole scene') / 1000)).toFixed(1),
};


console.log(JSON.stringify(report, null, 2));
