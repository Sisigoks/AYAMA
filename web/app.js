/* AYAMA viewer — Phase 4.
 *
 * No frameworks, no CDN, no build step, no network. The same reason every
 * raster is written as a COG: the deliverable has to be openable by someone who
 * will not install anything first. `python -m ayama.cli viewer <run>` serves
 * this directory and the tileset, and that is the whole toolchain.
 *
 * Two decisions worth stating, because both are correctness rather than taste.
 *
 * 1. Encoded layers are decoded on the CPU, not in a shader. Packing 24-bit
 *    values into RGB and unpacking them in GLSL invites two silent corruptions:
 *    the browser may colour-manage or premultiply a texture upload, and mediump
 *    floats cannot hold 16777215 exactly. Decoding through a 2D canvas gives
 *    back the exact bytes PIL wrote, and the same Float32Array then serves the
 *    cursor readout. The GPU only ever samples ordinary colour textures.
 *
 * 2. Nothing here recomputes elevation. Every metre on screen was decoded from
 *    a tile Phase 3 wrote from a raster Phase 2 produced. The viewer's job is
 *    to show that surface honestly, including when it is a bad one — which on
 *    the current benchmark it is, and `manifest.notes` says so on screen.
 */
'use strict';

var AYAMA = (function () {

// ── pure helpers ────────────────────────────────────────────────────────────

var MAX_CODE = 256 * 256 * 256 - 1;      // 16777215
var TERRAIN_BASE = -10000.0;
var TERRAIN_STEP = 0.1;

/** RGBA bytes -> metres, Mapbox Terrain-RGB. Mirrors ayama/mesh/encode.py. */
function decodeTerrainRGBA(data, out) {
  var n = data.length / 4;
  out = out || new Float32Array(n);
  for (var i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = TERRAIN_BASE +
      ((data[j] << 16) | (data[j + 1] << 8) | data[j + 2]) * TERRAIN_STEP;
  }
  return out;
}

/** RGBA bytes -> values, 24-bit linear over [vmin, vmax]. */
function decodeLinearRGBA(data, vmin, vmax, out) {
  var n = data.length / 4, span = (vmax - vmin) / MAX_CODE;
  out = out || new Float32Array(n);
  for (var i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = vmin + ((data[j] << 16) | (data[j + 1] << 8) | data[j + 2]) * span;
  }
  return out;
}

/* Colour ramps, anchor-for-anchor the same as ayama/dsm/cog.py's fallback LUT,
   so a PNG preview and this page never disagree about what a height looks like. */
var RAMPS = {
  terrain: [[51, 51, 153], [0, 153, 102], [243, 226, 137], [140, 92, 61], [255, 255, 255]],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma:   [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  rdbu_r:  [[5, 48, 97], [146, 197, 222], [247, 247, 247], [244, 165, 130], [103, 0, 31]],
  gray:    [[0, 0, 0], [255, 255, 255]]
};

/** 256-entry RGB lookup table for a named ramp. */
function lut(name) {
  var anchors = RAMPS[name] || RAMPS.gray, out = new Uint8Array(256 * 3);
  var last = anchors.length - 1;
  for (var i = 0; i < 256; i++) {
    var t = i / 255 * last, k = Math.min(Math.floor(t), last - 1), f = t - k;
    for (var c = 0; c < 3; c++) {
      out[i * 3 + c] = Math.round(anchors[k][c] * (1 - f) + anchors[k + 1][c] * f);
    }
  }
  return out;
}

function cssRamp(name) {
  var a = RAMPS[name] || RAMPS.gray;
  return 'linear-gradient(to right,' + a.map(function (c) {
    return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
  }).join(',') + ')';
}

/** Which ramp and display range a layer gets. Error is symmetric about zero. */
function layerStyle(key, spec) {
  var st = (spec && spec.stats) || {};
  var ramp = { dsm: 'terrain', ndsm: 'viridis', sigma: 'magma', error: 'rdbu_r' }[key] || 'gray';
  var lo, hi;
  if (key === 'error') {
    var m = Math.max(Math.abs(st.p1 || 0), Math.abs(st.p99 || 0)) || 1;
    lo = -m; hi = m;
  } else {
    lo = (st.p1 !== undefined ? st.p1 : st.min) || 0;
    hi = (st.p99 !== undefined ? st.p99 : st.max);
    if (hi === undefined || hi - lo < 1e-9) hi = lo + 1;
  }
  return { ramp: ramp, lo: lo, hi: hi };
}

/** Default LOD: the finest level that stays under a vertex budget. */
function pickLodIndex(manifest, budget) {
  budget = budget || 1600000;
  var lods = manifest.lods || [];
  for (var i = 0; i < lods.length; i++) {
    if (lods[i].width * lods[i].height <= budget) return i;
  }
  return Math.max(0, lods.length - 1);
}

function fmt(v, unit, digits) {
  if (v === null || v === undefined || !isFinite(v)) return '–';
  return v.toFixed(digits === undefined ? 2 : digits) + (unit ? ' ' + unit : '');
}

// ── 4x4 matrices, column-major, the GL convention ───────────────────────────

function mat4Perspective(fovy, aspect, near, far) {
  var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0]);
}

function normalize3(v) {
  var l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }

function mat4LookAt(eye, target, up) {
  var f = normalize3(sub3(target, eye));
  var s = normalize3(cross3(f, up));
  var u = cross3(s, f);
  return new Float32Array([
    s[0], u[0], -f[0], 0,
    s[1], u[1], -f[1], 0,
    s[2], u[2], -f[2], 0,
    -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]),
    -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]),
    f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2], 1]);
}

function mat4Mul(a, b) {
  var o = new Float32Array(16);
  for (var c = 0; c < 4; c++) {
    for (var r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
                     a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

// ── DOM panels (no WebGL — this half runs under jsdom) ───────────────────────

function $(sel) { return document.querySelector(sel); }

function el(tag, cls, text) {
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function kv(table, rows) {
  table.innerHTML = '';
  rows.forEach(function (r) {
    if (r[1] === null || r[1] === undefined || r[1] === '') return;
    var tr = el('tr');
    tr.appendChild(el('td', null, r[0]));
    tr.appendChild(el('td', null, String(r[1])));
    table.appendChild(tr);
  });
}

function renderNotes(manifest) {
  var host = $('#notes');
  if (!host) return 0;
  host.innerHTML = '';
  var notes = manifest.notes || [];
  notes.forEach(function (n) {
    var d = el('div', 'note ' + (n.level || 'info'));
    d.appendChild(el('b', null, n.level === 'critical' ? 'Read this first'
      : n.level === 'warning' ? 'Warning' : 'Note'));
    d.appendChild(el('span', null, n.text));
    host.appendChild(d);
  });
  return notes.length;
}

function renderStats(manifest) {
  var g = manifest.grid || {}, L = manifest.layers || {};
  var dsm = (L.dsm && L.dsm.stats) || {}, nd = (L.ndsm && L.ndsm.stats) || {};
  var sg = (L.sigma && L.sigma.stats) || {};
  kv($('#stats'), [
    ['raster', g.width + ' × ' + g.height + ' px'],
    ['GSD', fmt(g.gsd_m, 'm', 3)],
    ['extent', g.extent_m ? fmt(g.extent_m[0], null, 0) + ' × ' + fmt(g.extent_m[1], 'm', 0) : null],
    ['CRS', manifest.crs || 'none'],
    ['elevation', dsm.min !== undefined ? fmt(dsm.min, null, 1) + ' … ' + fmt(dsm.max, 'm', 1) : null],
    ['above ground', nd.max !== undefined ? fmt(nd.min, null, 2) + ' … ' + fmt(nd.max, 'm', 2) : null],
    ['mean 1σ', sg.mean !== undefined ? fmt(sg.mean, 'm') : null],
    ['tier', manifest.tier ? 'Tier ' + manifest.tier : null]
  ]);
}

function renderMetrics(manifest) {
  var m = manifest.metrics || {};
  kv($('#metrics'), [
    ['MAE', fmt(m.mae_m, 'm')],
    ['RMSE', fmt(m.rmse_m, 'm')],
    ['bias', fmt(m.bias_m, 'm')],
    ['Pearson r', fmt(m.pearson_r, null, 3)],
    ['edge F1', fmt(m.edge_f1, null, 3)],
    ['1σ coverage', fmt(m.coverage_1s, null, 3)],
    ['ECE', fmt(m.ece_m, 'm')]
  ]);
  var note = $('#metrics-note');
  if (!note) return;
  note.textContent = Object.keys(m).length
    ? 'Measured against the reference DSM by Phase 2. edge F1 and 1σ coverage are the ' +
      'two rows that describe structure and honesty; MAE alone cannot tell a working ' +
      'surface model from a DEM interpolator.'
    : 'This run was not validated against a reference DSM, so there are no metrics.';
}

function renderProvenance(manifest) {
  var p = manifest.provenance || {};
  kv($('#prov'), [
    ['backbone', p.backbone], ['segmentation', p.segmentation], ['DEM', p.dem],
    ['tier', p.tier], ['chip', p.chip], ['lattice', p.lattice_stride],
    ['bootstraps', p.n_bootstrap], ['built', manifest.generated_utc]
  ]);
}

function renderDownloads(manifest) {
  var host = $('#downloads');
  if (!host) return;
  host.innerHTML = '';
  var items = [['tileset.json', dataBase() + 'tileset.json']];
  if (manifest.mesh && manifest.mesh.obj) {
    items.push(['surface.obj — ' + (manifest.mesh.triangles || 0).toLocaleString() +
                ' triangles', dataBase() + manifest.mesh.obj]);
    if (manifest.mesh.mtl) items.push(['surface.mtl', dataBase() + manifest.mesh.mtl]);
    if (manifest.mesh.texture) items.push(['surface.jpg', dataBase() + manifest.mesh.texture]);
  }
  items.forEach(function (it) {
    var li = el('li'), a = el('a', null, it[0]);
    a.href = it[1];
    a.setAttribute('download', '');
    li.appendChild(a);
    host.appendChild(li);
  });
  var li = el('li');
  li.appendChild(el('span', 'sz', 'source run: ' + (manifest.source_run || 'unknown')));
  host.appendChild(li);
}

function renderLayerButtons(manifest, onPick) {
  var host = $('#layer-buttons');
  if (!host) return [];
  host.innerHTML = '';
  var order = ['texture', 'dsm', 'ndsm', 'sigma', 'error'];
  var keys = order.filter(function (k) { return manifest.layers && manifest.layers[k]; });
  keys.forEach(function (k) {
    var spec = manifest.layers[k];
    var b = el('button', null, spec.label || k);
    b.dataset.layer = k;
    b.setAttribute('aria-pressed', String(k === (manifest.default_layer || keys[0])));
    b.addEventListener('click', function () {
      host.querySelectorAll('button').forEach(function (o) {
        o.setAttribute('aria-pressed', String(o === b));
      });
      if (onPick) onPick(k);
    });
    host.appendChild(b);
  });
  return keys;
}

function renderLodOptions(manifest, onPick) {
  var sel = $('#lod');
  if (!sel) return;
  sel.innerHTML = '';
  (manifest.lods || []).forEach(function (l, i) {
    var o = el('option', null,
      'LOD ' + l.lod + ' — ' + l.width + '×' + l.height + ' px, ' +
      l.tiles.length + ' tile' + (l.tiles.length === 1 ? '' : 's') +
      ', ' + fmt(l.gsd_m, 'm', 2) + '/px');
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.addEventListener('change', function () { if (onPick) onPick(parseInt(sel.value, 10)); });
}

function renderLegend(key, spec) {
  var box = $('#legend');
  if (!box) return;
  if (key === 'texture' || !spec) { box.hidden = true; return; }
  var s = layerStyle(key, spec);
  box.hidden = false;
  $('#lg-title').textContent = (spec.label || key) + (spec.units ? ' (' + spec.units + ')' : '');
  $('#lg-ramp').style.background = cssRamp(s.ramp);
  $('#lg-min').textContent = fmt(s.lo, null, 2);
  $('#lg-max').textContent = fmt(s.hi, null, 2);
}

function renderSun(manifest) {
  var box = $('#sun-ctl'), note = $('#sun-note');
  if (!box || !note) return;
  var t = (manifest.provenance && manifest.provenance.sun) || null;
  if (!t) { box.hidden = true; return; }
  box.hidden = false;
  note.textContent = t;
}

/** Everything the panel shows, from the manifest alone. No GL required. */
function renderPanels(manifest, onLayer, onLod) {
  var id = $('#scene-id');
  if (id) id.textContent = (manifest.source_run || '') +
    '  ·  ' + (manifest.grid ? manifest.grid.width + '×' + manifest.grid.height : '');
  renderNotes(manifest);
  renderStats(manifest);
  renderMetrics(manifest);
  renderProvenance(manifest);
  renderDownloads(manifest);
  renderSun(manifest);
  var keys = renderLayerButtons(manifest, onLayer);
  renderLodOptions(manifest, onLod);
  var start = manifest.default_layer || keys[0];
  renderLegend(start, manifest.layers && manifest.layers[start]);
  var en = $('#exagg-note');
  if (en) {
    var nd = manifest.layers && manifest.layers.ndsm && manifest.layers.ndsm.stats;
    en.textContent = nd && nd.max < 1
      ? 'This surface has ' + fmt(nd.max, 'm') + ' of relief above ground. At 1× it is ' +
        'a flat plain, because that is what Phase 2 produced. Exaggeration makes the ' +
        'defect visible; it does not fix it.'
      : 'Scales elevation only. Horizontal distances stay true.';
  }
  return start;
}

// ── image loading and CPU decode ────────────────────────────────────────────

// ── where the tiles live ────────────────────────────────────────────────────
// One viewer, two deployments: a prebuilt tileset served beside the page by
// `ayama viewer`, or a freshly reconstructed job served by `ayama serve`. The
// only difference is a base URL, so it is resolved once here rather than
// threaded through every fetch.
function dataBase() {
  var job = new URLSearchParams(location.search).get('job');
  if (job && /^[a-f0-9]{6,32}$/i.test(job)) return 'api/jobs/' + job + '/tiles/';
  return 'data/';
}

function loadImage(url) {
  return new Promise(function (resolve, reject) {
    var img = new Image();
    img.onload = function () { resolve(img); };
    img.onerror = function () { reject(new Error('cannot load ' + url)); };
    img.src = url;
  });
}

/** Exact bytes back out of a PNG. No colour management, no premultiply. */
function imagePixels(img) {
  var c = document.createElement('canvas');
  c.width = img.naturalWidth || img.width;
  c.height = img.naturalHeight || img.height;
  var ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
}

/** Values -> an RGBA colour image, using the same ramps as the PNG previews. */
function colourize(values, w, h, ramp, lo, hi) {
  var table = lut(ramp), out = new Uint8ClampedArray(w * h * 4), span = (hi - lo) || 1;
  for (var i = 0; i < w * h; i++) {
    var t = (values[i] - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    var k = (t * 255) | 0;
    out[i * 4] = table[k * 3];
    out[i * 4 + 1] = table[k * 3 + 1];
    out[i * 4 + 2] = table[k * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  return new ImageData(out, w, h);
}

// ── WebGL ───────────────────────────────────────────────────────────────────

var VERT = [
  'attribute vec2 aPos;',
  'attribute float aH;',
  'attribute vec2 aUV;',
  'uniform mat4 uMVP;',
  'uniform float uExagg;',
  'uniform float uZBase;',
  'varying vec2 vUV;',
  'void main() {',
  '  vUV = aUV;',
  '  gl_Position = uMVP * vec4(aPos.x, aPos.y, (aH - uZBase) * uExagg, 1.0);',
  '}'
].join('\n');

var FRAG = [
  'precision highp float;',
  'varying vec2 vUV;',
  'uniform sampler2D uTex;',
  'uniform sampler2D uNrm;',
  'uniform vec3 uLight;',
  'uniform float uShade;',
  'uniform float uWire;',
  'void main() {',
  '  vec3 base = texture2D(uTex, vUV).rgb;',
  '  vec3 n = normalize(texture2D(uNrm, vUV).rgb * 2.0 - 1.0);',
  '  float lam = max(dot(n, normalize(uLight)), 0.0);',
  '  float shade = mix(1.0, 0.30 + 0.85 * lam, uShade);',
  '  vec3 c = base * shade;',
  '  gl_FragColor = vec4(mix(c, vec3(0.55, 0.75, 1.0), uWire), 1.0);',
  '}'
].join('\n');

function compile(gl, type, src) {
  var sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl) {
  var p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

function texture(gl, source) {
  var t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

/** Grid geometry for one tile, in metres relative to the scene centre. */
function tileGeometry(heights, w, h, gsd, x0, y0, totalH, centre) {
  var pos = new Float32Array(w * h * 2), uv = new Float32Array(w * h * 2);
  var i = 0;
  for (var r = 0; r < h; r++) {
    for (var c = 0; c < w; c++) {
      // +Y is north, and raster row 0 is the northernmost row.
      pos[i * 2] = (x0 + c) * gsd - centre[0];
      pos[i * 2 + 1] = (totalH - 1 - (y0 + r)) * gsd - centre[1];
      uv[i * 2] = w > 1 ? c / (w - 1) : 0.5;
      uv[i * 2 + 1] = h > 1 ? r / (h - 1) : 0.5;
      i++;
    }
  }
  return { pos: pos, uv: uv, heights: heights, w: w, h: h };
}

function gridIndices(w, h, uint) {
  var quads = (w - 1) * (h - 1);
  var arr = uint ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6);
  var k = 0;
  for (var r = 0; r < h - 1; r++) {
    for (var c = 0; c < w - 1; c++) {
      var tl = r * w + c, tr = tl + 1, bl = tl + w, br = bl + 1;
      arr[k++] = bl; arr[k++] = br; arr[k++] = tr;
      arr[k++] = bl; arr[k++] = tr; arr[k++] = tl;
    }
  }
  return arr;
}

function wireIndices(w, h, uint) {
  var arr = [], step = Math.max(1, Math.floor(Math.min(w, h) / 64));
  for (var r = 0; r < h; r += step) {
    for (var c = 0; c < w - step; c += step) { arr.push(r * w + c, r * w + c + step); }
  }
  for (var c2 = 0; c2 < w; c2 += step) {
    for (var r2 = 0; r2 < h - step; r2 += step) { arr.push(r2 * w + c2, (r2 + step) * w + c2); }
  }
  return uint ? new Uint32Array(arr) : new Uint16Array(arr);
}

// ── application ─────────────────────────────────────────────────────────────

var state = {
  manifest: null, gl: null, prog: null, tiles: [], lodIndex: 0,
  layer: 'texture', exagg: 1, shade: true, wire: false,
  cam: { az: -0.6, el: 0.85, dist: 900, target: [0, 0, 0] },
  zBase: 0, centre: [0, 0], uintIndex: false, needsDraw: true, fps: 0
};

function sliderToExagg(v) {           // 0..300 -> 1..200, log-ish and smooth
  return Math.round(Math.pow(10, (v / 300) * Math.log10(200)) * 100) / 100;
}
function exaggToSlider(e) {
  return Math.round(Math.log10(Math.max(1, e)) / Math.log10(200) * 300);
}

function setStatusError(msg, detail) {
  var box = document.getElementById('boot-error');
  if (!box) return;
  box.hidden = false;
  box.innerHTML = '<strong>' + msg + '</strong>' +
    (detail ? '<br><code>' + detail + '</code>' : '');
}

function initGL(canvas) {
  var gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
  if (gl) {
    state.uintIndex = true;
  } else {
    gl = canvas.getContext('webgl', { antialias: true, alpha: false }) ||
         canvas.getContext('experimental-webgl');
    if (gl) state.uintIndex = !!gl.getExtension('OES_element_index_uint');
  }
  if (!gl) return null;
  state.gl = gl;
  state.prog = program(gl);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.05, 0.06, 0.08, 1);
  return gl;
}

function resize() {
  var canvas = document.getElementById('gl');
  if (!canvas || !state.gl) return;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  var h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    state.gl.viewport(0, 0, w, h);
    state.needsDraw = true;
  }
}

function cameraEye() {
  var c = state.cam, t = c.target;
  return [
    t[0] + c.dist * Math.cos(c.el) * Math.sin(c.az),
    t[1] + c.dist * Math.cos(c.el) * Math.cos(c.az),
    t[2] + c.dist * Math.sin(c.el)
  ];
}

function viewProj(aspect) {
  var eye = cameraEye();
  var span = state.manifest ? Math.max.apply(null, state.manifest.grid.extent_m) : 1000;
  var proj = mat4Perspective(50 * Math.PI / 180, aspect, Math.max(1, span / 2000), span * 12);
  return mat4Mul(proj, mat4LookAt(eye, state.cam.target, [0, 0, 1]));
}

function draw() {
  var gl = state.gl;
  if (!gl || !state.tiles.length) return;
  var canvas = gl.canvas;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.prog);

  var mvp = viewProj(canvas.width / Math.max(1, canvas.height));
  var P = state.prog;
  gl.uniformMatrix4fv(gl.getUniformLocation(P, 'uMVP'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(P, 'uExagg'), state.exagg);
  gl.uniform1f(gl.getUniformLocation(P, 'uZBase'), state.zBase);
  gl.uniform1f(gl.getUniformLocation(P, 'uShade'), state.shade ? 1 : 0);
  gl.uniform3fv(gl.getUniformLocation(P, 'uLight'), state.light || [0.4, 0.5, 0.75]);

  var aPos = gl.getAttribLocation(P, 'aPos');
  var aH = gl.getAttribLocation(P, 'aH');
  var aUV = gl.getAttribLocation(P, 'aUV');
  var uWire = gl.getUniformLocation(P, 'uWire');
  var tris = 0;

  state.tiles.forEach(function (t) {
    gl.bindBuffer(gl.ARRAY_BUFFER, t.bufPos);
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, t.bufH);
    gl.enableVertexAttribArray(aH); gl.vertexAttribPointer(aH, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, t.bufUV);
    gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, t.texColour);
    gl.uniform1i(gl.getUniformLocation(P, 'uTex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, t.texNormal);
    gl.uniform1i(gl.getUniformLocation(P, 'uNrm'), 1);

    var type = state.uintIndex ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    gl.uniform1f(uWire, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, t.bufIdx);
    gl.drawElements(gl.TRIANGLES, t.nIdx, type, 0);
    tris += t.nIdx / 3;
    if (state.wire) {
      gl.uniform1f(uWire, 0.85);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, t.bufWire);
      gl.drawElements(gl.LINES, t.nWire, type, 0);
    }
  });

  var hud = document.getElementById('hud-tris');
  if (hud) hud.textContent = Math.round(tris).toLocaleString() + ' tris';
}

function loop() {
  resize();
  if (state.needsDraw) { draw(); state.needsDraw = false; }
  requestAnimationFrame(loop);
}

/** Load one LOD: decode every tile, build buffers, upload textures. */
function loadLod(index) {
  var m = state.manifest, gl = state.gl;
  var lod = m.lods[index];
  state.lodIndex = index;
  var hud = document.getElementById('hud-lod');
  if (hud) hud.textContent = 'LOD ' + lod.lod + ' · ' + lod.width + '×' + lod.height;

  var gsd = lod.gsd_m;
  state.centre = [lod.width * gsd / 2, lod.height * gsd / 2];

  return Promise.all(lod.tiles.map(function (t) {
    var base = dataBase();
    var jobs = [loadImage(base + t.layers.dsm), loadImage(base + t.layers.normal)];
    ['ndsm', 'sigma', 'error'].forEach(function (k) {
      jobs.push(t.layers[k] ? loadImage(base + t.layers[k]) : Promise.resolve(null));
    });
    jobs.push(t.layers.texture ? loadImage(base + t.layers.texture) : Promise.resolve(null));
    return Promise.all(jobs).then(function (imgs) {
      var px = imagePixels(imgs[0]);
      var heights = decodeTerrainRGBA(px.data);
      var tile = {
        spec: t, w: px.w, h: px.h,
        heights: heights, images: { normal: imgs[1], texture: imgs[5] },
        values: {}
      };
      ['ndsm', 'sigma', 'error'].forEach(function (k, i) {
        var img = imgs[2 + i];
        if (!img || !m.layers[k]) return;
        var p = imagePixels(img);
        tile.values[k] = decodeLinearRGBA(p.data, m.layers[k].vmin, m.layers[k].vmax);
      });
      return tile;
    });
  })).then(function (tiles) {
    disposeTiles();
    var zs = [];
    tiles.forEach(function (t) { zs.push(t.heights[0], t.heights[t.heights.length - 1]); });
    var dsmStats = (m.layers.dsm && m.layers.dsm.stats) || {};
    state.zBase = dsmStats.mean !== undefined ? dsmStats.mean : (zs[0] || 0);

    state.tiles = tiles.map(function (t) {
      var g = tileGeometry(t.heights, t.w, t.h, gsd, t.spec.x0, t.spec.y0,
                           lod.height, state.centre);
      var idx = gridIndices(t.w, t.h, state.uintIndex);
      var wir = wireIndices(t.w, t.h, state.uintIndex);
      var o = {
        spec: t.spec, w: t.w, h: t.h, heights: t.heights, values: t.values,
        images: t.images, x0: t.spec.x0, y0: t.spec.y0,
        bufPos: buffer(gl, gl.ARRAY_BUFFER, g.pos),
        bufUV: buffer(gl, gl.ARRAY_BUFFER, g.uv),
        bufH: buffer(gl, gl.ARRAY_BUFFER, t.heights),
        bufIdx: buffer(gl, gl.ELEMENT_ARRAY_BUFFER, idx),
        bufWire: buffer(gl, gl.ELEMENT_ARRAY_BUFFER, wir),
        nIdx: idx.length, nWire: wir.length,
        texNormal: texture(gl, t.images.normal)
      };
      o.texColour = null;
      return o;
    });
    applyLayer(state.layer);
    frameScene();
    state.needsDraw = true;
  });
}

function buffer(gl, target, data) {
  var b = gl.createBuffer();
  gl.bindBuffer(target, b);
  gl.bufferData(target, data, gl.STATIC_DRAW);
  return b;
}

function disposeTiles() {
  var gl = state.gl;
  if (!gl) return;
  state.tiles.forEach(function (t) {
    [t.bufPos, t.bufUV, t.bufH, t.bufIdx, t.bufWire].forEach(function (b) {
      if (b) gl.deleteBuffer(b);
    });
    [t.texColour, t.texNormal].forEach(function (x) { if (x) gl.deleteTexture(x); });
  });
  state.tiles = [];
}

/** Swap what is draped on the surface. Colours come from the decoded values. */
function applyLayer(key) {
  var gl = state.gl, m = state.manifest;
  if (!gl || !state.tiles.length) return;
  state.layer = key;
  var spec = m.layers[key];
  var style = spec ? layerStyle(key, spec) : null;

  state.tiles.forEach(function (t) {
    if (t.texColour) { gl.deleteTexture(t.texColour); t.texColour = null; }
    if (key === 'texture' && t.images.texture) {
      t.texColour = texture(gl, t.images.texture);
      return;
    }
    var vals = key === 'dsm' ? t.heights : t.values[key];
    if (!vals) {                                   // asked for a layer this run lacks
      t.texColour = texture(gl, t.images.texture || blankImage());
      return;
    }
    t.texColour = texture(gl, colourize(vals, t.w, t.h, style.ramp, style.lo, style.hi));
  });
  renderLegend(key, spec);
  state.needsDraw = true;
}

function blankImage() { return new ImageData(new Uint8ClampedArray([120, 120, 120, 255]), 1, 1); }

function frameScene() {
  var m = state.manifest;
  var span = Math.max(m.grid.extent_m[0], m.grid.extent_m[1]);
  state.cam.target = [0, 0, 0];
  state.cam.dist = span * 1.25;
  state.needsDraw = true;
}

/** Ray from the cursor, refined against the height field. */
function pick(mx, my) {
  var canvas = state.gl && state.gl.canvas;
  if (!canvas || !state.tiles.length) return null;
  var rect = canvas.getBoundingClientRect();
  var ndcX = (mx - rect.left) / rect.width * 2 - 1;
  var ndcY = 1 - (my - rect.top) / rect.height * 2;

  var eye = cameraEye();
  var fwd = normalize3(sub3(state.cam.target, eye));
  var right = normalize3(cross3(fwd, [0, 0, 1]));
  var up = cross3(right, fwd);
  var tanF = Math.tan(50 * Math.PI / 180 / 2);
  var aspect = rect.width / Math.max(1, rect.height);
  var dir = normalize3([
    fwd[0] + right[0] * ndcX * tanF * aspect + up[0] * ndcY * tanF,
    fwd[1] + right[1] * ndcX * tanF * aspect + up[1] * ndcY * tanF,
    fwd[2] + right[2] * ndcX * tanF * aspect + up[2] * ndcY * tanF
  ]);
  if (Math.abs(dir[2]) < 1e-6) return null;

  // Start on the z = 0 plane (the mean elevation) and iterate: sample the
  // surface under the current guess, step the ray to that height, repeat.
  // Converges in a few passes on any surface without overhangs, and this one
  // has none by construction - it is a height field.
  var z = 0, hit = null;
  for (var it = 0; it < 12; it++) {
    var tRay = (z - eye[2]) / dir[2];
    if (tRay < 0) return null;
    var wx = eye[0] + dir[0] * tRay, wy = eye[1] + dir[1] * tRay;
    hit = sample(wx, wy);
    if (!hit) return null;
    var zNew = (hit.elevation - state.zBase) * state.exagg;
    if (Math.abs(zNew - z) < 1e-3) break;
    z = zNew;
  }
  return hit;
}

/** Look up every loaded layer at a world (x, y) in metres. */
function sample(wx, wy) {
  var lod = state.manifest.lods[state.lodIndex], gsd = lod.gsd_m;
  var col = (wx + state.centre[0]) / gsd;
  var row = (lod.height - 1) - (wy + state.centre[1]) / gsd;
  if (!(col >= 0 && col < lod.width && row >= 0 && row < lod.height)) return null;
  var c = Math.round(col), r = Math.round(row);
  for (var i = 0; i < state.tiles.length; i++) {
    var t = state.tiles[i];
    var lc = c - t.x0, lr = r - t.y0;
    if (lc < 0 || lr < 0 || lc >= t.w || lr >= t.h) continue;
    var k = lr * t.w + lc;
    return {
      col: c, row: r, gsd: gsd,
      elevation: t.heights[k],
      ndsm: t.values.ndsm ? t.values.ndsm[k] : null,
      sigma: t.values.sigma ? t.values.sigma[k] : null,
      error: t.values.error ? t.values.error[k] : null
    };
  }
  return null;
}

function showReadout(hit) {
  var box = document.getElementById('readout');
  if (!box) return;
  if (!hit) { box.hidden = true; return; }
  box.hidden = false;
  var tr = state.manifest.transform;
  var e = tr ? tr[2] + tr[0] * (hit.col + 0.5) : hit.col * hit.gsd;
  var n = tr ? tr[5] + tr[4] * (hit.row + 0.5) : hit.row * hit.gsd;
  document.getElementById('rd-x').textContent = fmt(e, null, 1);
  document.getElementById('rd-y').textContent = fmt(n, null, 1);
  document.getElementById('rd-z').textContent = fmt(hit.elevation, 'm');
  document.getElementById('rd-n').textContent = hit.ndsm === null ? '–' : fmt(hit.ndsm, 'm');
  document.getElementById('rd-s').textContent = hit.sigma === null ? '–' : fmt(hit.sigma, 'm');
}

function bindControls() {
  var canvas = document.getElementById('gl');
  var drag = null;

  canvas.addEventListener('pointerdown', function (e) {
    drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', function (e) {
    drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ }
  });
  canvas.addEventListener('pointermove', function (e) {
    if (drag) {
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        var s = state.cam.dist / 900;
        var az = state.cam.az;
        state.cam.target[0] -= (dx * Math.cos(az) + dy * Math.sin(az)) * s;
        state.cam.target[1] += (dx * Math.sin(az) - dy * Math.cos(az)) * s;
      } else {
        state.cam.az -= dx * 0.006;
        state.cam.el = Math.max(0.04, Math.min(1.5, state.cam.el + dy * 0.006));
      }
      state.needsDraw = true;
    } else {
      showReadout(pick(e.clientX, e.clientY));
    }
  });
  canvas.addEventListener('pointerleave', function () { showReadout(null); });
  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    state.cam.dist = Math.max(20, Math.min(60000,
      state.cam.dist * Math.pow(1.0015, e.deltaY)));
    state.needsDraw = true;
  }, { passive: false });

  var ex = document.getElementById('exagg'), exv = document.getElementById('exagg-val');
  function setExagg(v) {
    state.exagg = v;
    if (exv) exv.textContent = (v < 10 ? v.toFixed(1) : Math.round(v)) + '×';
    if (ex) ex.value = String(exaggToSlider(v));
    state.needsDraw = true;
  }
  if (ex) ex.addEventListener('input', function () { setExagg(sliderToExagg(+ex.value)); });
  document.querySelectorAll('.presets button').forEach(function (b) {
    b.addEventListener('click', function () { setExagg(+b.dataset.exagg); });
  });
  setExagg(1);

  var w = document.getElementById('wire');
  if (w) w.addEventListener('change', function () { state.wire = w.checked; state.needsDraw = true; });
  var sh = document.getElementById('shade');
  if (sh) sh.addEventListener('change', function () { state.shade = sh.checked; state.needsDraw = true; });

  var bp = document.getElementById('btn-panel');
  if (bp) bp.addEventListener('click', function () {
    var app = document.getElementById('app');
    var hidden = app.classList.toggle('no-side');
    bp.textContent = hidden ? 'Show panel' : 'Hide panel';
    bp.setAttribute('aria-expanded', String(!hidden));
    state.needsDraw = true;
  });

  window.addEventListener('resize', function () { state.needsDraw = true; });
}

/** Light direction from the run's own sun angles, when it recorded them. */
function lightFromSun(manifest) {
  var p = manifest.provenance || {};
  var az = p.sun_azimuth_deg, el = p.sun_elevation_deg;
  if (az === undefined || el === undefined || az === null || el === null) {
    return [0.4, 0.5, 0.75];
  }
  var a = az * Math.PI / 180, e = el * Math.PI / 180;
  return [Math.cos(e) * Math.sin(a), Math.cos(e) * Math.cos(a), Math.sin(e)];
}

function boot() {
  return fetch(dataBase() + 'tileset.json')
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (manifest) {
      state.manifest = manifest;
      state.light = lightFromSun(manifest);
      var start = renderPanels(manifest,
        function (k) { applyLayer(k); },
        function (i) { loadLod(i); });
      state.layer = start;

      var canvas = document.getElementById('gl');
      var gl = canvas && initGL(canvas);
      if (!gl) {
        var err = document.getElementById('gl-error');
        if (err) err.hidden = false;
        return manifest;
      }
      bindControls();
      var idx = pickLodIndex(manifest);
      var sel = document.getElementById('lod');
      if (sel) sel.value = String(idx);
      requestAnimationFrame(loop);
      return loadLod(idx).then(function () { return manifest; });
    })
    .catch(function (e) {
      setStatusError('Could not load ' + dataBase() + 'tileset.json.',
        'Build one with: python -m ayama.cli mesh &lt;run&gt; — or serve this page with ' +
        'python -m ayama.cli viewer &lt;run&gt;. (' + e.message + ')');
      throw e;
    });
}

if (typeof document !== 'undefined' && document.addEventListener) {
  document.addEventListener('DOMContentLoaded', function () {
    // `serve` shows an upload form first and boots the viewer itself once a
    // job finishes; `viewer` has a tileset sitting there and boots immediately.
    if (window.AYAMA_NO_AUTOBOOT) return;
    if (document.getElementById('landing') && !new URLSearchParams(location.search).get('job')) return;
    boot().catch(function () { /* surfaced in the DOM */ });
  });
}

return {
  decodeTerrainRGBA: decodeTerrainRGBA, decodeLinearRGBA: decodeLinearRGBA,
  lut: lut, cssRamp: cssRamp, colourize: colourize, layerStyle: layerStyle,
  pickLodIndex: pickLodIndex,
  fmt: fmt, mat4Perspective: mat4Perspective, mat4LookAt: mat4LookAt, mat4Mul: mat4Mul,
  tileGeometry: tileGeometry, gridIndices: gridIndices,
  renderPanels: renderPanels, renderNotes: renderNotes, renderLegend: renderLegend,
  sliderToExagg: sliderToExagg, exaggToSlider: exaggToSlider,
  boot: boot, state: state, dataBase: dataBase,
  MAX_CODE: MAX_CODE, TERRAIN_BASE: TERRAIN_BASE, TERRAIN_STEP: TERRAIN_STEP
};

})();

/* Explicit global. `'use strict'` means a `var` in eval'd code stays in the eval
   scope, so scripts/check_app.js would otherwise never see the module. */
if (typeof window !== 'undefined') { window.AYAMA = AYAMA; }
if (typeof module !== 'undefined' && module.exports) { module.exports = AYAMA; }
