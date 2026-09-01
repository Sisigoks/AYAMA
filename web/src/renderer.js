/* TRAKSHA viewer - the WebGL renderer.
 *
 * Lifted out of the old single-file app when the front end moved to React. The
 * split runs along one line: this module touches a canvas, a manifest and
 * nothing else, while every `document.getElementById` went to a component.
 *
 * React does not own a WebGL context. It owns the tree around it, and reaches
 * this through a ref - which is the supported escape hatch for imperative
 * graphics, not a workaround. `createViewer(canvas)` returns the handle.
 */
'use strict';

// ── pure helpers ────────────────────────────────────────────────────────────

var MAX_CODE = 256 * 256 * 256 - 1;      // 16777215
var TERRAIN_BASE = -10000.0;
var TERRAIN_STEP = 0.1;

/** RGBA bytes -> metres, Mapbox Terrain-RGB. Mirrors traksha/mesh/encode.py. */
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

/* Colour ramps, anchor-for-anchor the same as traksha/dsm/cog.py's fallback LUT,
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

/** Everything the panel shows, from the manifest alone. No GL required. */

// ── image loading and CPU decode ────────────────────────────────────────────

// ── where the tiles live ────────────────────────────────────────────────────
// One viewer, two deployments: a prebuilt tileset served beside the page by
// `traksha viewer`, or a freshly reconstructed job served by `traksha serve`. The
// only difference is a base URL, so it is resolved once here rather than
// threaded through every fetch.

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
  'attribute vec3 aN;',
  'uniform mat4 uMVP;',
  'uniform float uExagg;',
  'uniform float uZBase;',
  'uniform vec3 uEye;',
  'varying vec2 vUV;',
  'varying float vDist;',
  'varying float vHeight;',
  'varying vec3 vN;',
  'void main() {',
  '  vUV = aUV;',
  '  // Scaling z by uExagg tilts every surface, so the normal transforms by',
  '  // the inverse transpose - for diag(1,1,e) that is diag(1,1,1/e). A wall',
  '  // has nz = 0 and therefore correctly stays vertical.',
  '  vN = normalize(vec3(aN.x, aN.y, aN.z / max(uExagg, 1e-3)));',
  '  float z = (aH - uZBase) * uExagg;',
  '  vec3 world = vec3(aPos.x, aPos.y, z);',
  '  vDist = length(world - uEye);',
  '  vHeight = aH - uZBase;',
  '  gl_Position = uMVP * vec4(world, 1.0);',
  '}'
].join('\n');

var FRAG = [
  'precision highp float;',
  'varying vec2 vUV;',
  'varying float vDist;',
  'varying float vHeight;',
  'uniform sampler2D uTex;',
  'uniform sampler2D uNrm;',
  'uniform vec3 uLight;',
  'uniform float uShade;',
  'uniform float uWire;',
  'uniform float uFog;',
  'uniform float uTexel;',
  'uniform float uRelief;',
  'uniform float uVertexNormal;',
  'varying vec3 vN;',
  'void main() {',
  '  vec3 base = texture2D(uTex, vUV).rgb;',
  '  // A normal map is indexed by (u, v), and a wall shares its UV with the',
  '  // pavement under it - so a facade would shade as though it were ground.',
  '  // A mesh that carries per-vertex normals overrides the map.',
  '  vec3 n = mix(normalize(texture2D(uNrm, vUV).rgb * 2.0 - 1.0),',
  '               normalize(vN), uVertexNormal);',
  '',
  '  // direct sun',
  '  float lam = max(dot(n, normalize(uLight)), 0.0);',
  '',
  '  // sky above, warm bounce from the ground below: separates up-facing from',
  '  // down-facing surfaces even where no sun reaches, which flat ambient cannot.',
  '  float up = n.z * 0.5 + 0.5;',
  '  vec3 hemi = mix(vec3(0.26, 0.29, 0.35), vec3(0.92, 0.94, 1.00), up);',
  '',
  '  // curvature from the normal map: darkens creases between solids. Four taps.',
  '  vec3 nx0 = texture2D(uNrm, vUV - vec2(uTexel, 0.0)).rgb * 2.0 - 1.0;',
  '  vec3 nx1 = texture2D(uNrm, vUV + vec2(uTexel, 0.0)).rgb * 2.0 - 1.0;',
  '  vec3 ny0 = texture2D(uNrm, vUV - vec2(0.0, uTexel)).rgb * 2.0 - 1.0;',
  '  vec3 ny1 = texture2D(uNrm, vUV + vec2(0.0, uTexel)).rgb * 2.0 - 1.0;',
  '  float curv = (dot(normalize(nx0 + nx1 + ny0 + ny1), n) - 1.0);',
  '  float ao = mix(clamp(1.0 + curv * 2.4, 0.35, 1.0), 1.0, uVertexNormal);',
  '',
  '  vec3 lit = base * (hemi * ao + vec3(0.85) * lam);',
  '  vec3 c = mix(base, lit, uShade);',
  '',
  '  // a faint cool tint on tall structure, so relief survives a grey texture',
  '  c = mix(c, c * vec3(1.04, 1.02, 0.96), clamp(vHeight / max(uRelief, 1.0), 0.0, 1.0) * 0.5);',
  '',
  '  // aerial perspective: the strongest distance cue the eye has outdoors',
  '  float f = 1.0 - exp(-vDist * uFog);',
  '  c = mix(c, vec3(0.62, 0.70, 0.82), clamp(f, 0.0, 0.85));',
  '',
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
  layer: 'texture', exagg: 1, shade: true, wire: false, fog: true,
  fly: null,
  cam: { az: -0.6, el: 0.85, dist: 900, target: [0, 0, 0] },
  zBase: 0, centre: [0, 0], uintIndex: false, needsDraw: true, fps: 0,
  structural: null, showStructural: false
};

function sliderToExagg(v) {           // 0..300 -> 1..200, log-ish and smooth
  return Math.round(Math.pow(10, (v / 300) * Math.log10(200)) * 100) / 100;
}
function exaggToSlider(e) {
  return Math.round(Math.log10(Math.max(1, e)) / Math.log10(200) * 300);
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
  var canvas = state.gl && state.gl.canvas;
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
  var structural = state.showStructural && state.structural;
  if (!gl || (!state.tiles.length && !structural)) return;
  var canvas = gl.canvas;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.prog);

  var mvp = viewProj(canvas.width / Math.max(1, canvas.height));
  var P = state.prog;
  var span = state.manifest ? Math.max.apply(null, state.manifest.grid.extent_m) : 1000;
  var relief = 30;
  if (state.manifest && state.manifest.layers && state.manifest.layers.ndsm) {
    var st = state.manifest.layers.ndsm.stats || {};
    if (isFinite(st.max) && st.max > 0) relief = st.max;
  }
  gl.uniformMatrix4fv(gl.getUniformLocation(P, 'uMVP'), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(P, 'uEye'), cameraEye());
  gl.uniform1f(gl.getUniformLocation(P, 'uExagg'), state.exagg);
  gl.uniform1f(gl.getUniformLocation(P, 'uZBase'), state.zBase);
  gl.uniform1f(gl.getUniformLocation(P, 'uShade'), state.shade ? 1 : 0);
  // Tuned to the scene so haze reads the same on a 500 m tile and a 5 km one.
  gl.uniform1f(gl.getUniformLocation(P, 'uFog'), state.fog ? 1.0 / (span * 2.2) : 0.0);
  gl.uniform1f(gl.getUniformLocation(P, 'uTexel'), 1.0 / 512.0);
  gl.uniform1f(gl.getUniformLocation(P, 'uRelief'), relief * state.exagg);
  gl.uniform3fv(gl.getUniformLocation(P, 'uLight'), state.light || [0.4, 0.5, 0.75]);

  var aPos = gl.getAttribLocation(P, 'aPos');
  var aH = gl.getAttribLocation(P, 'aH');
  var aUV = gl.getAttribLocation(P, 'aUV');
  var aN = gl.getAttribLocation(P, 'aN');
  var uWire = gl.getUniformLocation(P, 'uWire');
  var uVN = gl.getUniformLocation(P, 'uVertexNormal');
  var tris = 0;

  if (structural) {
    var m = state.structural;
    gl.uniform1f(uVN, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.bufPos);
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.bufH);
    gl.enableVertexAttribArray(aH); gl.vertexAttribPointer(aH, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.bufUV);
    gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
    if (aN >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, m.bufN);
      gl.enableVertexAttribArray(aN); gl.vertexAttribPointer(aN, 3, gl.FLOAT, false, 0, 0);
    }
    if (m.tex) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, m.tex);
      gl.uniform1i(gl.getUniformLocation(P, 'uTex'), 0);
    }
    gl.uniform1f(uWire, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.bufIdx);
    gl.drawElements(gl.TRIANGLES, m.nIdx, gl.UNSIGNED_INT, 0);
    tris += m.nIdx / 3;

    // Painted facades on top. Drawn a second time over the same triangles
    // rather than carved out of the first pass: an index buffer per refined
    // building would be the alternative, and at eight buildings the depth test
    // does the same job for the cost of eight small draws. GL_EQUAL keeps them
    // from z-fighting the pass underneath.
    if (m.painted && m.painted.length && m.groupTex) {
      gl.depthFunc(gl.LEQUAL);
      for (var gi = 0; gi < m.painted.length; gi++) {
        var grp = m.painted[gi];
        var img = m.groupTex[grp.tex];
        if (!img) continue;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, img);
        gl.uniform1i(gl.getUniformLocation(P, 'uTex'), 0);
        gl.drawElements(gl.TRIANGLES, grp.count * 3, gl.UNSIGNED_INT,
                        grp.first * 3 * 4);
      }
      gl.depthFunc(gl.LESS);
      if (m.tex) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, m.tex);
      }
    }
    if (state.wire && m.bufWire) {
      gl.uniform1f(uWire, 0.85);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.bufWire);
      gl.drawElements(gl.LINES, m.nWire, gl.UNSIGNED_INT, 0);
    }
    if (aN >= 0) gl.disableVertexAttribArray(aN);
    state.tris = Math.round(tris);
    // The tile path reports at the end of draw(); this path returns before it,
    // so it has to report for itself or the triangle count stays stale and the
    // reader is told the height field is still on screen.
    if (state.onStats) state.onStats(state.tris, state.lodIndex);
    state.needsDraw = false;
    return;
  }
  gl.uniform1f(uVN, 0);

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

  state.tris = Math.round(tris);
  if (state.onStats) state.onStats(state.tris, state.lodIndex);
}

// ── flythrough ──────────────────────────────────────────────────────────────
// An orbit control answers "what is the shape of this surface". It does not
// answer "how tall is that, standing next to it", which is the question a
// reconstruction exists to answer. So the tour descends from a survey view to
// eye level, crosses the scene low and oblique where parallax between near and
// far structure is strongest, and rises again.
//
// The camera height is clamped against the height field on every frame rather
// than baked into the keyframes, so the path cannot pass through a building on
// a scene it was not designed for - including scenes with far more relief than
// this one, which is the direction the calibration is meant to move.

function easeInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** Surface elevation under (x, y), or null outside the grid.
 *  Delegates to the same sampler the cursor readout uses, so the tour and the
 *  numbers on screen can never disagree about where the ground is. */
function surfaceAt(x, y) {
  if (!state.manifest || !state.tiles.length) return null;
  var hit = sample(x, y);
  return hit && isFinite(hit.elevation) ? hit.elevation : null;
}

function buildTour() {
  var m = state.manifest;
  if (!m) return null;
  var ex = m.grid.extent_m;
  var span = Math.max(ex[0], ex[1]);
  var r = span * 0.30;
  return {
    t0: performance.now(),
    duration: 34000,
    keys: [
      // survey: the whole extent, high and steep
      { az: -0.60, el: 0.95, dist: span * 1.25, tx: 0, ty: 0 },
      // descend and tilt toward the horizon
      { az: -0.20, el: 0.42, dist: span * 0.62, tx: -r * 0.5, ty: r * 0.4 },
      // low oblique pass: this is where relief reads
      { az: 0.55, el: 0.17, dist: span * 0.30, tx: r * 0.5, ty: r * 0.2 },
      // across the other diagonal, still low
      { az: 1.70, el: 0.15, dist: span * 0.26, tx: r * 0.2, ty: -r * 0.6 },
      { az: 2.90, el: 0.28, dist: span * 0.42, tx: -r * 0.4, ty: -r * 0.3 },
      // pull back out to where it started, so it loops cleanly
      { az: 3.90, el: 0.70, dist: span * 0.95, tx: 0, ty: 0 },
      { az: -0.60 + Math.PI * 2, el: 0.95, dist: span * 1.25, tx: 0, ty: 0 }
    ]
  };
}

function tourStep(now) {
  var f = state.fly;
  if (!f) return false;
  var u = (now - f.t0) / f.duration;
  if (u >= 1) { stopTour('finished'); return false; }

  var keys = f.keys;
  var seg = u * (keys.length - 1);
  var i = Math.min(keys.length - 2, Math.floor(seg));
  var lt = easeInOut(seg - i);
  var a = keys[i], b = keys[i + 1];
  var mix = function (p, q) { return p + (q - p) * lt; };

  state.cam.az = mix(a.az, b.az);
  state.cam.el = mix(a.el, b.el);
  state.cam.dist = mix(a.dist, b.dist);
  state.cam.target[0] = mix(a.tx, b.tx);
  state.cam.target[1] = mix(a.ty, b.ty);

  // Keep the target on the surface, and the eye above it. Without this the
  // low passes clip through roofs the moment a scene has real height.
  var gh = surfaceAt(state.cam.target[0], state.cam.target[1]);
  if (gh !== null) state.cam.target[2] = (gh - state.zBase) * state.exagg;

  var eye = cameraEye();
  var eh = surfaceAt(eye[0], eye[1]);
  if (eh !== null) {
    var floor = (eh - state.zBase) * state.exagg + 25;
    if (eye[2] < floor) {
      // raise the elevation angle just enough to clear the surface
      var need = (floor - state.cam.target[2]) / Math.max(state.cam.dist, 1);
      state.cam.el = Math.max(state.cam.el, Math.min(1.5, Math.asin(Math.min(1, need))));
    }
  }
  state.needsDraw = true;
  return true;
}

function loop() {
  resize();
  if (state.fly) tourStep(performance.now());
  if (state.needsDraw) { draw(); state.needsDraw = false; }
  requestAnimationFrame(loop);
}

/** Load one LOD: decode every tile, build buffers, upload textures. */
function loadLod(index) {
  var m = state.manifest, gl = state.gl;
  var lod = m.lods[index];
  state.lodIndex = index;
  if (state.onStats) state.onStats(state.tris || 0, index);

  var gsd = lod.gsd_m;
  state.centre = [lod.width * gsd / 2, lod.height * gsd / 2];

  return Promise.all(lod.tiles.map(function (t) {
    // Where tiles come from is the caller's business, not the renderer's:
    // a job serves them from api/jobs/<id>/tiles/ and a static build from
    // data/. `load()` records it, so there is nothing here to guess.
    var base = state.base || 'data/';
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
/* The selected layer as a single scene-wide texture.
 *
 * Tiles carry one texture each, which is fine for a grid of tiles and no use to
 * a mesh whose UVs span the whole scene in one piece: the structural mesh needs
 * one image. So the tiles' own colourisation is composited into one canvas at
 * the LOD's resolution, which reuses the ramps and ranges exactly rather than
 * computing a second, subtly different version of the same picture.
 *
 * Without this the structural mesh showed the orthophoto whatever layer was
 * selected - the buttons moved and the model did not.
 */
function compositeLayer(key) {
  var m = state.manifest;
  if (!m || !state.tiles.length) return null;
  var lod = m.lods[state.lodIndex];
  var cv = document.createElement('canvas');
  cv.width = lod.width; cv.height = lod.height;
  var ctx = cv.getContext('2d');
  if (!ctx) return null;

  var spec = m.layers[key];
  var style = spec ? layerStyle(key, spec) : null;
  var drew = 0;
  state.tiles.forEach(function (t) {
    var x0 = t.spec.x0, y0 = t.spec.y0;
    if (key === 'texture') {
      if (t.images.texture) { ctx.drawImage(t.images.texture, x0, y0, t.w, t.h); drew++; }
      return;
    }
    var vals = key === 'dsm' ? t.heights : t.values[key];
    if (!vals || !style) {
      if (t.images.texture) { ctx.drawImage(t.images.texture, x0, y0, t.w, t.h); drew++; }
      return;
    }
    ctx.putImageData(colourize(vals, t.w, t.h, style.ramp, style.lo, style.hi), x0, y0);
    drew++;
  });
  return drew ? cv : null;
}

function refreshStructuralTexture() {
  var gl = state.gl, m = state.structural;
  if (!gl || !m) return;
  var cv = compositeLayer(state.layer || 'texture');
  if (!cv) return;
  if (m.tex) gl.deleteTexture(m.tex);
  m.tex = texture(gl, cv);
}

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
  // The structural mesh is one piece of geometry with one texture, so it needs
  // the whole scene composited rather than a tile's worth.
  refreshStructuralTexture();
  // The legend is a component now; it reads the same layer spec from the
  // manifest, so there is nothing for the renderer to push.
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


/* ── canvas interaction ──────────────────────────────────────────────────────
 * Orbit, pan, zoom and the hover readout. These belong to the renderer, not to
 * a component: they read and write the camera directly at pointer rate, and
 * routing sixty of those a second through React state would be both slower and
 * wrong. React learns the result through `onPick`.
 */
function bindCanvas(canvas) {
  var drag = null;

  var down = function (e) {
    if (state.fly) stopTour('user');
    drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
    try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* not captured */ }
  };
  var up = function (e) {
    drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ }
  };
  var move = function (e) {
    if (drag) {
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        var sc = state.cam.dist / 900;
        var az = state.cam.az;
        state.cam.target[0] -= (dx * Math.cos(az) + dy * Math.sin(az)) * sc;
        state.cam.target[1] += (dx * Math.sin(az) - dy * Math.cos(az)) * sc;
      } else {
        state.cam.az -= dx * 0.006;
        state.cam.el = Math.max(0.04, Math.min(1.5, state.cam.el + dy * 0.006));
      }
      state.needsDraw = true;
    } else if (state.onPick) {
      state.onPick(pick(e.clientX, e.clientY));
    }
  };
  var leave = function () { if (state.onPick) state.onPick(null); };
  var menu = function (e) { e.preventDefault(); };
  var wheel = function (e) {
    if (state.fly) stopTour('user');
    e.preventDefault();
    state.cam.dist = Math.max(20, Math.min(60000,
      state.cam.dist * Math.pow(1.0015, e.deltaY)));
    state.needsDraw = true;
  };

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerleave', leave);
  canvas.addEventListener('contextmenu', menu);
  canvas.addEventListener('wheel', wheel, { passive: false });

  return function unbind() {
    canvas.removeEventListener('pointerdown', down);
    canvas.removeEventListener('pointerup', up);
    canvas.removeEventListener('pointermove', move);
    canvas.removeEventListener('pointerleave', leave);
    canvas.removeEventListener('contextmenu', menu);
    canvas.removeEventListener('wheel', wheel);
  };
}

/* ── the handle React holds ────────────────────────────────────────────────
 * Every method is safe to call before a tileset has loaded; the viewer simply
 * has nothing to draw yet. That matters because React will mount the canvas
 * and set options in whatever order the user clicks, not in load order.
 */
/* The structural mesh: terrain with footprint holes, and one solid per
 * building with real vertical facades. The tiles above are a height field and
 * cannot express a wall - one z per (x, y) is the definition - so this is a
 * second, separate piece of geometry drawn in place of them.
 *
 * `TKM1` is written by traksha/mesh/webmesh.py: a small header then five typed
 * arrays that go straight into GL buffers. No parsing loop, because a
 * quarter-million triangles of JSON or OBJ text would cost more to read than
 * to draw.
 */
function parseTKM1(buf) {
  var dv = new DataView(buf);
  var magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                                  dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'TKM1') throw new Error('not a TRAKSHA mesh: ' + magic);
  var version = dv.getUint32(4, true);
  if (version !== 2) {
    throw new Error('structural.bin is format v' + version
      + ', this viewer reads v2 - rebuild it with `traksha mesh`');
  }
  var nv = dv.getUint32(8, true), ni = dv.getUint32(12, true);
  var ng = dv.getUint32(16, true), ntex = dv.getUint32(20, true);
  var o = 44;                                  // 4 + 5*4 + 5*4 bytes
  var pos = new Float32Array(buf, o, nv * 2); o += nv * 8;
  var hgt = new Float32Array(buf, o, nv); o += nv * 4;
  var uv = new Float32Array(buf, o, nv * 2); o += nv * 8;
  var nrm = new Float32Array(buf, o, nv * 3); o += nv * 12;
  var idx = new Uint32Array(buf, o, ni); o += ni * 4;
  var groups = [];
  for (var i = 0; i < ng; i++) {
    groups.push({ first: dv.getUint32(o + i * 20, true),
                  count: dv.getUint32(o + i * 20 + 4, true),
                  kind: dv.getUint32(o + i * 20 + 8, true),
                  id: dv.getUint32(o + i * 20 + 12, true),
                  tex: dv.getInt32(o + i * 20 + 16, true) });
  }
  o += ng * 20;
  // The texture table: a name per painted facade, relative to structural.bin.
  var texNames = [], dec = new TextDecoder();
  for (var t = 0; t < ntex; t++) {
    var len = dv.getUint32(o, true); o += 4;
    texNames.push(dec.decode(new Uint8Array(buf, o, len))); o += len;
  }
  return { pos: pos, heights: hgt, uv: uv, normals: nrm, indices: idx,
           groups: groups, textures: texNames, version: version,
           zMin: dv.getFloat32(36, true), zMax: dv.getFloat32(40, true) };
}

function disposeStructural() {
  var gl = state.gl, m = state.structural;
  if (!gl || !m) return;
  (m.groupTex || []).forEach(function (t) { if (t) gl.deleteTexture(t); });
  [m.bufPos, m.bufH, m.bufUV, m.bufN, m.bufIdx, m.bufWire].forEach(function (b) {
    if (b) gl.deleteBuffer(b);
  });
  if (m.tex) gl.deleteTexture(m.tex);
  state.structural = null;
}

async function loadStructural(base, manifest) {
  var gl = state.gl;
  var spec = manifest && manifest.mesh && manifest.mesh.structural
    && manifest.mesh.structural.web;
  if (!gl || !spec) return null;
  if (state.structural) return state.structural;

  // The mesh has far more than 65 535 vertices, so 32-bit indices are not
  // optional. WebGL 1 without OES_element_index_uint cannot draw it at all,
  // and silently drawing a wrapped-around subset would be worse than saying so.
  if (!state.uintIndex) {
    throw new Error('this browser has no 32-bit mesh indices '
      + '(OES_element_index_uint); the structural mesh cannot be drawn');
  }

  var r = await fetch(base + 'structural.bin', { cache: 'no-cache' });
  if (!r.ok) throw new Error('structural.bin: HTTP ' + r.status);
  var m = parseTKM1(await r.arrayBuffer());

  // Positions arrive as absolute metres from the south-west corner; the scene
  // is drawn about its own centre, so shift once here rather than per frame.
  var xy = new Float32Array(m.pos.length);
  for (var i = 0; i < m.pos.length; i += 2) {
    xy[i] = m.pos[i] - state.centre[0];
    xy[i + 1] = m.pos[i + 1] - state.centre[1];
  }

  var buf = function (data, target) {
    var b = gl.createBuffer();
    var t = target || gl.ARRAY_BUFFER;
    gl.bindBuffer(t, b);
    gl.bufferData(t, data, gl.STATIC_DRAW);
    return b;
  };
  var out = {
    bufPos: buf(xy), bufH: buf(m.heights), bufUV: buf(m.uv),
    bufN: buf(m.normals), bufIdx: buf(m.indices, gl.ELEMENT_ARRAY_BUFFER),
    nIdx: m.indices.length, groups: m.groups,
    triangles: m.indices.length / 3, tex: null,
  };

  // Wireframe needs its own index buffer: the tiles build one from their grid,
  // and this mesh has no grid. Three edges per triangle, drawn as lines -
  // duplicates along shared edges cost a second line over the same pixels and
  // save building a unique-edge set for a quarter of a million faces.
  var wire = new Uint32Array(m.indices.length * 2);
  for (var t = 0, o = 0; t < m.indices.length; t += 3) {
    var a0 = m.indices[t], b0 = m.indices[t + 1], c0 = m.indices[t + 2];
    wire[o++] = a0; wire[o++] = b0;
    wire[o++] = b0; wire[o++] = c0;
    wire[o++] = c0; wire[o++] = a0;
  }
  out.bufWire = buf(wire, gl.ELEMENT_ARRAY_BUFFER);
  out.nWire = wire.length;

  // Painted facades: one image per refined building, drawn in its own pass.
  // Everything else shares the scene texture, so a scene with eight refined
  // buildings costs nine draw calls rather than one per group.
  out.groupTex = [];
  for (var t = 0; t < (m.textures || []).length; t++) {
    try {
      out.groupTex.push(texture(gl, await loadImage(base + m.textures[t])));
    } catch (e) {
      out.groupTex.push(null);                 // missing image: fall back
    }
  }
  out.painted = m.groups.filter(function (g) { return g.tex >= 0; });

  state.structural = out;
  // One texture for the whole mesh, composited from the tiles so that the
  // layer buttons apply here exactly as they do to the height field.
  refreshStructuralTexture();
  return out;
}

export function createViewer(canvas, opts) {
  opts = opts || {};
  if (!initGL(canvas)) return null;
  state.onPick = opts.onPick || null;
  state.onStats = opts.onStats || null;
  state.onTour = opts.onTour || null;
  var unbind = bindCanvas(canvas);
  requestAnimationFrame(loop);

  return {
    async load(manifest, base) {
      state.manifest = manifest;
      state.base = base;
      const ex = manifest.grid.extent_m;
      state.centre = [ex[0] / 2, ex[1] / 2];
      state.light = lightFromSun(manifest);
      await loadLod(pickLodIndex(manifest));
      frameScene();
      return { lods: manifest.lods.length, lod: state.lodIndex };
    },
    setLod(i) { return loadLod(i); },
    /* Switch between the height field and the structural rebuild. The mesh is
     * fetched the first time it is asked for - it is several megabytes, and a
     * reader who never opens it should never pay for it. */
    async setStructural(on) {
      if (on && !state.structural) {
        await loadStructural(state.base, state.manifest);
      }
      state.showStructural = !!on && !!state.structural;
      state.needsDraw = true;
      return state.showStructural;
    },
    hasStructural() {
      var m = state.manifest;
      return !!(m && m.mesh && m.mesh.structural && m.mesh.structural.web)
        && !!state.uintIndex;
    },
    setLayer(k) { state.layer = k; applyLayer(k); state.needsDraw = true; },
    setExagg(v) { state.exagg = v; state.needsDraw = true; },
    setShade(v) { state.shade = !!v; state.needsDraw = true; },
    setWire(v) { state.wire = !!v; state.needsDraw = true; },
    setFog(v) { state.fog = !!v; state.needsDraw = true; },
    fly() { startTour(); },
    stopFly() { stopTour('user'); },
    flying() { return !!state.fly; },
    frame() { frameScene(); },
    pick(x, y) { return pick(x, y); },
    lodIndex() { return state.lodIndex; },
    stats() {
      const lod = state.manifest && state.manifest.lods[state.lodIndex];
      return { lod: lod ? lod.lod : null,
               width: lod ? lod.width : 0, height: lod ? lod.height : 0,
               triangles: state.tris || 0, fps: state.fps || 0 };
    },
    dispose() {
      state.disposed = true; unbind(); disposeTiles(); disposeStructural();
    },
  };
}

/* The tour lives here because it drives the camera, but the button that starts
 * it is a React component - so these are exported rather than bound to a DOM id. */
function startTour() {
  state.fly = buildTour();
  if (state.onTour) state.onTour(true);
}

function stopTour(why) {
  state.fly = null;
  if (state.onTour) state.onTour(false, why);
}

/* Also exported: the pure arithmetic, so scripts/check_app.mjs can hold it
 * against traksha/mesh/encode.py. These two are independent implementations of
 * one packing; if they drift the viewer draws a confidently wrong surface and
 * nothing else in the suite would notice. */
export {
  pickLodIndex, layerStyle, cssRamp, fmt,
  decodeTerrainRGBA, decodeLinearRGBA, lut, sliderToExagg, exaggToSlider, MAX_CODE,
  // measured by scripts/bench_viewer.mjs: the CPU work before the GPU is involved
  tileGeometry, gridIndices, colourize,
};
