// Headless render check for the Phase 4 viewer.
//
//   npm install jsdom      (once, anywhere on the path)
//   node scripts/check_app.js [tileset-dir]
//
// jsdom has no WebGL, and that is the point rather than a limitation: this
// asserts the viewer degrades to a readable page instead of a blank canvas when
// GL is missing, and it exercises the half of app.js that must be correct
// before a single triangle is drawn - manifest parsing, the panels, the decode
// arithmetic that turns 8-bit channels back into metres.
//
// The decode check matters most. `web/app.js` and `traksha/mesh/encode.py` are two
// independent implementations of the same packing; if they ever disagree the
// viewer draws a confidently wrong surface, and nothing else in the suite would
// notice.
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');

let JSDOM, VirtualConsole;
try {
  ({ JSDOM, VirtualConsole } = require('jsdom'));
} catch (e) {
  console.log('skip: jsdom is not installed (npm install jsdom)');
  process.exit(0);
}

// Default to the committed demo tileset, so this runs on a fresh clone with
// nothing built. Pass a directory to check a tileset you produced yourself.
const tilesetDir = process.argv[2] || path.join(ROOT, 'web/data');
const manifestPath = path.join(tilesetDir, 'tileset.json');
if (!fs.existsSync(manifestPath)) {
  console.log(`skip: no tileset at ${manifestPath}`);
  console.log('      build one: python -m traksha.cli mesh <run> --out web/data --no-mesh --bits 12');
  process.exit(0);
}

const html = fs.readFileSync(path.join(ROOT, 'web/index.html'), 'utf8');
const app = fs.readFileSync(path.join(ROOT, 'web/app.js'), 'utf8');
const manifestText = fs.readFileSync(manifestPath, 'utf8');
const manifest = JSON.parse(manifestText);   // throws on NaN, which is the point

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => {
  if (/not implemented/i.test(e.message)) return;      // canvas, as expected
  errors.push('jsdomError: ' + e.message);
});
vc.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));

const dom = new JSDOM(html, {
  runScripts: 'outside-only', virtualConsole: vc, url: 'http://localhost:8020/',
});
const w = dom.window;

w.TRAKSHA_NO_AUTOBOOT = true;
w.HTMLCanvasElement.prototype.getContext = function () { return null; };  // no WebGL here
w.requestAnimationFrame = () => 0;
w.fetch = (url) => Promise.resolve({
  ok: url.includes('tileset.json'),
  status: url.includes('tileset.json') ? 200 : 404,
  json: () => Promise.resolve(JSON.parse(manifestText)),
});

try { w.eval(app); } catch (e) { errors.push('eval threw: ' + e.stack); }

const U = w.TRAKSHA;
function check(name, cond, detail) {
  if (cond) { console.log('  ok   ' + name); } else { errors.push(name + (detail ? ' — ' + detail : '')); }
}

if (!U) {
  errors.push('app.js did not expose window.TRAKSHA');
} else {
  // ── decode arithmetic must match traksha/mesh/encode.py exactly ──────────────
  // Tolerances are float32's, not float64's: the decoders write into a
  // Float32Array on purpose (it is what goes straight into a GL buffer), so
  // ~1e-7 relative is the floor. A wrong formula is off by orders of magnitude,
  // which this still catches; a tighter bound would only fail on storage.
  const f32 = (expected) => Math.max(1e-4, Math.abs(expected) * 1e-6);

  const code = 123456;                       // an arbitrary 24-bit value
  const rgba = new Uint8ClampedArray([(code >> 16) & 255, (code >> 8) & 255, code & 255, 255]);
  const h = U.decodeTerrainRGBA(rgba)[0], hWant = -10000 + code * 0.1;
  check('terrain-rgb decode', Math.abs(h - hWant) < f32(hWant), `got ${h}, want ${hWant}`);

  const lin = U.decodeLinearRGBA(rgba, -5, 5)[0], linWant = -5 + code / U.MAX_CODE * 10;
  check('linear decode', Math.abs(lin - linWant) < f32(linWant), `got ${lin}, want ${linWant}`);

  const full = U.decodeLinearRGBA(new Uint8ClampedArray([255, 255, 255, 255]), 0, 42)[0];
  check('linear decode saturates at vmax', Math.abs(full - 42) < f32(42), `got ${full}`);

  const zero = U.decodeLinearRGBA(new Uint8ClampedArray([0, 0, 0, 255]), 7, 9)[0];
  check('linear decode floors at vmin', Math.abs(zero - 7) < f32(7), `got ${zero}`);

  check('lut has 256 rgb entries', U.lut('viridis').length === 768);
  check('exaggeration slider roundtrips',
    Math.abs(U.sliderToExagg(U.exaggToSlider(25)) - 25) < 0.5);

  // ── panels render from the real manifest ──────────────────────────────────
  let start;
  try { start = U.renderPanels(manifest); }
  catch (e) { errors.push('renderPanels threw: ' + e.stack); }

  const doc = w.document;
  const txt = id => (doc.getElementById(id) || {}).textContent || '';
  check('layer buttons rendered', doc.querySelectorAll('#layer-buttons button').length >= 2);
  check('lod options rendered',
    doc.querySelectorAll('#lod option').length === manifest.lods.length);
  check('stats table filled', doc.querySelectorAll('#stats tr').length >= 4);
  check('provenance table filled', doc.querySelectorAll('#prov tr').length >= 3);
  check('downloads listed', doc.querySelectorAll('#downloads a').length >= 1);
  check('default layer chosen', !!start, `got ${start}`);

  const notes = doc.querySelectorAll('#notes .note');
  check('notes rendered', notes.length === (manifest.notes || []).length,
    `${notes.length} vs ${(manifest.notes || []).length}`);

  // The critical note is the one that must never be quietly dropped: it is what
  // stops a flattened city being presented as a finished 3D deliverable.
  const critical = (manifest.notes || []).filter(n => n.level === 'critical');
  if (critical.length) {
    check('critical note is visible',
      doc.querySelectorAll('#notes .note.critical').length === critical.length);
  }

  if (manifest.metrics && manifest.metrics.mae_m !== undefined) {
    check('metrics table shows MAE', /MAE/.test(txt('metrics')));
  }

  // ── boot without WebGL must explain itself, not blank out ─────────────────
  const done = U.boot().then(() => {
    check('gl-error shown when WebGL is absent', doc.getElementById('gl-error').hidden === false);
    check('panel still populated after boot',
      doc.querySelectorAll('#stats tr').length >= 4);
    finish();
  }).catch(e => { errors.push('boot rejected: ' + e.message); finish(); });
}

function finish() {
  if (errors.length) {
    console.error('\nFAILED:');
    errors.forEach(e => console.error('  - ' + e));
    process.exit(1);
  }
  console.log('\nviewer check passed');
  process.exit(0);
}

if (!U) finish();
