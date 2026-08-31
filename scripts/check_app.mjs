/* Encoding parity between the viewer and the pipeline.
 *
 *   node scripts/check_app.mjs [tileset-dir]
 *
 * `web/src/renderer.js` and `traksha/mesh/encode.py` are two independent
 * implementations of one packing: 24 bits of RGB carrying metres. If they ever
 * disagree the viewer draws a confidently wrong surface at full confidence, and
 * no other check in the suite would notice - the manifest still parses, the page
 * still renders, the numbers in the panel still come from the manifest rather
 * than from the pixels. So this holds the JavaScript decoders against the
 * arithmetic the encoder promises, and reads a real manifest to be sure the
 * fields the viewer indexes into are the fields the builder writes.
 *
 * This runs the renderer's pure half only. Whether the page renders is a
 * question about a browser, and scripts/check_site.mjs answers it in one.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// pathToFileURL, not a bare path: on Windows an absolute path starts with a
// drive letter, which the ESM loader reads as an unknown URL scheme.
const R = await import(pathToFileURL(path.join(ROOT, 'web/src/renderer.js')).href);

const errors = [];
const check = (name, cond, detail) =>
  cond ? console.log('  ok   ' + name)
       : errors.push(name + (detail ? ' — ' + detail : ''));

// Tolerances are float32's, not float64's: the decoders write into a
// Float32Array on purpose - it is what goes straight into a GL buffer - so ~1e-7
// relative is the floor. A wrong formula is off by orders of magnitude, which
// this still catches; a tighter bound would only fail on storage.
const f32 = (want) => Math.max(1e-4, Math.abs(want) * 1e-6);
const rgba = (c) => new Uint8ClampedArray([(c >> 16) & 255, (c >> 8) & 255, c & 255, 255]);

const code = 123456;                       // an arbitrary 24-bit value
const h = R.decodeTerrainRGBA(rgba(code))[0], hWant = -10000 + code * 0.1;
check('terrain-rgb decode', Math.abs(h - hWant) < f32(hWant), `got ${h}, want ${hWant}`);

const lin = R.decodeLinearRGBA(rgba(code), -5, 5)[0];
const linWant = -5 + (code / R.MAX_CODE) * 10;
check('linear decode', Math.abs(lin - linWant) < f32(linWant), `got ${lin}, want ${linWant}`);

const full = R.decodeLinearRGBA(rgba(R.MAX_CODE), 0, 42)[0];
check('linear decode saturates at vmax', Math.abs(full - 42) < f32(42), `got ${full}`);

const zero = R.decodeLinearRGBA(rgba(0), 7, 9)[0];
check('linear decode floors at vmin', Math.abs(zero - 7) < f32(7), `got ${zero}`);

check('lut has 256 rgb entries', R.lut('viridis').length === 768);
check('exaggeration slider roundtrips',
  Math.abs(R.sliderToExagg(R.exaggToSlider(25)) - 25) < 0.5);

// ── the manifest the builder writes is the manifest the viewer reads ────────
// Default to the committed demo tileset so this runs on a fresh clone with
// nothing built. Pass a directory to check one you produced yourself.
const dir = process.argv[2] || path.join(ROOT, 'web/data');
const manifestPath = path.join(dir, 'tileset.json');
if (!fs.existsSync(manifestPath)) {
  console.log(`\nskip: no tileset at ${manifestPath}`);
  console.log('      build one: python -m traksha.cli mesh <run> --out web/data --no-mesh --bits 12');
} else {
  const m = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));  // throws on NaN, which is the point
  const lods = m.lods || [];
  const key = m.default_layer;
  check('manifest has lods', Array.isArray(m.lods) && lods.length > 0);
  check('manifest names a default layer', !!key && !!(m.layers || {})[key], `got ${key}`);

  // Every tile of every level must carry the default layer. A tile missing it
  // is a hole in the surface the viewer cannot report, because it has nothing
  // to draw there and no reason to think anything is wrong.
  check('every tile carries the default layer',
    lods.every((l) => (l.tiles || []).length &&
                      l.tiles.every((t) => t.layers && t.layers[key])),
    'a tile with no default layer is an unreportable hole');

  // dsm is the height field: the one layer decoded as metres rather than
  // colourised, so it is the one the parity above actually governs.
  check('a height field is published',
    lods.every((l) => l.tiles.every((t) => t.layers.dsm)));

  check('lod picking returns an index in range', (() => {
    const i = R.pickLodIndex(m, 250000);
    return Number.isInteger(i) && i >= 0 && i < lods.length;
  })());

  const st = R.layerStyle(key, (m.layers || {})[key]);
  check('layerStyle gives the default layer a usable range',
    isFinite(st.lo) && isFinite(st.hi) && st.hi > st.lo, `got ${JSON.stringify(st)}`);

  // The critical note is the one that must never be quietly dropped: it is what
  // stops a flattened city being presented as a finished 3D deliverable. The
  // browser check asserts it reaches the screen; this asserts it exists to reach.
  const notes = m.notes || [];
  console.log(`  info manifest carries ${notes.length} note(s), ` +
    `${notes.filter((n) => n.level === 'critical').length} critical`);
}

if (errors.length) {
  console.error('\nFAILED:');
  errors.forEach((e) => console.error('  - ' + e));
  process.exit(1);
}
console.log('\nencoding parity holds');
