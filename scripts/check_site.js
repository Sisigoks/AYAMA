// Headless render check for the results site.
//
//   npm install jsdom      (once, anywhere on the path)
//   node scripts/check_site.js
//
// Loads web/results.html, runs web/results.js against the real results/study.json
// and asserts every panel actually rendered. This exists because a results file
// containing a bare NaN token - which Python writes by default - parses fine in
// Python and makes JSON.parse throw, deploying a completely blank page. Nothing
// short of executing the page catches that.
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'web/results.html'), 'utf8');
const app  = fs.readFileSync(path.join(ROOT, 'web/results.js'), 'utf8');
const study = fs.readFileSync(path.join(ROOT, 'results/study.json'), 'utf8');

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + e.message));
vc.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));

const dom = new JSDOM(html, {
  runScripts: 'outside-only', virtualConsole: vc,
  url: 'https://someuser.github.io/ayama/',
});
const w = dom.window;
w.fetch = (url) => Promise.resolve({
  ok: url.includes('study.json'), status: url.includes('study.json') ? 200 : 404,
  json: () => Promise.resolve(JSON.parse(study)),
});
w.navigator.clipboard = { writeText: () => Promise.resolve() };
w.Element.prototype.setPointerCapture = function () {};
w.Element.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, width: 800, height: 800, right: 800, bottom: 800 };
};

try { w.eval(app); } catch (e) { errors.push('eval threw: ' + e.stack); }

setTimeout(() => {
  const $ = s => w.document.querySelector(s);
  const txt = s => ($(s) ? $(s).textContent.replace(/\s+/g, ' ').trim() : '<<MISSING>>');
  const checks = [
    ['hero metrics',   () => w.document.querySelectorAll('#hero-metrics .metric').length, n => n >= 4],
    ['headline rows',  () => w.document.querySelectorAll('#headline-table tbody tr').length, n => n >= 8],
    ['floor verdict',   () => w.document.querySelectorAll('#floor-verdict .verdict').length, n => n === 1],
    ['class bars',     () => w.document.querySelectorAll('#class-chart .bar-row').length, n => n === 5],
    ['coverage gauge', () => w.document.querySelectorAll('#coverage-chart svg').length, n => n === 1],
    ['ablation rows',  () => w.document.querySelectorAll('#ablation-chart tbody tr').length, n => n >= 6],
    ['sun chart path', () => w.document.querySelectorAll('#sun-chart svg path.ln').length, n => n === 1],
    ['lambda dots',    () => w.document.querySelectorAll('#lambda-chart circle.dot').length, n => n >= 8],
    ['bench rows',     () => w.document.querySelectorAll('#bench-table tbody tr').length, n => n === 2],
    ['scene options',  () => w.document.querySelectorAll('#scene-select option').length, n => n === 3],
    ['scene facts',    () => w.document.querySelectorAll('#scene-facts .fact').length, n => n >= 10],
    ['layer options',  () => w.document.querySelectorAll('#left-layer option').length, n => n === 7],
  ];
  let bad = 0;
  for (const [name, get, ok] of checks) {
    let v; try { v = get(); } catch (e) { v = 'threw: ' + e.message; }
    const good = typeof v === 'number' && ok(v);
    if (!good) bad++;
    console.log(`  ${good ? 'ok  ' : 'FAIL'} ${name.padEnd(16)} ${v}`);
  }

  console.log('\n  rendered values:');
  console.log('   hero      ', txt('#hero-metrics .metric .v'), '|', txt('#hero-note').slice(0, 70));
  console.log('   img src   ', $('#img-left') ? $('#img-left').getAttribute('src') : '?');
  console.log('   wizard    ', txt('#wiz-tier'));
  console.log('   wiz cmd   ', txt('#wiz-cmd').slice(0, 90));
  console.log('   repo link ', $('#repo-link').href);
  console.log('   colab     ', $('#colab-btn').href);
  console.log('   footer    ', txt('#footer-meta').slice(0, 90));
  console.log('   lambda note', txt('#lambda-chart .panel-sub').slice(0, 110));

  
// ── the 3D section and the viewer it embeds ─────────────────────────────────
// The front page promises a live 3D view. That promise is only kept if the
// section is there, the iframe points somewhere real, and the assembled site
// actually contains the viewer and its tileset - three separate things, each of
// which has its own way of silently not shipping.
(function checkViewerEmbed() {
  const section = w.document.getElementById('three-d');
  if (!section) { errors.push('the 3D section is missing from index.html'); return; }

  const frame = w.document.getElementById('viewer-embed');
  if (!frame) { errors.push('the 3D section has no viewer iframe'); return; }
  const src = frame.getAttribute('src') || '';
  if (!src) { errors.push('the viewer iframe has no src'); return; }

  // The iframe src is relative to the assembled site root, not to the repo.
  const viewerDir = path.join(ROOT, 'web');
  const wants = [
    ['index.html', 'the viewer page'],
    ['app.js', 'the viewer script'],
    ['style.css', 'the viewer styles'],
    ['data/tileset.json', 'the committed demo tileset'],
  ];
  for (const [rel, what] of wants) {
    if (!fs.existsSync(path.join(viewerDir, rel))) {
      errors.push(`${what} is missing (web/${rel}), so ${src} would 404`);
    }
  }

  const manifestPath = path.join(viewerDir, 'data/tileset.json');
  if (fs.existsSync(manifestPath)) {
    let m;
    try { m = JSON.parse(fs.readFileSync(manifestPath, 'utf8')); }
    catch (e) { errors.push('web/data/tileset.json is not valid JSON: ' + e.message); }
    if (m) {
      let bytes = 0;
      (function walk(d) {
        for (const e of fs.readdirSync(d, { withFileTypes: true })) {
          const f = path.join(d, e.name);
          if (e.isDirectory()) walk(f); else bytes += fs.statSync(f).size;
        }
      })(path.join(viewerDir, 'data'));
      if (bytes > 6e6) {
        errors.push(`the demo tileset is ${(bytes / 1e6).toFixed(1)} MB; too heavy to embed`);
      }
      if (m.mesh) errors.push('the demo tileset ships the OBJ; it should not');
      // Every tile the manifest promises must exist, or the viewer draws holes.
      let missing = 0;
      for (const lod of m.lods || []) {
        for (const t of lod.tiles || []) {
          for (const rel of Object.values(t.layers || {})) {
            if (!fs.existsSync(path.join(viewerDir, 'data', rel))) missing++;
          }
        }
      }
      if (missing) errors.push(`${missing} tile(s) named in the manifest are missing`);
      console.log(`  viewer: ${(bytes / 1e6).toFixed(2)} MB, ${(m.lods || []).length} LODs, ` +
                  `${m.grid && m.grid.quantise_bits}-bit linear layers`);
    }
  }

  // The nav has to reach it, or nobody finds it.
  const nav = w.document.querySelector('nav');
  if (nav && !/#three-d/.test(nav.innerHTML)) {
    errors.push('the nav does not link to the 3D section');
  }
})();

if (errors.length) { console.log('\n  JS ERRORS:'); errors.forEach(e => console.log('   ' + e)); }
  console.log(`\n  ${bad === 0 && errors.length === 0 ? 'PAGE RENDERS CLEAN' : 'PROBLEMS: ' + (bad + errors.length)}`);
}, 400);
