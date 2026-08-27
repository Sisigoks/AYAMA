// Headless render check for the results site.
//
//   npm install jsdom      (once, anywhere on the path)
//   node scripts/check_site.js
//
// Loads site/index.html, runs site/app.js against the real results/study.json
// and asserts every panel actually rendered. This exists because a results file
// containing a bare NaN token - which Python writes by default - parses fine in
// Python and makes JSON.parse throw, deploying a completely blank page. Nothing
// short of executing the page catches that.
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'site/index.html'), 'utf8');
const app  = fs.readFileSync(path.join(ROOT, 'site/app.js'), 'utf8');
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

  if (errors.length) { console.log('\n  JS ERRORS:'); errors.forEach(e => console.log('   ' + e)); }
  console.log(`\n  ${bad === 0 && errors.length === 0 ? 'PAGE RENDERS CLEAN' : 'PROBLEMS: ' + (bad + errors.length)}`);
}, 400);
