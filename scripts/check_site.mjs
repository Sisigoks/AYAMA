/* Render check for the whole front end, in a real browser.
 *
 *   npm --prefix web install && npm --prefix web run build
 *   npx playwright install chromium
 *   node scripts/check_site.mjs
 *
 * A real browser, not jsdom, because the React conversion was landed against a
 * jsdom harness that reported every page as fine while the deployed site was
 * blank: an unclosed IIFE, an undefined `dataBase`, a dropped `bindControls`
 * that silently removed orbit, pan and zoom. jsdom evaluates scripts but does
 * not load ES modules, does not lay anything out and has no WebGL, so each of
 * those failures looked identical to success. Everything asserted below is
 * something jsdom could not have seen.
 *
 * It serves the build the way it deploys - one static directory, results/
 * copied in beside it - so a path that only works under the dev server's
 * middleware, or only from the Python service, fails here.
 */
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'web/dist');

if (!fs.existsSync(path.join(DIST, 'index.html'))) {
  console.log('skip: no build at web/dist');
  console.log('      cd web && npm install && npm run build');
  process.exit(0);
}
let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.log('skip: playwright is not installed (npm install -D playwright)'); process.exit(0); }

// ── serve the build exactly as it deploys ──────────────────────────────────
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml' };
const ROOTS = [DIST, path.join(ROOT, 'web')];          // web/ supplies data/
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
  const candidates = rel.startsWith('results/')
    ? [path.resolve(ROOT, rel)]
    : ROOTS.map((r) => path.resolve(r, rel));
  for (const f of candidates) {
    if (fs.existsSync(f) && fs.statSync(f).isFile()) {
      res.setHeader('Content-Type', TYPES[path.extname(f).toLowerCase()] || 'application/octet-stream');
      return fs.createReadStream(f).pipe(res);
    }
  }
  res.statusCode = 404;
  res.end('not found');
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const B = `http://127.0.0.1:${server.address().port}`;

const errors = [];
const check = (name, cond, detail) =>
  cond ? console.log('  ok   ' + name)
       : errors.push(name + (detail ? ' — ' + detail : ''));

// SwiftShader, so this needs no GPU and gives the same answer on a CI runner as
// on a laptop. Without it Chromium falls back to a null context and the viewer
// would be checked in exactly the state jsdom already failed to check it in.
const browser = await chromium.launch({ args: [
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const pageErrs = [];
page.on('pageerror', (e) => pageErrs.push('pageerror: ' + e.message));
// Chromium's console line for a failed request carries no URL, so the HTTP
// failures are tracked from the response side where the URL is knowable, and
// the matching console noise is dropped rather than reported twice without it.
page.on('console', (m) => {
  if (m.type() === 'error' && !/^Failed to load resource/.test(m.text())) {
    pageErrs.push('console: ' + m.text());
  }
});
page.on('requestfailed', (r) =>
  pageErrs.push('failed: ' + r.url().replace(B, '') + ' ' + (r.failure()?.errorText || '')));
page.on('response', (r) => {
  const u = r.url().replace(B, '');
  // /api/health 404s on the static site by design - that is the signal the app
  // uses to choose the fallback - so it is not an error to report.
  if (r.status() >= 400 && !/\/api\/health/.test(u)) pageErrs.push(`HTTP ${r.status()} ${u}`);
});

try {
  // ── with a service behind it: upload -> reconstruct -> 3D ───────────────
  // The reconstruction is Python and cannot move into the browser, so the app
  // asks /api/health whether a service is there. Stubbing it is the difference
  // between the two deployments this file has to keep working, and both are
  // real: `traksha serve` has a backend, GitHub Pages does not.
  await page.route('**/api/health', (route) => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({ ok: true, backbones: ['dav2-vits', 'dav2-vitb', 'dav2-vitl'] }),
  }));
  await page.goto(B + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  check('app mounts', (await page.locator('#root > *').count()) > 0,
    'an empty #root is what a module that threw on import looks like');
  check('upload form rendered', await page.locator('form.upload').count() === 1);
  check('drop zone rendered', await page.locator('.drop').count() === 1);
  check('backbones come from the service',
    (await page.locator('select').first().locator('option').count()) === 3,
    'the list must come from the service, not a hardcoded copy that can go stale');

  // ── with no service: the published static site ──────────────────────────
  // On Pages there is no backend at all. The page must not present a dead
  // upload form; it falls back to the tileset committed at web/data, so the
  // 3D is real on a site that can compute nothing.
  await page.unroute('**/api/health');
  await page.goto(B + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  check('falls back to the shipped tileset when no service answers',
    (await page.locator('canvas#gl').count()) === 1);
  check('WebGL drew rather than erroring', (await page.locator('.gl-error').count()) === 0);
  check('the fallback is a real surface',
    /tris/.test(await page.locator('.chip').first().innerText()),
    'the triangle count comes from the manifest that was actually loaded');
  const cv = await page.locator('canvas#gl').boundingBox();
  const tb = await page.locator('.topbar').boundingBox();
  check('canvas sits below the topbar', cv && tb && cv.y >= tb.y + tb.height - 1,
    'the viewer once drew under the header, and only layout shows it');

  // ── the results dashboard ───────────────────────────────────────────────
  const r = await page.goto(B + '/results.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  check('results.html served', r.status() === 200, `HTTP ${r.status()}`);
  const hasStudy = fs.existsSync(path.join(ROOT, 'results/dataset.json'));
  if (!hasStudy) {
    console.log('  info no results/dataset.json — checking the empty state instead');
    check('empty state explains itself', /No results/i.test(await page.locator('main').innerText()));
  } else {
    // A results file containing a bare NaN token - which Python writes by
    // default - parses in Python and makes JSON.parse throw, deploying a
    // blank page. Only executing the page catches that.
    check('tables rendered', (await page.locator('table.data').count()) >= 3);
    check('rows filled', (await page.locator('table.data tbody tr').count()) >= 4);
    check('verdict computed', (await page.locator('.verdict').count()) === 1,
      'the relief banner is derived from the data, not written by hand');
    check('scene explorer rendered', (await page.locator('.compare img').count()) === 2);
    const body = await page.locator('body').innerText();
    check('no undefined leaked into the copy', !/\bundefined\b|\bNaN\b/.test(body),
      'a missing field must read as "–", not as NaN');
  }

  // ── layout: the bug a DOM-only harness cannot see ───────────────────────
  check('page does not scroll sideways', await page.evaluate(
    () => document.documentElement.scrollWidth <= window.innerWidth + 1));

  check('no browser errors', pageErrs.length === 0, pageErrs.slice(0, 4).join(' | '));
} finally {
  await browser.close();
  server.close();
}

if (errors.length) {
  console.error('\nFAILED:');
  errors.forEach((e) => console.error('  - ' + e));
  process.exit(1);
}
console.log('\nsite check passed');
