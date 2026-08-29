/* AYAMA results site.
 *
 * Renders the real-imagery study: four Swiss city centres with airborne lidar
 * truth, read live from results/cpu/<arm>/dataset.json. It invents nothing. If
 * a field is missing the panel says so rather than showing a plausible number.
 *
 * Re-run `python -m ayama.cli dataset data/real --layout generic` and the page
 * shows the new measurements.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const fmt = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '–' : v.toFixed(d);
const pm = (o, d = 2) => o ? `${fmt(o.mean, d)} <small>± ${fmt(o.std, d)}</small>` : '–';
const mean = (o) => (o && isFinite(o.mean)) ? o.mean : null;

// The arms of the study. The fitted one leads because it is the result; the
// others are what the same pipeline does without a structural scale, and the
// gap between them is the finding.
const ARMS = [
  { id: 'real_vitl_learned', backbone: 'dav2-vitl', calib: 'dual branch + fitted scale', primary: true },
  { id: 'real_vitl_h1', backbone: 'dav2-vitl', calib: 'single affine (H1)' },
  { id: 'real_vitl_h2', backbone: 'dav2-vitl', calib: 'dual branch (H2)' },
  { id: 'real_vits_h1', backbone: 'dav2-vits', calib: 'single affine (H1)' },
  { id: 'real_vits_h2', backbone: 'dav2-vits', calib: 'dual branch (H2)' },
];

// Layers the explorer can show. Each is a PNG the pipeline writes per scene.
const LAYERS = [
  { id: 'texture.jpg', name: 'Orthophoto',   note: 'The only input, besides a public DEM. SWISSIMAGE 10 cm, resampled to 0.5 m.' },
  { id: 'dsm.png',     name: 'Predicted DSM', note: 'Metric elevation as delivered.' },
  { id: 'ndsm.png',    name: 'Height above ground', note: 'What a planner actually asks for. Compare it with the relief panel above.' },
  { id: 'error.png',   name: 'Error vs lidar', note: 'Predicted minus the swissSURFACE3D DSM. Blue is too low, red is too high.' },
  { id: 'sigma.png',   name: 'Uncertainty σ', note: 'Per-pixel 1σ. Bright where anchors are sparse.' },
];

const state = { arms: {}, arm: 'real_vitl_learned' };

// Quick-look rasters are committed for the primary arm only. Across arms they
// differ by amounts no eye can see - the finding is in the metrics, not the
// pictures - and four copies would put 28 MB in the repository for nothing.
const IMAGE_ARM = 'real_vitl_learned';

async function init() {
  wireStaticUI();
  const loaded = await Promise.all(ARMS.map(async (a) => {
    try {
      const res = await fetch(`results/cpu/${a.id}/dataset.json`, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return [a.id, await res.json()];
    } catch (err) { return [a.id, null]; }
  }));
  loaded.forEach(([id, d]) => { if (d) state.arms[id] = d; });

  if (!state.arms[state.arm]) {
    const first = Object.keys(state.arms)[0];
    if (!first) { showMissingResults(new Error('no dataset.json could be loaded')); return; }
    state.arm = first;
  }
  render();
}

function render() {
  const study = state.arms[state.arm];
  const agg = study.aggregate || {};
  renderHero(agg, study);
  renderReliefVerdict(agg);
  renderHeadline(agg);
  renderScenes(study);
  renderClassChart(study);
  renderAnchors(study);
  renderArms();
  renderExplorer(study);
  renderFooter(study);
}

// ── hero ─────────────────────────────────────────────────────────────────────
function renderHero(agg, study) {
  const host = $('#hero-metrics');
  if (!agg.mae_m) { host.innerHTML = '<div class="loading">No aggregate metrics in dataset.json.</div>'; return; }

  const mae = agg.mae_m;
  const dem = mean(agg.dem_metrics_mae_m);
  const base = mean(agg.baseline_metrics_mae_m);
  const nd = mean(agg.ndsm_metrics_mae_m);
  const flat = mean(agg.zero_baseline_metrics_mae_m);
  const trueH = mean(agg.true_mean_height_m);
  const predH = mean(agg.pred_mean_height_m);
  const recovered = (trueH && predH !== null) ? (100 * predH / trueH) : null;

  const cards = [
    { k: 'Elevation MAE', v: `${fmt(mae.mean)} <small>± ${fmt(mae.std)} m</small>`,
      n: `over ${agg.n_scenes} real scenes`, hi: true },
    { k: 'DEM alone — the floor', v: `${fmt(dem)} <small>m</small>`,
      n: (dem === null || mae.mean === null) ? ''
         : (mae.mean < dem ? `beaten by ${fmt(100 * (1 - mae.mean / dem), 1)}%`
                           : 'not beaten') },
    { k: 'Global-affine baseline', v: `${fmt(base)} <small>m</small>`,
      n: base ? `spatial calibration is ${fmt(100 * (1 - mae.mean / base), 0)}% better` : '' },
    { k: 'Relief recovered', v: recovered === null ? '–' : `${fmt(recovered, 1)}<small>%</small>`,
      n: (trueH === null) ? '' : `${fmt(predH)} m of a true ${fmt(trueH)} m`, bad: true },
  ];
  host.innerHTML = cards.map(c => `
    <div class="metric${c.hi ? ' hi' : ''}${c.bad ? ' bad' : ''}">
      <div class="k">${c.k}</div>
      <div class="v">${c.v}</div>
      <div class="n">${c.n}</div>
    </div>`).join('');

  const cfg = study.config || {};
  $('#hero-note').textContent =
    `${cfg.backbone || '?'} · ${agg.n_scenes} scenes of 1024×1024 px at 0.5 m · ` +
    `${cfg.device || 'cpu'} · ${fmt(study.wall_s, 0)} s · ` +
    `truth: swissSURFACE3D lidar`;
}

// ── the honesty banner ───────────────────────────────────────────────────────
// The elevation MAE looks respectable because most of a scene is ground and the
// DEM already knows the ground. The number that matters is how much height above
// ground came back. If that is near zero the page has to say so, and nobody gets
// to bury it by forgetting to update the copy: it is computed from the data.
function renderReliefVerdict(agg) {
  const host = $('#floor-verdict');
  if (!host) return;
  const nd = mean(agg.ndsm_metrics_mae_m);
  const flat = mean(agg.zero_baseline_metrics_mae_m);
  const trueH = mean(agg.true_mean_height_m);
  const predH = mean(agg.pred_mean_height_m);
  if (nd === null || flat === null) { host.innerHTML = ''; return; }

  const pct = trueH ? 100 * predH / trueH : null;
  const beatsFlat = flat > 0 ? 100 * (1 - nd / flat) : 0;
  const bad = pct !== null && pct < 10;
  host.innerHTML = `
    <div class="verdict ${bad ? 'bad' : 'good'}">
    <strong>${bad ? 'Height above ground is not being recovered.' :
                    'Height above ground is being recovered.'}</strong>
    Against the lidar nDSM the pipeline scores <b>${fmt(nd)} m</b> MAE.
    Predicting <em>zero height everywhere</em> scores <b>${fmt(flat)} m</b> —
    a difference of ${fmt(beatsFlat, 1)}%.
    It returns <b>${fmt(predH)} m</b> of an average <b>${fmt(trueH)} m</b> of true
    structure${pct === null ? '' : ` (${fmt(pct, 1)}%)`}.
    See README §4 for the mechanism.</div>`;
}

// ── headline table ───────────────────────────────────────────────────────────
function renderHeadline(agg) {
  const host = $('#headline-table');
  if (!host) return;
  const rows = [
    ['MAE (m)', pm(agg.mae_m), fmt(mean(agg.baseline_metrics_mae_m)), fmt(mean(agg.dem_metrics_mae_m))],
    ['RMSE (m)', pm(agg.rmse_m), '–', '–'],
    ['Pearson r', pm(agg.pearson_r, 3), '–', '–'],
    ['bias (m)', pm(agg.bias_m), '–', '–'],
    ['edge F1', pm(agg.edge_f1, 3), '–', '–'],
    ['1σ coverage', pm(agg.coverage_1s, 3), '–', '–'],
    ['ECE (m)', pm(agg.ece_m), '–', '–'],
    ['<b>nDSM MAE (m)</b>', `<b>${fmt(mean(agg.ndsm_metrics_mae_m))}</b>`, '–',
     `<b>${fmt(mean(agg.zero_baseline_metrics_mae_m))}</b> <small>flat ground</small>`],
  ];
  host.innerHTML = `<table class="data">
    <thead><tr><th>metric</th><th>ĀYĀMA</th><th>global affine</th><th>floor</th></tr></thead>
    <tbody>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join('')}</tbody>
  </table>`;
}

// ── per-scene table ──────────────────────────────────────────────────────────
function renderScenes(study) {
  const host = $('#scene-table');
  if (!host) return;
  const rows = (study.scenes || []).map(s => {
    const r = s.relief || {};
    const pct = r.true_mean_height_m ? 100 * r.pred_mean_height_m / r.true_mean_height_m : null;
    return `<tr>
      <td>${s.name}</td><td>${s.tier}</td>
      <td>${fmt(s.metrics && s.metrics.mae_m)}</td>
      <td>${fmt(s.dem_metrics && s.dem_metrics.mae_m)}</td>
      <td>${fmt(s.ndsm_metrics && s.ndsm_metrics.mae_m)}</td>
      <td>${fmt(s.zero_baseline_metrics && s.zero_baseline_metrics.mae_m)}</td>
      <td>${fmt(r.true_max_height_m, 1)}</td>
      <td>${fmt(r.pred_max_height_m, 1)}</td>
      <td>${pct === null ? '–' : fmt(pct, 1) + '%'}</td>
    </tr>`;
  }).join('');
  host.innerHTML = `<table class="data">
    <thead><tr>
      <th>scene</th><th>tier</th><th>MAE</th><th>DEM floor</th>
      <th>nDSM MAE</th><th>flat ground</th>
      <th>true max</th><th>pred max</th><th>relief</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="fineprint">Metres. “flat ground” is the MAE of predicting zero height
      everywhere — the floor any reconstruction must clear.</p>`;
}

// ── error by class ───────────────────────────────────────────────────────────
function renderClassChart(study) {
  const host = $('#class-chart');
  if (!host) return;
  const acc = {};
  (study.scenes || []).forEach(s => {
    Object.entries(s.metrics_by_class || {}).forEach(([k, m]) => {
      if (!isFinite(m.mae_m)) return;
      (acc[k] = acc[k] || []).push(m.mae_m);
    });
  });
  const rows = Object.entries(acc)
    .map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length])
    .sort((a, b) => a[1] - b[1]);
  if (!rows.length) { host.innerHTML = '<div class="loading">No per-class metrics.</div>'; return; }
  const max = Math.max(...rows.map(r => r[1]));
  host.innerHTML = rows.map(([k, v]) => `
    <div class="bar-row">
      <span class="bar-k">${k}</span>
      <span class="bar"><i style="width:${(100 * v / max).toFixed(1)}%"></i></span>
      <span class="bar-v">${fmt(v)} m</span>
    </div>`).join('') +
    `<p class="fineprint">Classes come from the colour heuristic, not from labels —
      none ship with these tiles, and on real imagery the heuristic is unreliable.</p>`;
}

// ── anchor composition ───────────────────────────────────────────────────────
function renderAnchors(study) {
  const host = $('#anchor-table');
  if (!host) return;
  const rows = (study.scenes || []).map(s => {
    const a = s.anchors || {};
    return `<tr><td>${s.name}</td><td>${a.dem ?? '–'}</td><td>${a.water ?? '–'}</td>
      <td class="${a.shadow ? '' : 'zero'}">${a.shadow ?? '–'}</td><td>${a.total ?? '–'}</td></tr>`;
  }).join('');
  const anyShadow = (study.scenes || []).some(s => (s.anchors || {}).shadow > 0);
  host.innerHTML = `<table class="data">
    <thead><tr><th>scene</th><th>DEM</th><th>water</th><th>shadow</th><th>total</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="fineprint">${anyShadow ? '' :
      '<b>Every anchor is a ground anchor.</b> swisstopo publishes no acquisition ' +
      'time for these products, so no sun angle can be derived and shadow physics ' +
      'is disabled. README §3.7.4 shows the conclusion does not depend on it.'}</p>`;
}

// ── the four arms ────────────────────────────────────────────────────────────
function renderArms() {
  const host = $('#arm-table');
  if (!host) return;
  const rows = ARMS.filter(a => state.arms[a.id]).map(a => {
    const agg = state.arms[a.id].aggregate || {};
    const trueH = mean(agg.true_mean_height_m), predH = mean(agg.pred_mean_height_m);
    return `<tr class="${a.id === state.arm ? 'sel' : ''}">
      <td><button class="linkish" data-arm="${a.id}">${a.backbone}</button></td>
      <td>${a.calib}</td>
      <td>${pm(agg.mae_m)}</td>
      <td>${fmt(mean(agg.edge_f1), 3)}</td>
      <td>${fmt(mean(agg.ndsm_metrics_mae_m))}</td>
      <td>${trueH ? fmt(100 * predH / trueH, 1) + '%' : '–'}</td>
    </tr>`;
  }).join('');
  host.innerHTML = `<table class="data">
    <thead><tr><th>backbone</th><th>calibration</th><th>MAE (m)</th>
      <th>edge F1</th><th>nDSM MAE (m)</th><th>relief</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="fineprint">Click a row to render the rest of this page from that run.
      Without a fitted structural scale every arm recovers under 1.3% of the true
      relief &mdash; two backbones and two calibrations agree. Supplying one fitted
      constant is what moves it.</p>`;
  $$('button[data-arm]', host).forEach(b => b.addEventListener('click', () => {
    state.arm = b.dataset.arm; render();
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  }));
}

// ── scene explorer ───────────────────────────────────────────────────────────
function renderExplorer(study) {
  const sceneSel = $('#scene-select'), left = $('#left-layer'), right = $('#right-layer');
  if (!sceneSel || !left || !right) return;
  const scenes = study.scenes || [];
  if (!scenes.length) return;

  sceneSel.innerHTML = scenes.map(s => `<option value="${s.name}">${s.name}</option>`).join('');
  const opts = LAYERS.map(l => `<option value="${l.id}">${l.name}</option>`).join('');
  left.innerHTML = opts; right.innerHTML = opts;
  left.value = 'texture.jpg'; right.value = 'ndsm.png';

  const update = () => {
    const name = sceneSel.value;
    const base = `results/cpu/${IMAGE_ARM}/${name}/`;
    $('#img-left').src = base + left.value;
    $('#img-right').src = base + right.value;
    $('#lbl-left').textContent = (LAYERS.find(l => l.id === left.value) || {}).name || '';
    $('#lbl-right').textContent = (LAYERS.find(l => l.id === right.value) || {}).name || '';
    const notes = [left.value, right.value].map(id => LAYERS.find(l => l.id === id))
      .filter(Boolean).map(l => `<b>${l.name}.</b> ${l.note}`);
    $('#layer-legend').innerHTML = notes.join(' ');
    renderSceneFacts(scenes.find(s => s.name === name));
  };
  [sceneSel, left, right].forEach(el => el.addEventListener('change', update));
  update();
  wireCompare();
}

function renderSceneFacts(s) {
  const host = $('#scene-facts');
  if (!host || !s) return;
  const m = s.metrics || {}, r = s.relief || {};
  const facts = [
    ['tier', `${s.tier} — ${s.tier_reason || ''}`],
    ['MAE', `${fmt(m.mae_m)} m`],
    ['nDSM MAE', `${fmt((s.ndsm_metrics || {}).mae_m)} m`],
    ['flat-ground floor', `${fmt((s.zero_baseline_metrics || {}).mae_m)} m`],
    ['true max height', `${fmt(r.true_max_height_m, 1)} m`],
    ['predicted max height', `${fmt(r.pred_max_height_m, 1)} m`],
    ['anchors', Object.entries(s.anchors || {}).filter(([k]) => k !== 'total')
      .map(([k, v]) => `${k} ${v}`).join(', ')],
  ];
  host.innerHTML = facts.map(([k, v]) =>
    `<div class="fact"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}

function wireCompare() {
  const wrap = $('#compare'), clip = $('#clip'), handle = $('#handle');
  if (!wrap || !clip || !handle || wrap.dataset.wired) return;
  wrap.dataset.wired = '1';
  let pct = 50;
  const set = (p) => {
    pct = Math.max(0, Math.min(100, p));
    clip.style.width = pct + '%';
    handle.style.left = pct + '%';
    handle.setAttribute('aria-valuenow', String(Math.round(pct)));
  };
  const fromEvent = (e) => {
    const rect = wrap.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    set(100 * x / rect.width);
  };
  let dragging = false;
  wrap.addEventListener('pointerdown', (e) => { dragging = true; fromEvent(e); });
  window.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
  window.addEventListener('pointerup', () => { dragging = false; });
  handle.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { set(pct - 2); e.preventDefault(); }
    if (e.key === 'ArrowRight') { set(pct + 2); e.preventDefault(); }
  });
  set(50);
}

// ── footer ───────────────────────────────────────────────────────────────────
function renderFooter(study) {
  const host = $('#footer-meta');
  if (!host) return;
  const cfg = study.config || {};
  host.textContent =
    `Rendered from results/cpu/${state.arm}/dataset.json · ` +
    `${study.n_ok}/${study.n_found} scenes · backbone ${cfg.backbone} · ` +
    `imagery and elevation truth © swisstopo, used under Swiss OGD terms.`;
}

// ── static UI ────────────────────────────────────────────────────────────────
function wireStaticUI() {
  wireWizard();
  const repo = 'https://github.com/Sisigoks/AYAMA';
  const link = $('#repo-link'); if (link) link.href = repo;
  $$('.copy').forEach(btn => btn.addEventListener('click', () => {
    const code = btn.previousElementSibling;
    if (!code) return;
    navigator.clipboard.writeText(code.textContent.trim()).then(() => {
      const was = btn.textContent; btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = was; }, 1200);
    });
  }));
}

// ── tier wizard ────────────────────────
// Independent of the study data: it answers "what tier will my own image get?"
function wireWizard() {
  const wiz = { geo: 'yes', dem: 'file', extra: 'none' };

  $$('.opts').forEach(group => group.addEventListener('click', e => {
    const btn = e.target.closest('.opt');
    if (!btn) return;
    $$('.opt', group).forEach(b => b.classList.toggle('on', b === btn));
    wiz[group.dataset.q] = btn.dataset.v;
    updateWizard(wiz);
  }));
  updateWizard(wiz);
}

function updateWizard(s) {
  const hasGcp = s.extra === 'gcp' || s.extra === 'both';
  const hasRef = s.extra === 'ref' || s.extra === 'both';
  const demUsable = s.geo === 'yes' && s.dem === 'file';

  let tier, expl;
  if (hasGcp && demUsable) {
    tier = 'B';
    expl = 'Ground control points pin the datum that a DEM can only approximate, and the DEM ' +
           'still supplies dense terrain anchors between them. This is the best the method does.';
  } else if (hasGcp) {
    tier = 'B';
    expl = 'Your control points fix the datum. Without a DEM there are no dense terrain anchors, ' +
           'so accuracy away from the control points rests on shadow physics and the depth prior.';
  } else if (demUsable) {
    tier = 'A';
    expl = 'DEM samples are admitted only where segmentation says bare ground, road or water, ' +
           'and cast-shadow heights fill in on structures. Terrain comes out well; absolute ' +
           'building height is the weaker half.';
  } else {
    tier = 'C';
    expl = s.geo === 'no'
      ? 'Without a CRS a public DEM cannot be located, so calibration falls back to shadow ' +
        'trigonometry and an assumed ground plane. Relative structure is trustworthy; the ' +
        'datum is arbitrary and the output says so.'
      : 'With no DEM to anchor terrain, calibration rests on shadow physics and an assumed ' +
        'ground plane. Relative structure is trustworthy; the datum is arbitrary.';
  }

  const badge = $('#wiz-tier');
  badge.className = `tier-badge ${tier.toLowerCase()}`;
  badge.textContent = `Tier ${tier} — ${{ A: 'automatic (DEM)', B: 'GCP-assisted', C: 'physics only' }[tier]}`;
  $('#wiz-expl').textContent = expl;

  const img = s.geo === 'yes' ? 'my_scene.tif' : 'my_photo.jpg';
  const lines = [`python -m ayama.cli run ${img} --out out/mine \\`,
                 `    --backbone dav2-vitl \\`];
  if (demUsable) lines.push(`    --dem my_dem.tif \\`);
  if (hasGcp) lines.push(`    --gcps my_gcps.csv \\`);
  if (hasRef) lines.push(`    --ref my_reference_dsm.tif \\`);
  lines.push(`    --bootstrap 24 --json out/mine.json`);
  $('#wiz-cmd').textContent = lines.join('\n');

  const fine = [];
  if (s.geo === 'no') fine.push('Without a CRS the ground sample distance is assumed to be 1 m, which scales every height linearly — pass a real GSD or georeference the image first.');
  if (!hasRef) fine.push('With no reference DSM there are no accuracy metrics, only the σ field; the run still writes DSM, nDSM and σ.');
  if (hasGcp) fine.push('GCP csv columns: row,col,elev_m[,label] in pixel coordinates.');
  fine.push('Sun azimuth and elevation are read from the GeoTIFF tags, or computed from EXIF GPS and timestamp. Without them the shadow branch is skipped rather than guessed.');
  $('#wiz-fine').textContent = fine.join(' ');
}


function showMissingResults(err) {
  const host = $('#hero-metrics');
  if (host) {
    host.innerHTML = `<div class="loading">
      Could not load <code>results/cpu/&lt;arm&gt;/dataset.json</code> (${err.message}).<br>
      Run <code>python scripts/fetch_swisstopo.py --out data/real/zurich</code> then
      <code>python -m ayama.cli dataset data/real --layout generic --out results/cpu/real_vitl_h1</code>.
    </div>`;
  }
}

document.addEventListener('DOMContentLoaded', init);
