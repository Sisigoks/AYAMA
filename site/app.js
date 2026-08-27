/* AYAMA results site.
 *
 * This file renders results/study.json. It invents nothing: if a field is
 * missing the panel says so rather than showing a plausible number. Re-run
 * `python -m ayama.cli study` and the page shows the new measurements.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const fmt = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '–' : v.toFixed(d);
const pm = (o, d = 2) => o ? `${fmt(o.mean, d)} <small>± ${fmt(o.std, d)}</small>` : '–';

// ── layers the explorer can show ─────────────────────────────────────────────
const LAYERS = [
  { id: 'rgb.jpg',       name: 'RGB image',        note: 'The only input the method sees, besides a public DEM.' },
  { id: 'dsm_pred.png',  name: 'Predicted DSM',    note: 'Metric elevation, terrain colour ramp shared with the reference below.' },
  { id: 'dsm_truth.png', name: 'Reference DSM',    note: 'The exact surface that generated the image. Same colour scale as the prediction.' },
  { id: 'error.png',     name: 'Error vs reference', note: 'Predicted minus reference, ±15 m. Blue is too low, red is too high.' },
  { id: 'sigma.png',     name: 'Uncertainty σ',    note: 'Per-pixel 1σ. Bright where anchors are sparse, dark where they cluster.' },
  { id: 'ndsm.png',      name: 'Height above ground', note: 'nDSM, 0–30 m. What a planner actually asks for: how tall is that.' },
  { id: 'shadow.png',    name: 'Cast shadow',      note: 'The mask the height physics is measured from.' },
];

// ── paper figures ────────────────────────────────────────────────────────────
const PAPER_FIGURES = [
  { file: 'fig1_ablation',           title: 'Which components earn their place',
    note: 'One inference per scene; every variant re-solves only the calibration.' },
  { file: 'fig6_qualitative',        title: 'A reconstruction, end to end',
    note: 'Input, reference, prediction, error and σ on one colour scale per quantity.' },
  { file: 'fig3_sun_window',         title: 'The shadow physics window',
    note: 'Height from shadow length alone, against sun elevation. Two panels, one x-axis — never two y-scales.' },
  { file: 'fig2_error_by_class',     title: 'Where the error lives',
    note: 'Terrain is close to solved; buildings and canopy carry the error.' },
  { file: 'fig5_reliability',        title: 'Does σ predict the error?',
    note: 'Coverage says whether the bars are the right size; the spread says whether σ can rank pixels.' },
  { file: 'fig4_lambda_sensitivity', title: 'Sensitivity to the one free parameter',
    note: 'A parameter that must be hunted for per scene is a knob, not a method.' },
];

function renderFigures() {
  const host = $('#figure-grid');
  if (!host) return;
  host.innerHTML = PAPER_FIGURES.map(f => `
    <figure class="paper-fig">
      <a href="results/figures/${f.file}.png" target="_blank" rel="noopener">
        <img src="results/figures/${f.file}.png" alt="${f.title}" loading="lazy">
      </a>
      <figcaption>
        <strong>${f.title}</strong>
        <span>${f.note}</span>
        <span class="dl"><a href="results/figures/${f.file}.pdf">PDF</a> ·
                         <a href="results/figures/${f.file}.png">PNG</a></span>
      </figcaption>
    </figure>`).join('');
  // A figure that was never rendered (no reference DSM, say) removes its own card
  // rather than leaving a broken image on the page.
  $$('.paper-fig img', host).forEach(img => {
    img.addEventListener('error', () => img.closest('.paper-fig').remove());
  });
}

// ── boot ─────────────────────────────────────────────────────────────────────
init();

async function init() {
  wireStaticUI();
  let study = null;
  try {
    const res = await fetch('results/study.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    study = await res.json();
  } catch (err) {
    showMissingResults(err);
    return;
  }
  render(study);
}

function render(study) {
  const agg = study.aggregate || {};
  renderHero(agg, study);
  renderHeadline(agg);
  renderClassChart(agg);
  renderCoverage(agg);
  renderAblation(study);
  renderSun(study);
  renderLambda(study);
  renderBench(study);
  renderExplorer(study);
  renderFigures();
  renderFooter(study);
}

// ── hero ─────────────────────────────────────────────────────────────────────
function renderHero(agg, study) {
  const host = $('#hero-metrics');
  const mae = agg.mae_m, base = agg.baseline_mae_m, r = agg.pearson_r, cov = agg.coverage_1s;
  if (!mae) { host.innerHTML = '<div class="loading">No aggregate metrics in study.json.</div>'; return; }

  const dem = agg.dem_mae_m;
  const improve = (base && base.mean) ? (1 - mae.mean / base.mean) * 100 : null;
  const overFloor = (dem && dem.mean) ? (1 - mae.mean / dem.mean) * 100 : null;
  const cards = [
    { k: 'MAE vs ground truth', v: `${fmt(mae.mean)} <small>± ${fmt(mae.std)} m</small>`,
      n: `over ${agg.n_scenes} independent scenes`, hi: true },
    { k: 'Global-affine baseline', v: `${fmt(base && base.mean)} <small>m</small>`,
      n: improve === null ? '' : `AGMC is ${fmt(improve, 0)}% better` },
    { k: 'DEM alone — the floor', v: `${fmt(dem && dem.mean)} <small>m</small>`,
      n: overFloor === null ? 'no DEM baseline recorded'
         : (overFloor > 5 ? `AGMC clears it by ${fmt(overFloor, 0)}%`
                          : 'AGMC does not clear it') },
    { k: '1σ coverage', v: fmt(cov && cov.mean),
      n: 'a calibrated σ lands near 0.68' },
  ];
  host.innerHTML = cards.map(c => `
    <div class="metric${c.hi ? ' hi' : ''}">
      <div class="k">${c.k}</div>
      <div class="v">${c.v}</div>
      <div class="n">${c.n}</div>
    </div>`).join('');

  renderFloorVerdict(agg);

  const cfg = study.config || {};
  const env = study.environment || {};
  $('#hero-note').textContent =
    `${cfg.backbone || '?'} · ${cfg.size}×${cfg.size} px at ${cfg.gsd_m} m · ` +
    `${env.gpu || 'CPU'} · measured ${(env.timestamp_utc || '').replace('T', ' ').replace('Z', ' UTC')}`;
}

// ── the honesty banner ───────────────────────────────────────────────────────
// If the pipeline does not beat the DEM it was anchored to, the depth model is
// contributing nothing and the page has to say so. Nobody gets to bury that by
// forgetting to update the copy: it is computed from the same numbers.
function renderFloorVerdict(agg) {
  const host = document.getElementById('floor-verdict');
  if (!host) return;
  const mae = agg.mae_m, dem = agg.dem_mae_m;
  if (!mae || !dem) { host.innerHTML = ''; return; }
  const gain = (1 - mae.mean / dem.mean) * 100;

  // A margin on MAE alone is not enough: the surface can win on average error
  // while losing on RMSE and matching the DEM's correlation exactly, which is
  // what "it reproduced the DEM" looks like from the outside.
  const rGain = (agg.pearson_r && agg.dem_pearson_r)
    ? agg.pearson_r.mean - agg.dem_pearson_r.mean : 0;
  const clearlyBetter = gain > 10 && rGain > 0.02;

  if (clearlyBetter) {
    host.innerHTML = `<div class="verdict good">
      <strong>Clears the floor.</strong> The reconstruction is ${fmt(gain, 0)}% better than the
      public DEM it was anchored to (${fmt(mae.mean)} m against ${fmt(dem.mean)} m), so the
      depth model is contributing real information rather than being interpolated over.</div>`;
  } else if (gain > -5) {
    host.innerHTML = `<div class="verdict warn">
      <strong>Level with the floor.</strong> At ${fmt(mae.mean)} m against ${fmt(dem.mean)} m,
      the reconstruction is not meaningfully better than resampling the public DEM it was
      anchored to${rGain <= 0.02 ? ', and its correlation with the truth is the same' : ''}.
      On these scenes the anchor graph is carrying the result and the depth model contributes
      little: the large gain over the global-affine baseline measures how badly a single scale
      fits a contaminated depth field, not how much that field knows. Treat the headline as a
      working figure, not a claim — and read the edge F1 and δ rows, which are what a surface
      with no structures in it looks like.</div>`;
  } else {
    host.innerHTML = `<div class="verdict bad">
      <strong>Below the floor.</strong> The reconstruction (${fmt(mae.mean)} m) is worse than
      the public DEM it was anchored to (${fmt(dem.mean)} m). The depth model is actively
      costing accuracy on these scenes.</div>`;
  }
}

// ── headline table ───────────────────────────────────────────────────────────
function renderHeadline(agg) {
  const rows = [
    ['MAE (m)', agg.mae_m, agg.baseline_mae_m, 'lower', agg.dem_mae_m],
    ['RMSE (m)', agg.rmse_m, agg.baseline_rmse_m, 'lower', agg.dem_rmse_m],
    ['Pearson r', agg.pearson_r, agg.baseline_pearson_r, 'higher', agg.dem_pearson_r],
    ['Spearman ρ', agg.spearman_r, null, 'higher'],
    ['Bias (m)', agg.bias_m, null, null],
    ['Median AE (m)', agg.median_ae_m, null, 'lower'],
    ['Slope MAE (deg)', agg.slope_mae_deg, null, 'lower'],
    ['Edge F1', agg.edge_f1, null, 'higher'],
    ['δ &lt; 1.25', agg.delta1, null, 'higher'],
    ['1σ coverage', agg.coverage_1s, null, null],
    ['ECE (m)', agg.ece_m, null, 'lower'],
  ].filter(row => row[1]);

  $('#headline-table').innerHTML = `
    <div class="tbl-scroll"><table>
      <thead><tr><th>metric</th><th>AGMC</th><th>global affine</th>
        <th>DEM alone <span class="dim">(floor)</span></th></tr></thead>
      <tbody>${rows.map(([label, a, b, dir, floor]) => {
        const d = label.includes('r') && !label.includes('MAE') && !label.includes('RMSE') ? 3 : 2;
        const win = b && dir && ((dir === 'lower' && a.mean < b.mean) || (dir === 'higher' && a.mean > b.mean));
        return `<tr>
          <td>${label}</td>
          <td class="num${win ? ' win' : ''}">${pm(a, d)}</td>
          <td class="num dim">${b ? pm(b, d) : '–'}</td>
          <td class="num dim">${floor ? pm(floor, d) : '–'}</td>
        </tr>`;
      }).join('')}</tbody>
    </table></div>`;
}

// ── per-class bars ───────────────────────────────────────────────────────────
function renderClassChart(agg) {
  const byClass = agg.by_class_mae_m || {};
  const entries = Object.entries(byClass).sort((a, b) => a[1].mean - b[1].mean);
  if (!entries.length) { $('#class-chart').innerHTML = '<div class="loading">No per-class metrics.</div>'; return; }
  const max = Math.max(...entries.map(e => e[1].mean));
  $('#class-chart').innerHTML = `<div class="bars">${entries.map(([name, v]) => {
    const pct = Math.max(2, (v.mean / max) * 100);
    const hot = v.mean > max * 0.6;
    return `<div class="bar-row">
      <span class="lab">${name}</span>
      <span class="bar-track"><span class="bar-fill${hot ? ' warm' : ''}" style="width:${pct}%"></span></span>
      <span class="val">${fmt(v.mean)} m</span>
    </div>`;
  }).join('')}</div>`;
}

// ── coverage gauge ───────────────────────────────────────────────────────────
function renderCoverage(agg) {
  const cov = agg.coverage_1s, ece = agg.ece_m;
  if (!cov) { $('#coverage-chart').innerHTML = '<div class="loading">σ was not evaluated.</div>'; return; }
  const off = Math.abs(cov.mean - 0.68);
  const ok = off < 0.08;
  const verdict = ok
    ? 'Within 8 points of the Gaussian expectation: the error bars mean what they say.'
    : (cov.mean > 0.68
      ? 'Above expectation — σ is conservative, so the bars are wider than the errors warrant.'
      : 'Below expectation — σ is overconfident and is understating the real error.');

  const W = 360, H = 90, pad = 22, y = 46;
  const x = v => pad + v * (W - 2 * pad);
  $('#coverage-chart').innerHTML = `
    <div class="gauge">
      <div class="big ${ok ? 'ok' : 'off'}">${fmt(cov.mean)}</div>
      <div class="note">${verdict}${ece ? ` ECE ${fmt(ece.mean)} m.` : ''}</div>
    </div>
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="One sigma coverage against the 0.68 target">
      <rect class="zone" x="${x(0.60)}" y="${y - 12}" width="${x(0.76) - x(0.60)}" height="24" rx="4"/>
      <line class="axis" x1="${pad}" y1="${y}" x2="${W - pad}" y2="${y}"/>
      <line class="target" x1="${x(0.68)}" y1="${y - 18}" x2="${x(0.68)}" y2="${y + 18}"/>
      <text x="${x(0.68)}" y="${y - 24}" text-anchor="middle">0.68 target</text>
      <circle class="dot" cx="${x(Math.min(1, Math.max(0, cov.mean)))}" cy="${y}" r="7"/>
      <text x="${pad}" y="${y + 30}">0.0</text>
      <text x="${W - pad}" y="${y + 30}" text-anchor="end">1.0</text>
      <text class="lbl" x="${x(Math.min(1, Math.max(0, cov.mean)))}" y="${y + 30}" text-anchor="middle">measured</text>
    </svg>`;
}

// ── ablation ─────────────────────────────────────────────────────────────────
const VARIANT_NOTE = {
  dem_only: 'no depth model at all — the public DEM resampled, i.e. the floor',
  global_affine: 'one a, b for the whole tile — the published baseline',
  agmc_no_gate: 'DEM anchors taken everywhere, rooftops included',
  agmc_no_shadow: 'no shadow-derived height anchors',
  agmc_no_water: 'no water flatness constraint',
  agmc: 'everything on',
  agmc_bootstrap: 'everything on, plus the bootstrap σ field',
};

function renderAblation(study) {
  const byVariant = {};
  Object.values(study.ablation || {}).forEach(rows => {
    (rows || []).forEach(r => {
      if (!r || r.mae_m === undefined) return;
      (byVariant[r.variant] = byVariant[r.variant] || []).push(r);
    });
  });
  const names = Object.keys(byVariant);
  if (!names.length) { $('#ablation-chart').innerHTML = '<div class="loading">No ablation in study.json.</div>'; return; }

  const mean = (rows, key) => rows.reduce((s, r) => s + (r[key] || 0), 0) / rows.length;
  const data = names.map(v => ({
    variant: v, note: VARIANT_NOTE[v] || '',
    mae: mean(byVariant[v], 'mae_m'), rmse: mean(byVariant[v], 'rmse_m'),
    r: mean(byVariant[v], 'pearson_r'), anchors: Math.round(mean(byVariant[v], 'n_anchors')),
  }));
  const worst = Math.max(...data.map(d => d.mae));
  const best = Math.min(...data.map(d => d.mae));

  $('#ablation-chart').innerHTML = `
    <div class="bars">${data.map(d => {
      const pct = Math.max(3, (d.mae / worst) * 100);
      const isBest = Math.abs(d.mae - best) < 1e-9;
      const isBase = d.variant === 'global_affine' || d.variant === 'dem_only';
      return `<div class="bar-row${isBase ? ' muted' : ''}">
        <span class="lab"><code>${d.variant}</code></span>
        <span class="bar-track"><span class="bar-fill${isBest ? ' good' : (isBase ? ' warm' : '')}" style="width:${pct}%"></span></span>
        <span class="val">${fmt(d.mae)} m</span>
      </div>`;
    }).join('')}</div>
    <div class="tbl-scroll" style="margin-top:1.2rem"><table>
      <thead><tr><th>variant</th><th>what it removes</th><th>MAE (m)</th><th>RMSE (m)</th><th>r</th><th>anchors</th></tr></thead>
      <tbody>${data.map(d => `<tr>
        <td><code>${d.variant}</code></td>
        <td class="dim" style="text-align:left">${d.note}</td>
        <td class="num">${fmt(d.mae)}</td>
        <td class="num">${fmt(d.rmse)}</td>
        <td class="num">${fmt(d.r, 3)}</td>
        <td class="num dim">${d.anchors}</td>
      </tr>`).join('')}</tbody>
    </table></div>`;
}

// ── sun sweep ────────────────────────────────────────────────────────────────
function renderSun(study) {
  const rows = (study.sun_sweep || []).filter(r => isFinite(r.sun_elevation_deg));
  if (!rows.length) { $('#sun-chart').innerHTML = '<div class="loading">No sun sweep.</div>'; return; }

  const W = 460, H = 240, L = 44, R = 40, T = 16, B = 34;
  const xs = rows.map(r => r.sun_elevation_deg);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const errs = rows.map(r => r.median_abs_height_error_m).filter(isFinite);
  const emax = Math.max(6, ...errs);

  const px = v => L + ((v - xmin) / Math.max(xmax - xmin, 1e-6)) * (W - L - R);
  const pyF1 = v => T + (1 - v) * (H - T - B);
  const pyErr = v => T + (1 - Math.min(v, emax) / emax) * (H - T - B);

  const line = (pts) => pts.length ? 'M' + pts.map(p => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' L ') : '';
  const f1pts = rows.map(r => [px(r.sun_elevation_deg), pyF1(r.f1)]);
  const errpts = rows.filter(r => isFinite(r.median_abs_height_error_m))
                     .map(r => [px(r.sun_elevation_deg), pyErr(r.median_abs_height_error_m)]);

  const bandL = px(Math.max(xmin, 20)), bandR = px(Math.min(xmax, 75));

  $('#sun-chart').innerHTML = `
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Shadow detection F1 and height error against sun elevation">
      <rect class="zone" x="${bandL}" y="${T}" width="${Math.max(0, bandR - bandL)}" height="${H - T - B}"/>
      <text x="${(bandL + bandR) / 2}" y="${T + 12}" text-anchor="middle">usable band 20–75°</text>
      ${[0, 0.25, 0.5, 0.75, 1].map(v =>
        `<line class="grid" x1="${L}" y1="${pyF1(v)}" x2="${W - R}" y2="${pyF1(v)}"/>
         <text x="${L - 8}" y="${pyF1(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`).join('')}
      <line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/>
      ${rows.map(r => `<text x="${px(r.sun_elevation_deg)}" y="${H - B + 16}" text-anchor="middle">${r.sun_elevation_deg}</text>`).join('')}
      <text class="lbl" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">sun elevation (degrees)</text>
      <path class="ln" d="${line(f1pts)}"/>
      ${f1pts.map(p => `<circle class="dot" cx="${p[0]}" cy="${p[1]}" r="3"/>`).join('')}
      <path class="ln2" d="${line(errpts)}"/>
      ${errpts.map(p => `<circle class="dot2" cx="${p[0]}" cy="${p[1]}" r="3"/>`).join('')}
      <text x="${W - R + 8}" y="${T + 10}">${emax.toFixed(0)} m</text>
      <text x="${W - R + 8}" y="${H - B}">0 m</text>
    </svg>
    <div class="chart-key">
      <span>shadow detection F1</span>
      <span class="b">median height error (m)</span>
    </div>`;
}

// ── lambda sweep ─────────────────────────────────────────────────────────────
function renderLambda(study) {
  const rows = study.lambda_sweep || [];
  const agmc = rows.filter(r => r.lam !== null && r.lam !== undefined);
  const base = rows.find(r => r.lam === null || r.lam === undefined);
  if (!agmc.length) { $('#lambda-chart').innerHTML = '<div class="loading">No lambda sweep.</div>'; return; }

  const W = 460, H = 220, L = 46, R = 18, T = 18, B = 40;
  const lx = agmc.map(r => Math.log10(r.lam));
  const xmin = Math.min(...lx), xmax = Math.max(...lx);
  const maes = agmc.map(r => r.mae_m).concat(base ? [base.mae_m] : []);
  const ymin = Math.min(...maes) * 0.9, ymax = Math.max(...maes) * 1.05;

  const px = v => L + ((v - xmin) / Math.max(xmax - xmin, 1e-6)) * (W - L - R);
  const py = v => T + (1 - (v - ymin) / Math.max(ymax - ymin, 1e-6)) * (H - T - B);
  const pts = agmc.map(r => [px(Math.log10(r.lam)), py(r.mae_m)]);
  const bestMae = Math.min(...agmc.map(r => r.mae_m));
  const flat = agmc.filter(r => r.mae_m <= bestMae * 1.1);
  const flatLo = Math.min(...flat.map(r => r.lam)), flatHi = Math.max(...flat.map(r => r.lam));

  $('#lambda-chart').innerHTML = `
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="MAE against the smoothness weight">
      ${base ? `<line class="target" x1="${L}" y1="${py(base.mae_m)}" x2="${W - R}" y2="${py(base.mae_m)}" style="stroke:var(--warm)"/>
                <text x="${W - R}" y="${py(base.mae_m) - 6}" text-anchor="end">baseline ${fmt(base.mae_m)} m</text>` : ''}
      <line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/>
      <line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>
      ${agmc.map(r => `<text x="${px(Math.log10(r.lam))}" y="${H - B + 16}" text-anchor="middle">${r.lam}</text>`).join('')}
      <text class="lbl" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">smoothness weight λ (log scale)</text>
      <text x="${L - 8}" y="${py(ymin) + 4}" text-anchor="end">${fmt(ymin, 1)}</text>
      <text x="${L - 8}" y="${py(ymax) + 4}" text-anchor="end">${fmt(ymax, 1)}</text>
      <text class="lbl" x="${L - 8}" y="${T - 4}" text-anchor="end">MAE m</text>
      <path class="ln" d="M${pts.map(p => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' L ')}"/>
      ${pts.map(p => `<circle class="dot" cx="${p[0]}" cy="${p[1]}" r="3.5"/>`).join('')}
    </svg>
    <div class="chart-key"><span>AGMC</span>${base ? '<span class="b">global affine</span>' : ''}</div>
    <p class="panel-sub" style="margin-top:.9rem">
      Within 10% of the best result for λ from <code>${flatLo}</code> to <code>${flatHi}</code>
      — the default is 1.0, inside that range. A parameter you have to hunt for is a knob;
      this one is a scale.
    </p>`;
}

// ── bench ────────────────────────────────────────────────────────────────────
function renderBench(study) {
  const rows = (study.bench && study.bench.results) || [];
  if (!rows.length) { $('#bench-table').innerHTML = '<div class="loading">No throughput data.</div>'; return; }
  const env = (study.bench && study.bench.environment) || {};
  $('#bench-table').innerHTML = `
    <p class="panel-sub"><code>${env.gpu || 'CPU only'}</code> · torch ${env.torch || '?'}
      ${env.vram_total_gb ? `· ${env.vram_total_gb} GB VRAM` : ''}</p>
    <div class="tbl-scroll"><table>
      <thead><tr><th>backbone</th><th>chip</th><th>batch</th><th>chips</th>
        <th>wall (s)</th><th>s/chip</th><th>MPix/s</th><th>peak VRAM (MB)</th></tr></thead>
      <tbody>${rows.map(r => r.error ? `<tr>
          <td><code>${r.backbone}</code></td><td class="num">${r.chip}</td>
          <td class="num">${r.batch_size}</td>
          <td colspan="5" class="dim" style="text-align:left">${r.error}</td></tr>`
        : `<tr>
          <td><code>${r.backbone}</code></td>
          <td class="num">${r.chip}</td>
          <td class="num">${r.batch_size}</td>
          <td class="num dim">${r.n_chips}</td>
          <td class="num">${fmt(r.wall_s)}</td>
          <td class="num">${fmt(r.s_per_chip, 3)}</td>
          <td class="num">${fmt(r.mpix_per_s)}</td>
          <td class="num dim">${r.peak_vram_mb ? fmt(r.peak_vram_mb, 0) : '–'}</td>
        </tr>`).join('')}</tbody>
    </table></div>`;
}

// ── explorer ─────────────────────────────────────────────────────────────────
function renderExplorer(study) {
  const scenes = study.scenes || [];
  const sel = $('#scene-select');
  if (!scenes.length) { $('#compare').innerHTML = '<div class="loading">No scenes in study.json.</div>'; return; }

  sel.innerHTML = scenes.map((s, i) =>
    `<option value="${i}">seed ${s.seed} — MAE ${fmt(s.metrics && s.metrics.mae_m)} m</option>`).join('');

  const left = $('#left-layer'), right = $('#right-layer');
  const opts = LAYERS.map(l => `<option value="${l.id}">${l.name}</option>`).join('');
  left.innerHTML = opts; right.innerHTML = opts;
  left.value = 'dsm_pred.png';
  right.value = 'dsm_truth.png';

  const update = () => {
    const s = scenes[+sel.value];
    const dir = `results/seed${s.seed}/preview`;
    $('#img-left').src = `${dir}/${left.value}`;
    $('#img-right').src = `${dir}/${right.value}`;
    const nameOf = id => (LAYERS.find(l => l.id === id) || {}).name || id;
    $('#img-left').alt = `${nameOf(left.value)}, scene seed ${s.seed}`;
    $('#img-right').alt = `${nameOf(right.value)}, scene seed ${s.seed}`;
    $('#lbl-left').textContent = nameOf(left.value);
    $('#lbl-right').textContent = nameOf(right.value);
    const noteOf = id => (LAYERS.find(l => l.id === id) || {}).note || '';
    $('#layer-legend').innerHTML =
      `<strong>${nameOf(left.value)}</strong> — ${noteOf(left.value)}<br>` +
      `<strong>${nameOf(right.value)}</strong> — ${noteOf(right.value)}`;
    renderSceneFacts(s);
  };

  [sel, left, right].forEach(el => el.addEventListener('change', update));
  update();
  wireCompare();
}

function renderSceneFacts(s) {
  const m = s.metrics || {}, t = s.scene_truth || {}, a = s.anchors || {};
  const facts = [
    ['MAE', `${fmt(m.mae_m)} m`],
    ['RMSE', `${fmt(m.rmse_m)} m`],
    ['bias', `${fmt(m.bias_m)} m`],
    ['tier', s.tier || '–'],
    ['anchors used', `${s.anchors_used ?? '–'}`],
    ['rejected', `${s.anchors_rejected ?? '–'}`],
    ['shadow anchors', `${a.shadow ?? '–'}`],
    ['sun elevation', t.sun_elevation_deg ? `${fmt(t.sun_elevation_deg, 1)}°` : '–'],
    ['true relief', (t.elev_min_m && t.elev_max_m) ? `${fmt(t.elev_max_m - t.elev_min_m, 0)} m` : '–'],
    ['tallest object', t.max_object_height_m ? `${fmt(t.max_object_height_m, 1)} m` : '–'],
    ['wall time', s.wall_s ? `${fmt(s.wall_s, 0)} s` : '–'],
  ];
  $('#scene-facts').innerHTML = facts.map(([k, v]) =>
    `<div class="fact"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}

function wireCompare() {
  const box = $('#compare'), clip = $('#clip'), handle = $('#handle');
  let dragging = false;

  const set = pct => {
    const p = Math.max(0, Math.min(100, pct));
    clip.style.clipPath = `inset(0 ${100 - p}% 0 0)`;
    handle.style.left = `${p}%`;
    handle.setAttribute('aria-valuenow', Math.round(p));
  };
  const fromEvent = e => {
    const rect = box.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    return (x / rect.width) * 100;
  };

  box.addEventListener('pointerdown', e => { dragging = true; box.setPointerCapture(e.pointerId); set(fromEvent(e)); });
  box.addEventListener('pointermove', e => { if (dragging) set(fromEvent(e)); });
  box.addEventListener('pointerup', () => { dragging = false; });
  box.addEventListener('pointercancel', () => { dragging = false; });
  handle.addEventListener('keydown', e => {
    const cur = +handle.getAttribute('aria-valuenow');
    if (e.key === 'ArrowLeft') { set(cur - 4); e.preventDefault(); }
    if (e.key === 'ArrowRight') { set(cur + 4); e.preventDefault(); }
  });
  set(50);
}

// ── footer + repo links ──────────────────────────────────────────────────────
function renderFooter(study) {
  const env = study.environment || {}, cfg = study.config || {};
  $('#footer-meta').textContent =
    `study generated ${env.timestamp_utc || '?'} · ${env.platform || ''} · ` +
    `torch ${env.torch || '–'} · rasterio ${env.rasterio || '–'} · GDAL ${env.gdal || '–'} · ` +
    `backbone ${cfg.backbone || '?'} · ${fmt(study.wall_s, 0)} s wall`;
}

function wireStaticUI() {
  // Repo + Colab links inferred from the Pages URL, so a fork needs no edits.
  const host = location.hostname, parts = location.pathname.split('/').filter(Boolean);
  let repo = 'https://github.com';
  if (host.endsWith('github.io')) {
    const user = host.split('.')[0];
    repo = parts.length ? `https://github.com/${user}/${parts[0]}` : `https://github.com/${user}`;
  }
  $('#repo-link').href = repo;
  const nb = repo.replace('https://github.com/', '') + '/blob/main/notebooks/ayama_gpu_harness.ipynb';
  $('#colab-btn').href = `https://colab.research.google.com/github/${nb}`;

  $$('.repro-tabs button').forEach(btn => btn.addEventListener('click', () => {
    $$('.repro-tabs button').forEach(b => b.classList.toggle('on', b === btn));
    $$('.tabpane').forEach(p => p.classList.toggle('on', p.dataset.pane === btn.dataset.tab));
  }));

  document.addEventListener('click', e => {
    const btn = e.target.closest('.copy');
    if (!btn) return;
    const code = btn.parentElement.querySelector('code');
    navigator.clipboard.writeText(code.textContent.trim()).then(() => {
      const was = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = was; }, 1400);
    });
  });

  wireWizard();
}

// ── "what will my image get" ─────────────────────────────────────────────────
function wireWizard() {
  const state = { geo: 'yes', dem: 'file', extra: 'none' };

  $$('.opts').forEach(group => group.addEventListener('click', e => {
    const btn = e.target.closest('.opt');
    if (!btn) return;
    $$('.opt', group).forEach(b => b.classList.toggle('on', b === btn));
    state[group.dataset.q] = btn.dataset.v;
    updateWizard(state);
  }));
  updateWizard(state);
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
                 `    --backbone dav2-vitl --device cuda --batch 0 \\`];
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

// ── no results yet ───────────────────────────────────────────────────────────
function showMissingResults(err) {
  $('#hero-metrics').innerHTML = `
    <div class="error-note" style="grid-column:1/-1">
      <strong>No results found.</strong> This page renders <code>results/study.json</code>,
      which is produced by <code>python -m ayama.cli study --out results</code>.
      Run it, commit <code>results/</code>, and every number and image here fills in.
      <br><span style="font-family:var(--mono);font-size:.8rem">(${err})</span>
    </div>`;
  ['headline-table', 'class-chart', 'coverage-chart', 'ablation-chart', 'sun-chart',
   'lambda-chart', 'bench-table'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<div class="loading">Waiting on results/study.json</div>';
  });
  const cmp = $('#compare');
  if (cmp) cmp.innerHTML = '<div class="loading" style="padding:2rem">Waiting on results/study.json</div>';
}
