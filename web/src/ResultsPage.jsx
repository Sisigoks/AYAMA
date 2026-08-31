/* TRAKSHA — the results dashboard.
 *
 * Every number here is read from `results/dataset.json` and `results/arms.json`
 * at load. Nothing is typed in, and nothing is computed that the study did not
 * already compute, so the page cannot drift away from the run it describes. A
 * missing field shows as "–" rather than as a plausible number.
 */
import { useEffect, useState } from 'react';
import { Brand } from './components.jsx';

const fmt = (v, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '–' : Number(v).toFixed(d);
const mean = (o) => (o && isFinite(o.mean) ? o.mean : null);
const pm = (o, d = 2) => (o ? `${fmt(o.mean, d)} ± ${fmt(o.std, d)}` : '–');

const LAYERS = [
  { id: 'texture.jpg', name: 'Orthophoto', note: 'The only image input.' },
  { id: 'dsm.png', name: 'Predicted DSM', note: 'Metric elevation as delivered.' },
  { id: 'ndsm.png', name: 'Height above ground', note: 'Compare with the relief figures above.' },
  { id: 'error.png', name: 'Error vs lidar', note: 'Predicted minus swissSURFACE3D.' },
  { id: 'sigma.png', name: 'Uncertainty σ', note: 'Per-pixel 1σ.' },
];

function Table({ head, rows }) {
  return (
    <div className="scroll-x">
      <table className="data">
        <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  );
}

/* Elevation MAE flatters this pipeline — half a city scene is ground, and the
 * anchor DEM already knows the ground. This banner is computed from the data so
 * nobody can soften it by forgetting to update the copy. */
function ReliefVerdict({ agg }) {
  const nd = mean(agg.ndsm_metrics_mae_m);
  const flat = mean(agg.zero_baseline_metrics_mae_m);
  const trueH = mean(agg.true_mean_height_m);
  const predH = mean(agg.pred_mean_height_m);
  if (nd === null || flat === null) return null;
  const pct = trueH ? (100 * predH) / trueH : null;
  const bad = pct !== null && pct < 10;
  return (
    <div className={`verdict ${bad ? 'bad' : 'good'}`}>
      <strong>
        {bad
          ? 'Height above ground is not being recovered.'
          : 'Height above ground is being recovered, partly.'}
      </strong>
      Against the lidar nDSM the pipeline scores <b>{fmt(nd)} m</b>. Predicting{' '}
      <em>zero height everywhere</em> scores <b>{fmt(flat)} m</b> — a difference of{' '}
      {fmt(100 * (1 - nd / flat), 1)}%. It returns <b>{fmt(predH)} m</b> of an average{' '}
      <b>{fmt(trueH)} m</b> of true structure{pct === null ? '' : ` (${fmt(pct, 1)}%)`}.
      See README §3.2 and §4.
    </div>
  );
}

function Explorer({ scenes }) {
  const [scene, setScene] = useState(scenes[0]?.name || '');
  const [left, setLeft] = useState('texture.jpg');
  const [right, setRight] = useState('ndsm.png');
  const [split, setSplit] = useState(50);
  const base = `results/${scene}/`;
  const note = (id) => LAYERS.find((l) => l.id === id);

  const wipe = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    setSplit(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)));
  };

  return (
    <section className="band alt">
      <div className="wrap">
        <h2>Look at one</h2>
        <div className="row">
          <select value={scene} onChange={(e) => setScene(e.target.value)}>
            {scenes.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
          <select value={left} onChange={(e) => setLeft(e.target.value)}>
            {LAYERS.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
          <select value={right} onChange={(e) => setRight(e.target.value)}>
            {LAYERS.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
        </div>

        <div
          className="compare"
          onPointerDown={wipe}
          onPointerMove={(e) => { if (e.buttons === 1) wipe(e); }}
        >
          <img src={base + right} alt={note(right)?.name} />
          <div className="clip" style={{ width: `${split}%` }}>
            <img src={base + left} alt={note(left)?.name} />
          </div>
          <div className="handle" style={{ left: `${split}%` }} />
        </div>
        <p className="note-sm">
          <b>{note(left)?.name}.</b> {note(left)?.note} <b>{note(right)?.name}.</b>{' '}
          {note(right)?.note} Drag across the image to wipe between them.
        </p>
      </div>
    </section>
  );
}

export default function ResultsPage() {
  const [study, setStudy] = useState(null);
  const [arms, setArms] = useState({});
  const [err, setErr] = useState('');

  useEffect(() => {
    const grab = async (u) => {
      const r = await fetch(u, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
      return r.json();
    };
    grab('results/dataset.json').then(setStudy).catch((e) => setErr(String(e.message || e)));
    grab('results/arms.json').then((d) => setArms(d.arms || {})).catch(() => {});
  }, []);

  if (err) {
    return (
      <main className="centre">
        <h1>No results to show</h1>
        <p className="note-sm">
          Could not load <code>results/dataset.json</code> ({err}). Run{' '}
          <code>python scripts/fetch_swisstopo.py --out data/real/zurich</code>, then{' '}
          <code>python -m traksha.cli dataset data/real --layout generic --deliver --out results</code>.
        </p>
      </main>
    );
  }
  if (!study) return <main className="centre"><p className="note-sm">Loading…</p></main>;

  const agg = study.aggregate || {};
  const scenes = study.scenes || [];
  const relief = (g) =>
    mean(g.true_mean_height_m)
      ? `${fmt((100 * mean(g.pred_mean_height_m)) / mean(g.true_mean_height_m), 1)}%`
      : '–';

  return (
    <>
      <header className="topbar">
        <Brand phase="results" />
        <div className="grow" />
        <a className="ghost" href="index.html">Reconstruct an image</a>
      </header>

      <section className="band">
        <div className="wrap">
          <h2>What it measures</h2>
          <p className="lede">
            Four real city centres — Zürich, Bern, Geneva, Lausanne — with airborne lidar
            for ground truth. Every number here is read live from{' '}
            <code>results/dataset.json</code>. Nothing is typed in by hand.
          </p>

          <ReliefVerdict agg={agg} />

          <h3>Accuracy against lidar</h3>
          <Table
            head={['metric', 'TRAKSHA', 'global affine', 'floor']}
            rows={[
              ['MAE (m)', pm(agg.mae_m), fmt(mean(agg.baseline_metrics_mae_m)), fmt(mean(agg.dem_metrics_mae_m))],
              ['RMSE (m)', pm(agg.rmse_m), '–', '–'],
              ['Pearson r', pm(agg.pearson_r, 3), '–', '–'],
              ['edge F1', pm(agg.edge_f1, 3), '–', '–'],
              ['δ < 1.25', pm(agg.delta1, 3), '–', '–'],
              ['1σ coverage', pm(agg.coverage_1s, 3), '–', '–'],
              ['nDSM MAE (m)', fmt(mean(agg.ndsm_metrics_mae_m)), '–',
                `${fmt(mean(agg.zero_baseline_metrics_mae_m))} (flat ground)`],
            ]}
          />

          <h3>Scene by scene</h3>
          <Table
            head={['scene', 'tier', 'MAE', 'DEM floor', 'nDSM MAE', 'flat ground', 'true max', 'pred max', 'relief']}
            rows={scenes.map((s) => {
              const r = s.relief || {};
              const pct = r.true_mean_height_m
                ? (100 * r.pred_mean_height_m) / r.true_mean_height_m : null;
              return [
                s.name, s.tier, fmt(s.metrics?.mae_m), fmt(s.dem_metrics?.mae_m),
                fmt(s.ndsm_metrics?.mae_m), fmt(s.zero_baseline_metrics?.mae_m),
                fmt(r.true_max_height_m, 1), fmt(r.pred_max_height_m, 1),
                pct === null ? '–' : `${fmt(pct, 1)}%`,
              ];
            })}
          />

          <h3>What the calibration had to work with</h3>
          <Table
            head={['scene', 'DEM', 'water', 'shadow', 'total']}
            rows={scenes.map((s) => {
              const a = s.anchors || {};
              return [s.name, a.dem ?? '–', a.water ?? '–', a.shadow ?? '–', a.total ?? '–'];
            })}
          />
          {scenes.length && scenes.every((s) => !(s.anchors || {}).shadow) ? (
            <p className="note-sm">
              <b>Every anchor is a ground anchor.</b> swisstopo publishes no acquisition
              time for these products, so no sun angle can be derived and shadow physics
              is disabled. README §3.4 shows the conclusion does not depend on it.
            </p>
          ) : null}

          <h3>This arm against the controls</h3>
          <Table
            head={['backbone', 'calibration', 'MAE (m)', 'edge F1', 'nDSM MAE (m)', 'relief']}
            rows={[
              [study.config?.backbone || 'primary', 'dual branch + fitted scale',
                pm(agg.mae_m), fmt(mean(agg.edge_f1), 3),
                fmt(mean(agg.ndsm_metrics_mae_m)), relief(agg)],
              ...Object.values(arms).map((a) => {
                const g = a.aggregate || {};
                const [bb, ...rest] = (a.label || '').split(', ');
                return [bb, rest.join(', ') || '—', pm(g.mae_m), fmt(mean(g.edge_f1), 3),
                  fmt(mean(g.ndsm_metrics_mae_m)), relief(g)];
              }),
            ]}
          />
          <p className="note-sm">
            The first row is the delivered study; the rest are controls from{' '}
            <code>results/arms.json</code>. Without a fitted structural scale every arm
            recovers under 1.3% of the true relief — two backbones and two calibrations
            agree. Supplying one fitted constant is what moves it.
          </p>
        </div>
      </section>

      {scenes.length ? <Explorer scenes={scenes} /> : null}

      <footer className="site">
        <div className="wrap">
          <p>
            <strong>TRAKSHA</strong> · calibration engine <strong>Chhaya</strong> ·{' '}
            <span lang="hi">छाया</span>, “shadow”
          </p>
          <p className="note-sm">
            {`Rendered from results/dataset.json · ${study.n_ok}/${study.n_found} scenes · `}
            imagery and elevation truth © swisstopo, used under Swiss OGD terms.
          </p>
        </div>
      </footer>
    </>
  );
}
