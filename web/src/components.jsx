/* TRAKSHA front end — components.
 *
 * The WebGL renderer is the one thing React does not own. It lives in
 * `renderer.js`, keeps its own camera and draw loop, and is reached through a
 * ref — the supported escape hatch for imperative graphics, not a workaround.
 * Routing pointer moves at sixty a second through React state would be both
 * slower and wrong; the component learns the result through `onPick`.
 */
import { useState, useEffect, useRef } from 'react';

export const LAYERS = [
  { id: 'texture', name: 'Orthophoto', note: 'The image the reconstruction was made from.' },
  { id: 'dsm', name: 'Elevation', note: 'Metric surface, terrain colour ramp.' },
  { id: 'ndsm', name: 'Height above ground', note: 'What a planner asks for: how tall is that.' },
  { id: 'sigma', name: 'Uncertainty σ', note: 'Per-pixel 1σ. Bright where anchors are sparse.' },
  { id: 'error', name: 'Error vs reference', note: 'Only present when a reference DSM was supplied.' },
];

const fmt = (v, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '–' : Number(v).toFixed(d);

export function Brand({ phase }) {
  return (
    <div className="brand">
      <span className="mark" aria-hidden="true" />
      <span className="name">TRAKSHA</span>
      {phase ? <span className="phase">{phase}</span> : null}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="kv-row">
      <span className="kv-k">{k}</span>
      <span className="kv-v mono">{v}</span>
    </div>
  );
}

function Panel({ title, note, children }) {
  return (
    <section className="ctl">
      <h2>{title}</h2>
      {note ? <p className="note-sm">{note}</p> : null}
      {children}
    </section>
  );
}

/* The tileset carries its own verdict on the surface inside it. Showing it is
 * not decoration: a 3D view of a flattened city that does not say so is worse
 * than no 3D view. */
export function Notes({ notes }) {
  if (!notes || !notes.length) return null;
  return (
    <div className="notes">
      {notes.map((n, i) => (
        <div key={i} className={'note ' + (n.level || 'info')}>
          <b>{n.level === 'critical' ? 'Defect' : n.level === 'warning' ? 'Warning' : 'Note'}</b>
          <span>{n.text}</span>
        </div>
      ))}
    </div>
  );
}

export function Upload({ onStarted, backbones }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [opts, setOpts] = useState({ backbone: 'dav2-vits', chip: 512, bootstrap: 8, mesh: true });
  const drop = useRef(null);

  useEffect(() => {
    const el = drop.current;
    if (!el) return undefined;
    const over = (e) => { e.preventDefault(); el.classList.add('over'); };
    const out = () => el.classList.remove('over');
    const dropped = (e) => {
      e.preventDefault(); out();
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) { setErr(''); setFile(f); }
    };
    el.addEventListener('dragover', over);
    el.addEventListener('dragleave', out);
    el.addEventListener('drop', dropped);
    return () => {
      el.removeEventListener('dragover', over);
      el.removeEventListener('dragleave', out);
      el.removeEventListener('drop', dropped);
    };
  }, []);

  async function submit(e) {
    e.preventDefault();
    if (!file || busy) return;
    setBusy(true); setErr('');
    try {
      const fd = new FormData();
      fd.append('image', file);
      Object.entries(opts).forEach(([k, v]) => fd.append(k, String(v)));
      const r = await fetch('api/jobs', { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`server said ${r.status}: ${(await r.text()).slice(0, 200)}`);
      onStarted((await r.json()).id);
    } catch (e2) {
      setErr(String(e2.message || e2));
      setBusy(false);
    }
  }

  return (
    <form className="upload" onSubmit={submit}>
      <div className="drop" ref={drop}>
        <input
          id="file" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(e) => { const f = e.target.files && e.target.files[0]; if (f) { setErr(''); setFile(f); } }}
        />
        <label htmlFor="file">
          <b>{file ? file.name : 'Choose an image, or drop one here'}</b>
          <span>
            {file
              ? `${(file.size / 1e6).toFixed(1)} MB`
              : 'A GeoTIFF keeps its CRS and scale. PNG or JPG works too, at Tier C.'}
          </span>
        </label>
      </div>

      <div className="opts">
        <label>
          Backbone
          <select value={opts.backbone} onChange={(e) => setOpts({ ...opts, backbone: e.target.value })}>
            {(backbones || ['dav2-vits', 'dav2-vitb', 'dav2-vitl']).map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </label>
        <label>
          Chip
          <select value={opts.chip} onChange={(e) => setOpts({ ...opts, chip: Number(e.target.value) })}>
            {[384, 512, 1024].map((c) => <option key={c} value={c}>{c} px</option>)}
          </select>
        </label>
        <label className="chk">
          <input
            type="checkbox" checked={opts.mesh}
            onChange={(e) => setOpts({ ...opts, mesh: e.target.checked })}
          />
          {' '}also build a downloadable OBJ
        </label>
      </div>

      {err ? <div className="note critical"><b>Upload failed</b><span>{err}</span></div> : null}

      <button className="primary" disabled={!file || busy}>
        {busy ? 'Starting…' : 'Reconstruct'}
      </button>
    </form>
  );
}

/* The reconstruction screen.
 *
 * Everything drawn here is read from the job record the backend sends; nothing
 * is interpolated, estimated or animated forward on a timer. The overall figure
 * is weighted by measured stage duration (traksha/api/phases.py), which matters
 * because depth is 80% of a run: with equal weights the bar would reach 10% and
 * then stand still for two and a half minutes, and a bar that stands still is
 * indistinguishable from one that has died.
 */
const PHASE_ICON = { done: '✓', failed: '×', skipped: '–' };

function Bar({ value, tone }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  return (
    <div className={`bar${tone ? ' ' + tone : ''}`}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Progress({ job, onCancel }) {
  const phases = job.phases || [];
  const failed = job.status === 'failed';
  const overall = failed ? job.progress || 0 : (job.status === 'done' ? 1 : job.progress || 0);

  return (
    <div className="progress">
      <h2>Reconstructing</h2>
      <p className="note-sm">
        The pipeline runs server-side, one scene at a time. A 1024 px image is a
        couple of minutes on CPU, nearly all of it depth.
      </p>

      <div className="overall">
        <div className="pct">{Math.round(overall * 100)}%</div>
        <Bar value={overall} tone={failed ? 'bad' : null} />
      </div>

      {job.phase && !failed ? (
        <div className="current">
          <div className="hd">
            <span className="nm">{job.phase}</span>
            <span className="dt">{Math.round((job.phase_progress || 0) * 100)}%</span>
          </div>
          <Bar value={job.phase_progress} />
          {job.message ? <p className="msg">{job.message}</p> : null}
        </div>
      ) : null}

      <ol className="stages">
        {phases.map((p) => (
          <li key={p.name} className={p.status}>
            <span className="dot" />
            <span className="nm">{p.name}</span>
            {p.status === 'running' && p.message ? (
              <span className="dt">{p.message}</span>
            ) : (
              <span className="dt">
                {PHASE_ICON[p.status] || ''}
                {p.duration_s != null && p.status === 'done' ? ` ${p.duration_s}s` : ''}
              </span>
            )}
          </li>
        ))}
      </ol>

      {job.error ? (
        <div className="note critical">
          <b>Failed{job.phase ? ` during ${job.phase}` : ''}</b>
          <span>{job.error}</span>
        </div>
      ) : null}
      <button className="ghost" onClick={onCancel}>Start over</button>
    </div>
  );
}


/* Shown when nothing answered on /api/health.
 *
 * The viewer falls back to the tileset committed at web/data, which is correct
 * on GitHub Pages where there is no backend and correct in development where
 * the service simply has not been started. What is not correct is looking
 * identical in both cases: a reader who wants to upload an image needs to know
 * that the scene on screen is a bundled demo and that reconstruction needs a
 * process nobody started.
 */
function OfflineNote() {
  return (
    <div className="note warn">
      <b>Demo scene — no pipeline service</b>
      <span>
        Nothing answered on <code>/api/health</code>, so this is the tileset
        committed at <code>web/data</code>, not a reconstruction. Uploading an
        image needs the Python service: run <code>npm run dev</code>, which
        starts both halves, or{' '}
        <code>python -m traksha.cli serve --port 8000</code> beside the UI.
      </span>
    </div>
  );
}

export function SidePanel(p) {
  const { manifest: m, base, layer, setLayer, lod, setLod, exagg, setExagg,
          shade, setShade, fog, setFog, wire, setWire, flying, onFly, readout,
          offline, structural, setStructural, hasStructural } = p;
  const g = m.grid || {};
  const layers = LAYERS.filter((l) => m.layers && m.layers[l.id]);
  const active = LAYERS.find((l) => l.id === layer);
  const met = m.metrics || {};
  const prov = m.provenance || {};
  const mesh = m.mesh;
  const st = (k) => (m.layers && m.layers[k] ? m.layers[k].stats : null);

  return (
    <aside className="side">
      {offline ? <OfflineNote /> : null}
      <Notes notes={m.notes} />

      <Panel title="Layer" note={active ? active.note : null}>
        <div className="layer-btns">
          {layers.map((l) => (
            <button key={l.id} className={l.id === layer ? 'on' : ''} onClick={() => setLayer(l.id)}>
              {l.name}
            </button>
          ))}
        </div>
      </Panel>

      {hasStructural ? (
        <Panel
          title="Geometry"
          note={structural
            ? 'Buildings are separate solids with real vertical facades, cut along the segmentation. Terrain is holed under each footprint.'
            : 'A height field: one elevation per ground position. It cannot represent a wall, so a facade is a steep ramp welded to the ground.'}
        >
          <div className="layer-btns">
            <button className={structural ? '' : 'on'} onClick={() => setStructural(false)}>
              Height field
            </button>
            <button className={structural ? 'on' : ''} onClick={() => setStructural(true)}>
              Structural mesh
            </button>
          </div>
          {structural && m.mesh && m.mesh.structural ? (
            <div className="kv">
              <Row k="buildings" v={m.mesh.structural.buildings ?? '–'} />
              <Row k="separated" v={
                m.mesh.structural.quality
                  ? `${Math.round((m.mesh.structural.quality.separation_score || 0) * 100)}%`
                  : '–'} />
              <Row k="facade area" v={
                m.mesh.structural.quality
                  ? `${((m.mesh.structural.quality.wall_area_frac || 0) * 100).toFixed(1)}%`
                  : '–'} />
            </div>
          ) : null}
        </Panel>
      ) : null}

      <Panel title="Vertical exaggeration" note="Scales elevation only. Horizontal distances stay true.">
        <div className="row">
          <input type="range" min="1" max="60" step="1" value={exagg}
                 onChange={(e) => setExagg(Number(e.target.value))} />
          <output className="mono">{exagg}×</output>
        </div>
      </Panel>

      <Panel title="Detail">
        <div className="row">
          <select value={lod} onChange={(e) => setLod(Number(e.target.value))}>
            {(m.lods || []).map((l, i) => (
              <option key={i} value={i}>{`LOD ${l.lod} · ${l.width}×${l.height}`}</option>
            ))}
          </select>
        </div>
        <label className="chk">
          <input type="checkbox" checked={shade} onChange={(e) => setShade(e.target.checked)} />
          {' '}shade from normals
        </label>
        <label className="chk">
          <input type="checkbox" checked={fog} onChange={(e) => setFog(e.target.checked)} />
          {' '}aerial perspective
        </label>
        <label className="chk">
          <input type="checkbox" checked={wire} onChange={(e) => setWire(e.target.checked)} />
          {' '}wireframe
        </label>
      </Panel>

      <Panel
        title="Flythrough"
        note="Descends to eye level and crosses the scene low, where parallax between near and far structure is strongest. Press F, or touch the view, to stop."
      >
        <button className={'wide' + (flying ? ' on' : '')} onClick={onFly}>
          {flying ? 'Stop' : 'Fly through'}
        </button>
      </Panel>

      <Panel title="This surface">
        <div className="kv">
          <Row k="GSD" v={`${fmt(g.gsd_m, 3)} m`} />
          <Row k="extent" v={`${(g.extent_m || []).map((x) => Math.round(x)).join(' × ')} m`} />
          <Row k="CRS" v={m.crs || 'none'} />
          <Row k="elevation" v={st('dsm') ? `${fmt(st('dsm').min, 1)} … ${fmt(st('dsm').max, 1)} m` : '–'} />
          <Row k="above ground" v={st('ndsm') ? `${fmt(st('ndsm').min)} … ${fmt(st('ndsm').max)} m` : '–'} />
          <Row k="mean 1σ" v={st('sigma') ? `${fmt(st('sigma').mean)} m` : '–'} />
        </div>
      </Panel>

      <Panel title="Phase 2 result">
        {Object.keys(met).length ? (
          <div className="kv">
            <Row k="MAE" v={`${fmt(met.mae_m)} m`} />
            <Row k="RMSE" v={`${fmt(met.rmse_m)} m`} />
            <Row k="bias" v={`${fmt(met.bias_m)} m`} />
            <Row k="Pearson r" v={fmt(met.pearson_r, 3)} />
            <Row k="edge F1" v={fmt(met.edge_f1, 3)} />
            <Row k="1σ coverage" v={fmt(met.coverage_1s, 3)} />
            <Row k="ECE" v={`${fmt(met.ece_m)} m`} />
          </div>
        ) : (
          <p className="note-sm">
            This run was not validated against a reference DSM, so there are no metrics.
          </p>
        )}
      </Panel>

      <Panel title="Provenance">
        <div className="kv">
          <Row k="backbone" v={prov.backbone || '–'} />
          <Row k="semantics" v={prov.segmentation || '–'} />
          {/* The structural segmentation, which is a different thing from the
              five-class semantics above: SAM 2 supplies instances, not classes. */}
          <Row k="instances" v={
            !prov.instances || prov.instances === 'off'
              ? 'off'
              : `${prov.instances.replace('sam2:', '')} · ${prov.instance_count ?? '–'}`} />
          <Row k="DEM" v={prov.dem || 'none'} />
          <Row k="tier" v={prov.tier || '–'} />
        </div>
      </Panel>

      <Panel title="Download">
        <ul className="dl">
          <li><a href={base + 'tileset.json'} download>tileset.json</a></li>
          {mesh ? (
            <>
              <li>
                <a href={base + mesh.obj} download>
                  {`surface.obj — ${(mesh.triangles || 0).toLocaleString()} triangles`}
                </a>
              </li>
              {mesh.mtl ? <li><a href={base + mesh.mtl} download>surface.mtl</a></li> : null}
              {mesh.texture ? <li><a href={base + mesh.texture} download>surface.jpg</a></li> : null}
              {/* The structural rebuild at full resolution. Named with its size
                  because it is two orders of magnitude larger than the tileset
                  and a reader should know that before clicking. */}
              {/* The refined model, when a GPU has painted its facades. Named
                  as synthesised in the link itself, because a reader clicking
                  it is downloading walls no camera ever saw. */}
              {mesh.structural && mesh.structural.refined ? (
                <li>
                  <a href={base + mesh.structural.refined.obj} download>
                    {`structural_refined.obj — ${mesh.structural.refined.buildings_refined || 0}`}
                    {` of ${mesh.structural.refined.buildings_total || 0} buildings with `}
                    {mesh.structural.refined.synthesised
                      ? 'SYNTHESISED walls' : 'measured texture only'}
                  </a>
                </li>
              ) : null}
              {mesh.structural ? (
                <li>
                  <a href={base + mesh.structural.obj} download>
                    {`structural.obj — ${(mesh.structural.triangles || 0).toLocaleString()} triangles, `}
                    {`${mesh.structural.buildings || 0} buildings`}
                    {mesh.structural.bytes
                      ? ` (${Math.round(mesh.structural.bytes / 1e6)} MB)` : ''}
                  </a>
                </li>
              ) : null}
            </>
          ) : (
            <li className="note-sm">No OBJ was requested for this run.</li>
          )}
        </ul>
      </Panel>

      {readout ? (
        <Panel title="Cursor">
          <div className="kv">
            <Row k="elevation" v={`${fmt(readout.elevation)} m`} />
            <Row k="above ground" v={readout.ndsm === null ? '–' : `${fmt(readout.ndsm)} m`} />
            <Row k="1σ" v={readout.sigma === null ? '–' : `${fmt(readout.sigma)} m`} />
          </div>
        </Panel>
      ) : null}
    </aside>
  );
}
