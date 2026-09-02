/* TRAKSHA — the app shell.
 *
 * Three screens behind one piece of state: upload, progress, viewer. The job id
 * lives in the URL (`?job=…`) so a reconstruction survives a reload and can be
 * handed to someone else.
 *
 * Progress arrives over SSE with polling underneath. The fallback is not
 * defensive habit: this gets run behind Colab's port proxy, which buffers
 * `text/event-stream`, and without it the page sits on "queued" until the job
 * finishes and looks broken the entire time.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { createViewer } from './renderer.js';
import { Brand, Upload, Progress, SidePanel } from './components.jsx';

const jobBase = (id) => `api/jobs/${id}/tiles/`;

function useJob(id) {
  const [job, setJob] = useState(null);
  useEffect(() => {
    if (!id) return undefined;
    let live = true;
    let es = null;

    /* Every frame - streamed or polled - is the whole job record, so applying
     * one is a replacement rather than a merge. The two guards are what keep a
     * late poll response from undoing a fresher stream frame: a finished job
     * never un-finishes, and the overall figure never goes backwards. A bar
     * that retreats tells the reader the number is invented. */
    const apply = (d) => {
      if (!live || !d) return;
      setJob((prev) => {
        if (!prev) return d;
        const over = (j) => (j.status === 'done' || j.status === 'failed');
        if (over(prev) && !over(d)) return prev;
        if (!over(d) && (d.progress || 0) < (prev.progress || 0)) return prev;
        return d;
      });
    };

    const poll = async () => {
      try {
        const r = await fetch(`api/jobs/${id}`, { cache: 'no-store' });
        if (r.ok) apply(await r.json());
      } catch { /* transient; the interval retries */ }
    };

    /* addEventListener, not onmessage. The server names its frames, and
     * `onmessage` fires only for unnamed ones - which is why this screen used
     * to render nothing at all while the server sent every event. The names
     * are SSE_EVENTS in traksha/api/server.py and a test pins them together. */
    try {
      es = new EventSource(`api/jobs/${id}/events`);
      const onFrame = (ev) => {
        let d;
        try { d = JSON.parse(ev.data); } catch { return; }
        apply(d);
      };
      es.addEventListener('progress', onFrame);
      es.addEventListener('end', onFrame);
      es.onerror = () => { /* the poll carries it */ };
    } catch { es = null; }

    poll();
    const t = setInterval(poll, 4000);
    return () => { live = false; clearInterval(t); if (es) es.close(); };
  }, [id]);
  return job;
}

function Viewer({ manifest, base, layer, lod, exagg, shade, fog, wire,
                  onStats, onReadout, onFlying, flyNonce,
                  structural, onStructuralReady, onStructuralFallback }) {
  const ref = useRef(null);
  const api = useRef(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (!ref.current) return undefined;
    const v = createViewer(ref.current, { onPick: onReadout, onStats, onTour: onFlying });
    if (!v) { setErr('This browser reports no WebGL context.'); return undefined; }
    api.current = v;
    v.load(manifest, base)
      .then(() => onStructuralReady?.(v.hasStructural()))
      .catch((e) => setErr(String(e.message || e)));
    return () => { v.dispose(); api.current = null; };
  }, [manifest, base]);

  useEffect(() => { api.current?.setLayer(layer); }, [layer]);
  useEffect(() => { api.current?.setLod(lod); }, [lod]);
  useEffect(() => { api.current?.setExagg(exagg); }, [exagg]);
  useEffect(() => { api.current?.setShade(shade); }, [shade]);
  useEffect(() => { api.current?.setFog(fog); }, [fog]);
  useEffect(() => { api.current?.setWire(wire); }, [wire]);
  useEffect(() => {
    // Several megabytes, fetched on first use. A failure falls back to the
    // height field and says why - the viewer keeps working, and the panel does
    // not claim a geometry it is not drawing. Silently showing one while the
    // control says the other is the kind of lie this project avoids.
    const v = api.current;
    if (!v) return;
    v.setStructural(structural)
      .then((on) => {
        if (structural && !on) {
          onStructuralFallback?.('this run has no structural mesh');
        }
      })
      .catch((e) => onStructuralFallback?.(String(e.message || e)));
  }, [structural]);
  useEffect(() => {
    if (!flyNonce || !api.current) return;
    api.current.flying() ? api.current.stopFly() : api.current.fly();
  }, [flyNonce]);

  useEffect(() => {
    const key = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        api.current?.flying() ? api.current.stopFly() : api.current?.fly();
      }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, []);

  return (
    <div className="canvas-wrap">
      <canvas id="gl" ref={ref} />
      {err ? <div className="gl-error"><b>Cannot draw</b><span>{err}</span></div> : null}
      <div className="hint">drag to orbit · shift-drag to pan · scroll to zoom · F to fly</div>
    </div>
  );
}

export default function App() {
  const [jobId, setJobId] = useState(() => new URLSearchParams(location.search).get('job') || '');
  const [manifest, setManifest] = useState(null);
  const [base, setBase] = useState('');
  const [layer, setLayer] = useState('texture');
  const [lod, setLod] = useState(0);
  const [exagg, setExagg] = useState(1);
  const [shade, setShade] = useState(true);
  const [fog, setFog] = useState(true);
  const [wire, setWire] = useState(false);
  // The height field or the structural rebuild. Two different meshes, not
  // two styles: a height field has one z per (x, y) and cannot show a wall.
  // The structural rebuild is the default: it is the same calibrated heights as
  // the height field, with the buildings actually separated and their walls
  // vertical. A reader arriving at the viewer should see the better geometry
  // first and be able to switch back, not the other way round.
  const [structural, setStructural] = useState(true);
  const [hasStructural, setHasStructural] = useState(false);
  const [flying, setFlying] = useState(false);
  const [flyNonce, setFlyNonce] = useState(0);
  const [stats, setStats] = useState({ triangles: 0, lod: null });
  const [readout, setReadout] = useState(null);
  const [backbones, setBackbones] = useState(null);
  // null while unknown, then true/false. Drives the one thing a viewer with no
  // service behind it must not do: look identical to one that has a service.
  const [service, setService] = useState(null);
  // Why the structural mesh is not showing, when it was asked for and could not
  // be drawn. Silence would leave the panel claiming a geometry it is not using.
  const [structuralNote, setStructuralNote] = useState('');

  const job = useJob(manifest ? '' : jobId);

  useEffect(() => {
    fetch('api/health')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d) => { setService(true); if (d.backbones) setBackbones(d.backbones); })
      .catch(() => setService(false));
  }, []);

  /* Where a tileset comes from depends on how the page is being served, and
   * getting it wrong is why the upload form once never appeared: the app found
   * the bundled demo at `data/` and went straight to the viewer.
   *
   *   ?job=…              a reconstruction — load it when the job finishes
   *   job service present the landing screen is the upload form
   *   static hosting      no API, so show the demo tileset at `data/`
   */
  useEffect(() => {
    let live = true;
    const tryLoad = async (b) => {
      try {
        const r = await fetch(`${b}tileset.json`, { cache: 'no-cache' });
        if (!r.ok) return false;
        const m = await r.json();
        if (live) {
          setManifest(m);
          setBase(b);
          // The builder chooses the opening layer - texture when there is one,
          // the height field when there is not. Hardcoding `texture` here
          // opened a textureless tileset on a layer it does not carry, and the
          // viewer has nothing to draw and no reason to think that is wrong.
          setLayer(m.default_layer || 'texture');
        }
        return true;
      } catch { return false; }
    };
    if (jobId) {
      if (job && job.status === 'done') tryLoad(jobBase(jobId));
      return () => { live = false; };
    }
    fetch('api/health', { cache: 'no-store' })
      .then((r) => { if (!r.ok) throw new Error('no api'); })
      .catch(() => { if (live) tryLoad('data/'); });
    return () => { live = false; };
  }, [jobId, job && job.status]);

  const start = useCallback((id) => {
    setJobId(id);
    history.replaceState(null, '', `?job=${id}`);
  }, []);

  const reset = useCallback(() => {
    setJobId(''); setManifest(null);
    history.replaceState(null, '', location.pathname);
  }, []);

  return (
    <>
      <header className="topbar">
        <Brand phase={manifest ? 'viewer' : null} />
        <div className="grow">
          {manifest ? (
            <span className="scene-id mono">
              {`${(manifest.source_run || '').split(/[\\/]/).slice(-2).join('/')} · ${manifest.grid.width}×${manifest.grid.height}`}
            </span>
          ) : null}
        </div>
        {manifest ? (
          <span className="chip mono">
            {`LOD ${stats.lod === null ? '–' : stats.lod} · ${stats.triangles.toLocaleString()} tris`}
          </span>
        ) : null}
        <a className="ghost" href="results.html">Study &amp; results</a>
        {jobId || manifest ? <button className="ghost" onClick={reset}>New reconstruction</button> : null}
      </header>

      {manifest ? (
        <div className="stage">
          <Viewer
            manifest={manifest} base={base} layer={layer} lod={lod} exagg={exagg}
            shade={shade} fog={fog} wire={wire} flyNonce={flyNonce}
            structural={structural} onStructuralReady={setHasStructural}
            onStructuralFallback={(why) => { setStructural(false); setStructuralNote(why); }}
            onStats={(tris, l) => setStats({ triangles: tris, lod: l })}
            onReadout={setReadout}
            onFlying={setFlying}
          />
          <SidePanel
            offline={service === false && !jobId}
            manifest={manifest} base={base}
            layer={layer} setLayer={setLayer} lod={lod} setLod={setLod}
            exagg={exagg} setExagg={setExagg} shade={shade} setShade={setShade}
            fog={fog} setFog={setFog} wire={wire} setWire={setWire}
            structural={structural} setStructural={setStructural}
            hasStructural={hasStructural} structuralNote={structuralNote}
            flying={flying} onFly={() => setFlyNonce((n) => n + 1)} readout={readout}
          />
        </div>
      ) : jobId ? (
        <main className="centre">
          {job ? <Progress job={job} onCancel={reset} />
               : <p className="note-sm">{`Looking for job ${jobId}…`}</p>}
        </main>
      ) : (
        <main className="centre">
          <h1>Metric elevation from a single image</h1>
          <p className="lede">
            Upload a nadir satellite or aerial image. TRAKSHA estimates relative depth,
            harvests metric anchors from the scene, solves a spatially varying
            calibration, and returns a surface you can inspect in 3D and download as a
            textured mesh.
          </p>
          <div className="note critical">
            <b>Read this before trusting a result</b>
            <span>
              A research proof of concept with a diagnosed failure. Without a fitted
              structural scale the calibration returns a near-flat surface; with one it
              recovers about a third of true building height and misses individual
              structures by tens of per cent. Every reconstruction states its own
              defects on the result screen. See sections 3.2 and 4 of the README.
            </span>
          </div>
          <Upload onStarted={start} backbones={backbones} />
        </main>
      )}
    </>
  );
}
