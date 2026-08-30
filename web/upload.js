/* TRAKSHA — the upload and progress half of the app.
 *
 * Deliberately separate from app.js: that file renders a tileset and knows
 * nothing about jobs, and this one drives a job and knows nothing about WebGL.
 * The join between them is a single call to TRAKSHA.boot() once a tileset exists.
 *
 * The progress display is the reason this uses SSE rather than polling. A
 * reconstruction takes tens of seconds to minutes, and the stages have names
 * that mean something - `anchors`, then `calibration`, then `uncertainty`. A
 * bar that names the stage it is in turns a wait into an explanation.
 */
'use strict';

var TRAKSHA_JOB = (function () {

var STAGE_LABEL = {
  ingest: 'reading the image and its metadata',
  depth: 'monocular depth inference',
  segmentation: 'semantic classes',
  shadow: 'cast-shadow detection',
  anchors: 'harvesting metric anchors',
  calibration: 'anchor-graph calibration (AGMC)',
  uncertainty: 'bootstrap uncertainty',
  assemble: 'DSM / DTM / nDSM',
  artifacts: 'writing GeoTIFFs',
  validation: 'metrics vs reference',
  tiles: 'building the 3D tileset',
};

function $(id) { return document.getElementById(id); }

function show(which) {
  ['landing', 'progress', 'app'].forEach(function (id) {
    var el = $(id);
    if (el) el.hidden = (id !== which);
  });
}

// ── stage list ──────────────────────────────────────────────────────────────
function renderStages(stages, state) {
  var host = $('stage-list');
  if (!host) return;
  host.innerHTML = '';
  stages.forEach(function (name) {
    var st = state[name] || { status: 'pending' };
    var row = document.createElement('div');
    row.className = 'stage ' + st.status;
    var dot = document.createElement('span');
    dot.className = 'dot';
    var label = document.createElement('span');
    label.className = 'stage-name';
    label.textContent = STAGE_LABEL[name] || name;
    var detail = document.createElement('span');
    detail.className = 'stage-detail';
    detail.textContent = st.detail || '';
    row.appendChild(dot);
    row.appendChild(label);
    row.appendChild(detail);
    host.appendChild(row);
  });
}

// ── notes: the honesty layer, same source as the viewer's ──────────────────
function renderNotes(notes) {
  var host = $('job-notes');
  if (!host) return;
  host.innerHTML = '';
  (notes || []).forEach(function (n) {
    var d = document.createElement('div');
    d.className = 'note ' + (n.level || 'info');
    var b = document.createElement('b');
    b.textContent = n.level === 'critical' ? 'Read this first'
      : n.level === 'warning' ? 'Warning' : 'Note';
    var s = document.createElement('span');
    s.textContent = n.text;
    d.appendChild(b);
    d.appendChild(s);
    host.appendChild(d);
  });
}

function renderSummary(job) {
  var host = $('job-summary');
  if (!host) return;
  var s = job.summary || {};
  var m = s.metrics || {};
  var g = s.grid || {};
  var rows = [
    ['job', job.id],
    ['elapsed', job.elapsed_s + ' s'],
    ['tier', s.tier ? 'Tier ' + s.tier : null],
    ['why', s.tier_reason],
    ['raster', g.width ? g.width + ' x ' + g.height + ' px' : null],
    ['GSD', g.gsd_m ? g.gsd_m.toFixed(3) + ' m' : null],
    ['anchors', s.anchors ? JSON.stringify(s.anchors) : null],
    ['MAE', m.mae_m !== undefined ? m.mae_m.toFixed(2) + ' m' : null],
    ['1σ coverage', m.coverage_1s !== undefined ? m.coverage_1s.toFixed(3) : null],
  ];
  host.innerHTML = '';
  rows.forEach(function (r) {
    if (r[1] === null || r[1] === undefined || r[1] === '') return;
    var tr = document.createElement('tr');
    var a = document.createElement('td'); a.textContent = r[0];
    var b = document.createElement('td'); b.textContent = String(r[1]);
    tr.appendChild(a); tr.appendChild(b); host.appendChild(tr);
  });
}

// ── follow one job to completion ───────────────────────────────────────────
function follow(job) {
  show('progress');
  var state = {};
  var stages = job.stages || [];
  renderStages(stages, state);
  $('job-id').textContent = job.id;

  var log = $('job-log');
  var es = new EventSource('api/jobs/' + job.id + '/events');

  es.addEventListener('stage', function (e) {
    var ev = JSON.parse(e.data);
    state[ev.stage] = ev;
    // A stage that reports "running" after another was running means the
    // previous one finished, even if its `done` event is still in flight.
    stages.forEach(function (n) {
      if (state[n] && state[n].status === 'running' && n !== ev.stage) {
        state[n] = Object.assign({}, state[n], { status: 'done' });
      }
    });
    renderStages(stages, state);
    if (log) {
      var line = document.createElement('div');
      line.textContent = '[' + ev.t.toFixed(1) + 's] ' + ev.stage + ' ' +
                         ev.status + (ev.detail ? ' — ' + ev.detail : '');
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    }
  });

  es.addEventListener('end', function (e) {
    es.close();
    var done = JSON.parse(e.data);
    if (done.status === 'failed') {
      show('progress');
      var err = $('job-error');
      if (err) {
        err.hidden = false;
        err.textContent = done.error || 'the reconstruction failed';
      }
      return;
    }
    renderNotes(done.notes);
    renderSummary(done);
    openViewer(done);
  });

  es.onerror = function () {
    es.close();
    var err = $('job-error');
    if (err) {
      err.hidden = false;
      err.textContent = 'lost the connection to the server while it was working';
    }
  };
}

// ── hand off to the viewer ─────────────────────────────────────────────────
function openViewer(job) {
  // The viewer reads its tileset base from ?job=, so put it in the URL: the
  // result becomes a link someone can send, and a reload does not re-run the
  // pipeline.
  var url = new URL(location.href);
  url.searchParams.set('job', job.id);
  history.replaceState({}, '', url);

  show('app');
  var dl = $('job-downloads');
  if (dl) {
    dl.innerHTML = '';
    [['dsm.tif', 'surface elevation'], ['ndsm.tif', 'height above ground'],
     ['sigma.tif', 'uncertainty'], ['error.tif', 'error vs reference']
    ].forEach(function (f) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = 'api/jobs/' + job.id + '/artifacts/' + f[0];
      a.textContent = f[0] + ' — ' + f[1];
      a.setAttribute('download', '');
      li.appendChild(a);
      dl.appendChild(li);
    });
  }
  if (window.TRAKSHA && window.TRAKSHA.boot) {
    window.TRAKSHA.boot().catch(function () { /* surfaced in the DOM */ });
  }
}

// ── submit ─────────────────────────────────────────────────────────────────
function submit(file, opts) {
  var fd = new FormData();
  fd.append('image', file);
  Object.keys(opts).forEach(function (k) { fd.append(k, opts[k]); });

  var btn = $('go');
  if (btn) { btn.disabled = true; btn.textContent = 'uploading…'; }

  return fetch('api/jobs', { method: 'POST', body: fd })
    .then(function (r) {
      return r.json().then(function (body) {
        if (!r.ok) throw new Error(body.detail || ('HTTP ' + r.status));
        return body;
      });
    })
    .then(follow)
    .catch(function (e) {
      var err = $('upload-error');
      if (err) { err.hidden = false; err.textContent = e.message; }
      if (btn) { btn.disabled = false; btn.textContent = 'Reconstruct'; }
    });
}

// ── wiring ─────────────────────────────────────────────────────────────────
function init() {
  if (!$('landing')) return;                 // the standalone viewer page

  var existing = new URLSearchParams(location.search).get('job');
  if (existing) {                            // a shared link to a finished job
    show('app');
    fetch('api/jobs/' + existing).then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { if (j) { renderNotes(j.notes); renderSummary(j); openViewer(j); } })
      .catch(function () { /* the viewer will report its own failure */ });
    return;
  }

  show('landing');
  var input = $('file');
  var drop = $('drop');
  var chosen = null;

  function choose(f) {
    chosen = f;
    var label = $('chosen');
    if (label) {
      label.textContent = f ? (f.name + '  (' + (f.size / 1e6).toFixed(1) + ' MB)') : '';
    }
    var btn = $('go');
    if (btn) btn.disabled = !f;
  }

  if (input) input.addEventListener('change', function () { choose(input.files[0]); });
  if (drop) {
    ['dragenter', 'dragover'].forEach(function (ev) {
      drop.addEventListener(ev, function (e) {
        e.preventDefault(); drop.classList.add('over');
      });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      drop.addEventListener(ev, function (e) {
        e.preventDefault(); drop.classList.remove('over');
      });
    });
    drop.addEventListener('drop', function (e) {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) choose(e.dataTransfer.files[0]);
    });
    drop.addEventListener('click', function () { if (input) input.click(); });
  }

  var form = $('upload-form');
  if (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (!chosen) return;
      var err = $('upload-error');
      if (err) err.hidden = true;
      submit(chosen, {
        backbone: ($('backbone') || {}).value || 'dav2-vits',
        chip: ($('chip') || {}).value || 512,
        bootstrap: ($('bootstrap') || {}).value || 12,
        mesh: ($('want-mesh') || {}).checked ? 'true' : 'false',
      });
    });
  }
}

if (typeof document !== 'undefined' && document.addEventListener) {
  document.addEventListener('DOMContentLoaded', function () {
    if (!window.TRAKSHA_NO_AUTOBOOT) init();
  });
}

return { init: init, follow: follow, submit: submit, show: show,
         renderStages: renderStages, renderNotes: renderNotes,
         renderSummary: renderSummary, STAGE_LABEL: STAGE_LABEL };

})();

if (typeof window !== 'undefined') { window.TRAKSHA_JOB = TRAKSHA_JOB; }
if (typeof module !== 'undefined' && module.exports) { module.exports = TRAKSHA_JOB; }
