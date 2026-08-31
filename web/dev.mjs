/* `npm run dev` - the UI and the pipeline it talks to, together.
 *
 * The reconstruction is Python and cannot move into the browser, so development
 * needs two processes: Vite on :5173 and the TRAKSHA service on :8000. Starting
 * only the first is the obvious thing to do and it fails in the least helpful
 * way available - Vite comes up fine, the page loads, and the console fills with
 * `ECONNREFUSED 127.0.0.1:8000`, which says what happened but not what to do.
 *
 * Two things this has to get right, both learned from getting them wrong.
 *
 * **Which interpreter.** A relative path to `.venv` is resolved against
 * whichever directory the spawn happens to use, so it silently missed and fell
 * through to whatever `python` was on PATH. On a machine where the shell has no
 * venv active that is a different interpreter with a different set of packages,
 * and the failure surfaces much later as a missing dependency. The venv is now
 * resolved absolutely from this file, and every candidate is *probed* for the
 * modules the service needs before anything is started - so a missing package
 * is reported by name, against a named interpreter, instead of as a stack trace
 * from inside FastAPI.
 *
 * **Whether it actually came up.** Waiting a couple of seconds and assuming
 * success is not a check: the service prints its banner and then raises, which
 * takes longer than the wait. This waits for the port to accept a connection,
 * and treats the child exiting as the failure it is.
 */
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import process from 'node:process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const API_PORT = Number(process.env.TRAKSHA_API_PORT || 8000);
const UI_PORT = Number(process.env.TRAKSHA_UI_PORT || 5173);
const WIN = process.platform === 'win32';

// What `traksha serve` needs beyond the core install. Import names, not pip
// names: `python-multipart` imports as `python_multipart`.
const NEEDS = { traksha: 'the package itself (pip install -e .)',
                fastapi: 'traksha[api]', uvicorn: 'traksha[api]',
                python_multipart: 'traksha[api]', sse_starlette: 'traksha[api]' };

const say = (tag, colour) => (line) =>
  process.stdout.write(`\x1b[${colour}m${tag}\x1b[0m ${line}`);
const api = say('[api]', '36');
const warn = say('[api]', '33');
const bad = say('[api]', '31');

function listening(port, ms = 1200) {
  return new Promise((resolve) => {
    const s = net.connect({ port, host: '127.0.0.1' })
      .on('connect', () => { s.destroy(); resolve(true); })
      .on('error', () => resolve(false));
    setTimeout(() => { s.destroy(); resolve(false); }, ms);
  });
}

function candidates() {
  const venv = WIN ? path.join(ROOT, '.venv', 'Scripts', 'python.exe')
                   : path.join(ROOT, '.venv', 'bin', 'python');
  return [process.env.TRAKSHA_PYTHON, existsSync(venv) ? venv : null,
          'python', 'python3'].filter(Boolean);
}

/* Ask an interpreter what it is missing. Returns null if it cannot run at all. */
function probe(exe) {
  const code = 'import importlib.util as u,sys;'
    + `print(",".join(m for m in ${JSON.stringify(Object.keys(NEEDS))} `
    + 'if u.find_spec(m) is None))';
  return new Promise((resolve) => {
    let out = '';
    const c = spawn(exe, ['-c', code], { cwd: ROOT, stdio: ['ignore', 'pipe', 'ignore'] });
    c.stdout.on('data', (d) => { out += d.toString(); });
    c.on('error', () => resolve(null));
    c.on('exit', (code_) => resolve(code_ === 0
      ? out.trim().split(',').filter(Boolean) : null));
    setTimeout(() => { c.kill(); resolve(null); }, 20000);
  });
}

const children = [];
function stop() {
  for (const c of children) {
    try { WIN ? spawn('taskkill', ['/pid', String(c.pid), '/f', '/t']) : c.kill('SIGTERM'); }
    catch { /* already gone */ }
  }
}
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { stop(); process.exit(0); });
process.on('exit', stop);

async function startApi() {
  if (await listening(API_PORT)) {
    api(`already listening on :${API_PORT}, using it\n`);
    return true;
  }

  const tried = [];
  let chosen = null;
  for (const exe of candidates()) {
    const missing = await probe(exe);
    if (missing === null) { tried.push([exe, ['could not run']]); continue; }
    if (missing.length === 0) { chosen = exe; break; }
    tried.push([exe, missing]);
  }

  if (!chosen) {
    bad('no interpreter here can run the pipeline service.\n');
    for (const [exe, missing] of tried) {
      warn(`  ${exe}: missing ${missing.join(', ')}\n`);
    }
    const wanted = new Set(tried.flatMap(([, m]) => m).map((m) => NEEDS[m]).filter(Boolean));
    if (wanted.size) warn(`  install: pip install -e ".[api]"   (${[...wanted].join('; ')})\n`);
    return false;
  }

  const child = spawn(chosen, ['-m', 'traksha.cli', 'serve', '--port', String(API_PORT)],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
  children.push(child);
  child.stdout.on('data', (d) => api(d.toString()));
  child.stderr.on('data', (d) => { if (d.toString().trim()) api(d.toString()); });

  let died = false;
  child.on('exit', (code) => {
    died = true;
    bad(`the pipeline service exited (code ${code}). Uploading will not work; `
      + 'the viewer falls back to the bundled demo scene.\n');
  });

  api(`${chosen} -m traksha.cli serve --port ${API_PORT}\n`);
  // Wait for the port, not for a timer. Imports are slow and a failure can
  // arrive well after any fixed wait would have declared success. The pause
  // matters: a refused connection resolves immediately, so without it the
  // whole loop runs in milliseconds and reports a healthy service as dead.
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 60 && !died; i++) {
    if (await listening(API_PORT, 500)) { api(`up on :${API_PORT}\n`); return true; }
    await pause(500);
  }
  if (!died) warn(`started, but nothing is answering on :${API_PORT} after 30 s\n`);
  return !died;
}

await startApi();

const vite = spawn(WIN ? 'npx.cmd' : 'npx', ['vite', '--port', String(UI_PORT)],
  { stdio: 'inherit', shell: WIN, cwd: HERE });
children.push(vite);
vite.on('exit', (code) => { stop(); process.exit(code ?? 0); });
