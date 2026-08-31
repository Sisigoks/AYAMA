/* `npm run dev` - the UI and the pipeline it talks to, together.
 *
 * The reconstruction is Python and cannot move into the browser, so development
 * needs two processes: Vite on :5173 and the TRAKSHA service on :8000. Starting
 * only the first is the obvious thing to do and it fails in the least helpful
 * way available - Vite comes up fine, the page loads, and the console fills with
 *
 *     http proxy error: /api/health
 *     Error: connect ECONNREFUSED 127.0.0.1:8000
 *
 * repeated once per poll, which says what happened but not what to do. So this
 * starts both, and if the API cannot be started it says so once, in words, and
 * leaves the UI running rather than failing outright: the viewer still works
 * against the committed demo tileset, only the upload path needs the service.
 *
 * An API already listening on the port is adopted rather than duplicated, so
 * running the service by hand in another terminal keeps working.
 */
import { spawn } from 'node:child_process';
import net from 'node:net';
import process from 'node:process';

const API_PORT = Number(process.env.TRAKSHA_API_PORT || 8000);
const UI_PORT = Number(process.env.TRAKSHA_UI_PORT || 5173);
const WIN = process.platform === 'win32';

const paint = (tag, colour) => (line) =>
  process.stdout.write(`\x1b[${colour}m${tag}\x1b[0m ${line}`);

function listening(port) {
  return new Promise((resolve) => {
    const s = net.connect({ port, host: '127.0.0.1' })
      .on('connect', () => { s.destroy(); resolve(true); })
      .on('error', () => resolve(false));
    setTimeout(() => { s.destroy(); resolve(false); }, 1500);
  });
}

/* The interpreter that has traksha installed. A bare `python` is right when a
 * virtualenv is active, which is the normal case, but a checkout with a .venv
 * and no activation is common enough to be worth trying too - otherwise the
 * error is `No module named traksha`, which reads like a broken install. */
function interpreters() {
  const venv = WIN ? '../.venv/Scripts/python.exe' : '../.venv/bin/python';
  return [process.env.TRAKSHA_PYTHON, venv, 'python', 'python3'].filter(Boolean);
}

const children = [];
function stop() {
  for (const c of children) {
    try { WIN ? spawn('taskkill', ['/pid', c.pid, '/f', '/t']) : c.kill('SIGTERM'); }
    catch { /* already gone */ }
  }
}
process.on('SIGINT', () => { stop(); process.exit(0); });
process.on('SIGTERM', () => { stop(); process.exit(0); });
process.on('exit', stop);

async function startApi() {
  if (await listening(API_PORT)) {
    paint('[api]', '36')(`already listening on :${API_PORT}, using it\n`);
    return true;
  }
  for (const exe of interpreters()) {
    const child = spawn(exe, ['-m', 'traksha.cli', 'serve', '--port', String(API_PORT)],
      { cwd: '..', stdio: ['ignore', 'pipe', 'pipe'] });
    const failed = await new Promise((resolve) => {
      let settled = false;
      child.on('error', () => { if (!settled) { settled = true; resolve(true); } });
      // Give it long enough to fail on a missing module, which is immediate,
      // without waiting for the model imports, which are not.
      setTimeout(() => { if (!settled) { settled = true; resolve(false); } }, 2500);
    });
    if (failed || child.exitCode !== null) continue;

    children.push(child);
    child.stdout.on('data', (d) => paint('[api]', '36')(d.toString()));
    child.stderr.on('data', (d) => {
      const t = d.toString();
      if (/No module named traksha/.test(t)) {
        paint('[api]', '31')('traksha is not installed in this interpreter. '
          + 'Run `pip install -e .` in the repository root.\n');
      } else if (!/^\s*$/.test(t)) {
        paint('[api]', '36')(t);
      }
    });
    paint('[api]', '36')(`${exe} -m traksha.cli serve --port ${API_PORT}\n`);
    return true;
  }
  return false;
}

const ok = await startApi();
if (!ok) {
  paint('[api]', '33')('could not start the pipeline service.\n');
  paint('[api]', '33')('  The UI will run and the 3D viewer will show the bundled demo scene,\n');
  paint('[api]', '33')('  but uploading an image needs the service. Start it yourself with:\n');
  paint('[api]', '33')(`      python -m traksha.cli serve --port ${API_PORT}\n`);
}

const vite = spawn(WIN ? 'npx.cmd' : 'npx', ['vite', '--port', String(UI_PORT)],
  { stdio: 'inherit', shell: WIN });
children.push(vite);
vite.on('exit', (code) => { stop(); process.exit(code ?? 0); });
