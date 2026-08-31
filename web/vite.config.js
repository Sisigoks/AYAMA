import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

/* Serve ../results during `npm run dev`.
 *
 * The results page reads results/dataset.json, which lives at the repository
 * root - outside Vite's root, so it is invisible to the dev server and the page
 * would come up empty with only a 404 in the console. The Pages workflow and
 * scripts/serve.py both copy that directory in beside the build, so this makes
 * development match deployment rather than adding a path only dev knows about.
 */
function results() {
  const dir = path.resolve(import.meta.dirname ?? '.', '..', 'results');
  const TYPES = { '.json': 'application/json', '.png': 'image/png',
                  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml' };
  return {
    name: 'traksha-results',
    configureServer(server) {
      server.middlewares.use('/results', (req, res, next) => {
        // Reject anything that climbs out of results/ - this is a dev server,
        // but it still should not hand out arbitrary files from the disk.
        const rel = decodeURIComponent((req.url || '/').split('?')[0]);
        const file = path.resolve(dir, '.' + rel);
        if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
          return next();
        }
        res.setHeader('Content-Type', TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream');
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

/* TRAKSHA front end.
 *
 * `npm run dev` serves the app on :5173 and proxies the API and the job tile
 * store to the Python service on :8000. That split is the point of the proxy:
 * the reconstruction runs in Python and cannot move into the browser, so during
 * development two servers are running and the front end must not care which one
 * answered. Every fetch in the app is a relative path for the same reason.
 *
 *   python -m traksha.cli serve      # :8000, the pipeline
 *   npm run dev                      # :5173, the UI  <- open this one
 *
 * `npm run build` emits ../web/dist, which the Python service serves on its own
 * when it is present, so production is one process and one port.
 *
 * `base: './'` keeps every asset reference relative. Absolute paths break both
 * GitHub Pages (served from /<repo>/) and Colab's port proxy (served from a
 * long generated prefix), and that second one is how this was found.
 */
const API = 'http://127.0.0.1:8000';

/* Say it once, in words, and answer the request.
 *
 * Vite's default proxy error handler prints a Node stack trace per failed
 * request. The page polls health every few seconds, so a service that is simply
 * not running produces a wall of ECONNREFUSED that says what happened but not
 * what to do, and buries the Vite banner. This collapses that into one line
 * with the command that fixes it, and replies 503 with a readable body so the
 * browser gets an answer instead of a hanging request.
 */
let toldAt = 0;
function quietly(proxy) {
  // Vite attaches its own error listener after `configure` runs, and that one
  // prints the Node stack trace. Deferring to the next tick lets it attach
  // first, so both can be dropped and replaced with this one. Reaching into
  // Vite's logging is worth it here: the page polls health every few seconds,
  // so the default behaviour is dozens of identical stack traces that bury the
  // one line saying which command to run.
  setTimeout(() => {
    proxy.removeAllListeners('error');
    proxy.on('error', onError);
  }, 0);
  proxy.on('error', onError);
}

function onError(err, _req, res) {
  const now = Date.now();
  if (now - toldAt > 10000) {
    toldAt = now;
    const why = err && err.code === 'ECONNREFUSED'
      ? `no pipeline service on ${API}`
      : `proxy error: ${err && err.message}`;
    console.log(`\n  \x1b[33m${why}\x1b[0m`);
    console.log('  Uploading an image needs it. Start it with:');
    console.log('      python -m traksha.cli serve --port 8000');
    console.log('  or run `npm run dev`, which starts both.\n');
  }
  // `res` is a socket for websocket upgrades, and has no writeHead.
  if (res && typeof res.writeHead === 'function' && !res.headersSent) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ detail: 'the TRAKSHA pipeline service is not running' }));
  } else if (res && typeof res.end === 'function') {
    res.end();
  }
}

export default defineConfig({
  base: './',
  plugins: [react(), results()],
  server: {
    port: 5173,
    proxy: {
      // Only /api. `/data` is the demo tileset committed at web/data, which
      // Vite already serves as a static file from its own root - proxying it
      // sent the fallback to a server that serves nothing there, so with no
      // service running the viewer had nothing to fall back to either.
      '/api': { target: API, changeOrigin: true, configure: quietly },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: new URL('./index.html', import.meta.url).pathname,
        results: new URL('./results.html', import.meta.url).pathname,
      },
    },
  },
});
