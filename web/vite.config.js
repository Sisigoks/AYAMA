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
export default defineConfig({
  base: './',
  plugins: [react(), results()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      // the demo tileset the standalone viewer falls back to
      '/data': { target: 'http://127.0.0.1:8000', changeOrigin: true },
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
