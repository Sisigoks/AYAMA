"""Preview the results site exactly as GitHub Pages will serve it.

    python scripts/serve.py            # http://localhost:8000
    python scripts/serve.py --port 9000 --no-open

The Pages workflow assembles `web/` plus the JSON and previews from `results/`
into one directory. This does the same into a temporary folder and serves it, so
what you see locally is what deploys — including the `results/dataset.json` fetch,
which fails from a file:// URL and would otherwise only break once published.
"""
from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import tempfile
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = (".json", ".md", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tex")


def assemble(dest: str) -> str:
    site = os.path.join(ROOT, "web")
    if not os.path.isdir(site):
        raise SystemExit("web/ not found")
    shutil.copytree(site, dest, dirs_exist_ok=True)

    # The Phase 4 viewer ships as a sub-page. web/ is self-contained - its own
    # tileset lives in web/data - so copying the directory is the whole job, and
    # what is served locally is exactly what deploys.
    web = os.path.join(ROOT, "web")
    if os.path.isdir(web):
        shutil.copytree(web, os.path.join(dest, "viewer"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("node_modules", "dist"))
        tiles = os.path.join(dest, "viewer", "data", "tileset.json")
        print(f"  viewer:  /viewer/  ({'with' if os.path.exists(tiles) else 'NO'} tileset)")
    else:
        print("  web/: not found - the 3D section will not load")

    results = os.path.join(ROOT, "results")
    if os.path.isdir(results):
        out = os.path.join(dest, "results")
        n = 0
        for base, _dirs, files in os.walk(results):
            rel = os.path.relpath(base, results)
            for f in files:
                # Rasters are large and the page never reads them; the workflow
                # excludes them too, so the preview matches the deployment.
                if not f.lower().endswith(KEEP):
                    continue
                target = os.path.join(out, rel) if rel != "." else out
                os.makedirs(target, exist_ok=True)
                shutil.copy2(os.path.join(base, f), os.path.join(target, f))
                n += 1
        print(f"  results: {n} files")
    else:
        print("  results/: not found — the page will show its 'no results' state")
    return dest


def _hosted() -> bool:
    """Colab and friends: localhost is inside the VM, not in the browser."""
    import sys

    return "google.colab" in sys.modules or os.path.isdir("/content")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="traksha-site-")
    print(f"assembling into {tmp}")
    assemble(tmp)

    os.chdir(tmp)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quieter than the default
            if "200" not in fmt % a:
                super().log_message(fmt, *a)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}/"
        print(f"serving {url}   (ctrl-c to stop)")
    if _hosted():
        print()
        print("  ! On Colab this address is inside the VM and your browser")
        print("    cannot open it. Run this from a PYTHON cell instead:")
        print()
        print("      import subprocess, time")
        print("      from google.colab.output import eval_js")
        print("      subprocess.Popen(['python', 'scripts/serve.py',")
        print(f"                        '--port', '{args.port}', '--no-open'])")
        print("      time.sleep(4)")
        print(f"      print(eval_js('google.colab.kernel.proxyPort({args.port})'))")
        print()
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
