"""The web service: upload a scene, watch it reconstruct, look at the result.

    python -m traksha.cli serve --port 8000

Three surfaces, and the split matters:

    /                     the app - upload form, live progress, 3D viewer
    /api/...              JSON + SSE, so the front end has no privileged access
    /api/jobs/{id}/tiles  the tileset, served exactly as the viewer expects

The viewer is the same `web/app.js` the local `traksha viewer` command serves. It
takes its tileset base from a query parameter, so one implementation covers a
prebuilt local tileset and a freshly reconstructed job. Nothing about the
rendering path is web-service specific.

**What this does not do.** No authentication, no rate limiting beyond a single
worker slot, no persistence across restarts, no HTTPS. It is a demo server for a
research artifact, and putting it on the public internet unchanged would be a
mistake. The upload validation in `jobs.py` is real, but it is the only hard
edge here.
"""
# NOTE: deliberately no `from __future__ import annotations` here.
# FastAPI resolves endpoint annotations at decoration time, and the fastapi
# names are imported lazily inside create_app() so the core install stays thin.
# With PEP 563 the annotations become strings that pydantic then tries to
# resolve against module globals - where those names do not exist - and every
# upload endpoint fails with "is not fully defined". Evaluated eagerly, they
# resolve against the enclosing scope and are real classes.

import asyncio
import json
import os
from typing import Optional

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")


def create_app(jobs_root: str = "out/jobs", web_dir: Optional[str] = None,
               max_concurrent: int = 1):
    """Build the FastAPI app. Imported lazily so the core install stays thin."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the web service needs the api extra:\n"
            "  pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.29' "
            "'python-multipart>=0.0.9'"
        ) from exc

    from .jobs import JobStore, UploadRejected

    web = os.path.abspath(web_dir or WEB_DIR)
    store = JobStore(jobs_root, max_concurrent=max_concurrent)
    app = FastAPI(title="TRAKSHA", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store

    # ---------------------------------------------------------------- api
    @app.post("/api/jobs")
    async def create_job(
        image: UploadFile = File(...),
        backbone: str = Form("dav2-vits"),
        chip: int = Form(512),
        bootstrap: int = Form(12),
        sun_azimuth: Optional[float] = Form(None),
        sun_elevation: Optional[float] = Form(None),
        mesh: bool = Form(False),
    ):
        data = await image.read()
        try:
            job = store.create(data, image.filename or "upload", {
                "backbone": backbone, "chip": chip, "bootstrap": bootstrap,
                "mesh": mesh,
                "sun_azimuth": sun_azimuth, "sun_elevation": sun_elevation,
            })
        except UploadRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(job.public(), status_code=202)

    @app.get("/api/jobs")
    async def list_jobs():
        return {"jobs": store.list()}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        """Server-sent events: one message per StageEvent, then a final state.

        Polling would work and would be simpler, but the pipeline's stages are
        the interesting part of a two-minute wait - a progress bar that names
        `anchors` and then `calibration` is the difference between a wait and a
        black box.
        """
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")

        async def stream():
            sent = 0
            while True:
                while sent < len(job.events):
                    ev = job.events[sent]
                    sent += 1
                    yield f"event: stage\ndata: {json.dumps(ev)}\n\n"
                if job.status in ("done", "failed") and sent >= len(job.events):
                    yield f"event: end\ndata: {json.dumps(job.public())}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/tiles/{path:path}")
    async def job_tiles(job_id: str, path: str):
        target = store.path(job_id, "tiles", *path.split("/"))
        if target is None or not os.path.isfile(target):
            raise HTTPException(404, "not found")
        return FileResponse(target)

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    async def job_artifact(job_id: str, name: str):
        """The GeoTIFFs, so a result can leave the browser and enter QGIS."""
        target = store.path(job_id, "run", name)
        if target is None or not os.path.isfile(target):
            raise HTTPException(404, "not found")
        return FileResponse(target, filename=f"traksha_{job_id}_{name}")

    @app.get("/api/health")
    async def health():
        from ..eval.bench import device_report

        rep = device_report()
        return {"ok": True,
                "cpu_count": rep.get("cpu_count"),
                "torch": rep.get("torch"), "jobs": len(store.list())}

    # ---------------------------------------------------------------- app
    @app.get("/")
    async def index():
        return FileResponse(os.path.join(web, "index.html"))

    @app.get("/{name:path}")
    async def static_file(name: str):
        parts = [p for p in name.split("/") if p not in ("", ".", "..")]
        if not parts:
            raise HTTPException(404, "not found")
        target = os.path.abspath(os.path.join(web, *parts))
        if os.path.commonpath([target, web]) != web or not os.path.isfile(target):
            raise HTTPException(404, "not found")
        return FileResponse(target)

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, jobs_root: str = "out/jobs",
          max_concurrent: int = 1, reload: bool = False) -> int:
    import uvicorn

    app = create_app(jobs_root=jobs_root, max_concurrent=max_concurrent)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
