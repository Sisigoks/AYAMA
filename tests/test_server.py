"""The web service: upload validation, the job lifecycle, and path safety.

This is the only component that takes untrusted input, so the tests weight
accordingly: roughly half of them are about refusing things rather than doing
things. A reconstruction that works is worth little if the endpoint that starts
it will also read `/etc/passwd`.

The pipeline runs on the smallest real backbone throughout - these tests
are about the service, and a real backbone would make them a weights download
and a minute of inference each.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("fastapi")

from traksha.api.jobs import (MAX_UPLOAD_BYTES, JobStore,  # noqa: E402
                            UploadRejected, validate_upload)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
TIFF = b"II*\x00" + b"\x00" * 64


# ------------------------------------------------------------ upload gate
def test_accepts_the_formats_it_claims_to():
    assert validate_upload("a.png", PNG) == ".png"
    assert validate_upload("a.jpg", JPEG) == ".jpg"
    assert validate_upload("a.tif", TIFF) == ".tif"
    assert validate_upload("A.TIFF", TIFF) == ".tiff"


def test_rejects_an_unsupported_extension():
    with pytest.raises(UploadRejected, match="unsupported extension"):
        validate_upload("payload.exe", PNG)


def test_rejects_content_that_contradicts_the_extension():
    """A suffix is a claim by the uploader; the magic bytes are evidence."""
    with pytest.raises(UploadRejected, match="not a PNG, JPEG or TIFF"):
        validate_upload("innocent.png", b"#!/bin/sh\nrm -rf /\n")


def test_rejects_empty_and_oversized_uploads():
    with pytest.raises(UploadRejected, match="empty"):
        validate_upload("a.png", b"")
    with pytest.raises(UploadRejected, match="limit"):
        validate_upload("a.png", PNG + b"\x00" * MAX_UPLOAD_BYTES)


# ------------------------------------------------------------ path safety
def test_job_paths_cannot_escape_the_job_directory(tmp_path):
    store = JobStore(str(tmp_path))
    job = store.create(PNG, "a.png", {"backbone": "dav2-vits"})
    time.sleep(0.1)

    inside = store.path(job.id, "tiles", "tileset.json")
    assert inside is not None and inside.startswith(os.path.abspath(job.dir))

    for attempt in (("..", "..", "etc", "passwd"), ("tiles", "..", "..", "secret"),
                    ("", "..", "x")):
        assert store.path(job.id, *attempt) is None, f"escaped with {attempt}"
    assert store.path("no-such-job", "tiles") is None


def test_the_uploaders_filename_is_never_used_on_disk(tmp_path):
    """Filenames are a path-traversal vector and we have no need to keep one."""
    store = JobStore(str(tmp_path))
    job = store.create(PNG, "../../../evil.png", {"backbone": "dav2-vits"})
    names = os.listdir(job.dir)
    assert "input.png" in names
    assert not any("evil" in n for n in names)
    # The original is retained as metadata only, where it cannot address a file.
    assert job.params["original_name"] == "../../../evil.png"


# ------------------------------------------------------------- lifecycle
@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from traksha.dsm.cog import write_rgb
    from traksha.data.sample import load_sample_scene

    sc = load_sample_scene(size=256, sun=(210.0, 45.0))
    p = tmp_path_factory.mktemp("up") / "scene.tif"
    write_rgb(str(p), sc.rgb, sc.meta)
    return str(p)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from traksha.api.server import create_app

    return TestClient(create_app(jobs_root=str(tmp_path_factory.mktemp("jobs"))))


def _run(client, scene, **form):
    data = {"backbone": "dav2-vits", "chip": "256", "bootstrap": "3"}
    data.update(form)
    with open(scene, "rb") as fh:
        r = client.post("/api/jobs", files={"image": ("s.tif", fh, "image/tiff")},
                        data=data)
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    for _ in range(300):
        job = client.get(f"/api/jobs/{jid}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.3)
    pytest.fail("job did not finish")


def test_upload_reconstructs_and_serves_a_viewable_tileset(client, scene):
    """The whole point: an image goes in, something the viewer can draw comes out."""
    job = _run(client, scene)
    assert job["status"] == "done", job.get("error")

    man = client.get(f"/api/jobs/{job['id']}/tiles/tileset.json")
    assert man.status_code == 200
    manifest = man.json()
    assert manifest["lods"] and manifest["layers"]

    # Every tile the manifest promises must actually be served.
    for lod in manifest["lods"]:
        for t in lod["tiles"]:
            for rel in t["layers"].values():
                assert client.get(f"/api/jobs/{job['id']}/tiles/{rel}").status_code == 200


def test_the_result_reports_its_own_tier_and_defects(client, scene):
    """A reconstruction that will not say how it was calibrated is not a result."""
    job = _run(client, scene)
    assert job["summary"]["tier"] in ("A", "B", "C")
    assert job["summary"]["tier_reason"]
    assert job["summary"]["anchors"]["total"] > 0
    # derive_notes runs on the served surface, so the page can be honest.
    assert isinstance(job["notes"], list)


def test_artifacts_are_downloadable_as_geotiff(client, scene):
    job = _run(client, scene)
    r = client.get(f"/api/jobs/{job['id']}/artifacts/dsm.tif")
    assert r.status_code == 200
    assert r.content[:4] in (b"II*\x00", b"MM\x00*")      # a real TIFF
    assert client.get(f"/api/jobs/{job['id']}/artifacts/nope.tif").status_code == 404


def test_progress_events_stream_and_terminate(client, scene):
    """SSE must end, or the browser waits forever on a finished job."""
    job = _run(client, scene)
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "event: progress" in body
    assert "event: end" in body
    # The frame carries the phase record, which is what the screen draws.
    assert '"name": "depth"' in body
    assert '"progress": 1.0' in body


def test_http_rejections_are_readable(client):
    bad = client.post("/api/jobs", files={"image": ("x.exe", b"MZ\x00", "application/x-msdownload")})
    assert bad.status_code == 400 and "unsupported extension" in bad.json()["detail"]

    fake = client.post("/api/jobs", files={"image": ("x.png", b"not a png", "image/png")})
    assert fake.status_code == 400 and "not a PNG" in fake.json()["detail"]

    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_static_files_are_served_and_cannot_escape(client):
    """The service serves the Vite build, and only the Vite build.

    Asset names are content-hashed, so they cannot be listed here. They are read
    out of the index the build emitted instead - which also asserts the two
    halves agree: an index referencing a bundle the service will not serve is a
    blank page, and that is exactly how the React conversion first shipped.
    """
    import re

    index = client.get("/")
    assert index.status_code == 200
    assert "<div id=\"root\"" in index.text

    refs = re.findall(r'(?:src|href)="([^"]+)"', index.text)
    assets = [r for r in refs if r.endswith((".js", ".css"))]
    assert assets, "the built index references no bundle"
    for ref in assets:
        got = client.get("/" + ref.lstrip("./"))
        assert got.status_code == 200, ref

    assert client.get("/results.html").status_code == 200

    for path in ("/../pyproject.toml", "/../../etc/passwd", "/nope.js"):
        assert client.get(path).status_code == 404, path


def test_health_reports_the_machine(client):
    h = client.get("/api/health").json()
    assert h["ok"] is True
    assert "cpu_count" in h and "torch" in h


def test_an_upload_comes_back_with_relief_not_a_flat_sheet(client):
    """The failure a user would actually see, and the one that shipped.

    The job service built its own Config and never set `scale_model` or
    `dual_branch`, so an upload ran the anchors-only path: 0.4 m of relief on a
    scene carrying 33 m of it. The pipeline was working exactly as README
    section 3.2 describes and serving the result as if it were the product.
    """
    import io as _io
    import os as _os
    import tempfile
    import time

    from traksha.data.sample import load_sample_scene
    from traksha.dsm.cog import write_rgb

    sc = load_sample_scene(size=384)
    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "s.tif")
        write_rgb(path, sc.rgb, sc.meta)
        with open(path, "rb") as fh:
            buf = _io.BytesIO(fh.read())

    r = client.post("/api/jobs",
                    files={"image": ("s.tif", buf, "image/tiff")},
                    data={"backbone": "dav2-vits", "chip": "384",
                          "bootstrap": "3", "mesh": "true"})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(600):
        job = client.get(f"/api/jobs/{jid}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")

    m = client.get(f"/api/jobs/{jid}/tiles/tileset.json").json()
    assert m["layers"]["ndsm"]["stats"]["max"] > 5.0, (
        "an upload came back flat - the job service lost the fitted "
        "structural scale (traksha/api/jobs.py)")
    # and the mesh a viewer can actually download
    mesh = m.get("mesh")
    assert mesh and mesh["triangles"] > 1000
    for key in ("obj", "mtl", "texture"):
        assert client.get(f"/api/jobs/{jid}/tiles/{mesh[key]}").status_code == 200


# --------------------------------------------------------------- progress
# The progress screen reported nothing at all: the server framed every message
# as a named SSE event that the client's `onmessage` could not receive, and the
# poll fallback returned a job record with no current phase in it. Both halves
# are pinned here. See docs/ARCHITECTURE.md section 2.1.
def test_the_poll_endpoint_carries_the_phase_while_the_job_runs(tmp_path):
    """The defect: `GET /api/jobs/{id}` used to answer `stage: null` throughout."""
    from fastapi.testclient import TestClient

    from traksha.api.server import create_app

    c = TestClient(create_app(jobs_root=str(tmp_path)))
    r = c.post("/api/jobs", files={"image": ("a.png", PNG, "image/png")},
               data={"backbone": "dav2-vits"})
    body = c.get(f"/api/jobs/{r.json()['id']}").json()

    for key in ("phase", "phase_status", "phase_progress", "progress", "phases"):
        assert key in body, f"the poll response has no '{key}'"
    assert isinstance(body["phases"], list) and body["phases"]
    assert {p["name"] for p in body["phases"]} >= {"ingest", "depth", "tiles"}
    assert 0.0 <= body["progress"] <= 1.0


def test_the_sse_frames_are_named_what_the_client_listens_for(tmp_path):
    """A named frame never reaches `EventSource.onmessage`.

    This is the whole bug, and it is a contract between two languages that
    nothing else in the suite would notice breaking: the server can rename a
    frame and every Python test still passes while the page goes blank.
    """
    import re

    from fastapi.testclient import TestClient

    from traksha.api.server import SSE_EVENTS, create_app

    c = TestClient(create_app(jobs_root=str(tmp_path)))
    r = c.post("/api/jobs", files={"image": ("a.png", PNG, "image/png")},
               data={"backbone": "dav2-vits"})
    job_id = r.json()["id"]

    lines = []
    with c.stream("GET", f"/api/jobs/{job_id}/events") as s:
        for line in s.iter_lines():
            lines.append(line)
            if len(lines) > 60 or line.startswith("event: end"):
                break

    on_wire = {ln.split("event: ", 1)[1] for ln in lines if ln.startswith("event: ")}
    assert on_wire, "the stream sent no named frames"
    assert on_wire <= set(SSE_EVENTS), f"undeclared frame names: {on_wire - set(SSE_EVENTS)}"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_jsx = open(os.path.join(root, "web", "src", "App.jsx"), encoding="utf-8").read()
    listened = set(re.findall(r"addEventListener\('([\w-]+)'", app_jsx))
    assert on_wire <= listened, (
        f"the server sends {sorted(on_wire)} but the client listens for "
        f"{sorted(listened)}; frames the client does not name are discarded")
    assert "es.onmessage" not in app_jsx, (
        "onmessage only fires for unnamed frames - this is the original bug")


def test_a_frame_is_the_whole_state_so_a_late_client_is_correct(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from traksha.api.server import create_app

    c = TestClient(create_app(jobs_root=str(tmp_path)))
    r = c.post("/api/jobs", files={"image": ("a.png", PNG, "image/png")},
               data={"backbone": "dav2-vits"})
    with c.stream("GET", f"/api/jobs/{r.json()['id']}/events") as s:
        payload = None
        for line in s.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                break
    assert payload is not None, "the stream sent no data frame"
    for key in ("id", "status", "phase", "progress", "phases"):
        assert key in payload, f"a frame with no '{key}' cannot stand alone"


# -------------------------------------------------------------- recovery
def test_jobs_are_adopted_from_disk_when_the_store_restarts(tmp_path):
    store = JobStore(str(tmp_path))
    job = store.create(PNG, "a.png", {"backbone": "dav2-vits"})
    job._done.wait(timeout=60)

    fresh = JobStore(str(tmp_path))
    assert fresh.get(job.id) is not None, "a restart lost a completed job"
    assert fresh.get(job.id).public()["phases"], "the phase record did not survive"


def test_a_job_left_running_by_a_crash_comes_back_failed(tmp_path):
    """A job that reports `running` forever is indistinguishable from a hang."""
    import json

    store = JobStore(str(tmp_path))
    job = store.create(PNG, "a.png", {"backbone": "dav2-vits"})
    job._done.wait(timeout=60)

    # Rewrite the record the way a killed process would leave it.
    with open(job.state_path(), encoding="utf-8") as fh:
        rec = json.load(fh)
    rec["status"] = "running"
    rec["finished"] = None
    rec["error"] = None
    rec["progress"]["current"] = "depth"
    for p in rec["progress"]["phases"]:
        p["status"] = "running" if p["name"] == "depth" else "pending"
    with open(job.state_path(), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)

    revived = JobStore(str(tmp_path)).get(job.id)
    assert revived.status == "failed"
    assert "restart" in (revived.error or "")
    assert revived.public()["phase_status"] == "failed"
