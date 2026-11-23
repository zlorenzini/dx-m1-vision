"""server.py — phone web app for the DEEPX DX-M1.

Run with the DX-RT Python (the venv that has dx_engine):
    venv-dx-runtime/bin/python3 server.py [--port 8090]

Standard library only (plus the cv2/numpy/dx_engine that DX-RT already has).

API
  GET  /                      phone page
  GET  /api/modes             available modes
  POST /api/jobs?kind=image|video&mode=<id>&conf=0.35   body = raw file bytes
                              -> {"id": ...}
  GET  /api/jobs/<id>         -> {"state": queued|running|done|error, "progress": 0..1, ...}
  GET  /api/jobs/<id>/result  annotated JPEG or MP4
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
WORK = os.environ.get("DXAPP_WORK", os.path.join(HERE, "work"))
MAX_UPLOAD = 400 * 1024 * 1024
MAX_SECONDS = 10.0
MAX_FPS = 30.0
MAX_SIDE = 1280          # long side of processed video frames
MAX_IMAGE_SIDE = 2400    # long side of returned photos
JOB_TTL = 30 * 60

runner = engine.Runner()
jobs = {}
jobs_lock = threading.Lock()
work_q = queue.Queue()


def new_job(kind, mode, conf, src_path):
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "kind": kind, "mode": mode, "conf": conf, "src": src_path,
           "state": "queued", "progress": 0.0, "created": time.time(),
           "result": None, "error": None, "summary": None}
    with jobs_lock:
        jobs[jid] = job
    work_q.put(jid)
    return job


def public(job):
    keys = ("id", "kind", "mode", "state", "progress", "error", "summary")
    out = {k: job[k] for k in keys}
    out["position"] = 0
    if job["state"] == "queued":
        with jobs_lock:
            out["position"] = sum(1 for j in jobs.values()
                                  if j["state"] == "queued" and j["created"] < job["created"])
    return out


def fit(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
    return img


def process_image(job, jdir):
    img = cv2.imread(job["src"], cv2.IMREAD_COLOR)  # applies EXIF orientation
    if img is None:
        # e.g. HEIC: let ffmpeg convert it
        conv = os.path.join(jdir, "conv.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job["src"], "-frames:v", "1", conv],
                       check=False, timeout=60)
        img = cv2.imread(conv, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read that image format.")
    img = fit(img, MAX_IMAGE_SIDE)
    rows, ms = runner.infer(job["mode"], img, job["conf"])
    engine.draw(job["mode"], img, rows)
    out = os.path.join(jdir, "result.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    job["summary"] = {"counts": engine.summarize(job["mode"], rows), "npuMs": round(ms, 1),
                      "width": img.shape[1], "height": img.shape[0]}
    return out, "image/jpeg"


def process_video(job, jdir):
    cap = cv2.VideoCapture(job["src"])  # FFmpeg backend auto-rotates phone video
    if not cap.isOpened():
        raise ValueError("Could not read that video format.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 <= src_fps <= 240):
        src_fps = 30.0
    step = max(1, round(src_fps / MAX_FPS))
    out_fps = src_fps / step
    total_src = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    limit_src = int(src_fps * MAX_SECONDS)
    expected = max(1, int(min(total_src or limit_src, limit_src) / step))

    out = os.path.join(jdir, "result.mp4")
    enc = None
    idx = written = 0
    counts_max, npu_ms = {}, 0.0
    try:
        while idx < limit_src:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % step:
                continue
            frame = fit(frame, MAX_SIDE)
            if frame.shape[0] % 2 or frame.shape[1] % 2:
                frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            if enc is None:
                h, w = frame.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{out_fps:.4f}",
                     "-i", "-", "-i", job["src"],
                     "-map", "0:v", "-map", "1:a?", "-t", str(MAX_SECONDS), "-shortest",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            rows, ms = runner.infer(job["mode"], frame, job["conf"])
            npu_ms += ms
            engine.draw(job["mode"], frame, rows)
            for k, v in engine.summarize(job["mode"], rows).items():
                counts_max[k] = max(counts_max.get(k, 0), v)
            enc.stdin.write(frame.tobytes())
            written += 1
            job["progress"] = min(0.97, written / expected)
    finally:
        cap.release()
        if enc is not None:
            enc.stdin.close()
            err = enc.stderr.read().decode(errors="replace")
            if enc.wait(timeout=120) != 0:
                raise RuntimeError(f"Video encoding failed: {err.strip()[-300:]}")
    if written == 0:
        raise ValueError("No frames could be read from that video.")
    job["summary"] = {"counts": counts_max, "frames": written, "fps": round(out_fps, 2),
                      "seconds": round(written / out_fps, 1),
                      "npuMsPerFrame": round(npu_ms / written, 1),
                      "trimmed": idx >= limit_src}
    return out, "video/mp4"


def worker():
    while True:
        jid = work_q.get()
        with jobs_lock:
            job = jobs.get(jid)
        if not job:
            continue
        jdir = os.path.dirname(job["src"])
        job["state"] = "running"
        t0 = time.time()
        try:
            fn = process_image if job["kind"] == "image" else process_video
            job["result"] = fn(job, jdir)
            job["summary"]["totalSec"] = round(time.time() - t0, 1)
            job["progress"] = 1.0
            job["state"] = "done"
        except Exception as e:  # report to the phone instead of dying
            job["error"] = str(e) or e.__class__.__name__
            job["state"] = "error"
        finally:
            try:
                os.remove(job["src"])
            except OSError:
                pass


def janitor():
    while True:
        time.sleep(60)
        cutoff = time.time() - JOB_TTL
        with jobs_lock:
            old = [j for j in jobs.values() if j["created"] < cutoff and j["state"] in ("done", "error")]
            for j in old:
                del jobs[j["id"]]
        for j in old:
            shutil.rmtree(os.path.dirname(j["src"]), ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "dx-m1-phone-app"

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/jobs/") or self.command != "GET":
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype, download_name=None):
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):  # iOS Safari needs ranges for <video>
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else end
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'inline; filename="{download_name}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self):
        url = urlparse(self.path)
        p = url.path
        if p in ("/", "/index.html"):
            return self.send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
        if p == "/api/modes":
            return self.send_json({"modes": engine.available_modes(), "maxSeconds": MAX_SECONDS})
        if p.startswith("/api/jobs/"):
            parts = p.split("/")[3:]
            with jobs_lock:
                job = jobs.get(parts[0]) if parts else None
            if not job:
                return self.send_json({"error": "No such job (it may have expired)."}, 404)
            if len(parts) == 1:
                return self.send_json(public(job))
            if parts[1] == "result" and job["state"] == "done":
                path, ctype = job["result"]
                ext = "jpg" if ctype == "image/jpeg" else "mp4"
                return self.send_file(path, ctype, f"dx-m1-{job['mode']}-{job['id']}.{ext}")
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/jobs":
            return self.send_json({"error": "Not found"}, 404)
        q = parse_qs(url.query)
        kind = q.get("kind", ["image"])[0]
        mode = q.get("mode", ["detect"])[0]
        try:
            conf = min(0.95, max(0.05, float(q.get("conf", ["0.35"])[0])))
        except ValueError:
            conf = 0.35
        if kind not in ("image", "video"):
            return self.send_json({"error": "kind must be image or video"}, 400)
        if mode not in {m["id"] for m in engine.available_modes()}:
            return self.send_json({"error": f"Unknown mode {mode!r}"}, 400)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.send_json({"error": "Empty upload"}, 400)
        if length > MAX_UPLOAD:
            return self.send_json({"error": "File too large (400 MB max)."}, 413)

        os.makedirs(WORK, exist_ok=True)
        jdir = tempfile.mkdtemp(prefix="job-", dir=WORK)
        src = os.path.join(jdir, "input")
        left = length
        with open(src, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            shutil.rmtree(jdir, ignore_errors=True)
            return self.send_json({"error": "Upload interrupted"}, 400)
        job = new_job(kind, mode, conf, src)
        self.send_json(public(job), 202)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    args = ap.parse_args()
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=janitor, daemon=True).start()
    print(f"DX-M1 phone app on http://{args.host}:{args.port}  modes: "
          f"{[m['id'] for m in engine.available_modes()]}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
