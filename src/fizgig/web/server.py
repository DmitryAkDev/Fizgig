"""Fizgig's web UI server: the training workflow in a browser, for cloud GPU pods and headless machines.

Standard library only. The page is built from what Fizgig declares: every registered family's description and its
form (fizgig.families.form), so a new family, option or preset reaches the web UI with no change here. A run starts
through fizgig.families.launch, exactly as the desktop app starts it.

Access: on a pod (a password from --password, VNC_PASSWORD or FIZGIG_PASSWORD) it listens on every address and asks
for the password; without one it listens on localhost only. Either way every API call carries the page's session
token (in the served page, unreadable to other sites) and a localhost server checks the Host header, so a web page
elsewhere cannot drive it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families
from fizgig.web.schema import get_schema

logger = logging.getLogger("fizgig.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PREFS_FILE = os.path.join(BASE_DIR, "prefs.json")
STATE_FILE = os.path.join(BASE_DIR, "web_state.json")      # the web UI's own memory (git-ignored)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DEFAULT = os.path.join(BASE_DIR, "output_loras")
MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".wav", ".mp3", ".flac", ".mp4", ".webm"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

TOKEN = secrets.token_urlsafe(24)       # this server's session token, placed in the page it serves

_lock = threading.Lock()
_run = {"proc": None, "state": "idle", "stage": "", "lora": "", "output_dir": "", "family": "", "values": {},
        "image_folder": "", "pause": False}
_log: list[tuple[int, str]] = []        # (line number, text), the last 5000
_log_n = [0]
_log_event = threading.Condition(_lock)
_stats = {"vram": [0, 0], "ram": [0, 0], "vram_peak": 0, "ram_peak": 0, "device_name": "GPU"}


# ---- hardware meters (as the desktop status bar) ---------------------------------------------------------------
def _poll_stats_loop():
    try:
        import psutil
    except ImportError:
        psutil = None
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
    except Exception:
        pass
    while True:
        vu = vt = 0
        name = "GPU"
        try:
            if nvml is not None:
                h = nvml.nvmlDeviceGetHandleByIndex(0)
                m = nvml.nvmlDeviceGetMemoryInfo(h)
                vu, vt = int(m.used), int(m.total)
                name = nvml.nvmlDeviceGetName(h)
                name = name.decode("utf-8", "ignore") if isinstance(name, bytes) else name
        except Exception:
            pass
        if not vt:                                   # no pynvml: nvidia-smi, as the desktop status bar falls back
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                n, u, t = [x.strip() for x in out.stdout.splitlines()[0].split(",")][:3]
                name, vu, vt = n, int(u) << 20, int(t) << 20
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
        ru = rt = 0
        if psutil is not None:
            vm = psutil.virtual_memory()
            ru, rt = vm.total - vm.available, vm.total
        with _lock:
            _stats.update(vram=[vu, vt], ram=[ru, rt], device_name=name, vram_peak=max(_stats["vram_peak"], vu),
                          ram_peak=max(_stats["ram_peak"], ru))
        time.sleep(1.0)


def append_log(text: str):
    with _lock:
        _log_n[0] += 1
        _log.append((_log_n[0], text))
        del _log[:-5000]
        _log_event.notify_all()


# ---- prefs and the web UI's own state ---------------------------------------------------------------------------
def _read_json(path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_prefs() -> dict:
    return _read_json(PREFS_FILE)


def update_prefs(changes: dict):
    """Only the given keys, read fresh and written atomically: the desktop app may be open on the same files."""
    with _lock:
        p = load_prefs()
        p.update(changes)
        _write_json(PREFS_FILE, p)


def pref_path(prefs, key) -> str:
    """A Preferences path as the desktop resolves it (a folder inside Fizgig is stored relative)."""
    v = str(prefs.get(key) or "").strip()
    return v if not v or os.path.isabs(v) else os.path.normpath(os.path.join(BASE_DIR, v))


def load_state() -> dict:
    return _read_json(STATE_FILE)


def save_state(changes: dict):
    with _lock:
        s = load_state()
        s.update(changes)
        _write_json(STATE_FILE, s)


# ---- runs -------------------------------------------------------------------------------------------------------
def _plan(family_key, values, image_folder, resume=""):
    desc = get_family(family_key)
    if desc is None:
        raise ValueError(f"Unknown family: {family_key}")
    prefs = load_prefs()
    values = dict(values or {})
    if resume:
        values["RESUME_TRAINING"] = resume
    name = str(values.get("LORA_NAME") or "").strip() or "my_lora"
    out_dir = str(values.get("LORA_OUTPUT_DIR") or "").strip() or OUTPUT_DEFAULT
    values.update(LORA_OUTPUT_DIR=out_dir, DATASET_CONFIG=os.path.join(DATASET_DIR, f"{_safe(name)}_web.toml"))
    captioner = pref_path(prefs, "krea2_text_encoder")
    raw_ovr = prefs.get("caption_qwen_instructions")
    inputs = form.inputs(
        desc, values, models={k: pref_path(prefs, k) for k in form.model_keys(desc)}, python=sys.executable,
        repo_dir=BASE_DIR, image_folder=str(image_folder or "").strip(),
        cache_root=pref_path(prefs, "cache_dir"), samples_dir=os.path.join(out_dir, "sample"),
        inference_int8=str(prefs.get("inference_int8", "")).strip() in ("1", "True", "true"),
        captioner=captioner if captioner and os.path.isfile(captioner) else "",
        caption_overrides=dict(raw_ovr) if isinstance(raw_ovr, dict) else {})
    return desc, inputs, launch.plan(desc, inputs)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80] or "run"


def latest_state(out_dir, name) -> str:
    """The run's newest resumable state folder (<name>-NNNNNN-state), or ""."""
    try:
        found = [d for d in os.listdir(out_dir) if re.fullmatch(re.escape(name) + r"-\d{6}-state", d)]
    except OSError:
        return ""
    return os.path.join(out_dir, max(found)) if found else ""


def _start(family_key, values, image_folder, resume=""):
    with _lock:
        if _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running"):
            return {"error": "A run is already going. Stop or pause it first."}
    desc, inputs, plan = _plan(family_key, values, image_folder, resume)
    if plan.problems:
        return {"problems": plan.problems}
    out_dir = inputs["LORA_OUTPUT_DIR"]
    for stale in (".pause_requested", ".sample_override.json"):          # as the desktop's Start clears them
        try:
            os.remove(os.path.join(out_dir, stale))
        except OSError:
            pass
    for d in plan.dirs:
        os.makedirs(d, exist_ok=True)
    for path, text in plan.files:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with _lock:
        _run.update(state="starting", stage="", lora=inputs["LORA_NAME"], output_dir=out_dir, family=family_key,
                    values=dict(values), image_folder=image_folder, pause=False)
    save_state({"last_run": {"family": family_key, "values": values, "image_folder": image_folder}})
    for line in plan.console:
        append_log(line + "\n")
    threading.Thread(target=_run_stages, args=(plan.stages, out_dir), daemon=True).start()
    return {"ok": True, "stages": [s.name for s in plan.stages]}


def _popen(cmd):
    """As the desktop's run_subprocess: UTF-8, unbuffered, the chosen GPU, its own process group (and below-normal
    priority on Windows, so the desktop stays smooth)."""
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    want = str(load_prefs().get("cuda_device") or "").strip()
    if want:
        env["CUDA_VISIBLE_DEVICES"] = want
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                               | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env, cwd=BASE_DIR, **kw)


def _run_stages(stages, out_dir):
    append_log(f"\n=== Fizgig run, {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    with _lock:
        _run["state"] = "running"
    for stage in stages:
        with _lock:
            if _run["state"] == "stopped":
                return
            _run["stage"] = stage.name
        append_log(f"\n▶ {stage.name}\n")
        try:
            proc = _popen(stage.cmd)
        except OSError as e:
            append_log(f"✖ {stage.name} could not start: {e}\n")
            with _lock:
                _run.update(state="failed", proc=None)
            return
        with _lock:
            _run["proc"] = proc
        for line in proc.stdout:
            append_log(line)
        ret = proc.wait()
        with _lock:
            stopped = _run["state"] == "stopped"
        if stopped:
            append_log(f"\n■ Stopped during {stage.name}.\n")
            return
        if ret != 0:
            append_log(f"\n✖ {stage.name} failed (exit code {ret}).\n")
            with _lock:
                _run.update(state="failed", proc=None)
            return
    paused = os.path.exists(os.path.join(out_dir, ".pause_requested")) or _run["pause"]
    with _lock:
        _run.update(state="paused" if paused else "completed", proc=None, stage="")
    append_log("\n⏸ Paused. Resume continues from the saved state.\n" if paused else "\n✔ Run complete.\n")


def _stop():
    with _lock:
        proc = _run["proc"]
        if proc is None or proc.poll() is not None:
            return False
        _run["state"] = "stopped"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    return True


# ---- the HTTP side ----------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    password: str | None = None
    local_only = True

    def log_message(self, fmt, *args):            # quiet: the console is for the run
        pass

    # -- replies
    def _send(self, code, body: bytes, ctype, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, code=200, extra=()):
        self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _file(self, path, ctype=None):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return self.json({"error": "not found"}, 404)
        self._send(200, data, ctype or mimetypes.guess_type(path)[0] or "application/octet-stream")

    # -- access
    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
        return not self.local_only or host in LOCAL_HOSTS

    def _session(self):
        return hmac.new(TOKEN.encode(), (self.password or "").encode(), hashlib.sha256).hexdigest()

    def _logged_in(self):
        if not self.password:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "fizgig_session" and hmac.compare_digest(v, self._session()):
                return True
        return False

    def _api_ok(self, qs=None):
        token = self.headers.get("X-Fizgig-Token") or ((qs or {}).get("t") or [""])[0]
        if not hmac.compare_digest(token, TOKEN):
            self.json({"error": "Reload the page."}, 403)
            return False
        if not self._logged_in():
            self.json({"error": "Log in first.", "login": True}, 401)
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    # -- GET
    def do_GET(self):
        if not self._host_ok():
            return self.json({"error": "bad host"}, 403)
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
                page = f.read().replace("__FIZGIG_TOKEN__", TOKEN)
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path in ("/app.js", "/style.css"):
            return self._file(os.path.join(STATIC_DIR, path[1:]),
                              "application/javascript; charset=utf-8" if path.endswith(".js") else "text/css")
        if path in ("/logo.jpg", "/icon.png"):
            return self._file(os.path.join(BASE_DIR, path[1:]))
        if not path.startswith("/api/") or not self._api_ok(qs):
            return None if path.startswith("/api/") else self.json({"error": "not found"}, 404)

        if path == "/api/schema":
            return self.json({**get_schema(), "logged_in": True})
        if path == "/api/prefs":
            p = load_prefs()
            return self.json({"models": {k: str(p.get(k) or "") for d in training_families()
                                         for k in form.model_keys(d)},
                              "cache_dir": str(p.get("cache_dir") or "")})
        if path == "/api/state":
            return self.json(load_state())
        if path == "/api/stats":
            with _lock:
                return self.json(dict(_stats))
        if path == "/api/run":
            with _lock:
                r = {k: v for k, v in _run.items() if k != "proc"}
                r["running"] = _run["proc"] is not None and _run["proc"].poll() is None
            r["resume"] = latest_state(r["output_dir"], r["lora"]) if r["lora"] else ""
            return self.json(r)
        if path == "/api/log":
            return self._log_stream(int((qs.get("since") or ["0"])[0] or 0))
        if path == "/api/browse":
            return self._browse((qs.get("path") or [""])[0])
        if path == "/api/samples":
            return self._samples((qs.get("dir") or [""])[0])
        if path == "/api/media":
            p = (qs.get("path") or [""])[0]
            if os.path.splitext(p)[1].lower() not in MEDIA or not os.path.isfile(p):
                return self.json({"error": "not a picture, sound or clip"}, 404)
            return self._file(p)
        if path == "/api/captions":
            return self._captions((qs.get("folder") or [""])[0], (qs.get("ext") or [".txt"])[0])
        return self.json({"error": "not found"}, 404)

    def _log_stream(self, since):
        """Server-sent events: every console line after `since`, then new ones as they arrive."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = since
        try:
            while True:
                with _lock:
                    fresh = [(n, t) for n, t in _log if n > last]
                    if not fresh:
                        _log_event.wait(timeout=15)
                        fresh = [(n, t) for n, t in _log if n > last]
                if fresh:
                    for n, t in fresh:
                        self.wfile.write(f"id: {n}\ndata: {json.dumps(t)}\n\n".encode("utf-8"))
                    last = fresh[-1][0]
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _browse(self, target):
        target = target or BASE_DIR
        if not os.path.isdir(target):
            target = os.path.dirname(target) if os.path.isdir(os.path.dirname(target)) else BASE_DIR
        dirs, files = [], []
        try:
            for e in sorted(os.scandir(target), key=lambda e: e.name.lower()):
                if e.name.startswith("."):
                    continue
                (dirs if e.is_dir() else files).append({"name": e.name, "path": e.path})
        except OSError:
            pass
        parent = os.path.dirname(target)
        return self.json({"path": target, "parent": parent if parent != target else "", "dirs": dirs,
                          "files": files[:2000]})

    def _samples(self, folder):
        try:
            items = [e for e in os.scandir(folder) if os.path.splitext(e.name)[1].lower() in MEDIA]
        except OSError:
            return self.json({"items": []})
        items.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        return self.json({"items": [{"name": e.name, "path": e.path, "time": e.stat().st_mtime}
                                    for e in items[:300]]})

    def _captions(self, folder, ext):
        out = []
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return self.json({"items": []})
        for n in names:
            stem, e = os.path.splitext(n)
            if e.lower() not in MEDIA:
                continue
            cap = os.path.join(folder, stem + ext)
            try:
                with open(cap, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                text = None
            out.append({"name": n, "path": os.path.join(folder, n), "caption": text})
        return self.json({"items": out})

    # -- POST
    def do_POST(self):
        if not self._host_ok():
            return self.json({"error": "bad host"}, 403)
        path = urlparse(self.path).path
        if path == "/api/login":
            if self.headers.get("X-Fizgig-Token") != TOKEN:
                return self.json({"error": "Reload the page."}, 403)
            pwd = str(self._body().get("password") or "")
            if self.password and not hmac.compare_digest(pwd, self.password):
                time.sleep(1.0)
                return self.json({"error": "Wrong password."}, 401)
            return self.json({"ok": True}, extra=[("Set-Cookie", f"fizgig_session={self._session()}; Path=/; "
                                                                 "HttpOnly; SameSite=Strict")])
        if not self._api_ok():
            return None
        b = self._body()
        try:
            if path == "/api/form/state":
                desc = get_family(b.get("family", ""))
                if desc is None:
                    return self.json({"error": "unknown family"}, 400)
                cap = pref_path(load_prefs(), "krea2_text_encoder")
                return self.json(form.state(desc, b.get("values") or {}, b.get("image_folder") or "",
                                            captioner=cap if cap and os.path.isfile(cap) else ""))
            if path == "/api/preset":
                desc = get_family(b.get("family", ""))
                preset = dict(desc.presets or ()).get(b.get("name")) if desc else None
                if preset is None:
                    return self.json({"error": "unknown preset"}, 400)
                return self.json({"values": form.apply_preset(desc, b.get("values") or {}, preset)})
            if path == "/api/plan":
                _, _, plan = _plan(b.get("family", ""), b.get("values") or {}, b.get("image_folder") or "")
                return self.json({"problems": plan.problems, "stages": [{"name": s.name, "cmd": s.cmd}
                                                                         for s in plan.stages],
                                  "dataset_toml": plan.dataset_toml})
            if path == "/api/start":
                return self.json(_start(b.get("family", ""), b.get("values") or {}, b.get("image_folder") or ""))
            if path == "/api/resume":
                with _lock:
                    last = {k: _run[k] for k in ("family", "values", "image_folder", "lora", "output_dir")}
                if not last["family"]:
                    lr = load_state().get("last_run") or {}
                    last = {"family": lr.get("family", ""), "values": lr.get("values") or {},
                            "image_folder": lr.get("image_folder", ""),
                            "lora": (lr.get("values") or {}).get("LORA_NAME", ""),
                            "output_dir": (lr.get("values") or {}).get("LORA_OUTPUT_DIR") or OUTPUT_DEFAULT}
                state = latest_state(last["output_dir"], last["lora"])
                if not state:
                    return self.json({"error": "No saved state to resume from in the output folder."}, 400)
                return self.json(_start(last["family"], last["values"], last["image_folder"], resume=state))
            if path == "/api/pause":
                with _lock:
                    out, running = _run["output_dir"], _run["proc"] is not None and _run["proc"].poll() is None
                if not running:
                    return self.json({"error": "Nothing is running."}, 400)
                with open(os.path.join(out, ".pause_requested"), "w", encoding="utf-8") as f:
                    f.write("pause\n")
                with _lock:
                    _run["pause"] = True
                append_log("\n⏸ Pause requested: the run stops cleanly at the end of this epoch.\n")
                return self.json({"ok": True})
            if path == "/api/stop":
                return self.json({"ok": _stop()})
            if path == "/api/prefs":
                allowed = {k for d in training_families() for k in form.model_keys(d)} | {"cache_dir"}
                update_prefs({k: str(v) for k, v in (b.get("changes") or {}).items() if k in allowed})
                return self.json({"ok": True})
            if path == "/api/state":
                save_state({k: v for k, v in b.items() if k in ("family", "image_folder", "values", "tab")})
                return self.json({"ok": True})
            if path == "/api/captions/save":
                folder, name, text = b.get("folder", ""), b.get("name", ""), b.get("text", "")
                stem = os.path.splitext(os.path.basename(name))[0]
                if not stem or not os.path.isdir(folder):
                    return self.json({"error": "bad folder or file"}, 400)
                with open(os.path.join(folder, stem + (b.get("ext") or ".txt")), "w", encoding="utf-8") as f:
                    f.write(text)
                return self.json({"ok": True})
        except ValueError as e:
            return self.json({"error": str(e)}, 400)
        return self.json({"error": "not found"}, 404)


def run_server(host, port, password=None):
    Handler.password = password or None
    Handler.local_only = not password
    threading.Thread(target=_poll_stats_loop, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    logger.info(f"Fizgig web UI on http://{shown}:{port}/" + (" (password required)" if password else ""))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fizgig's web UI")
    ap.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default=None, help="Address to listen on (default: localhost; every address with a "
                                                 "password)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--password", default=None, help="Login password (default: VNC_PASSWORD or FIZGIG_PASSWORD)")
    a = ap.parse_args(argv)
    pwd = a.password or os.environ.get("VNC_PASSWORD") or os.environ.get("FIZGIG_PASSWORD")
    host = a.host or ("0.0.0.0" if pwd else "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not pwd:
        sys.exit("Listening beyond localhost needs a password (--password, VNC_PASSWORD or FIZGIG_PASSWORD).")
    run_server(host, a.port, pwd)


if __name__ == "__main__":
    main()
