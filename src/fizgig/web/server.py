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
from fizgig.web import live
from fizgig.web.port import PORT_DEFAULT, PORT_WALK, listening, on_pod, saved_port
from fizgig.web.schema import get_schema

logger = logging.getLogger("fizgig.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PREFS_FILE = os.path.join(BASE_DIR, "prefs.json")
STATE_FILE = os.path.join(BASE_DIR, "web_state.json")      # the web UI's own memory (git-ignored)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DEFAULT = os.path.join(BASE_DIR, "output_loras")
MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif", ".wav", ".mp3", ".flac", ".m4a", ".mp4",
         ".webm"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

TOKEN = secrets.token_urlsafe(24)       # this server's session token, placed in the page it serves

_lock = threading.Lock()
_login_lock = threading.Lock()
_login_fails: dict = {}    # client address -> (wrong guesses, locked until, last miss)
_run = {"proc": None, "state": "idle", "stage": "", "lora": "", "output_dir": "", "family": "", "values": {},
        "image_folder": "", "pause": False}
_log: list[tuple[int, str]] = []        # (line number, text), the last 5000
_log_n = [0]
_log_event = threading.Condition(_lock)
_stats = {"vram": [0, 0], "ram": [0, 0], "vram_peak": 0, "ram_peak": 0, "device_name": "GPU"}


# no console window for helper commands: started from the desktop app the web UI has none, so each
# nvidia-smi / taskkill would flash one up on Windows
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# ---- hardware meters (as the desktop status bar) ---------------------------------------------------------------
def list_gpus():
    """[{index, name, total_gb, used_gb, uuid}] from nvidia-smi (every card, whatever CUDA_VISIBLE_DEVICES says)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,uuid",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
                             **_NO_WINDOW)
        rows = []
        for line in out.stdout.strip().splitlines():
            i, n, t, u, uid = [x.strip() for x in line.split(",")][:5]
            rows.append({"index": int(i), "name": n, "total_gb": int(t) / 1024, "used_gb": int(u) / 1024,
                         "uuid": uid})
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def _chosen_gpu(rows, want):
    """The row for Preferences' cuda_device (an index or a GPU-… uuid), else the first card."""
    want = str(want or os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    for r in rows:
        if want and (want == str(r["index"]) or want == r["uuid"]):
            return r
    return rows[0] if rows else None


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
        want = str(os.environ.get("CUDA_VISIBLE_DEVICES") or load_prefs().get("cuda_device") or "").split(",")[0]
        want = want.strip()
        try:
            if nvml is not None:
                h = (nvml.nvmlDeviceGetHandleByUUID(want) if want.startswith("GPU-") else
                     nvml.nvmlDeviceGetHandleByIndex(int(want) if want.isdigit() else 0))
                m = nvml.nvmlDeviceGetMemoryInfo(h)
                vu, vt = int(m.used), int(m.total)
                name = nvml.nvmlDeviceGetName(h)
                name = name.decode("utf-8", "ignore") if isinstance(name, bytes) else name
        except Exception:
            pass
        if not vt:                                   # no pynvml: nvidia-smi, as the desktop status bar falls back
            r = _chosen_gpu(list_gpus(), want)
            if r:
                name, vu, vt = r["name"], int(r["used_gb"] * 2 ** 30), int(r["total_gb"] * 2 ** 30)
        ru = rt = 0
        if psutil is not None:
            vm = psutil.virtual_memory()
            ru, rt = vm.total - vm.available, vm.total
        with _lock:
            _stats.update(vram=[vu, vt], ram=[ru, rt], device_name=name, vram_peak=max(_stats["vram_peak"], vu),
                          ram_peak=max(_stats["ram_peak"], ru))
            mine = {"running": _run["proc"] is not None and _run["proc"].poll() is None, "state": _run["state"],
                    "lora": _run["lora"], "output_dir": _run["output_dir"]}
        live.write(live.LIVE_WEB, mine)          # the desktop app reads this before starting a run
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


def _checkpoint_kind(path) -> str:
    """'finetune' (a whole model, the source file with the trained tensors replaced) or 'lora', from the file's
    metadata header - a few KB read, never the tensors."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < (64 << 20):
                return "lora"
            meta = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError):
        return "lora"
    return "finetune" if meta.get("fizgig_finetune") else "lora"


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
    if live.busy(live.desktop_state()):
        return {"error": "The desktop app is training. Pause or stop that run first (it shows above)."}
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
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")     # a GPU number means the same card as in nvidia-smi
    want = str(load_prefs().get("cuda_device") or "").strip()
    if want and not os.environ.get("CUDA_VISIBLE_DEVICES"):  # one set where Fizgig was started wins, as the desktop
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
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, **_NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    return True


# ---- uploads: a training folder sent from the browser's computer ------------------------------------------------
# what the trainers read (dataset/image_dataset.py: pictures, .mp4 clips, songs) + captions, ACE-Step's .json song
# details and zips. Not .gif / .webm: nothing trains on them.
UPLOAD_EXTS = (MEDIA - {".gif", ".webm"}) | {".txt", ".caption", ".json", ".zip"}
UPLOAD_DEFAULT = os.path.join(BASE_DIR, "training_data")     # when Preferences has no input folder set


def uploads_root() -> str:
    """Where uploaded folders go: the desktop's Preferences > Input Directories folder (a pod sets it to
    /workspace/datasets), else training_data/ inside Fizgig."""
    return pref_path(load_prefs(), "input_dataset_dir") or UPLOAD_DEFAULT


def _upload_name(name, what) -> str:
    """One plain name, kept as it is ("Don't Stop (Live) #1.wav" is fine): no folders, no '..', nothing hidden, and
    nothing Windows can't store, so every upload stays inside uploads_root()."""
    name = str(name or "").strip()
    bad = (not name or name.startswith(".") or name.endswith((".", " "))
           or re.search(r'[/\\<>:"|?*\x00-\x1f]', name)
           or os.path.splitext(name)[0].upper() in _WIN_RESERVED)
    if bad:
        raise ValueError(f"{what} {name!r}: names can't contain / \\ < > : \" | ? *, start with a dot, or end "
                         f"with a dot or space.")
    return name


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _save_upload(folder, name, rfile, length) -> dict:
    folder, name = _upload_name(folder, "Folder name"), _upload_name(name, "File name")
    ext = os.path.splitext(name)[1].lower()
    if ext not in UPLOAD_EXTS:
        raise ValueError(f"{name}: Fizgig doesn't train on {ext or 'this'} files. It takes pictures, .mp4 clips, "
                         f"songs (.wav .mp3 .flac .m4a), caption .txt files, .json song details and .zip files.")
    dest = os.path.join(uploads_root(), folder)
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(dest, f".{name}.part")
    left = length
    try:
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
    except OSError:
        left = left or 1
    if left:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise ValueError(f"{name}: the upload was cut off. Upload it again.")
    saved, skipped, replaced = [], [], []
    if ext == ".zip":
        import zipfile
        try:
            with zipfile.ZipFile(tmp) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if info.is_dir() or not base or "__MACOSX" in info.filename or base.startswith("."):
                        continue
                    if os.path.splitext(base)[1].lower() not in UPLOAD_EXTS - {".zip"}:
                        skipped.append(base)
                        continue
                    try:
                        base = _upload_name(base, "File name")
                    except ValueError:
                        skipped.append(base)
                        continue
                    if base in saved or os.path.exists(os.path.join(dest, base)):
                        replaced.append(base)
                    with z.open(info) as src, open(os.path.join(dest, base), "wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
                    saved.append(base)
        except zipfile.BadZipFile:
            raise ValueError(f"{name}: not a readable zip.")
        finally:
            os.remove(tmp)
    else:
        if os.path.exists(os.path.join(dest, name)):
            replaced.append(name)
        os.replace(tmp, os.path.join(dest, name))
        saved.append(name)
    return {"folder": dest, "saved": sorted(set(saved)), "skipped": skipped, "replaced": replaced}


# ---- model downloads: the desktop's "Download models for me" (fizgig.scripts.fetch_models) ----------------------
_fetch = {"proc": None, "state": "idle", "family": "", "file": "", "done": 0, "total": 0, "lines": []}


def _fetch_status() -> dict:
    with _lock:
        out = {k: v for k, v in _fetch.items() if k not in ("proc", "lines")}
        out["lines"] = _fetch["lines"][-40:]
    out["env_token"] = bool(os.environ.get("HF_TOKEN"))
    return out


def _start_fetch(family, token="", optional=False):
    """The family's model files plus the helper models, into <repo>/models (a pod links that to its volume), paths
    written to Preferences by the fetcher. A token is passed to that one download only, never written to disk."""
    desc = get_family(family)
    if desc is None:
        raise ValueError(f"Unknown family: {family}")
    token = (token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if any(f.gated and f.repo for f in desc.model_files) and not token:
        raise ValueError(f"{desc.display_name}'s files need a Hugging Face token.")
    with _lock:
        if _fetch["proc"] is not None and _fetch["proc"].poll() is None:
            raise ValueError("A model download is already running.")
    cmd = [sys.executable, "-m", "fizgig.scripts.fetch_models", "--progress", "--family", "tools", "--family", family]
    if optional or not desc.fetch_optional_label:
        cmd.append("--include-optional")          # no optional tick on the family: its whole list, as the desktop
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(BASE_DIR, "src"))
    if token:
        env["HF_TOKEN"] = token
    kw = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env, cwd=BASE_DIR, **kw)
    with _lock:
        _fetch.update(proc=proc, state="running", family=family, file="", done=0, total=0, lines=[])
    threading.Thread(target=_fetch_watch, args=(proc,), daemon=True).start()


def _fetch_watch(proc):
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = re.match(r"\s*\[progress\] (\d+) (\d+) (.+)$", line)
        with _lock:
            if m:
                _fetch.update(done=int(m.group(1)), total=int(m.group(2)), file=m.group(3))
            else:
                _fetch["lines"].append(line)
                del _fetch["lines"][:-200]
    code = proc.wait()
    with _lock:
        if _fetch["state"] == "running":
            _fetch["state"] = "done" if code == 0 else "failed"


def _cancel_fetch():
    """Partial files resume next time; Preferences are only written when a file is complete."""
    with _lock:
        proc = _fetch["proc"]
        if proc is None or proc.poll() is not None:
            return False
        _fetch["state"] = "cancelled"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, **_NO_WINDOW)
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

    def _stream(self, path, attachment=True):
        """A file sent in pieces (a fine-tune checkpoint is ~10 GB; it never sits in memory), honouring a byte
        range: a player dragged along a song or clip asks for "Range: bytes=N-" and gets just that (206), and a
        dropped download can carry on where it stopped. attachment: a download (else played in the page)."""
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            return self.json({"error": "not found"}, 404)
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:                                         # "bytes=-N": the last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                f.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            code = 206
        name = os.path.basename(path).replace('"', "")
        with f:
            f.seek(start)
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if attachment
                             else mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            left = end - start + 1
            try:
                while left > 0 and (chunk := f.read(min(left, 8 << 20))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

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
                              "cache_dir": str(p.get("cache_dir") or ""),
                              "web_port": saved_port(p), "port_now": self.server.server_address[1], "on_pod": on_pod(), "port_fixed": getattr(self, "port_fixed", False),
                              "cuda_device": str(p.get("cuda_device") or ""), "gpus": list_gpus(),
                              "gpu_env_locked": bool(os.environ.get("CUDA_VISIBLE_DEVICES"))})
        if path == "/api/state":
            return self.json(load_state())
        if path == "/api/fetch":
            return self.json(_fetch_status())
        if path == "/api/upload/root":
            return self.json({"root": uploads_root()})
        if path == "/api/stats":
            with _lock:
                return self.json(dict(_stats))
        if path == "/api/run":
            with _lock:
                r = {k: v for k, v in _run.items() if k != "proc"}
                r["running"] = _run["proc"] is not None and _run["proc"].poll() is None
            r["resume"] = latest_state(r["output_dir"], r["lora"]) if r["lora"] else ""
            d = live.desktop_state()
            r["desktop"] = ({k: d.get(k) for k in ("running", "state", "lora", "output_dir", "family", "stage",
                                                   "percent", "last_result", "ack", "unresponsive")} if d else None)
            r["queue"] = live.queue_summary() if d else []
            return self.json(r)
        if path == "/api/desktop/log":
            text, nxt = live.read_log(int((qs.get("offset") or ["0"])[0] or 0), tail=bool(qs.get("tail")))
            return self.json({"text": text, "offset": nxt})
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
            return self._stream(p, attachment=False)       # in pieces, with byte ranges: songs and clips seek
        if path == "/api/checkpoints":
            return self._checkpoints((qs.get("dir") or [""])[0])
        if path == "/api/download":
            p = (qs.get("path") or [""])[0]
            if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
                return self.json({"error": "not a checkpoint"}, 404)
            return self._stream(p)
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

    def _checkpoints(self, folder):
        """The run's saved LoRAs / checkpoints in its output folder, newest first."""
        try:
            items = [e for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(".safetensors")]
        except OSError:
            return self.json({"items": []})
        now = time.time()
        items = [e for e in items if now - e.stat().st_mtime > 5]       # not one still being written
        items.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        return self.json({"items": [{"name": e.name, "path": e.path, "size": e.stat().st_size,
                                     "time": e.stat().st_mtime, "kind": _checkpoint_kind(e.path)}
                                    for e in items]})

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
            if self.password:
                # wrong guesses lock out only the device making them, for longer each time (1 s, 2 s, ... 60 s),
                # forgotten after 15 minutes; nothing waits while holding the lock, so the owner's own login on
                # another device is never held up. Bytes, so a non-ASCII password compares.
                ip, now = self.client_address[0], time.time()
                with _login_lock:
                    n, until, last = _login_fails.get(ip, (0, 0.0, 0.0))
                    if now - last > 900:
                        n, until = 0, 0.0
                    if now < until:
                        return self.json({"error": f"Too many wrong passwords. Try again in {int(until - now) + 1} s."},
                                         429)
                    if not hmac.compare_digest(pwd.encode("utf-8"), self.password.encode("utf-8")):
                        n += 1
                        _login_fails[ip] = (n, now + min(60.0, float(n)), now)
                        if len(_login_fails) > 1000:                  # many addresses: keep the recent ones
                            for k in sorted(_login_fails, key=lambda k: _login_fails[k][2])[:500]:
                                _login_fails.pop(k, None)
                        return self.json({"error": "Wrong password."}, 401)
                    _login_fails.pop(ip, None)
            return self.json({"ok": True}, extra=[("Set-Cookie", f"fizgig_session={self._session()}; Path=/; "
                                                                 "HttpOnly; SameSite=Strict")])
        if not self._api_ok():
            return None
        if path == "/api/upload":                    # a raw file body, not JSON
            qs = parse_qs(urlparse(self.path).query)
            try:
                return self.json(_save_upload((qs.get("folder") or [""])[0], (qs.get("name") or [""])[0],
                                              self.rfile, int(self.headers.get("Content-Length") or 0)))
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
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
            if path == "/api/fetch":
                _start_fetch(b.get("family", ""), b.get("token", ""), bool(b.get("optional")))
                return self.json({"ok": True})
            if path == "/api/fetch/cancel":
                return self.json({"ok": _cancel_fetch()})
            if path == "/api/stop":
                return self.json({"ok": _stop()})
            if path == "/api/desktop":
                if not live.read_fresh(live.LIVE_DESKTOP):          # commands need the app itself answering
                    return self.json({"error": "The desktop app isn't open."}, 400)
                live.send_desktop(str(b.get("cmd") or ""))
                return self.json({"ok": True})
            if path == "/api/prefs":
                allowed = {k for d in training_families() for k in form.model_keys(d)} | {"cache_dir"}
                ch = b.get("changes") or {}
                upd = {k: str(v) for k, v in ch.items() if k in allowed}
                if "cuda_device" in ch:
                    v = str(ch["cuda_device"] or "").strip()
                    if v and v not in {g["uuid"] for g in list_gpus()} | {str(g["index"]) for g in list_gpus()}:
                        raise ValueError("That GPU isn't in this machine.")
                    upd["cuda_device"] = v
                if "web_port" in ch:
                    port = int(ch["web_port"])
                    if not 1024 <= port <= 65535:
                        raise ValueError("The port must be between 1024 and 65535.")
                    upd["web_port"] = str(port)
                update_prefs(upd)
                return self.json({"ok": True})
            if path == "/api/state":
                save_state({k: v for k, v in b.items() if k in ("family", "image_folder", "values", "tab")})
                return self.json({"ok": True})
            if path == "/api/captions/save":
                folder, name, text = b.get("folder", ""), b.get("name", ""), b.get("text", "")
                stem = os.path.splitext(os.path.basename(name))[0]
                ext = str(b.get("ext") or ".txt")
                if not re.fullmatch(r"(\.[A-Za-z0-9_]{1,12})?\.(txt|caption|tags)", ext):
                    return self.json({"error": "A caption file ends in .txt, .caption or .tags."}, 400)
                if not stem or not os.path.isdir(folder):
                    return self.json({"error": "bad folder or file"}, 400)
                with open(os.path.join(folder, stem + ext), "w", encoding="utf-8") as f:
                    f.write(text)
                return self.json({"ok": True})
        except ValueError as e:
            return self.json({"error": str(e)}, 400)
        return self.json({"error": "not found"}, 404)


class _Server(ThreadingHTTPServer):
    # Windows' SO_REUSEADDR lets a second program bind a port another is already serving on, so a taken port would
    # look free there; only POSIX needs it (a quick restart rebinding a port still in TIME_WAIT).
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def _bind(host, start, walk):
    """The server on `start`, or the nearest free port above it when `walk`. Returns (server, port)."""
    last = None
    for port in range(start, min(65536, start + (PORT_WALK if walk else 1))):
        if listening(port):
            continue
        try:
            return _Server((host, port), Handler), port
        except OSError as e:
            last = e
    if walk:
        sys.exit(f"No free port from {start} to {start + PORT_WALK - 1}; set one with --port.")
    sys.exit(f"Port {start} is in use{f' ({last})' if last else ''}; pick another with --port.")


def run_server(host, port=None, password=None):
    """`port` given = that port only. Otherwise the Preferences port (8081 until one is saved); when it is taken, the
    nearest free one above it, saved to Preferences so every later launch gets that same port. A pod always uses
    8081: it is the port its proxy forwards."""
    Handler.password = password or None
    Handler.local_only = not password
    Handler.port_fixed = port is not None
    if port is not None or on_pod():
        srv, port = _bind(host, port or PORT_DEFAULT, walk=False)
    else:
        want = saved_port(load_prefs())
        srv, port = _bind(host, want, walk=True)
        if port != want:
            logger.info(f"Port {want} is in use; the web UI takes {port} and keeps it (Preferences > Web UI)")
        if str(load_prefs().get("web_port") or "") != str(port):
            update_prefs({"web_port": str(port)})
    threading.Thread(target=_poll_stats_loop, daemon=True).start()
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    logger.info(f"Fizgig web UI on http://{shown}:{port}/" + (" (password required)" if password else ""))
    def _term(*_a):                            # a service manager's stop: end the run with it, as Ctrl+C does
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _term)
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
        live.remove(live.LIVE_WEB)
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fizgig's web UI")
    ap.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default=None, help="Address to listen on (default: localhost; every address with a "
                                                 "password)")
    ap.add_argument("--port", type=int, default=None, help="Port to serve on (default: the one in Preferences, "
                                                            "8081 at first; a taken one moves to the next free one)")
    ap.add_argument("--password", default=None, help="Login password, 8 characters or more (default: "
                                                     "FIZGIG_PASSWORD, then on a pod VNC_PASSWORD, then the one in "
                                                     "Preferences)")
    a = ap.parse_args(argv)
    # a pod: its VNC_PASSWORD, the remote desktop's, so one password opens both. Elsewhere VNC_PASSWORD is ignored (a
    # stray one would open the web UI to the network while Preferences says no password).
    saved = str(load_prefs().get("web_password") or "").strip()
    pwd = (a.password or os.environ.get("FIZGIG_PASSWORD")
           or ((os.environ.get("VNC_PASSWORD") or saved) if on_pod() else saved))
    pod_pwd = on_pod() and not a.password and not os.environ.get("FIZGIG_PASSWORD") and pwd == os.environ.get(
        "VNC_PASSWORD")
    if pwd and len(pwd) < 8 and not pod_pwd:          # a pod's own VNC password is taken at any length
        sys.exit("The web UI password needs at least 8 characters (Preferences > Web UI, web_password in "
                 "prefs.json, FIZGIG_PASSWORD or --password).")
    host = a.host or ("0.0.0.0" if pwd else "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not pwd:
        sys.exit("Listening beyond localhost needs a password: set one in Preferences > Web UI (web_password in "
                 "prefs.json), or use FIZGIG_PASSWORD or --password.")
    run_server(host, a.port, password=pwd)


if __name__ == "__main__":
    main()
