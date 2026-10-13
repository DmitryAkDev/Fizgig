"""The desktop app and the web UI showing each other's runs.

Each side writes a small heartbeat file while it is open: the desktop app its run (state, LoRA, output folder,
progress, and a copy of its training console), the web UI whether it is training. The web UI reads the desktop's to
show and control a desktop run and its queue; each side reads the other's to refuse a second run on the same GPU.
The web UI's Pause / Stop / Resume for a desktop run go back through a command file the desktop app picks up, so the
desktop's own state machine (queue, pause sidecar, buttons) does the work exactly as its own buttons would.
"""
import json
import os
import time

from fizgig.web.paths import user_file  # noqa: E402  (the web UI's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LIVE_DESKTOP = user_file("live_desktop.json")
LIVE_WEB = user_file("live_web.json")
DESKTOP_CMD = user_file("desktop_cmd.json")
DESKTOP_LOG = user_file("desktop_console.log")
DESKTOP_SAMPLES = user_file("desktop_samples.json")   # preview settings changed in the web UI
LAST_USED = os.path.join(BASE_DIR, ".last_used.json")
# the preview settings the two apps share, per model: steps, CFG, every N epochs and the Turbo (strength, or H3's
# steps and %). The web UI sends a change to the desktop app; the desktop keeps them in .last_used.json
SHARED_SAMPLE_KEYS = ("SAMPLE_STEPS", "SAMPLE_CFG_SCALE", "SAMPLE_EVERY_N_EPOCHS", "FAMILY_TURBO_STRENGTH",
                      "FAMILY_TURBO_STEPS", "FAMILY_TURBO_PACE")
_BOX = {"SAMPLE_STEPS": "steps", "SAMPLE_CFG_SCALE": "cfg", "SAMPLE_EVERY_N_EPOCHS": "every"}
QUEUE_FILE = os.path.join(BASE_DIR, "presets", "training_queue.json")
FRESH_S = 8.0               # a heartbeat older than this = that side is closed (or hung)
DESKTOP_CMDS = ("pause", "stop", "resume")


def write(path, data):
    data = dict(data, heartbeat=time.time(), pid=os.getpid())
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def read_fresh(path):
    """The other side's heartbeat file, or None when it is missing, stale or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and time.time() - float(d.get("heartbeat", 0)) < FRESH_S else None
    except (OSError, ValueError, TypeError):
        return None


def remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _pid_alive(pid, started=None) -> bool:
    """That process is running - and, given its start time, is the same process (Windows reuses PIDs)."""
    try:
        import psutil
        if not pid or not psutil.pid_exists(int(pid)):
            return False
        return started is None or abs(psutil.Process(int(pid)).create_time() - float(started)) < 1.0
    except Exception:
        return False


def desktop_state():
    """The desktop app's run, or None when the app is closed. A heartbeat gone quiet while its trainer is still
    running (the desktop busy for a moment - a model load) still counts, marked unresponsive, so the web UI keeps
    showing the run and never starts a second one on the GPU."""
    try:
        with open(LIVE_DESKTOP, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    if time.time() - float(d.get("heartbeat", 0) or 0) < FRESH_S:
        return d
    if _pid_alive(d.get("trainer_pid"), d.get("trainer_started")):
        return dict(d, unresponsive=True)
    return None


def busy(d) -> bool:
    """That side has a run going (training, caching, or pausing at the end of its epoch)."""
    return bool(d and (d.get("running") or d.get("state") in ("running", "pausing", "starting")))


def send_desktop(cmd):
    if cmd not in DESKTOP_CMDS:
        raise ValueError(f"unknown command {cmd!r}")
    tmp = DESKTOP_CMD + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cmd": cmd, "at": time.time()}, f)
    os.replace(tmp, DESKTOP_CMD)


def send_desktop_samples(family, values):
    """Preview settings changed in the web UI, for the open desktop app to take (merged with any not yet taken)."""
    vals = {k: str(v) for k, v in (values or {}).items() if k in SHARED_SAMPLE_KEYS}
    if not vals:
        return
    try:
        with open(DESKTOP_SAMPLES, encoding="utf-8") as f:
            pending = json.load(f)
    except (OSError, ValueError):
        pending = {}
    pending.setdefault(family, {}).update(vals)
    tmp = DESKTOP_SAMPLES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pending, f)
    os.replace(tmp, DESKTOP_SAMPLES)


def take_desktop_samples():
    """For the desktop app: {family key: {setting: value}} the web UI changed, claimed so it applies once."""
    claim = DESKTOP_SAMPLES + ".claimed"
    try:
        os.replace(DESKTOP_SAMPLES, claim)
    except OSError:
        return {}
    try:
        with open(claim, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    remove(claim)
    return d if isinstance(d, dict) else {}


def store_samples(last_used, family_key, gui_label, values):
    """Preview settings into the desktop's memory (its .last_used.json dict) as its Samples tab keeps them:
    sample_settings[label] (steps / cfg / every) and turbo_strengths[key]."""
    for k, v in values.items():
        if k in _BOX:
            last_used.setdefault("sample_settings", {}).setdefault(gui_label, {})[_BOX[k]] = str(v)
        elif k == "FAMILY_TURBO_STRENGTH":
            last_used.setdefault("turbo_strengths", {})[family_key] = str(v)
            try:
                last_used.setdefault("samples_turbo_live", {})[family_key] = float(v or 0) > 0
            except ValueError:
                pass


def write_samples_to_last_used(family_key, gui_label, values):
    """The desktop app is closed: the change goes straight into its .last_used.json, so it opens with it."""
    try:
        with open(LAST_USED, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    store_samples(d, family_key, gui_label, {k: v for k, v in values.items() if k in SHARED_SAMPLE_KEYS})
    tmp = LAST_USED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, LAST_USED)


def take_desktop_cmd():
    """For the desktop app: the waiting command (claimed, so it runs once), or None. Commands older than a minute
    are dropped - a Stop sent while the app was closed must not fire when it next opens."""
    claim = DESKTOP_CMD + ".claimed"
    try:
        os.replace(DESKTOP_CMD, claim)
    except OSError:
        return None
    try:
        with open(claim, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    remove(claim)
    if not isinstance(d, dict) or time.time() - float(d.get("at", 0) or 0) > 60:
        return None
    return d.get("cmd") if d.get("cmd") in DESKTOP_CMDS else None


def queue_summary():
    """The desktop's training queue as short rows for the web UI (it lives in presets/training_queue.json)."""
    try:
        with open(QUEUE_FILE, encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for it in items if isinstance(items, list) else ():
        if not isinstance(it, dict):
            continue
        p = it.get("preset") if isinstance(it.get("preset"), dict) else {}
        folder = str(it.get("image_folder") or "")
        out.append({"name": str(p.get("LORA_NAME") or os.path.basename(folder) or "(unnamed)"),
                    "family": str(it.get("architecture") or ""), "folder": folder,
                    "queued_at": str(it.get("queued_at") or "")})
    return out


def read_log(offset, limit=256 * 1024, tail=False):
    """New text in the desktop's console copy since byte `offset`: (text, next offset). A shorter file than the
    offset = a new run started it again, so reading starts from the top. `tail`: only the last 200 KB, for a page
    catching up (a phone should not read a whole day's log to see the current line)."""
    try:
        size = os.path.getsize(DESKTOP_LOG)
        if tail:
            offset = max(0, size - 200 * 1024)
        elif offset > size:
            offset = 0
        with open(DESKTOP_LOG, "rb") as f:
            f.seek(offset)
            data = f.read(limit)
        return data.decode("utf-8", "replace"), offset + len(data)
    except OSError:
        return "", 0
