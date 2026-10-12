"""Run history: every training run the web UI or the desktop app starts, newest first - for the web UI's Runs &
outputs tab (what ran, on what, how it ended, its previews and checkpoints, and its settings to start again with).

One file both apps write (.fizgig_runs.json in the Fizgig folder, git-ignored): each change re-reads it under a lock
file and writes it whole, so the two apps never lose each other's records. A record:

    {id, source: "web" | "desktop", family, lora, output_dir, image_folder, values (the settings), started_at,
     ended_at, result: "running" | "finished" | "failed" | "stopped" | "paused", progress (the last figures)}
"""
import json
import os
import time
import uuid

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HISTORY_FILE = os.path.join(BASE_DIR, ".fizgig_runs.json")
KEEP = 300                                  # the newest runs kept


class _Lock:
    """A lock file the two apps share; a stale one (a crashed writer) is taken over after a few seconds."""

    def __init__(self):
        self.path = HISTORY_FILE + ".lock"
        self.fd = None

    def __enter__(self):
        for _ in range(100):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 5:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        return self                         # could not lock: carry on rather than lose the record

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.remove(self.path)
            except OSError:
                pass


def _read():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _write(runs):
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(runs[-KEEP:], f, default=str)
    os.replace(tmp, HISTORY_FILE)


def start(source, family, lora, output_dir, image_folder, values):
    """Record a run that is starting; returns its id."""
    rec = {"id": uuid.uuid4().hex[:12], "source": source, "family": family, "lora": lora, "output_dir": output_dir,
           "image_folder": image_folder, "values": values or {}, "started_at": time.time(), "ended_at": None,
           "result": "running", "progress": None}
    try:
        with _Lock():
            runs = _read()
            runs.append(rec)
            _write(runs)
    except OSError:
        pass
    return rec["id"]


def end(run_id, result, progress=None):
    """A run ended (finished / failed / stopped / paused), with its last figures."""
    if not run_id:
        return
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id:
                    r.update(result=result, ended_at=time.time(), progress=progress or r.get("progress"))
            _write(runs)
    except OSError:
        pass


def runs():
    """Newest first. A run still marked running whose app is gone for good (a crash) reads as ended unknown, a week on."""
    out = list(reversed(_read()))
    for r in out:
        if r.get("result") == "running" and time.time() - float(r.get("started_at") or 0) > 7 * 86400:
            r["result"] = "unknown"
    return out


def get(run_id):
    return next((r for r in _read() if r.get("id") == run_id), None)
