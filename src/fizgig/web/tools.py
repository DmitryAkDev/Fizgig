"""The web UI's Extract tab: the desktop's Extract (exact weight SVD straight from the file, no model loaded - the
families/extract.py path) run as its CLI in a subprocess, so the server never loads torch; its progress and log come
back for the page. One extraction at a time."""

import os
import re
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_lock = threading.Lock()
_J = {"state": "idle", "log": [], "progress": None, "output": "", "error": "", "proc": None, "started": 0.0}


def families():
    """The models Extract offers: every family whose description lists the extract workbench (the desktop's list)."""
    from fizgig.families.registry import training_families
    return [d for d in training_families() if "extract" in (d.workbench or ()) and d.training_ready]


def info(desc, out_dir):
    groups = [{"label": g.label, "blocks": [{"id": b.id, "label": b.label} for b in g.blocks]}
              for g in desc.load_driver().block_map()] if desc.extract_presets else []
    return {"family": desc.key, "presets": [{"name": n, "blocks": sorted(dict(b))} for n, b in desc.extract_presets],
            "groups": groups, "out_dir": out_dir}


def free_name(path):
    """`path`, or path_2, path_3... - an extraction never overwrites a file."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def run(desc, source, output, rank, preset="", blocks=None):
    with _lock:
        if _J["state"] == "running":
            raise ValueError("An extraction is running - wait for it to finish.")
    if not source or not os.path.isfile(source):
        raise ValueError("Choose a source LoRA (that file isn't there).")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output = free_name(output)
    cmd = [sys.executable, os.path.join(BASE_DIR, "src", "fizgig", "scripts", "extract_lora.py"), "--family", desc.key,
           "--source", source, "--output", output, "--rank", str(int(rank))]
    if blocks:
        cmd += ["--blocks", ",".join(blocks)]
    elif preset:
        cmd += ["--preset", preset]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kw)
    with _lock:
        _J.update(state="running", log=[f"Extracting {os.path.basename(source)} at rank {rank} -> {output}"],
                  progress=None, output=output, error="", proc=proc, started=time.time())
    threading.Thread(target=_follow, args=(proc,), daemon=True).start()
    return output


def _follow(proc):
    buf = b""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in (b"\r", b"\n"):
            line = buf.decode("utf-8", "replace").strip()
            buf = b""
            if not line:
                continue
            m = re.match(r"^(\w+): (\d+)/(\d+)$", line)
            with _lock:
                if m:                                # the CLI's "\rSVD: i/n" progress
                    _J["progress"] = [int(m.group(2)), int(m.group(3))]
                else:
                    _J["log"] = (_J["log"] + [line])[-300:]
        else:
            buf += ch
    code = proc.wait()
    with _lock:
        ok = code == 0 and os.path.isfile(_J["output"])
        secs = time.time() - _J["started"]
        _J.update(state="done" if ok else "failed", proc=None,
                  error="" if ok else (_J["log"][-1] if _J["log"] else f"exit code {code}"))
        _J["log"].append(f"Done in {secs:.0f} s." if ok else "The extraction failed.")


def status():
    with _lock:
        return {k: _J[k] for k in ("state", "log", "progress", "output", "error")}
