"""The web UI's Repair Studio: the desktop tab's engine (fizgig/families/workbench.py), kept in a worker process
(scripts/studio_worker.py) so the model stays loaded between renders, driven from here.

One render runs at a time. A slider moved while it renders cancels it, and only the newest state renders next - the
desktop's cancel-and-refire. The page polls status() for progress and the newest pictures. The load plan (which model
files, preview checkpoint or speed LoRA, the preview settings) follows the desktop's (_repair_engine_plan_family).
"""
import json
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(BASE_DIR, ".fizgig_web_studio")
EV = "@@EV "

_lock = threading.Lock()
_S = {"proc": None, "sock": None, "port": 0, "state": "idle", "family": "", "primary": "", "donor": "", "primary_blocks": [],
      "donor_blocks": [], "hash": "", "status": "", "progress": None, "gen": 0, "shown": 0, "baseline": "",
      "tweaked": "", "pending": None, "rendering": False, "error": "", "saved": None, "profile": None,
      "used": 0.0, "log": []}


_send_lock = threading.Lock()


def _send(cmd):
    """A command to the worker, over its local socket (not stdin: see the worker's note on the C runtime's lock)."""
    p, sock = _S["proc"], _S["sock"]
    if p is None or p.poll() is not None or sock is None:
        raise RuntimeError("The Studio worker isn't running.")
    with _send_lock:
        sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))


def _reader(proc):
    for line in proc.stdout:
        if not line.startswith(EV):
            with _lock:
                _S["log"] = (_S["log"] + [line.rstrip()])[-200:]
            continue
        try:
            ev = json.loads(line[len(EV):])
        except ValueError:
            continue
        _event(ev)
    with _lock:
        if _S["proc"] is proc:
            _S.update(proc=None, state="idle", rendering=False, status="The Studio worker stopped.", progress=None)


def _event(ev):
    kind = ev.get("ev")
    nxt = None
    with _lock:
        if kind == "ready":
            _S["port"] = int(ev.get("port") or 0)
        elif kind == "status":
            _S["status"] = ev.get("text", "")
        elif kind == "step":
            _S["progress"] = [ev.get("done", 0), ev.get("total", 0)]
        elif kind in ("loaded", "donor"):
            _S.update({k: ev.get(k) for k in ("primary_blocks", "donor_blocks", "hash")})
            if kind == "loaded":
                _S.update(state="ready", family=ev.get("family", ""), primary=ev.get("primary", ""),
                          donor=ev.get("donor", ""), status="Loaded. Generating preview…", error="")
                _S["profile"] = _profile(ev.get("hash", ""))
            else:
                _S.update(donor=ev.get("donor", ""), status="Donor loaded. Enable it per block to mix it in."
                          if ev.get("donor") else "Donor unloaded.")
        elif kind == "rendered":
            _S.update(rendering=False, progress=None, baseline=ev.get("baseline", ""), tweaked=ev.get("tweaked", ""),
                      shown=ev.get("gen", 0), status="Ready.", error="")
            nxt, _S["pending"] = _S["pending"], None
        elif kind == "aborted":
            _S.update(rendering=False, progress=None)
            nxt, _S["pending"] = _S["pending"], None
        elif kind == "error":
            where = ev.get("where")
            _S.update(error=ev.get("msg", ""), progress=None,
                      status=("Preview error: " if where == "render" else "Couldn't load: " if where == "load" else "")
                      + ev.get("msg", ""))
            if where == "render":
                _S["rendering"] = False
                nxt, _S["pending"] = _S["pending"], None
            elif where == "load":
                _S["state"] = "idle"
        elif kind == "saved":
            _S["saved"] = {"out": ev.get("out"), "summary": ev.get("summary")}
            _S["status"] = "Saved."
        elif kind == "reset":
            _S.update(state="idle", primary="", donor="", primary_blocks=[], donor_blocks=[], hash="", profile=None,
                      rendering=False, pending=None, progress=None, status="Models unloaded.")
    if nxt is not None:
        render(nxt)


def _profile(h):
    """The Profiler's sidecar for this LoRA (same file hash), for the pills panel; None without one."""
    try:
        from fizgig.utils.lora_files import find_profile_for_hash
        from fizgig.web.server import load_prefs, pref_path
        d = pref_path(load_prefs(), "profiles_dir") or os.path.join(BASE_DIR, "output_loras", "profiles")
        prof = find_profile_for_hash(d, h)
        if not prof:
            return None
        rep = os.path.join(os.path.dirname(prof.get("_sidecar_path", "")), prof.get("html_report", "") or "")
        return {"lora_name": prof.get("lora_name", ""), "created": str(prof.get("created", ""))[:10],
                "blocks": [{"name": b.get("name"), "category": b.get("category"), "pct": b.get("pct")}
                           for b in (prof.get("top_active_blocks") or [])[:8]],
                "report": rep if os.path.isfile(rep) else ""}
    except Exception:
        return None


def _ensure_worker():
    with _lock:
        if _S["proc"] is not None and _S["proc"].poll() is None:
            return
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen([sys.executable, "-m", "fizgig.scripts.studio_worker"], cwd=BASE_DIR,
                            env={**env, "PYTHONPATH": os.path.join(BASE_DIR, "src") + os.pathsep + BASE_DIR},
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, **kw)
    with _lock:
        _S.update(proc=proc, sock=None, port=0, status="Starting the Studio worker…")
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    import socket
    for _ in range(600):                         # its first start imports torch: up to a couple of minutes
        if _S["port"] or proc.poll() is not None:
            break
        time.sleep(0.2)
    if not _S["port"]:
        raise RuntimeError("The Studio worker didn't start.")
    sock = socket.create_connection(("127.0.0.1", _S["port"]), timeout=10)
    sock.settimeout(None)
    with _lock:
        _S["sock"] = sock


# ---- the load plan, as the desktop builds it ---------------------------------------------------------------------
def plan(desc, prefs, pref_path, fast=True, samples=None, negative=None):
    """(load kwargs for the worker, preview settings or None, status line, problems)."""
    look = lambda k: pref_path(prefs, k)
    paths = {r: desc.model_path(r, look) for r in ("dit", "vae", "text_encoder")}
    labels = {f.role: f.label for f in desc.model_files}
    problems = [f"{labels.get(r, r)} isn't set (or isn't where Preferences says). Set it on the Preferences tab."
                for r, p in paths.items() if not p or not os.path.exists(p)]
    follow = desc.workbench_follows_samples
    fast = fast or follow
    dit, speed, use_ck, ck = paths["dit"], "", False, None
    have = lambda k: bool(k and look(k) and os.path.exists(look(k)))
    if fast:
        c = desc.preview_checkpoint()
        if c is not None and have(c[0].pref_key):
            dit, use_ck, ck = look(c[0].pref_key), True, c[1]
        else:
            sp = desc.preview_speed()
            if sp is not None and have(sp.pref_key):
                speed = look(sp.pref_key)
    sp = desc.preview_speed()
    steps = ck.steps if ck else (sp.settings.steps if speed else desc.default_sampling().steps)
    settings = None
    if follow:
        smp = samples or {}
        settings = {"steps": str(smp.get("steps") or desc.preview_steps),
                    "cfg": str(smp.get("cfg") or f"{desc.preview_cfg:g}"),
                    "turbo": str(smp.get("turbo") if smp.get("turbo") not in (None, "") else
                                 (desc.preview_speed_defaults() or (0, 0))[1]),
                    "negative": negative if negative is not None else str(smp.get("negative") or "")}
        steps = settings["steps"]
    swap = str(prefs.get("inference_blocks_to_swap") or "Auto").strip()
    kw = {"dit_path": dit, "vae_path": paths["vae"], "text_encoder_path": paths["text_encoder"],
          "speed_lora_path": speed, "use_checkpoint_sampling": use_ck, "device": "cuda",
          "precision": "int8" if str(prefs.get("inference_int8", "")).strip() in ("1", "True", "true") else "auto",
          "blocks_to_swap": "auto" if swap.lower().startswith("auto") else int(
              "".join(ch for ch in swap.split(" ")[0] if ch.isdigit()) or 0)}
    status = f"Loading {desc.display_name} ({ck.name + ', ' if ck else ''}{steps}-step previews)…"
    return kw, settings, status, problems


def load(desc, primary, donor, kw, settings, status, turbo):
    if not primary or not os.path.isfile(primary):
        raise ValueError("Choose a primary LoRA first (that file isn't there).")
    if donor and not os.path.isfile(donor):
        raise ValueError("That donor LoRA isn't there.")
    with _lock:
        _S.update(state="loading", status="Starting the Studio worker…", error="", progress=None, pending=None,
                  family=desc.key, used=time.time())

    def go():                                    # the worker's first start imports torch: not in the request
        try:
            _ensure_worker()
            with _lock:
                _S["status"] = status
            _send({"cmd": "load", "family": desc.key, "plan": kw, "primary": primary, "donor": donor or "",
                   "turbo": bool(turbo), "preview_settings": settings, "status": status})
        except Exception as e:
            with _lock:
                _S.update(state="idle", error=str(e), status=f"Couldn't start: {e}")
    threading.Thread(target=go, daemon=True).start()


def donor(path):
    if path and not os.path.isfile(path):
        raise ValueError("That donor LoRA isn't there.")
    _send({"cmd": "donor", "path": path or ""})


def settings(turbo=None, preview_settings=None):
    c = {"cmd": "settings"}
    if turbo is not None:
        c["turbo"] = bool(turbo)
    if preview_settings is not None:
        c["preview_settings"] = preview_settings
    _send(c)


def render(state):
    """Render this slider state: now if nothing renders, else it waits (replacing any older wait) and the render
    going is cancelled - the newest state is what renders next."""
    with _lock:
        if _S["state"] != "ready":
            return {"error": "Load a LoRA first."}
        _S["used"] = time.time()
        if _S["rendering"]:
            _S["pending"] = state
            busy = True
        else:
            _S["gen"] += 1
            gen = _S["gen"]
            _S.update(rendering=True, progress=None, status="Generating preview…")
            busy = False
    if busy:
        _send({"cmd": "cancel"})
        return {"ok": True, "queued": True}
    _send({"cmd": "render", "gen": gen, "state": state, "out_dir": OUT_DIR})
    return {"ok": True, "gen": gen}


def save(out, state, include_donor):
    with _lock:
        _S["saved"] = None
    _send({"cmd": "save", "out": out, "state": state, "include_donor": bool(include_donor)})


def reset():
    with _lock:
        running = _S["proc"] is not None and _S["proc"].poll() is None
        if running:
            _S.update(status="Unloading…", pending=None)
    if running:
        _send({"cmd": "cancel"})
        _send({"cmd": "reset"})


def stop_worker():
    """Quit the worker (the GPU is wanted for a training run, or the server is stopping)."""
    with _lock:
        p = _S["proc"]
    if p is None or p.poll() is not None:
        return
    try:
        _send({"cmd": "cancel"})
        _send({"cmd": "quit"})
        p.wait(timeout=20)
    except Exception:
        p.kill()


def holding_gpu() -> bool:
    with _lock:
        return _S["proc"] is not None and _S["proc"].poll() is None and _S["state"] in ("loading", "ready")


def status():
    with _lock:
        d = {k: _S[k] for k in ("state", "family", "primary", "donor", "primary_blocks", "donor_blocks", "status",
                                "progress", "shown", "rendering", "error", "saved", "profile")}
        d["pending"] = _S["pending"] is not None
        d["log"] = _S["log"][-12:]                 # the worker's own lines (the engine's log), for when it goes wrong
    return d


def image(which, gen):
    with _lock:
        p = _S.get(which) if which in ("baseline", "tweaked") else ""
    return p if p and os.path.isfile(p) else ""


# ---- what the page draws for a model (the desktop tab's layout, from the family's description) ----------------
CAT_COLOR = {"style_composition": "#5B9BD5", "style_ident_overlap": "#5BB3A6", "identity": "#70AD47",
             "ident_details_overlap": "#B8A547", "details": "#ED7D31", "look": "#5B9BD5"}
CAT_SHORT = {"style_composition": "Style+Comp", "style_ident_overlap": "Style/ID", "identity": "Identity",
             "ident_details_overlap": "ID/Detail", "details": "Details", "look": "Look"}
# Klein's (category_masters) built-in presets: which categories stay ticked
KLEIN_PRESETS = {"✨Reset All": None, "✨Identity Only": ["identity", "style_ident_overlap", "ident_details_overlap"],
                 "✨Style+Composition Only": ["style_composition", "style_ident_overlap"],
                 "✨Details Only": ["details", "ident_details_overlap"]}
STILLS_SIZES = ["256", "384", "512", "768", "1024"]


def families():
    """The models Repair Studio offers in the web UI: stills for now (the desktop lists H3 and ACE-Step too)."""
    from fizgig.families.registry import training_families
    return [d for d in training_families() if "repair" in (d.workbench or ()) and d.training_ready
            and not ({"clip", "voice"} & set(d.media))]


def layout(desc, prefs, pref_path):
    cats = dict(desc.block_categories or ())
    groups = []
    for g in desc.load_driver().block_map():
        groups.append({"label": g.label, "blocks": [
            {"id": b.id, "label": b.label, "cat": CAT_SHORT.get(cats.get(b.id), ""),
             "color": CAT_COLOR.get(cats.get(b.id), "")} for b in g.blocks]})
    if desc.category_masters:
        presets = [{"name": n, "keep": keep} for n, keep in KLEIN_PRESETS.items()]
    else:
        presets = [{"name": "✨Reset All", "keep": None}] + [{"name": n, "blocks": dict(b)} for n, b in
                                                            (desc.repair_presets or ())]
    look = lambda k: pref_path(prefs, k)
    have = lambda k: bool(k and look(k) and os.path.exists(look(k)))
    ck, sp = desc.preview_checkpoint(), desc.preview_speed()
    fast = (f"{ck[1].name} ({ck[1].steps}-step, default)" if ck is not None and have(ck[0].pref_key) else
            f"{sp.name} ({sp.settings.steps}-step, default)" if sp is not None and have(sp.pref_key) else "")
    ds = desc.default_sampling()
    return {"key": desc.key, "name": desc.display_name, "groups": groups,
            "categories": {b: CAT_SHORT.get(c, "") for b, c in cats.items()},
            "masters": [{"cat": c, "label": CAT_SHORT[c], "color": CAT_COLOR[c]} for c in
                        ("style_composition", "style_ident_overlap", "identity", "ident_details_overlap", "details")]
            if desc.category_masters else [],
            "category_of": cats if desc.category_masters else {},
            "presets": presets, "lucky_zones": [list(z) for z in (desc.lucky_zones or ())],
            "reference_kind": desc.reference_kind or "", "reference_strength": bool(desc.reference_strength),
            "size": str(desc.repair_size or 768), "sizes": STILLS_SIZES,
            "default_strength": float(desc.repair_default_strength or 1.0),
            "negative_default": desc.preview_negative, "turbo_preview": bool(desc.activation_cache),
            "follows_samples": bool(desc.workbench_follows_samples),
            "dit_fast": fast, "dit_full": f"Full ({ds.steps}-step, slow)" if ds else "",
            "cfg_fast": float(ck[1].cfg) if ck is not None and have(ck[0].pref_key) else
            float(sp.settings.cfg) if fast and sp is not None else float(ds.cfg if ds else 1.0),
            "cfg_full": float(ds.cfg if ds else 1.0)}
