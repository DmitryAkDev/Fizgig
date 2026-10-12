"""The web UI's Repair Studio worker: one process holding a family's workbench engine (fizgig.families.workbench, the
desktop's own) so the model stays loaded between renders. Commands arrive as JSON lines over a local socket (its port in the "ready" event); events leave as
"@@EV {json}" lines on stdout (anything else on stdout is the engine's own log). One render runs at a time in its own
thread, so a "cancel" is heard while it renders.

Commands (cmd=...): load {family, plan, primary, donor, turbo, preview_settings}, donor {path} / unload_donor,
settings {turbo, preview_settings}, render {gen, state, out_dir}, cancel, save {out, state, include_donor}, reset, quit.
Events (ev=...): status {text}, step {done, total}, loaded {...}, rendered {gen, baseline, tweaked},
aborted {gen}, error {where, msg}, saved {summary}, reset, donor {...}.
"""
import json
import os
import sys
import threading
import traceback

EV = "@@EV "
_out_lock = threading.Lock()


def emit(**d):
    with _out_lock:
        sys.stdout.write(EV + json.dumps(d, default=str) + "\n")
        sys.stdout.flush()


S = {"engine": None, "desc": None, "render": None}
_engine_lock = threading.RLock()


def _auto_swap():
    """The desktop's inference block-swap Auto: 16 GB and up none, 10-14 GB 12, under 10 GB 16."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 0 if gb >= 15 else 12 if gb >= 10 else 16
    except Exception:
        pass
    return 0


def _blocks(eng):
    return {"primary_blocks": sorted(eng.primary_block_ids or ()), "donor_blocks": sorted(eng.donor_block_ids or ()),
            "hash": getattr(eng, "primary_hash", "") or ""}


def cmd_load(c):
    from fizgig.families.registry import get as get_family
    desc = get_family(c["family"])
    if desc is None:
        raise ValueError(f"Unknown model: {c['family']}")
    plan = dict(c["plan"])
    if plan.pop("use_checkpoint_sampling", False):
        ck = desc.preview_checkpoint()
        plan["preview_sampling"] = ck[1] if ck else None
    else:
        plan["preview_sampling"] = None
    if str(plan.get("blocks_to_swap", "")).lower().startswith("auto"):
        plan["blocks_to_swap"] = _auto_swap()
    with _engine_lock:
        eng = S["engine"]
        if eng is not None and (eng.desc is not desc and getattr(eng.desc, "key", None) != desc.key):
            eng.reset()
            eng = None
        if eng is not None and eng.primary_network is not None:
            eng.reset()                          # a new primary: a fresh session, as the desktop's Start does
        if eng is None:
            eng = desc.make_workbench_engine()
        S["engine"], S["desc"] = eng, desc
        eng.turbo_preview = bool(c.get("turbo", True))
        if c.get("preview_settings") is not None:
            eng.preview_settings = c["preview_settings"]
        eng.on_step = lambda done, total: emit(ev="step", done=done, total=total)
        emit(ev="status", text=c.get("status") or f"Loading {desc.display_name}…")
        eng.ensure_pipeline(**plan)
        emit(ev="status", text="Loading the LoRA…")
        eng.load_primary(c["primary"])
        if c.get("donor"):
            eng.load_donor(c["donor"])
        emit(ev="loaded", family=desc.key, primary=c["primary"], donor=c.get("donor") or "", **_blocks(eng))


def cmd_donor(c):
    with _engine_lock:
        eng = S["engine"]
        if eng is None or eng.primary_network is None:
            raise ValueError("Load a primary LoRA first.")
        if eng.donor_network is not None:
            eng.unload_donor()
        if c.get("path"):
            eng.load_donor(c["path"])
        emit(ev="donor", donor=c.get("path") or "", **_blocks(eng))


def cmd_settings(c):
    eng = S["engine"]
    if eng is None:
        return
    if "turbo" in c:
        eng.turbo_preview = bool(c["turbo"])
    if c.get("preview_settings") is not None:
        eng.preview_settings = c["preview_settings"]
        eng._invalidate_baseline_cache()


def _state(d):
    from fizgig.repair_studio.state import BlockState, SliderState
    s = SliderState.from_json(d)
    eng = S["engine"]
    if eng is not None:                         # every block of the family, the ones the page left out at defaults
        for bid in eng.block_ids():
            s.blocks.setdefault(bid, BlockState())
    return s


def _aborted(e):
    names = {type(e).__name__}
    return bool(names & {"RenderCancelled", "SampleAborted", "PreviewAborted"})


def _render(c):
    gen = c["gen"]
    eng = S["engine"]
    try:
        with _engine_lock:
            if eng is None or eng.primary_network is None:
                raise ValueError("Load a LoRA first.")
            st = _state(c["state"])
            eng.clear_cancel()
            emit(ev="status", text=c.get("status") or "Generating preview…")
            base = eng.generate_baseline(st)
            tweak = eng.generate_preview(st)
            out = c["out_dir"]
            os.makedirs(out, exist_ok=True)
            paths = {}
            for name, img in (("baseline", base), ("tweaked", tweak)):
                p = os.path.join(out, f"{name}_{gen}.png")
                img.save(p)
                paths[name] = p
        emit(ev="rendered", gen=gen, **paths)
    except Exception as e:
        if _aborted(e):
            emit(ev="aborted", gen=gen)
        else:
            msg = str(e)
            emit(ev="error", where="render", gen=gen,
                 msg=("Out of VRAM for this preview: a smaller Res, or more inference block swap in Preferences."
                      if "out of memory" in msg.lower() else msg), trace=traceback.format_exc()[-2000:])


def cmd_render(c):
    t = S["render"]
    if t is not None and t.is_alive():         # the server sends one at a time; a stray one waits its turn
        t.join()
    S["render"] = threading.Thread(target=_render, args=(c,), daemon=True)
    S["render"].start()


def cmd_cancel(_c):
    eng = S["engine"]
    if eng is not None:
        eng.request_cancel()


def cmd_save(c):
    with _engine_lock:
        eng = S["engine"]
        if eng is None or eng.primary_network is None:
            raise ValueError("Load a LoRA first.")
        summary = eng.save_repaired(c["out"], _state(c["state"]), include_donor=bool(c.get("include_donor")))
    emit(ev="saved", out=c["out"], summary=summary)


def cmd_reset(_c):
    with _engine_lock:
        if S["engine"] is not None:
            S["engine"].reset()
        S["engine"] = S["desc"] = None
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    emit(ev="reset")


HANDLERS = {"load": cmd_load, "donor": cmd_donor, "settings": cmd_settings, "render": cmd_render,
            "cancel": cmd_cancel, "save": cmd_save, "reset": cmd_reset}


def _run(c):
    try:
        HANDLERS[c["cmd"]](c)
    except Exception as e:
        emit(ev="error", where=c.get("cmd"), msg=str(e) or type(e).__name__, trace=traceback.format_exc()[-2000:])


def main():
    if os.environ.get("FIZGIG_STUDIO_DUMP"):          # diagnosing a hang: every thread's stack after N seconds
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["FIZGIG_STUDIO_DUMP"]), exit=False, file=sys.stderr)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.abspath(os.path.join(here, "..", "..")), os.path.abspath(os.path.join(here, "..", "..", ".."))):
        if p not in sys.path:
            sys.path.insert(0, p)
    # the heavy libraries load HERE, on the main thread, before it waits on stdin: loading a native library on a
    # command thread while the main thread sits in a blocking pipe read deadlocks on Windows (numpy hung for ever)
    emit(ev="status", text="Starting the Studio worker…")
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import fizgig.families.workbench  # noqa: F401
    from fizgig.families.registry import training_families
    training_families()
    # commands come over a local socket, not stdin: a thread blocked reading stdin holds the C runtime's file lock,
    # and anything that loads a native library meanwhile (numpy, torch's kernels, PNG encode) waits on it for ever
    import socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    emit(ev="ready", port=srv.getsockname()[1])
    srv.settimeout(120)
    try:
        conn, _ = srv.accept()
    except OSError:
        return
    srv.close()
    for line in conn.makefile("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get("cmd") == "quit":
            break
        if c.get("cmd") == "cancel":
            cmd_cancel(c)                        # heard at once, even while a load or render holds the engine
            continue
        # load / save / reset can take a while: their own thread, so a cancel still gets through
        threading.Thread(target=_run, args=(c,), daemon=True).start()
    cmd_reset({})


if __name__ == "__main__":
    main()
