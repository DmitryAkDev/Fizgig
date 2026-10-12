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
    eq = {}
    try:                                         # a Model EQ row: its gain (the sliders' 0 / 2 = -g / +g)
        from fizgig.families.workbench import DONOR, PRIMARY
        gains = getattr(eng, "_eq", {}) or {}
        eq = {"primary": gains.get(PRIMARY), "donor": gains.get(DONOR)}
    except Exception:
        pass
    return {"primary_blocks": sorted(eng.primary_block_ids or ()), "donor_blocks": sorted(eng.donor_block_ids or ()),
            "hash": getattr(eng, "primary_hash", "") or "", "eq": eq}


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
        S["nolora_out"] = S["nolora_song"] = None
        eng.turbo_preview = bool(c.get("turbo", True))
        if getattr(desc, "video_workbench", False):
            eng.int8_attention = True                # as the desktop's Repair Studio (comfy-kitchen where it exists)
            eng.on_status = lambda text: emit(ev="status", text=str(text))
            S["kf"] = {}                             # keyframe / reference latents, per slot, photo, crop and canvas
        if c.get("preview_settings") is not None:
            eng.preview_settings = c["preview_settings"]
        eng.on_step = lambda done, total: emit(ev="step", done=done, total=total)
        emit(ev="status", text=c.get("status") or f"Loading {desc.display_name}…")
        eng.plan_canvas = tuple(c["canvas"]) if c.get("canvas") else None   # H3 Auto: room for this clip
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
            out = c["out_dir"]
            os.makedirs(out, exist_ok=True)
            desc = S["desc"]
            nolora = None                        # the No-LoRA pane's render, after the pair is up
            if getattr(desc, "video_workbench", False):
                paths, nolora = _render_clips(eng, desc, st, c, out, gen)
            elif getattr(desc, "audio_workbench", False):
                paths, nolora = _render_songs(eng, st, c, out, gen)
            else:
                base = eng.generate_baseline(st)
                tweak = eng.generate_preview(st)
                paths = {}
                for name, img in (("baseline", base), ("tweaked", tweak)):
                    p = os.path.join(out, f"{name}_{gen}.png")
                    img.save(p)
                    paths[name] = p
            emit(ev="rendered", gen=gen, nolora_pending=nolora is not None, notes=S.get("notes"), **paths)
            _tidy(out, gen)
            if nolora is not None:               # as the desktop: the pair first, then the base model's own
                try:
                    emit(ev="nolora", gen=gen, **nolora())
                except Exception as e:           # a newer edit took the engine (or it failed: say so)
                    emit(ev="nolora", gen=gen, aborted=_aborted(e), msg="" if _aborted(e) else str(e))
    except Exception as e:
        if _aborted(e):
            emit(ev="aborted", gen=gen)
        else:
            msg = str(e)
            emit(ev="error", where="render", gen=gen,
                 msg=("Out of VRAM for this preview: a smaller Res, or more inference block swap in Preferences."
                      if "out of memory" in msg.lower() else msg), trace=traceback.format_exc()[-2000:])


def _cond(eng, c, width, height, frames):
    """(keyframes, references) for the clip, as the desktop's _repair_h3_prepare_cond: under First/last frame each
    photo's crop is a keyframe (first at frame 0, last at the final frame; a still takes the first only); under
    Reference each is a reference. Latents are kept per slot / photo / crop / canvas."""
    from PIL import Image
    spec = [k for k in (c.get("keyframes") or []) if k.get("path") and os.path.isfile(k["path"])]
    if not spec:
        return None, None
    ref = bool(c.get("ref_mode"))
    out = []
    for k in spec:
        slot, path = k.get("slot"), k["path"]
        if not ref and slot == "last" and int(frames) <= 1:
            continue
        img = Image.open(path).convert("RGB")
        rect = k.get("rect")
        if rect:                                 # the box placed for another W x H: its shape now, centred inside it
            x0, y0, x1, y1 = (int(v) for v in rect)
            c = _centre_crop((x1 - x0, y1 - y0), width, height)
            rect = (x0 + c[0], y0 + c[1], x0 + c[2], y0 + c[3])
        else:
            rect = _centre_crop(img.size, width, height)
        key = ("ref" if ref else "kf", slot, path, os.path.getmtime(path), rect, int(width), int(height))
        hit = S["kf"].get(key)
        if hit is None:
            crop = img.crop(rect)
            hit = eng.encode_reference_image(crop, width, height) if ref else eng.encode_keyframe(crop, width, height)
            S["kf"][key] = hit
        out.append(hit if ref else (0 if slot == "first" else max(0, int(frames) - 1), hit))
    return (None, out or None) if ref else (out or None, None)


def _centre_crop(size, width, height):
    """The largest box of the clip's shape in the middle of the photo (the desktop's crop dialog starts there)."""
    w, h = size
    want = width / float(height)
    if w / float(h) > want:
        cw = int(round(h * want))
        x0 = (w - cw) // 2
        return (x0, 0, x0 + cw, h)
    ch = int(round(w / want))
    y0 = (h - ch) // 2
    return (0, y0, w, y0 + ch)


def _write_clip(clip, path_png, path_mp4, fps, rate):
    """The clip's middle frame as a picture and, for a clip, an MP4 with its sound (silent when it has none)."""
    clip["middle"].save(path_png)
    frames = clip.get("frames") or []
    if len(frames) <= 1:
        return ""
    import numpy as np
    import tempfile
    import torch
    from fizgig.minimax.common import _find_ffmpeg, write_preview_mp4, write_wav
    if not _find_ffmpeg():
        return ""
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    ten = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    wav = clip.get("wav")
    wav_path = os.path.join(tempfile.gettempdir(), f"fizgig_studio_{os.getpid()}.wav")
    write_wav(wav_path, wav if wav is not None else torch.zeros(2, int(rate * len(frames) / fps)), sample_rate=rate)
    write_preview_mp4(path_mp4, ten, wav_path, fps=fps)
    return path_mp4


def _render_clips(eng, desc, st, c, out, gen):
    """H3: the baseline and tweaked clips, as the desktop renders them (baseline_clip, then render_clip)."""
    v = c.get("video") or {}
    frames = int(v.get("frames") or 22)
    width, height = int(v.get("width") or 768), int(v.get("height") or 640)
    st.preview_width, st.preview_height, st.preview_frames = width, height, frames
    st.keyframes, st.references = _cond(eng, c, width, height, frames)
    opts = dict(frames=frames, regime="custom", steps=int(v.get("steps") or 4),
                turbo_strength=float(v.get("turbo") if v.get("turbo") not in (None, "") else 1.0),
                with_audio=bool(v.get("sound", True)))
    spec = getattr(desc, "clip_spec", None)
    fps, rate = (spec.fps, spec.audio_rate) if spec is not None else (24, 32000)
    import torch
    torch.cuda.reset_peak_memory_stats()
    rest = torch.cuda.memory_allocated() / 2**30
    base = eng.baseline_clip(st, cache=None, **opts)
    tweak = eng.render_clip(st, cache=None, **opts)
    print(f"[studio] clip {width}x{height} x{frames}, {len(st.keyframes or [])} keyframe(s), "
          f"{len(st.references or [])} reference(s): peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved, from {rest:.2f} GB at rest "
          f"(base {getattr(eng, 'base_plan', None)})", flush=True)
    paths = {}
    for name, clip in (("baseline", base), ("tweaked", tweak)):
        png = os.path.join(out, f"{name}_{gen}.png")
        mp4 = _write_clip(clip, png, os.path.join(out, f"{name}_{gen}.mp4"), fps, rate)
        paths[name], paths[name + "_media"] = png, mp4

    def nolora():
        """The same seed / prompt / canvas / length / photos on the base model with no LoRA (the engine keeps it, so
        a slider move doesn't re-render it - nor re-write it here)."""
        clip = eng.nolora_clip(st, cache=None, **opts)
        hit = S.get("nolora_out")
        if hit and hit[0] is clip and os.path.isfile(hit[1]["nolora"]):
            return hit[1]
        png = os.path.join(out, f"nolora_{gen}.png")
        got = {"nolora": png, "nolora_media": _write_clip(clip, png, os.path.join(out, f"nolora_{gen}.mp4"), fps, rate)}
        S["nolora_out"] = (clip, got)
        return got
    return paths, (nolora if v.get("nolora") and hasattr(eng, "nolora_clip") else None)


def _audio_inputs(eng, a, seconds):
    """Prompt B, the song to start from and the reference recording onto the engine (it reads them each render), as
    the desktop's _repair_pb_push / _repair_src_effective; returns the readouts beside its sliders."""
    pb = a.get("prompt_b") or None
    if hasattr(eng, "prompt_b"):
        eng.prompt_b, eng.prompt_blend = pb, (float(a.get("blend") or 0) if pb else 0.0)
    src = a.get("source") or {}
    path, mode = src.get("path") or None, src.get("mode") or "song"
    d, f, s = float(src.get("denoise", 1.0)), float(src.get("follow", 1.0)), float(src.get("strength", 0.5))
    shown_d, shown_f = d, f
    rp, dur = None, None
    live = bool(path and os.path.isfile(path))
    if live and mode != "song":
        try:
            from fizgig.acestep.driver import audio_seconds
            dur = float(audio_seconds(path))
        except Exception:
            dur = None
    if live and mode == "repaint":
        r = src.get("region")
        rp = (round(float(r[0]), 2), round(float(r[1]), 2)) if r else None
    elif live and mode == "extend":
        rp = (round(dur, 2), float(seconds)) if dur is not None and dur < seconds - 0.25 else None
    if mode != "song":
        d, f = 1.0, 0.0
        live = live and rp is not None
    elif d >= 1.0 and f <= 0.0:
        live = False
    if not live:
        path, d, f, rp = None, 1.0, 0.0, None
    ref = a.get("reference") or None
    if not (ref and os.path.isfile(ref)):
        ref = None
    if hasattr(eng, "source_path"):
        eng.source_path, eng.source_denoise, eng.source_follow = path, d, f
        eng.source_repaint, eng.repaint_strength = rp, s
        eng.reference_path = ref
    notes = {"strength": f"{s:.2f}"}
    if rp is not None:
        notes["strength"] += f" - remakes {rp[0]:.1f}\u2013{rp[1]:.1f} s, keeps the rest"
    elif mode == "repaint":
        notes["strength"] += " - drag across the Tweaked waveform to choose the section"
    elif mode == "extend":
        notes["strength"] += (" - pick a song" if dur is None else f" - set Length above the song's {dur:.0f} s to extend it")
    txt, ftxt = f"{shown_d:.2f}", f"{shown_f:.2f}"
    drv = getattr(eng, "driver", None)
    if drv is not None and hasattr(drv, "schedule") and hasattr(drv, "source_start"):
        try:
            ts = drv.schedule(drv.description.preview_steps)
            t0 = drv.source_start(ts, shown_d) if shown_d < 1.0 else ts[0]
            n = sum(1 for t in ts if t0 is not None and t <= t0 + 1e-9)
            txt += ((" - the source as it is" if t0 is None else f" - starts at {t0:.2f}, {n} of {len(ts)} steps")
                    if shown_d < 1.0 else " - starts from noise")
            ftxt += " - off" if shown_f <= 0.0 else f" - {int(n * shown_f)} of {n} steps follow"
        except Exception:
            pass
    notes.update(denoise=txt, follow=ftxt)
    return notes


def _render_songs(eng, st, c, out, gen):
    """ACE-Step: the baseline and tweaked songs (the length is the Res box, in seconds) as WAVs with their
    waveform pictures."""
    a = c.get("audio") or {}
    seconds = int(a.get("seconds") or st.preview_width or 30)
    try:
        eng.driver.set_options({"preview_seconds": seconds})
    except Exception:
        pass
    S["notes"] = _audio_inputs(eng, a, seconds)
    base = eng.generate_baseline(st)
    tweak = eng.generate_preview(st)
    from fizgig.minimax.common import write_wav
    paths = {}
    for name, song in (("baseline", base), ("tweaked", tweak)):
        png = os.path.join(out, f"{name}_{gen}.png")
        wav = os.path.join(out, f"{name}_{gen}.wav")
        song.picture().save(png)
        write_wav(wav, song.wave, sample_rate=48000)
        paths[name], paths[name + "_media"] = png, wav
    key = (eng.primary_path, st.seed, st.prompt, seconds,
           eng._prompt_b_key() if hasattr(eng, "_prompt_b_key") else None,
           eng._audio_key() if hasattr(eng, "_audio_key") else None)

    def nolora():
        hit = S.get("nolora_song")
        if hit and hit[0] == key and os.path.isfile(hit[1]["nolora"]):
            return hit[1]
        import copy
        base_st = copy.deepcopy(st)
        base_st.primary_scale, base_st.donor_scale = 0.0, 0.0
        song = eng.generate_preview(base_st)
        png, wav = os.path.join(out, f"nolora_{gen}.png"), os.path.join(out, f"nolora_{gen}.wav")
        song.picture().save(png)
        write_wav(wav, song.wave, sample_rate=48000)
        got = {"nolora": png, "nolora_media": wav}
        S["nolora_song"] = (key, got)
        return got
    return paths, nolora


def _tidy(out, gen):
    """Drop renders two or more generations old (the page shows the newest), keeping the No-LoRA files in use."""
    import re
    keep = {os.path.normcase(p) for k in ("nolora_out", "nolora_song") if S.get(k) for p in S[k][1].values() if p}
    try:
        for f in os.listdir(out):
            m = re.match(r"(baseline|tweaked|nolora)_(\d+)\.(png|mp4|wav)$", f)
            p = os.path.join(out, f)
            if m and int(m.group(2)) < gen - 1 and os.path.normcase(p) not in keep:
                os.remove(p)
    except OSError:
        pass


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
