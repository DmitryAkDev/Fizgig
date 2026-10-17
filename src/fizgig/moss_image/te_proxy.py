"""MossTextEncoder: the main-venv face of the Qwen3.5-9B text encoder, which runs in the side venv (te_env.py) as
te_worker.py. Fizgig's generic cache, trainer previews and workbench call load_text_encoder / encode_text /
unload_text_encoder as for any family; this class turns that into a worker process that lives from load to unload.

.encode(captions) -> [(L_i, 4096) bf16 CPU tensors], each the reference pipeline's unpadded last_hidden_state.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import sys
import threading

import torch

logger = logging.getLogger(__name__)

TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-Base"
CONFIG = os.path.join(os.path.dirname(__file__), "qwen35_9b_te_config.json")
WORKER = os.path.join(os.path.dirname(__file__), "te_worker.py")
FULL_VRAM_GB = 19.0          # bf16 text model ~16.8 GB + a 4096-token forward


def resolve_tokenizer(te_path: str) -> str:
    """The tokenizer folder: beside the official text_encoder/ folder (../tokenizer), else the cached helper repo's
    tokenizer/ (description helper_files), else the repo id (the worker fetches it once)."""
    base = te_path if os.path.isdir(te_path) else os.path.dirname(te_path)
    for cand in (os.path.join(base, "tokenizer"), os.path.join(os.path.dirname(base), "tokenizer")):
        if os.path.isfile(os.path.join(cand, "tokenizer.json")):
            return cand
    try:
        from fizgig.utils.hf_cache import cached_snapshot_dir
        snap = cached_snapshot_dir(TOKENIZER_REPO)
    except Exception:                                    # noqa: BLE001
        snap = None
    if snap and os.path.isfile(os.path.join(snap, "tokenizer", "tokenizer.json")):
        return os.path.join(snap, "tokenizer")
    return TOKENIZER_REPO


def resolve_config(te_path: str) -> str:
    base = te_path if os.path.isdir(te_path) else os.path.dirname(te_path)
    own = os.path.join(base, "config.json")
    return own if os.path.isfile(own) else CONFIG


def pick_mode(device) -> str:
    forced = os.environ.get("FIZGIG_MOSS_TE_MODE", "").strip()
    if forced:
        return forced
    if str(device) == "cpu" or not torch.cuda.is_available():
        return "cpu"
    from fizgig.families.quant import free_vram_gb
    return "full" if free_vram_gb() >= FULL_VRAM_GB else "stream"


class MossTextEncoder:
    def __init__(self, path, device="cuda", mode=None, dtype="bfloat16", out_dtype="bfloat16", python=None,
                 tokenizer=None, max_tokens=4096, timeout=None):
        from fizgig.moss_image import te_env
        self.path = path
        self.mode = mode or pick_mode(device)
        py = python or te_env.ensure()
        tok = tokenizer or resolve_tokenizer(path)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["PYTHONIOENCODING"] = "utf-8"    # Windows pipes default to the ANSI code page
        env.pop("PYTHONPATH", None)          # the worker imports nothing from Fizgig
        if os.path.isdir(tok):
            env["HF_HUB_OFFLINE"] = "1"      # weights, config and tokenizer are all local: never ask the Hub
        cmd = [py, WORKER, "--te", path, "--config", resolve_config(path), "--tokenizer", tok,
               "--device", str(device), "--mode", self.mode, "--dtype", dtype, "--out_dtype", out_dtype,
               "--max_tokens", str(int(max_tokens))]
        logger.info(f"[moss te] starting the text encoder ({self.mode}) in {os.path.dirname(os.path.dirname(py))}")
        self._tail = collections.deque(maxlen=40)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, encoding="utf-8", env=env, bufsize=1)
        self._err = threading.Thread(target=self._pump_stderr, daemon=True)
        self._err.start()
        self._next = 0
        msg = self._read()
        if not msg.get("ready"):
            self._fail("did not start")
        self.info = msg

    # ---- plumbing -----------------------------------------------------------------------------------------------
    def _pump_stderr(self):
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                self._tail.append(line)
                if line.startswith("[moss te]"):
                    logger.info(line)

    def _fail(self, what):
        try:
            self._proc.wait(timeout=5)
        except Exception:                                # noqa: BLE001
            pass
        self._err.join(timeout=2)
        tail = "\n".join(list(self._tail)[-12:])
        self._kill()
        raise RuntimeError(f"MOSS-Image text encoder {what} (exit {self._proc.returncode}). Last lines:\n{tail}")

    def _read(self):
        while True:
            line = self._proc.stdout.readline()
            if not line:
                self._fail("stopped")
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    pass
            if line:                         # stray output from a library: never a reply
                self._tail.append(line)

    def _kill(self):
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except Exception:                            # noqa: BLE001
                pass

    # ---- the encoder ------------------------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def encode(self, captions):
        from safetensors.torch import load_file
        captions = [captions] if isinstance(captions, str) else list(captions)
        if not captions:
            return []
        if not self.alive:
            self._fail("is not running")
        self._next += 1
        rid = self._next
        try:
            self._proc.stdin.write(json.dumps({"id": rid, "captions": captions}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._fail("stopped")
        msg = self._read()
        if msg.get("id") != rid:
            self._fail(f"answered out of order ({msg})")
        if "error" in msg:
            raise RuntimeError(f"MOSS-Image text encoder: {msg['error']}")
        sd = load_file(msg["path"])
        try:
            os.remove(msg["path"])
        except OSError:
            pass
        out = [sd[f"h{i}"] for i in range(len(captions))]
        if [int(h.shape[0]) for h in out] != list(msg["lens"]):
            raise RuntimeError("MOSS-Image text encoder returned mismatched lengths")
        return out

    def unload(self):
        if self._proc is None:
            return
        if self.alive:
            try:
                self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=60)
            except Exception:                            # noqa: BLE001
                pass
        self._kill()
        out_dir = getattr(self, "info", {}).get("out_dir")
        if out_dir and os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        self._proc = None

    def __del__(self):
        try:
            self.unload()
        except Exception:                                # noqa: BLE001
            pass
