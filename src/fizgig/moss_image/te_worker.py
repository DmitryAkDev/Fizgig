"""MOSS-Image text-encoder worker. Runs ONLY in the side venv (te_env.py: transformers >= 5.3) as a plain script -
it imports torch, transformers and safetensors, never Fizgig - and serves te_proxy.MossTextEncoder over stdin/stdout.

What it computes is the reference pipeline's MOSSImagePipeline.encode_prompt, per caption and unpadded:
    text = apply_chat_template([{"role": "user", "content": caption}], add_generation_prompt=True,
                               enable_thinking=True)   # == TEMPLATE.format(caption.strip()): the jinja trims
    ids = tokenizer(text, truncation=True, max_length=4096)
    hidden = Qwen3_5Model(ids, use_cache=False).last_hidden_state[0]
Only the language model is built (Qwen3_5TextModel; the vision tower and its 0.9 GB are never loaded): for text-only
input its positions equal the multimodal model's, which the tests check.

Protocol, one JSON object per line:
    -> {"id": 1, "captions": ["...", ...]}       <- {"id": 1, "path": "<tmp>/1.safetensors", "lens": [L0, ...]}
    -> {"id": 2, "quit": true}                   (exit)
    <- {"ready": true, "transformers": "5.12.1", "mode": "..."} once loaded; {"id", "error"} on a failed request.
The safetensors file holds h0..hN, each (L_i, 4096) in the output dtype. Logs go to stderr.

Modes: "full" (whole model on the device), "stream" (embeddings and norm resident, each decoder layer moved to the
device for its forward and back - exact, ~1-2 GB of VRAM; the default below ~19 GB free), "cpu".
"""
import argparse
import json
import os
import sys
import tempfile
import time

TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_TOKENS = 4096
_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")


def log(msg):
    print(f"[moss te] {msg}", file=sys.stderr, flush=True)


_PROTOCOL = sys.stdout       # the proxy's channel; everything else printing to stdout (transformers' own checks)
sys.stdout = sys.stderr       # goes to stderr instead, so it can never corrupt a reply


def send(obj):
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


def weight_files(path):
    if os.path.isdir(path):
        idx = [f for f in os.listdir(path) if f.endswith(".safetensors.index.json")]
        if idx:
            path = os.path.join(path, idx[0])
        else:
            return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(os.path.dirname(path), s) for s in shards]
    return [path]


def text_state_dict(files):
    """The language model's tensors only (bare names: embed_tokens..., layers.N..., norm.weight)."""
    from safetensors import safe_open
    sd = {}
    for fn in files:
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if "visual." in k or k.startswith(("lm_head.", "mtp.")) or ".mtp." in k:
                    continue
                name = next((k[len(p):] for p in _PREFIXES if k.startswith(p)), k)
                sd[name] = f.get_tensor(k)
    return sd


def load_model(te_path, config_path, dtype, device, mode):
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(config_path)
    text_cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = Qwen3_5TextModel(text_cfg)
    sd = text_state_dict(weight_files(te_path))
    expected = set(model.state_dict())
    missing, unexpected = sorted(expected - set(sd)), sorted(set(sd) - expected)
    if missing or unexpected:
        raise KeyError(f"not the Qwen3.5 text encoder: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    del sd
    model.eval().requires_grad_(False)
    # the rotary table's inv_freq is a non-persistent buffer, left on meta: build that module for real
    model.rotary_emb = type(model.rotary_emb)(config=text_cfg)
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f"buffer {name} still on meta after load")
    if mode == "stream" and str(device).startswith("cuda"):
        for name, child in model.named_children():
            if name != "layers":
                child.to(device)
        for buf_name, buf in model.named_buffers(recurse=False):
            setattr(model, buf_name, buf.to(device))
        pin = True
        for layer in model.layers:
            # each weight keeps one pinned CPU copy; a forward borrows a device copy and drops it after
            tensors = list(layer.parameters()) + list(layer.buffers())
            for t in tensors:
                t.data = t.data.to("cpu")
                if pin:
                    try:
                        t.data = t.data.pin_memory()
                    except RuntimeError:          # not enough page-locked RAM: plain copies, slower transfers
                        pin = False
                        log("pinned memory unavailable - streaming from pageable RAM")
            homes = [t.data for t in tensors]

            def pre(mod, args, kwargs, tensors=tensors, homes=homes):
                for t, h in zip(tensors, homes):
                    t.data = h.to(device, non_blocking=True)

            def post(mod, args, kwargs, out, tensors=tensors, homes=homes):
                for t, h in zip(tensors, homes):
                    t.data = h
                return out
            layer.register_forward_pre_hook(pre, with_kwargs=True)
            layer.register_forward_hook(post, with_kwargs=True)
    else:
        model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="full", choices=("full", "stream", "cpu"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_dtype", default="bfloat16")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 3):
        raise SystemExit(f"transformers {transformers.__version__} < 5.3 - run src/fizgig/moss_image/te_env.py with Fizgig's python")
    device = "cpu" if a.mode == "cpu" else a.device
    dtype, out_dtype = getattr(torch, a.dtype), getattr(torch, a.out_dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    probe = "a red fox, {braces} and \\n ünïcode"
    rendered = tok.apply_chat_template([{"role": "user", "content": probe}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    if rendered != TEMPLATE.format(probe.strip()):
        log(f"chat template differs from the expected string ({rendered!r}); using apply_chat_template")
    model = load_model(a.te, a.config, dtype, device, a.mode)
    log(f"loaded in {time.time() - t0:.1f}s ({a.mode}, {a.dtype}, {device})")
    out_dir = tempfile.mkdtemp(prefix="moss_te_")
    send({"ready": True, "transformers": transformers.__version__, "mode": a.mode, "out_dir": out_dir})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        rid = req.get("id")
        try:
            tensors, lens = {}, []
            for i, cap in enumerate(req["captions"]):
                text = tok.apply_chat_template([{"role": "user", "content": cap}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
                ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_tokens).input_ids
                with torch.inference_mode():
                    h = model(input_ids=ids.to(device), use_cache=False, return_dict=True).last_hidden_state[0]
                tensors[f"h{i}"] = h.to("cpu", out_dtype).contiguous()
                lens.append(int(h.shape[0]))
            path = os.path.join(out_dir, f"{rid}.safetensors")
            save_file(tensors, path)
            send({"id": rid, "path": path, "lens": lens})
        except Exception as e:                       # noqa: BLE001 - reported to the proxy, the worker lives on
            import traceback
            traceback.print_exc(file=sys.stderr)
            send({"id": rid, "error": f"{type(e).__name__}: {e}"})
    log("bye")


if __name__ == "__main__":
    main()
