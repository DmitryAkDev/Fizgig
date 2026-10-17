"""MOSS-Image feature tests beyond M4 (sliders, Royale, INT8 attention, compile, the SFT preview checkpoint, batch > 1,
full fine-tune): CPU, tiny random weights."""
import math
import os

import pytest
import torch
import torch.nn.functional as F

from moss_tiny import TINY, pair, rel

RESULTS = {}


def _desc():
    from fizgig.families import registry
    return registry.get("moss_image")


def _drv(options=None):
    d = _desc().load_driver()
    d.set_options(options or {})
    return d


def _lora(fz, drv, rank=4, seed=9):
    from fizgig.families.lora import FamilyLoRA
    net = FamilyLoRA(fz.requires_grad_(False), drv)
    net.add_trainable(rank, float(rank))
    g = torch.Generator().manual_seed(seed)
    for p in net.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return net


# ---- 1. sliders --------------------------------------------------------------------------------------------------
def test_slider_offered_with_a_preset():
    d = _desc()
    assert d.slider_training and d.slider_guidance == 2.0 and d.validate() == []
    name, p = next((n, p) for n, p in d.presets if "Slider" in n)
    assert p["FAMILY_SLIDER"] is True and p["NETWORK_TYPE"] == "LoRA (standard)" and p["DATASET_MEGAPIXELS"] == "0.5"
    assert all(p2.get("FAMILY_SLIDER") is False for n2, p2 in d.presets if n2 != name)
    from fizgig.families import form
    assert "slider" in [k for k, _ in form.kinds(d)]          # the Training tab's Kind of training row
    assert form.slider_sources(d) == ["pairs", "prompts"]
    assert form.form(d)                                        # the whole form still builds


def test_slider_diff_ref_weighting():
    """diff_weight 0 and identical poles are the plain loss; otherwise each token's error is weighted by how much the
    poles differ there, normalised to mean 1 (Krea 2's rule)."""
    from fizgig.moss_image import sampling as S
    drv = _drv()
    fz, _ = pair(seed=31)
    fz.requires_grad_(False)
    x0 = torch.randn(1, TINY["latent_channels"], 3, 4)
    other = x0.clone()
    other[..., 0, :] += 2.0                      # the poles differ only along the top row
    c = {"hidden_states": torch.randn(1, 6, TINY["text_dim"])}
    g = lambda: torch.Generator().manual_seed(5)
    with torch.no_grad():
        plain, info = drv.training_loss(fz, x0, c, g())
        w0, _ = drv.training_loss(fz, x0, c, g(), diff_ref=other, diff_weight=0.0)
        same, _ = drv.training_loss(fz, x0, c, g(), diff_ref=x0, diff_weight=1.0)
        got, _ = drv.training_loss(fz, x0, c, g(), diff_ref=other, diff_weight=0.75)
        gen = g()
        eps = torch.randn(x0.shape, generator=gen)
        s = S.sample_t(gen)
        xt = (1 - s) * x0 + s * eps
        err = (fz(xt, torch.tensor([1000 * s]), c["hidden_states"]).float() - (eps - x0)).pow(2).mean(dim=1)
        d = (x0 - other).abs().mean(dim=1)
        r = (d / d.mean()).clamp(max=8.0)
        wgt = 0.25 + 0.75 * r
        want = (err * wgt / wgt.mean()).mean()
    assert torch.equal(w0, plain) and rel(same, plain) <= 1e-6
    RESULTS["slider diff_ref vs hand weighting"] = rel(got, want)
    assert rel(got, want) <= 1e-6 and not torch.allclose(got, plain)


def test_slider_image_pair_and_prompt_steps_train_both_poles():
    """The trainer's two slider steps on the tiny model: an image-pair step (+1 toward the image, -1 toward its pair)
    and a prompt-pair step (_prompt_slider_step through noise_latents / predict) give finite losses and LoRA grads;
    predict at a noise_latents state is the training forward."""
    from fizgig.families.train import _prompt_slider_step
    drv = _drv()
    fz, _ = pair(seed=32)
    net = _lora(fz, drv)
    a = torch.randn(1, TINY["latent_channels"], 2, 3)
    b = a + 0.5 * torch.randn_like(a)
    c = {"hidden_states": torch.randn(1, 5, TINY["text_dim"])}
    gen = torch.Generator().manual_seed(0)
    total = 0.0
    for m, x, y in ((1.0, a, b), (-1.0, b, a)):
        net.set_trainable_multiplier(m)
        loss, _ = drv.training_loss(fz, x, c, gen, diff_ref=y, diff_weight=1.0)
        (0.5 * loss).backward()
        total += 0.5 * loss.item()
    net.set_trainable_multiplier(1.0)
    assert math.isfinite(total)
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(float(g.abs().sum()) > 0 for g in grads)
    for p in net.parameters():
        p.grad = None
    # predict == the training forward at the same state
    st = drv.noise_latents(a, torch.Generator().manual_seed(4))
    with torch.no_grad():
        p1 = drv.predict(fz, st, c)
        p2 = fz(st["xt"], torch.tensor([1000 * st["t"]]), c["hidden_states"])
    assert torch.equal(p1, p2)
    enc = [{"hidden_states": torch.randn(1, n, TINY["text_dim"])} for n in (4, 6, 3)]
    l2, t = _prompt_slider_step(drv, fz, net, a, enc, torch.Generator().manual_seed(1), guidance=2.0, min_t=0.0,
                                max_t=1.0)
    assert math.isfinite(l2) and 0.0 <= t <= 1.0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_slider_practice_latent_round_trip():
    """A prompt slider's practice bank: generate() -> decode -> encode gives a cache-shaped (1, 128, h, w) latent."""
    from fizgig.flux3.autoencoder import AutoEncoder
    drv = _drv()
    torch.manual_seed(0)
    fz = __import__("fizgig.moss_image.model", fromlist=["MossImageDiT"]).MossImageDiT(
        **dict(TINY, latent_channels=128)).eval().requires_grad_(False)
    vae = AutoEncoder().eval()
    with torch.no_grad():
        lat = drv.generate(fz, {"hidden_states": torch.randn(1, 4, TINY["text_dim"])}, 32, 32, steps=2, seed=1,
                           cfg=1.0)
        z = drv.practice_latent(vae, lat, 32)
    assert z.shape == (1, 128, 2, 2) and torch.isfinite(z).all()


# ---- 2. LoRA Royale -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """Tiny model files on disk, as the workbench loads them: a DiT (128 latent channels, ComfyUI's fused-qkv layout),
    a random FLUX.2 VAE and two LoRA "epochs" of one run."""
    from safetensors.torch import save_file

    from fizgig.flux3.autoencoder import AutoEncoder
    from fizgig.moss_image.model import MossImageDiT
    from moss_tiny import randomise
    d = tmp_path_factory.mktemp("moss_files")
    torch.manual_seed(0)
    dit = randomise(MossImageDiT(**dict(TINY, latent_channels=128)), seed=3)
    sd = {}
    for k, v in dit.state_dict().items():                   # ComfyUI's file: q/k/v fused into proj.qkv
        if k.endswith(".proj.q.weight"):
            stem = k[: -len("q.weight")]
            sd[stem + "qkv.weight"] = torch.cat([dit.state_dict()[stem + x + ".weight"] for x in "qkv"])
        elif not (k.endswith(".proj.k.weight") or k.endswith(".proj.v.weight")):
            sd[k] = v
    save_file({"model.diffusion_model." + k: v.contiguous() for k, v in sd.items()}, str(d / "dit.safetensors"))
    save_file({k: v.contiguous() for k, v in AutoEncoder().state_dict().items()}, str(d / "vae.safetensors"))
    drv = _drv()
    for e in (1, 2):
        net = _lora(MossImageDiT(**dict(TINY, latent_channels=128)), drv, seed=100 + e)
        net.save(str(d / f"run-{e:06d}.safetensors"), metadata={"ss_base_model_version": "moss-image"})
    return d


def _engine(files, settings=None):
    eng = _desc().make_workbench_engine()
    eng.ensure_pipeline(str(files / "dit.safetensors"), str(files / "vae.safetensors"), "unused-te", device="cpu",
                        lowmem=False, precision="bf16")
    eng.preview_settings = settings
    calls = []
    real = eng.driver.generate

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    eng.driver.generate = spy
    return eng, calls


def _prime(eng, prompts):
    """Conditioning straight into the engine's prompt cache (the real text encoder lives in the side venv)."""
    for i, p in enumerate(prompts):
        g = torch.Generator().manual_seed(1000 + i)
        eng._prompt_cache[(p,) + tuple(eng.driver.text_cache_key())] = {
            "hidden_states": torch.randn(3 + 2 * i, TINY["text_dim"], generator=g).to(torch.bfloat16)}


def test_royale_offered():
    from fizgig.web import royale
    d = _desc()
    assert "royale" in d.workbench and d in royale.families()
    assert royale.family_of([], {d.key: d}) is None


def test_royale_engine_paths(files):
    """What the studio worker's Royale commands do, on the tiny model: point at an epoch, render, swap to the next
    epoch in place, a seed-travel frame, a prompt-travel frame, a sheet's no-LoRA column. Renders carry MOSS's own
    sampling (shift 5, the negative branch skipping block 22) and equal the driver's own generate."""
    from fizgig.families import registry
    from fizgig.networks.lora import assert_lora_family_matches
    eng, calls = _engine(files, {"steps": "2", "cfg": "3", "negative": "bad", "turbo": "0"})
    _prime(eng, ["a fox", "bad", "a cat", "a fox, red"])
    e1, e2 = str(files / "run-000001.safetensors"), str(files / "run-000002.safetensors")
    assert registry.family_of_lora(e1).key == "moss_image"
    assert_lora_family_matches(e1, "moss_image", "Royale")
    eng.load_primary(e1)
    st = eng.default_state(32, 32)
    st.prompt, st.seed = "a fox", 7
    img1 = eng.generate_preview(st)
    assert img1.size == (32, 32)
    kw = calls[-1]
    assert kw["cfg"] == 3.0 and kw["steps"] == 2 and "neg_cond" in kw
    assert dict(kw["options"])["uncond_skip_layers"] == (22,) and dict(kw["options"])["shift"] == 5.0
    # the render is exactly the driver's own, given the same conditioning
    with torch.no_grad():
        want = eng.driver.generate(eng.dit, eng._cond_to_device(eng._prompt_cache[("a fox",)]), 32, 32, steps=2,
                                   seed=7, cfg=3.0, neg_cond=eng._cond_to_device(eng._prompt_cache[("bad",)]),
                                   options=kw["options"])
    import numpy as np
    assert np.array_equal(np.array(eng.driver.decode(eng.vae, want, 32, 32)), np.array(img1))
    assert eng.swap_primary_weights(e2)
    img2 = eng.generate_preview(st)
    assert not np.array_equal(np.array(img1), np.array(img2))            # another epoch, another picture
    # seed travel: t = 0 is seed a's own picture
    a0 = eng.generate_preview(st, seed_b=9, travel_t=0.0)
    assert np.array_equal(np.array(a0), np.array(img2))
    assert eng.generate_preview(st, seed_b=9, travel_t=0.5).size == (32, 32)
    # prompt travel: waypoints padded with masks; the endpoints are their own prompts
    assert eng.supports_prompt_travel
    ctx, _ = eng.encode_travel_prompts(["a fox", "a fox, red"])
    first = eng.generate_preview(st, override_ctx=eng.interp_waypoints(ctx, 0.0))
    assert np.array_equal(np.array(first), np.array(img2))
    last = eng.generate_preview(st, override_ctx=eng.interp_waypoints(ctx, 1.0))
    st2 = eng.default_state(32, 32)
    st2.prompt, st2.seed = "a fox, red", 7
    assert np.array_equal(np.array(last), np.array(eng.generate_preview(st2)))
    assert eng.generate_preview(st, override_ctx=eng.interp_waypoints(ctx, 0.4)).size == (32, 32)
    # the comparison sheet's no-LoRA column: every block at 0 is the bare model
    for bs in st.blocks.values():
        bs.primary_strength = 0.0
    bare = eng.generate_preview(st)
    eng.reset()
    eng2, _ = _engine(files, {"steps": "2", "cfg": "3", "negative": "bad", "turbo": "0"})
    _prime(eng2, ["a fox", "bad"])                     # no LoRA at all
    st3 = eng2.default_state(32, 32)
    st3.prompt, st3.seed = "a fox", 7
    assert np.array_equal(np.array(bare), np.array(eng2.generate_preview(st3)))
    eng2.reset()


# ---- 3. INT8 attention for workbench renders ---------------------------------------------------------------------
@pytest.fixture
def fake_kernel():
    """A stand-in for comfy-kitchen's kernel (exact SDPA, counted), the module's state restored after."""
    from fizgig.modules import int8_attention as i8
    saved = dict(i8._STATE)
    seen = []

    def fn(q, k, v):
        seen.append(tuple(q.shape))
        return F.scaled_dot_product_attention(q, k, v) * 1.0
    i8._STATE.update(checked=True, fn=fn, wanted=False)
    try:
        yield seen
    finally:
        i8._STATE.clear()
        i8._STATE.update(saved)


def test_int8_attention_flag_and_engine():
    d = _desc()
    assert d.int8_attention
    from fizgig.families.workbench import WorkbenchEngine
    assert WorkbenchEngine(d).int8_attention


def test_int8_attention_only_unmasked_no_grad_renders(fake_kernel):
    """The kernel takes the image + text attention of a render (one prompt, padded or not - the per-sample path drops
    the padding); a masked batch (prompts of different lengths), anything under grad and anything outside renders()
    stay on exact SDPA."""
    from fizgig.modules import int8_attention as i8
    fz, _ = pair(seed=41)
    fz.requires_grad_(False)
    n_attn = TINY["num_layers"] + 2                      # the blocks and the 2-block text refiner
    x = torch.randn(1, TINY["latent_channels"], 2, 3)
    c = torch.randn(1, 5, TINY["text_dim"])
    with torch.no_grad():
        exact = fz(x, 300.0, c)
        assert fake_kernel == []                         # switch off: never
        with i8.renders(True):
            got = fz(x, 300.0, c)
        assert len(fake_kernel) == n_attn and torch.allclose(got, exact, atol=1e-6)
        # one padded prompt: the padding is dropped first, so the kernel still runs
        cp = torch.cat([c, torch.zeros(1, 3, TINY["text_dim"])], 1)
        m = torch.tensor([[True] * 5 + [False] * 3])
        fake_kernel.clear()
        with i8.renders(True):
            got = fz(x, 300.0, cp, m)
        assert len(fake_kernel) == n_attn and torch.allclose(got, exact, atol=1e-6)
        # a masked batch of two lengths: exact SDPA with the key mask, never the kernel
        xb = torch.randn(2, TINY["latent_channels"], 2, 3)
        cb = torch.randn(2, 6, TINY["text_dim"])
        mb = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
        ref = fz(xb, torch.tensor([300.0, 600.0]), cb, mb)
        fake_kernel.clear()
        with i8.renders(True):
            got = fz(xb, torch.tensor([300.0, 600.0]), cb, mb)
        assert fake_kernel == [] and torch.equal(got, ref)
    # under grad (training never takes it, even with the switch on)
    with i8.renders(True):
        fz(x, 300.0, c)
    assert fake_kernel == []


def test_int8_attention_on_around_engine_renders(files, fake_kernel):
    eng, _ = _engine(files, {"steps": "1", "cfg": "3", "negative": "bad", "turbo": "0"})
    _prime(eng, ["a fox", "bad"])
    st = eng.default_state(32, 32)
    st.prompt, st.seed = "a fox", 3
    eng.generate_preview(st)
    assert fake_kernel                                   # the render ran through it
    from fizgig.modules import int8_attention as i8
    assert not i8.wanted()                               # and the switch is back off afterwards
    eng.reset()


# ---- 4. torch.compile ---------------------------------------------------------------------------------------------
def test_compile_offered_but_auto_waits_for_a_measurement():
    d = _desc()
    assert d.compiles and d.compile_boundary == "outside" and not d.compile_fullgraph
    drv = _drv()
    assert drv.compile_plan("auto", 100000, "int8", 0)[0] is False          # unmeasured: Auto stays off
    assert drv.compile_plan("on", 10, "int8", 0)[0] == "outside"            # On compiles


@pytest.mark.parametrize("boundary", ["inside", "outside"])
def test_compile_blocks_no_breaks_no_per_step_recompiles(boundary, monkeypatch):
    """The 42-block list compiled in place after the LoRA wrapped it (aot_eager on CPU - the same graphs inductor
    gets): no graph break in a block; after one dynamic-shape recompile, new caption lengths and a new bucket train
    without recompiling; sampling adds one no-grad graph, and the negative branch skipping a block adds none.
    Losses and LoRA grads equal the eager model's."""
    import torch._dynamo as dyn
    from torch._dynamo.testing import CompileCounterWithBackend

    from fizgig.families import compile as C
    dyn.reset()
    drv = _drv()
    fz, _ = pair(seed=51)
    eager, _ = pair(seed=51)
    net, net_e = _lora(fz, drv), _lora(eager, drv)
    for m in (fz, eager):
        m.enable_gradient_checkpointing(True)
        m.train()
    explain = dyn.explain(fz.layers[0])(torch.randn(1, 11, TINY["dim"]),
                                        torch.randn(1, 11, TINY["dim"] // TINY["num_heads"]),
                                        torch.randn(1, 6, TINY["dim"]), None)
    assert explain.graph_break_count == 0, [b.reason for b in explain.break_reasons]
    dyn.reset()
    cnt = CompileCounterWithBackend("aot_eager")
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda m, **kw: real(m, backend=cnt, **{k: v for k, v in kw.items()
                                                                                    if k != "backend"}))
    monkeypatch.setattr(C, "ready_to_compile", lambda *a, **k: True)
    drv.compile_blocks(fz, boundary=boundary)
    assert getattr(fz.layers[0], "_handles_checkpointing", False)

    def step(model, n_text, h, w, seed):
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(1, TINY["latent_channels"], h, w, generator=g)
        c = torch.randn(1, n_text, TINY["text_dim"], generator=torch.Generator().manual_seed(seed + 1))
        loss, _ = drv.training_loss(model, x0, {"hidden_states": c}, g)
        loss.backward()
        return loss.item()
    worst = 0.0
    seen = []
    for i, (n_text, h, w) in enumerate([(5, 2, 3), (7, 2, 3), (9, 2, 3), (6, 3, 4), (12, 3, 4)]):
        a, b = step(fz, n_text, h, w, i), step(eager, n_text, h, w, i)
        assert abs(a - b) <= 1e-5 * abs(b)
        seen.append(cnt.frame_count)
    for p, q in zip(net.parameters(), net_e.parameters()):
        worst = max(worst, rel(p.grad, q.grad))
    RESULTS[f"compile {boundary}: LoRA grads vs eager"] = worst
    assert worst <= 1e-4
    assert seen[1] == seen[-1], seen               # after the first dynamic recompile: none per step
    fz.eval()
    with torch.no_grad():
        for i in range(3):
            drv.generate(fz, {"hidden_states": torch.randn(1, 5 + i, TINY["text_dim"])}, 48, 32, steps=3, seed=1,
                         cfg=3.0, neg_cond={"hidden_states": torch.randn(1, 2, TINY["text_dim"])},
                         options=(("uncond_skip_layers", (1,)),))
            seen.append(cnt.frame_count)
    assert seen[-1] == seen[-3] and seen[-1] - seen[4] <= 2, seen
    RESULTS[f"compile {boundary}: graphs compiled"] = float(seen[-1])
    dyn.reset()


# ---- 5. the SFT preview checkpoint -------------------------------------------------------------------------------
def _convrot_file(path, dit, rot=16):
    """`dit` saved as ComfyUI's int8 ConvRot file: fused proj.qkv, the 4 block Linears int8 with per-row scales and
    a comfy_quant marker, the attention-kernel hint, the rest bf16 (Comfy-Org's moss_image_sft_int8_convrot layout;
    rotation group 16 since the tiny dims are 64 / 96)."""
    import json

    from safetensors.torch import save_file

    from fizgig.minimax.convrot import quantize_int8_convrot
    sd = {k: v for k, v in dit.state_dict().items()}
    out = {}
    marker = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                                           "convrot_groupsize": rot}).encode()), dtype=torch.uint8)
    for k, v in sd.items():
        if k.endswith(".proj.k.weight") or k.endswith(".proj.v.weight"):
            continue
        if k.endswith(".proj.q.weight"):
            stem = k[: -len("q.weight")]
            k, v = stem + "qkv.weight", torch.cat([sd[stem + x + ".weight"] for x in "qkv"])
        if ".proj." in k and k.startswith("layers."):
            base = k[: -len(".weight")]
            q, sc = quantize_int8_convrot(v.float(), rot)
            out[base + ".weight"], out[base + ".weight_scale"], out[base + ".comfy_quant"] = q, sc, marker.clone()
        else:
            out[k] = v.to(torch.bfloat16)
    for n in range(len(dit.layers)):
        out[f"layers.{n}.comfy_attention.config"] = torch.tensor(list(b'{"kernel": "sage"}'), dtype=torch.uint8)
    out["rope.bands"] = torch.zeros(4)
    save_file({k: v.contiguous() for k, v in out.items()}, str(path))
    return str(path)


def test_sft_preview_checkpoint_declared():
    from fizgig.families import form
    d = _desc()
    f, ck = d.preview_checkpoint()
    assert f.role == "preview_dit" and not f.required and f.fetch_is_optional and d.fetch_optional_label
    assert "int8_convrot" in f.path and "bf16" in f.alt_path
    assert d.train_preview_checkpoint and d.preview_checkpoint_on is False      # a Base LoRA carries over partly
    assert ck.cfg == 3.0 and ck.negative_prompt and dict(ck.options) == {"shift": 5.0, "uncond_skip_layers": (22,)}
    assert d.validate() == [] and form.form(d)


def test_sft_convrot_file_loads_as_stored(tmp_path):
    """ComfyUI's int8 ConvRot file: kept int8 (ConvRotInt8Linear, the fused qkv's rows and scales split) and equal to
    the same file decoded; both close to the model it was quantised from. The training base still refuses it."""
    from fizgig.minimax.convrot import ConvRotInt8Linear
    from fizgig.moss_image.model import is_convrot, load_moss_dit
    from moss_tiny import randomise
    from fizgig.moss_image.model import MossImageDiT
    torch.manual_seed(0)
    src = randomise(MossImageDiT(**TINY), seed=41).eval()
    path = _convrot_file(tmp_path / "sft_int8_convrot.safetensors", src)
    assert is_convrot(path)
    drv = _drv()
    kept, swapped = drv.load_planned(path, "cpu", "bf16", 0)
    assert swapped == 0 and kept._convrot
    n_rot = sum(isinstance(m, ConvRotInt8Linear) for m in kept.modules())
    assert n_rot == 6 * TINY["num_layers"]
    assert drv.load_planned(str(tmp_path / "missing.safetensors"), "cpu", "bf16", 0) is None
    dec = load_moss_dit(path, device="cpu", convrot="decode").eval()
    with pytest.raises(ValueError, match="bf16"):
        load_moss_dit(path, device="cpu")
    x = torch.randn(1, TINY["latent_channels"], 3, 4)
    t = torch.tensor([600.0])
    txt = torch.randn(1, 5, TINY["text_dim"])
    with torch.no_grad():
        a = kept(x.bfloat16(), t, txt.bfloat16()).float()
        b = dec(x.bfloat16(), t, txt.bfloat16()).float()
        c = src(x, t, txt)
    RESULTS["SFT convrot kept vs decoded"] = rel(a, b)
    RESULTS["SFT convrot vs source model"] = rel(a, c)
    assert rel(a, b) <= 2e-2 and rel(a, c) <= 6e-2
    with pytest.raises(RuntimeError, match="bf16 file"):
        drv.prepare_training(kept, None)


def test_sft_previews_through_checkpoint_previews(tmp_path, files):
    """The trainer's _CheckpointPreviews round on the tiny SFT file: the training model is parked and restored
    unchanged, the epoch's LoRA rides on the checkpoint (a different picture from the bare SFT), and the render
    takes the SFT recipe (CFG 3 with the negative, shift 5, block 22 skipped)."""
    from fizgig.families.lora import FamilyLoRA
    from fizgig.families.train import _CheckpointPreviews
    from fizgig.moss_image.model import MossImageDiT, load_moss_dit
    from moss_tiny import randomise
    drv = _drv()
    torch.manual_seed(0)
    sft = randomise(MossImageDiT(**dict(TINY, latent_channels=128)), seed=43).eval()
    path = _convrot_file(tmp_path / "sft.safetensors", sft)
    train = load_moss_dit(str(files / "dit.safetensors"), device="cpu", dtype=torch.float32)
    net = _lora(train, drv, seed=7)
    before = {k: v.clone() for k, v in train.state_dict().items()}
    ck = _desc().preview_checkpoint_sampling
    cond = {"hidden_states": torch.randn(1, 5, TINY["text_dim"])}
    neg = {"hidden_states": torch.randn(1, 3, TINY["text_dim"])}
    seen = {}

    def render(m, fl, swapped):
        kw = dict(steps=2, seed=3, cfg=ck.cfg, neg_cond=neg, options=ck.options)
        with torch.no_grad():
            seen["with"] = drv.generate(m, cond, 64, 64, **kw).float()
        seen["swapped"], seen["convrot"] = swapped, getattr(m, "_convrot", False)
        return True

    cp = _CheckpointPreviews(drv, path, "cpu", cache_mode="off")
    assert cp.render(train, net, None, render)
    bare, _ = drv.load_planned(path, "cpu", "bf16", 0)
    with torch.no_grad():
        seen["bare"] = drv.generate(bare, cond, 64, 64, steps=2, seed=3, cfg=ck.cfg, neg_cond=neg,
                                    options=ck.options).float()
    assert seen["convrot"] and seen["swapped"] == 0
    assert torch.isfinite(seen["with"]).all() and not torch.allclose(seen["with"], seen["bare"])
    after = train.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert train.blocks_to_swap == 0 and train.offloader is None


# ---- 6. batch size > 1 ------------------------------------------------------------------------------------------
def _collate(tmp_path, items):
    """The shared dataset's batch assembly (BucketBatchManager.__getitem__) over cache files of these
    (latent, hidden_states) items - the real right-padding + mask code."""
    import types

    from safetensors.torch import save_file

    from fizgig.dataset.image_dataset import BucketBatchManager
    infos = []
    for j, (lat, h) in enumerate(items):
        lp, tp = tmp_path / f"i{j}_lat.safetensors", tmp_path / f"i{j}_te.safetensors"
        save_file({f"latent_{lat.shape[-2]}x{lat.shape[-1]}": lat.contiguous()}, str(lp))
        save_file({"cond__hidden_states": h.contiguous()}, str(tp))
        infos.append(types.SimpleNamespace(latent_cache_path=str(lp), text_encoder_output_cache_path=str(tp),
                                           item_key=f"img{j}", still_only=False))
    fake = types.SimpleNamespace(bucket_batch_indices=[("r", 0)], buckets={"r": infos}, batch_size=len(items),
                                 timestep_pool=None)
    return BucketBatchManager.__getitem__(fake, 0)


def test_batch_collation_pads_ragged_captions(tmp_path):
    C = TINY["latent_channels"]
    lat = [torch.randn(C, 2, 3) for _ in range(3)]
    hs = [torch.randn(n, TINY["text_dim"]) for n in (4, 7, 5)]
    b = _collate(tmp_path, list(zip(lat, hs)))
    assert b["latents"].shape == (3, C, 2, 3) and b["cond__hidden_states"].shape == (3, 7, TINY["text_dim"])
    m = b["cond__hidden_states_mask"]
    assert m.dtype == torch.bool and m.sum(1).tolist() == [4, 7, 5] and not m[0, 4:].any()
    for j, h in enumerate(hs):
        assert torch.equal(b["cond__hidden_states"][j, : h.shape[0]], h)
        assert not b["cond__hidden_states"][j, h.shape[0]:].any()
    c = _drv().batch_cond(b, "cpu")
    assert set(c) == {"hidden_states", "mask"}
    same = _collate(tmp_path / "..", [(lat[0], hs[0]), (lat[1], hs[0])])   # equal lengths: plain stack, no mask
    assert "cond__hidden_states_mask" not in same
    assert _drv().supports_batch and not __import__("fizgig.families.driver", fromlist=["x"]).FamilyDriver.supports_batch


@pytest.mark.parametrize("dropout,seed", [(0.0, 1234), (0.5, 70), (0.5, 75)])   # 70 / 75: row 2 / row 1 dropped
def test_batch_of_two_equals_two_batch_one_steps(tmp_path, dropout, seed):
    """A batch of 2 with captions of different lengths (padded + mask through the real collation): its loss is the
    mean of the two batch-1 losses and its LoRA grads the mean of theirs - per-sample noise, sigma and caption
    dropout drawn in batch 1's order from the same generator."""
    drv = _drv({"caption_dropout": dropout} if dropout else {})
    fz, _ = pair(seed=51)
    net = _lora(fz, drv, seed=5)
    drv._uncond = torch.randn(1, 3, TINY["text_dim"]) if dropout else None
    C = TINY["latent_channels"]
    lat = [torch.randn(C, 2, 3, generator=torch.Generator().manual_seed(k)) for k in (1, 2)]
    hs = [torch.randn(n, TINY["text_dim"], generator=torch.Generator().manual_seed(10 + n)) for n in (4, 7)]
    batch = _collate(tmp_path, list(zip(lat, hs)))
    params = list(net.parameters())

    def grads():
        g = [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        return g

    gen = torch.Generator().manual_seed(seed)
    l1, i1 = drv.training_loss(fz, lat[0][None], {"hidden_states": hs[0][None]}, gen)
    l2, i2 = drv.training_loss(fz, lat[1][None], {"hidden_states": hs[1][None]}, gen)
    ((l1 + l2) / 2).backward()
    g_seq = grads()
    gen = torch.Generator().manual_seed(seed)
    lb, ib = drv.training_loss(fz, batch["latents"], drv.batch_cond(batch, "cpu"), gen)
    lb.backward()
    g_b = grads()
    RESULTS[f"batch-2 loss vs batch-1 mean p{dropout} s{seed}"] = rel(lb.detach(), (l1 + l2).detach() / 2)
    gr = max(rel(a, b) for a, b in zip(g_b, g_seq) if float(b.abs().sum()) > 0)
    RESULTS[f"batch-2 grads vs batch-1 mean p{dropout} s{seed}"] = gr
    assert rel(lb.detach(), (l1 + l2).detach() / 2) <= 1e-5 and gr <= 1e-4
    assert abs(ib["t"] - (i1["t"] + i2["t"]) / 2) <= 1e-6 and i1["t"] != i2["t"]


def test_batch_guard_in_the_trainer():
    """The trainer's batch rule: a family without supports_batch keeps the batch-1 error; MOSS passes, a slider or a
    fine-tune at batch > 1 is refused (the source text of the guard, as the trainer runs on CUDA only)."""
    import inspect

    from fizgig.families import train
    src = inspect.getsource(train.train_family)
    assert "driver.supports_batch" in src and "slider or finetune" in src
    assert "per-image loss watch, per-image LR, auto-recaption and look warm-up are off" in src
    from fizgig.families.registry import FAMILIES
    others = [k for k, d in FAMILIES.items() if k != "moss_image" and d.training_ready]
    for k in others[:4]:
        assert FAMILIES[k].load_driver().supports_batch is False, k


def test_zz_print_results():
    for k, v in RESULTS.items():
        print(f"{k[:50]:50s} {v:.3e}")
