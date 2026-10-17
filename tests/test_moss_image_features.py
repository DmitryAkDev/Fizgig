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


def test_zz_print_results():
    for k, v in RESULTS.items():
        print(f"{k[:50]:50s} {v:.3e}")
