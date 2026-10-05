"""ACE-Step 1.5 XL driver for Fizgig's standard layer (families/driver.py) - text-to-music: songs with vocals and
lyrics, 48 kHz stereo. Two bases: XL Turbo (default) and XL SFT (family option base=sft).

* model: acestep/model.py (adapted from the official repository), loaded from ComfyUI's single file; the DiT is the
  root module, its condition encoder (text projector, lyric and timbre encoders) and null condition beside it
* conditioning, cached per item: Qwen3-Embedding-0.6B's last hidden state for the official text-to-music prompt
  (instruction, caption, bpm / time signature / key / duration) -> {"text": (Lt, 1024)}, and the token embeddings of
  "# Languages / # Lyric" -> {"lyrics": (Ll, 1024)}. The condition encoder runs inside every forward (frozen, no
  grad), with the silence latent as the timbre reference - the official handler's text-to-music without a reference
* latents: the Oobleck VAE (acestep/vae.py), 64 channels at 25 per second, (64, T); songs over MAX_SECONDS are cut
* training: rectified flow, x_t = t noise + (1 - t) x0, target noise - x0, unweighted MSE. Turbo trains on its own 8
  timesteps (shift 3), as the official trainer and Side-Step's turbo presets do; SFT on continuous logit-normal
  timesteps (mu -0.4, sigma 1) with 15% of steps conditioned on the null embedding (Side-Step's corrected mode)
* sampling: Euler; Turbo over the official schedule (8 steps, shift 3, no CFG), SFT 30 steps at shift 1 with APG
  guidance 7 against the null condition (the XL SFT code's defaults); context = the silence latent + ones
* LoRA: q/k/v/o of each layer's self- and cross-attention (the official trainer's targets), keys
  base_model.model.layers.N.self_attn.q_proj.lora_A.weight + .alpha, which ComfyUI loads as ACE-Step 1.5 LoRAs
"""
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
HELPER = "ACE-Step/Ace-Step1.5"                    # tokenizer + silence latent (helper_files)
TOKENIZER = "Qwen3-Embedding-0.6B"
SILENCE = "acestep-v15-turbo/silence_latent.pt"
RATE = 25                                          # latent frames per second (48 kHz / hop 1920)
MAX_SECONDS = 240                                  # the official trainer's max_duration
REFER_FRAMES = 750                                 # the handler's timbre reference with no reference audio: 30 s
TEXT_TOKENS, LYRIC_TOKENS = 256, 2048              # the handler's truncation
INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"   # text2music (constants.py)
TURBO_T = (1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3)
SFT_STEPS, SFT_SHIFT, SFT_GUIDANCE = 30, 1.0, 7.0          # acestep-v15-xl-sft generate_audio defaults
TIMESTEP_MU, TIMESTEP_SIGMA, CFG_DROPOUT = -0.4, 1.0, 0.15  # config.json; Side-Step's corrected training
LYRICS_SUFFIX = ".lyrics.txt"
INSTRUMENTAL = "[Instrumental]"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def caption_prompt(caption, seconds, bpm=None, timesignature=None, keyscale=None):
    """The official SFT_GEN_PROMPT with the text-to-music instruction and _dict_to_meta_string's metadata."""
    metas = (f"- bpm: {bpm or 'N/A'}\n- timesignature: {timesignature or 'N/A'}\n- keyscale: {keyscale or 'N/A'}\n"
             f"- duration: {int(seconds)} seconds\n")
    return f"# Instruction\n{INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"


def lyrics_prompt(lyrics, language="en"):
    """The handler's _format_lyrics."""
    return f"# Languages\n{language or 'en'}\n\n# Lyric\n{lyrics or INSTRUMENTAL}<|endoftext|>"


def song_info(audio_path):
    """A training song's sidecars: <stem>.lyrics.txt (else instrumental) and an optional <stem>.json with bpm,
    timesignature, keyscale, language."""
    stem = os.path.splitext(audio_path)[0]
    info = {}
    if os.path.exists(stem + ".json"):
        with open(stem + ".json", encoding="utf-8") as f:
            info = {k: v for k, v in json.load(f).items() if v not in (None, "")}
    if os.path.exists(stem + LYRICS_SUFFIX):
        with open(stem + LYRICS_SUFFIX, encoding="utf-8") as f:
            info["lyrics"] = f.read().strip()
    return info


def parse_prompt(prompt):
    """A sample prompt line: "<caption>" or "<caption> | <lyrics>", lyric lines separated by " / "."""
    caption, _, lyrics = prompt.partition("|")
    lyrics = "\n".join(s.strip() for s in lyrics.split(" / ")).strip()
    return caption.strip(), lyrics or INSTRUMENTAL


def read_audio(path, seconds=MAX_SECONDS):
    """Any audio file -> float32 (2, n) at 48 kHz, the first `seconds` (ffmpeg decodes and resamples)."""
    import subprocess
    from fizgig.families.clips import ffmpeg
    cmd = [ffmpeg(), "-v", "error", "-i", path, "-t", str(seconds), "-f", "f32le", "-ac", "2", "-ar", "48000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def audio_seconds(path):
    """A song's length as trained (capped at MAX_SECONDS), from ffmpeg's banner - no decode."""
    import re
    import subprocess
    from fizgig.families.clips import ffmpeg
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise ValueError(f"{os.path.basename(path)}: not a readable audio file")
    h, mi, se = m.groups()
    return min(MAX_SECONDS, int(h) * 3600 + int(mi) * 60 + float(se))


class _Momentum:
    """APG's momentum buffer (apg_guidance.MomentumBuffer, momentum -0.75)."""

    def __init__(self, momentum=-0.75):
        self.momentum, self.average = momentum, 0

    def update(self, value):
        self.average = value + self.momentum * self.average


def apg(cond, uncond, scale, momentum, norm_threshold=2.5):
    """Adaptive projected guidance (the XL SFT code's apg_forward, eta 0, dims [1] = time): the momentum-averaged
    (cond - uncond), clamped to norm 2.5 along time, minus its part parallel to cond."""
    momentum.update(cond - uncond)
    diff = momentum.average
    diff = diff * torch.minimum(torch.ones_like(diff), norm_threshold / diff.norm(p=2, dim=1, keepdim=True))
    c = F.normalize(cond.double(), dim=1)
    d = diff.double()
    orthogonal = (d - (d * c).sum(dim=1, keepdim=True) * c).to(cond.dtype)
    return cond + (scale - 1) * orthogonal


def _helper_file(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HELPER, name)


def silence_latent():
    """The official silence latent, (1, 15000, 64) float32."""
    return torch.load(_helper_file(SILENCE), map_location="cpu").transpose(1, 2).float()


class _TextEncoder:
    """Qwen3-Embedding-0.6B from ComfyUI's single file, with its tokenizer from the ACE-Step repository."""

    def __init__(self, path, device):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        self.tok = AutoTokenizer.from_pretrained(HELPER, subfolder=TOKENIZER)
        model = Qwen3Model(Qwen3Config(
            vocab_size=151669, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128, max_position_embeddings=32768,
            rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, use_sliding_window=False))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"not ACE-Step's Qwen3 0.6B file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model = model.to(device, DTYPE).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, caption_text, lyric_text):
        c = self.tok(caption_text, truncation=True, max_length=TEXT_TOKENS, return_tensors="pt").input_ids
        ly = self.tok(lyric_text, truncation=True, max_length=LYRIC_TOKENS, return_tensors="pt").input_ids
        text = self.model(input_ids=c.to(self.device)).last_hidden_state[0]
        lyrics = self.model.embed_tokens(ly.to(self.device))[0]
        return {"text": text.to(DTYPE).cpu(), "lyrics": lyrics.to(DTYPE).cpu()}

    def unload(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AceStepDriver(FamilyDriver):
    _silence = None

    @property
    def silence(self):
        if AceStepDriver._silence is None:
            AceStepDriver._silence = silence_latent()
        return AceStepDriver._silence

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from fizgig.acestep.model import AceConfig, AceStep15
        with init_empty_weights():
            dit = AceStep15(AceConfig())
        missing, unexpected = dit.load_checkpoint({k: v.to(DTYPE) for k, v in load_file(path).items()}, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"not an ACE-Step 1.5 file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return dit.to(device).eval().requires_grad_(False)

    def load_vae(self, path, device):
        from fizgig.acestep.vae import load_vae
        return load_vae(path, device)

    def load_text_encoder(self, path, device):
        return _TextEncoder(path, device)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.gradient_checkpointing = bool(on)

    # ---- caching --------------------------------------------------------------------------------
    def cache_stage(self, stage, datasets, args, device, aux):
        from fizgig.families.cache import _clean, FORMAT_VERSION
        from safetensors import safe_open
        from safetensors.torch import save_file
        desc = self.description
        if stage == "latents":
            from fizgig.acestep import vae as V
            from fizgig.scripts.cache_latents import encode_datasets
            vae = self.load_vae(args.model, device)
            gen = torch.Generator().manual_seed(0)
            args.needs_reencode = lambda p: True       # skip_existing is decided below, by the source file

            def encode(batch):
                for it in batch:
                    stamp = f"{os.path.getsize(it.item_key)}:{int(os.path.getmtime(it.item_key))}:{MAX_SECONDS}"
                    if args.skip_existing and os.path.exists(it.latent_cache_path):
                        with safe_open(it.latent_cache_path, framework="pt") as f:
                            if (f.metadata() or {}).get("source") == stamp:
                                continue
                    wav = read_audio(it.item_key)
                    z = V.encode(vae, torch.from_numpy(wav), gen)[0].T            # (64, T)
                    logger.info(f"latent cache: {it.item_key} -> {tuple(z.shape)} ({wav.shape[1] / 48000:.1f} s)")
                    os.makedirs(os.path.dirname(it.latent_cache_path), exist_ok=True)
                    save_file({f"latent_{z.shape[1]}": _clean(z, "latent", it.item_key)}, it.latent_cache_path,
                              metadata={"architecture": desc.arch_id, "seconds": f"{wav.shape[1] / 48000:.3f}",
                                        "source": stamp, "format_version": FORMAT_VERSION})
            encode_datasets(datasets, encode, args)
            return True

        from fizgig.scripts.cache_text import post_process, prepare_cache_files_and_paths, process_batches
        files, paths = prepare_cache_files_and_paths(datasets)
        te = self.load_text_encoder(args.model, device)
        skip = args.skip_existing
        args.skip_existing = False                     # existence alone is not enough: the prompt is checked below

        def encode(batch):
            for it in batch:
                info = song_info(it.item_key)
                cap = caption_prompt(it.caption, int(audio_seconds(it.item_key)), info.get("bpm"),
                                     info.get("timesignature"), info.get("keyscale"))
                lyr = lyrics_prompt(info.get("lyrics"), info.get("language", "en"))
                key = cap + "\n" + lyr
                p = it.text_encoder_output_cache_path
                if skip and os.path.exists(p):
                    with safe_open(p, framework="pt") as f:
                        if (f.metadata() or {}).get("caption1") == key:
                            continue
                c = te.encode(cap, lyr)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                save_file({f"cond__{k}": _clean(v, k, it.item_key) for k, v in c.items()}, p,
                          metadata={"architecture": desc.arch_id, "caption1": key, "format_version": FORMAT_VERSION})
                logger.info(f"text cache: {it.item_key} ({c['text'].shape[0]} caption + {c['lyrics'].shape[0]} "
                            f"lyric tokens)")
        process_batches(args, datasets, files, paths, encode)
        self.unload_text_encoder(te)
        post_process(datasets, files, paths, args.keep_cache)
        return True

    def media_problem(self, path):
        stem = os.path.splitext(path)[0]
        if os.path.exists(stem + ".json"):
            try:
                with open(stem + ".json", encoding="utf-8") as f:
                    json.load(f)
            except ValueError as e:
                return f"{os.path.basename(stem)}.json is not valid JSON ({e})"
        return ""

    def encode_text(self, te, captions):
        """Sample prompts ("<caption> | <lyrics>"), at the preview length."""
        seconds = self._preview_seconds()
        return [te.encode(caption_prompt(c, seconds), lyrics_prompt(ly)) for c, ly in map(parse_prompt, captions)]

    def _preview_seconds(self):
        return int(float(self.options.get("preview_seconds", 30)))

    # ---- the model call -------------------------------------------------------------------------
    def _condition(self, dit, cond, frames, device):
        """The condition encoder's sequence for one item (no grad: the encoder is never trained)."""
        text = cond["text"].to(device, DTYPE)
        lyrics = cond["lyrics"].to(device, DTYPE)
        text = text if text.dim() == 3 else text[None]
        lyrics = lyrics if lyrics.dim() == 3 else lyrics[None]
        refer = self.silence[:, :REFER_FRAMES].to(device, DTYPE)
        with torch.no_grad():
            enc, _ = dit.condition(text, torch.ones(text.shape[:2], device=device), lyrics,
                                   torch.ones(lyrics.shape[:2], device=device), refer)
        return enc

    def _context(self, frames, device):
        from fizgig.acestep.model import context_latents
        return context_latents(self.silence, frames).to(device, DTYPE)

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _x0(latents):
        """Cached (1, 64, T) -> the model's (1, T, 64)."""
        return latents.float().transpose(1, 2)

    @property
    def sft(self):
        """Training (or previewing) on the XL SFT base (family option base=sft)."""
        return self.options.get("base") == "sft"

    def _draw_t(self, n, generator, min_t, max_t):
        """Turbo: its own 8 timesteps, uniformly (the official trainer); a window keeps those inside. SFT: continuous
        logit-normal (mu -0.4, sigma 1), rescaled into the window."""
        if self.sft:
            t = torch.sigmoid(torch.randn(n, generator=generator) * TIMESTEP_SIGMA + TIMESTEP_MU)
            return min_t + (max_t - min_t) * t
        ts = [t for t in TURBO_T if min_t <= t <= max_t] or list(TURBO_T)
        idx = torch.randint(len(ts), (n,), generator=generator)
        return torch.tensor([ts[i] for i in idx])

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        x0 = self._x0(latents)
        dev = x0.device
        t = self._draw_t(x0.shape[0], generator, min_t, max_t).to(dev)
        noise = torch.randn(x0.shape, generator=generator).to(dev)
        tb = t.view(-1, 1, 1)
        xt = tb * noise + (1 - tb) * x0
        enc = self._condition(dit, cond, x0.shape[1], dev)
        if self.sft and float(torch.rand(1, generator=generator)) < CFG_DROPOUT:
            enc = dit.null_condition_emb.to(enc).expand_as(enc)      # the guidance's null condition, learned too
        pred = dit(xt.to(DTYPE), t.to(DTYPE), enc, self._context(x0.shape[1], dev)).float()
        return F.mse_loss(pred, noise - x0), {"t": float(t.mean())}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = self._x0(latents)
        t = self._draw_t(1, generator, min_t, max_t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        return {"xt": t.view(-1, 1, 1) * noise + (1 - t.view(-1, 1, 1)) * x0, "t": t}

    def predict(self, dit, state, cond):
        xt = state["xt"]
        enc = self._condition(dit, cond, xt.shape[1], xt.device)
        return dit(xt.to(DTYPE), state["t"].to(DTYPE), enc, self._context(xt.shape[1], xt.device))

    # ---- sampling -------------------------------------------------------------------------------
    def _frames(self, seconds=None):
        return int(round((seconds or self._preview_seconds()) * RATE))

    @torch.no_grad()
    def initial_noise(self, seed, width=0, height=0, frames=None):
        g = torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn((1, frames or self._frames(), 64), generator=g, dtype=torch.float32)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, **_ignored):
        device = next(dit.parameters()).device
        x = (self.initial_noise(seed) if noise is None else noise).to(device).float()
        guided = self.sft
        if guided:
            ts = [1.0 - i / SFT_STEPS for i in range(SFT_STEPS)]
            if SFT_SHIFT != 1.0:
                ts = [SFT_SHIFT * t / (1 + (SFT_SHIFT - 1) * t) for t in ts]
        elif sigmas is not None:
            ts = [float(s) for s in sigmas if float(s) > 0.0]
        elif int(steps) == len(TURBO_T):
            ts = list(TURBO_T)
        else:
            shift = float(dict(options or ()).get("shift", 3.0))
            ts = [shift * t / (1 + (shift - 1) * t) for t in (1.0 - i / int(steps) for i in range(int(steps)))]
        ts = ts + [0.0]
        enc = self._condition(dit, cond, x.shape[1], device)
        ctx = self._context(x.shape[1], device)
        if guided:
            enc = torch.cat([enc, dit.null_condition_emb.to(enc).expand_as(enc)])
            ctx = ctx.expand(2, -1, -1)
            momentum = _Momentum()
        for i in range(len(ts) - 1):
            if on_step is not None:
                on_step(i, len(ts) - 1)
            xin = torch.cat([x, x]) if guided else x
            v = dit(xin.to(DTYPE), torch.full((xin.shape[0],), ts[i], device=device, dtype=DTYPE), enc, ctx).float()
            if guided:
                v = apg(v[:1], v[1:], SFT_GUIDANCE, momentum)
            x = x + (ts[i + 1] - ts[i]) * v
        return x

    @torch.no_grad()
    def decode(self, vae, latents, width=0, height=0):
        """-> a Song (waveform + a picture of it), which save_preview writes."""
        from fizgig.acestep import vae as V
        lat = latents if latents.shape[-1] == 64 else latents.transpose(1, 2)
        return Song(V.decode(vae, lat.float()))

    def decode_audio(self, vae, audio):
        return self.decode(vae, audio).wave

    def save_preview(self, result, path):
        from fizgig.minimax.common import write_wav
        from fizgig.acestep.vae import SAMPLE_RATE
        wav = os.path.splitext(path)[0] + ".wav"
        write_wav(wav, result.wave, sample_rate=SAMPLE_RATE)
        result.picture().save(path)                  # last: the gallery's "finished" signal
        return [wav, path]

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        blocks = []
        for i in range(self.description.n_blocks):
            mods = [f"layers.{i}.{a}.{p}" for a in ("self_attn", "cross_attn") for p in ATTN]
            if names is not None:
                mods = [m for m in mods if m in names]
            blocks.append(Block(f"layer_{i}", f"Layer {i}", mods))
        return [BlockGroup("DiT layers", blocks)]

    def alias_flat(self, flat):
        """ComfyUI's checkpoint naming (decoder_layers_N_...) -> the decoder-rooted name."""
        return flat[len("decoder_"):] if flat.startswith("decoder_layers_") else None


class Song:
    """A decoded preview: the waveform (2, n) at 48 kHz, and a waveform picture for the gallery tile."""

    def __init__(self, wave):
        self.wave = wave

    def picture(self, width=1024, height=256):
        from PIL import Image, ImageDraw
        w = self.wave.mean(0).numpy()
        cols = np.array_split(np.abs(w), width)
        peak = max(1e-6, float(np.abs(w).max()))
        img = Image.new("RGB", (width, height), (24, 26, 33))
        d = ImageDraw.Draw(img)
        mid = height // 2
        for x, c in enumerate(cols):
            a = float(c.max()) / peak if c.size else 0.0
            r = float(np.sqrt((c ** 2).mean())) / peak if c.size else 0.0
            d.line([(x, mid - a * mid * 0.95), (x, mid + a * mid * 0.95)], fill=(70, 110, 170))
            d.line([(x, mid - r * mid * 0.95), (x, mid + r * mid * 0.95)], fill=(140, 190, 250))
        return img
