"""MOSS-Image (OpenMOSS) through the standard layer: a 5.08B single-stream DiT (42 blocks, width 3072), the
Qwen3.5-9B text encoder (run in a side venv: it needs transformers >= 5.3, Fizgig pins 4.57) and the FLUX.2 VAE.
Apache-2.0. Driver: moss_image/driver.py. Hidden until it has been trained on a real GPU (milestone M4).

Facts from the reference pipeline (OpenMOSS-Team/MOSS-Image-Base pipeline_moss_image.py, configs), the paper
(App. A.2) and OpenMOSS/MOSS-Image USERGUIDE.md, Oct 2026. The text encoder and its refiner inside the DiT are never
trained (USERGUIDE: "keep the text encoder frozen"; Chris, 16 Oct 2026).
"""
from fizgig.families.description import FamilyDescription, FamilyOption, LoRAFormat, ModelFile, SamplingSettings

_REPO = "OpenMOSS-Team/MOSS-Image-Base"
_COMFY = "Comfy-Org/MOSS-Image"
_CARD = f"https://huggingface.co/{_REPO}"
_GUIDE = "https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md"
_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"      # the USERGUIDE's
_REF_OPTS = (("shift", 5.0), ("uncond_skip_layers", (22,)))


def _preset(rank, lr, epochs=30, mp="1.0", slider=False):
    # not yet measured on MOSS (M4): rank 8-16, fused AdamW, the shared 1e-4 / 2e-4 starting points
    return {
        "FAMILY_SLIDER": slider, "FAMILY_SLIDER_GUIDANCE": "2",
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "ADAPTIVE_LR": False,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "MOSS_CAPTION_DROPOUT": "0",
    }


MOSS_IMAGE_BASE = FamilyDescription(
    key="moss_image",
    arch_id="mossimage",
    display_name="MOSS-Image",
    gui_label="MOSS-Image",
    lora_name_suffix="moss",
    aliases=("moss-image", "moss_image", "moss"),
    experimental=True,
    hidden=False,                     # M4 passed on an RTX 4070 Ti 12 GB (2.9 s/step at 1024, rank 16, INT8)

    model_files=(
        ModelFile("moss_dit", "MOSS-Image Base DiT", True, _COMFY,
                  "diffusion_models/moss_image_base_bf16.safetensors", 10.17, role="dit",
                  hint="bf16 base (ComfyUI's single file: q/k/v fused, split on load). Fizgig applies its own INT8 / "
                       "NF4; ComfyUI's int8 ConvRot file is not a training base. The official diffusers shards "
                       "(transformer/ folder) load too.",
                  download_note="~10.2 GB - Apache-2.0"),
        ModelFile("moss_vae", "FLUX.2 VAE", True, _COMFY, "vae/flux2-vae.safetensors", 0.34, role="vae",
                  hint="The FLUX.2 autoencoder - the same file Flux 3 uses."),
        ModelFile("moss_text_encoder", "Qwen3.5-9B text encoder", True, _REPO,
                  "text_encoder/model.safetensors.index.json", 16.8, role="text_encoder",
                  hint="The official text_encoder/ folder (index + 4 shards + config.json). Runs in its own small "
                       "environment (envs/moss_te, transformers 5.x on Fizgig's own torch), made on first use; "
                       "unloaded before training.",
                  download_note="~16.8 GB in 4 shards"),
    ),
    prefs_title="Model Paths (MOSS-Image)",
    text_encoder_label="Qwen3.5-9B",
    vae_label="FLUX.2 VAE",

    latent_channels=128,              # FLUX.2's 32 channels 2x2-patchified, BatchNorm-normalised
    spatial_factor=16,
    bucket_step=16,
    native_megapixels=1.0,            # USERGUIDE: 1-2 MP outputs, 0.25-4 MP works

    n_blocks=42,                      # transformer/config.json num_layers
    block_prefix="layers",
    block_note="42 identical single-stream blocks (image and text tokens together; one modulation shared by all). "
               "The 2-block text refiner in front of them stays frozen, as does the text encoder.",
    workbench=("repair", "profiler", "extract"),

    lora=LoRAFormat(
        key_template="lora_unet_layers_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys; ComfyUI maps lora_unet_layers_N_proj_q to diffusion_model.layers.N.proj.q, and on its "
             "fused-qkv file onto that block's qkv rows.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet; reference pipeline module names",
    ),

    driver="fizgig.moss_image.driver:MossImageDriver",
    modelspec_arch="moss-image",
    implementation=_CARD,
    ema_default="0.98",
    precisions=("int8", "nf4", "bf16"),
    auto_precisions=("int8", "nf4"),  # bf16 is 10.2 GB of weights: past 12-16 GB cards with activations
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    # sliders (the shared trainer's two kinds): image pairs through training_loss(diff_ref=, diff_weight=), prompt
    # pairs through noise_latents / predict. The push strength is Qwen's 2 until a MOSS slider is measured
    slider_training=True,
    slider_guidance=2.0,
    helper_files=((_REPO, ("tokenizer/*", "text_encoder/config.json")),),
    workbench_follows_samples=True,

    options=(
        FamilyOption("MOSS_CAPTION_DROPOUT", "Caption dropout", kind="entry", tokens="caption_dropout={}",
                     default="0", width=8,
                     hint="Share of steps trained on the empty prompt instead of the caption (0 = off, the default; "
                          "0.05-0.1 keeps CFG working on a LoRA trained long). The empty prompt is encoded with "
                          "the captions.",
                     summary="caption dropout {}", section="other"),
        FamilyOption("MOSS_TIMESTEPS", "Timesteps", choices=(
            ("Paper (noisy-weighted)", "timesteps=paper"), ("Logit-normal", "timesteps=logit_normal"),
            ("Uniform", "timesteps=uniform")),
            hint="Paper: the base's own final-stage mix (a truncated logit-normal leaning noisy, 1% uniform).",
            section="other", width=24),
    ),

    sampling=(
        SamplingSettings("MOSS-Image Base", steps=50, cfg=3.0, sampler="euler", scheduler="simple", options=_REF_OPTS,
                         negative_prompt=True,
                         note="Shift 5, CFG 3 (3-6; more for short prompts), >= 50 steps. The negative branch skips "
                              "block 22 (two forwards per step, as the reference).",
                         source=f"{_GUIDE}; reference pipeline MOSSImagePipeline.__call__ defaults"),
    ),
    preview_steps=28,
    preview_cfg=3.0,
    preview_negative=_NEGATIVE,
    preview_cfg_note="3 for long prompts, up to 6 for short ones. MOSS-Image reads long natural-language captions best.",
    preview_width=1024,
    preview_height=1024,
    repair_size=1024,

    presets=(
        ("✨ MOSS-Image Character (rank 16, 1e-4)", _preset(16, 1e-4)),
        ("✨ MOSS-Image Style (rank 16, 5e-5)", _preset(16, 5e-5)),
        # sliders run hot and short (Qwen's / Krea 2's recipe: rank 8, 2e-4, 0.5 MP); not yet measured on MOSS
        ("✨ MOSS-Image Slider (rank 8, 2e-4)", _preset(8, 2e-4, epochs=20, mp="0.5", slider=True)),
    ),

    notes=(
        ("Licence: Apache-2.0 (weights and code).", _CARD),
        ("Keep the text encoder frozen (OpenMOSS); Fizgig also keeps the DiT's text refiner frozen.", _GUIDE),
        ("Text: Qwen3.5-9B last hidden state of the chat template with thinking on, per caption without padding "
         "(up to 4096 tokens); the image's positions start after the prompt's length.", "pipeline_moss_image.py"),
        ("Training: flow matching (sigma = 1000 timestep), velocity noise - x0, per-image token-mean MSE, no "
         "weighting; timesteps on the 1000-step grid.", "MOSS-Image paper App. A.2"),
    ),
)
