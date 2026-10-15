"""Flux 3 Image Klein (Black Forest Labs, released 15 Oct 2026): a 5.6B DiT for text-to-image and editing with up to
10 reference images, Qwen3-VL-4B text encoder, the FLUX.2 autoencoder.

Sources: the model cards (huggingface.co/black-forest-labs/flux-3-image-klein, -base, -comfy) and the reference code
(github.com/black-forest-labs/flux-3-image-klein, Apache-2.0).
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_COMFY = "black-forest-labs/flux-3-image-klein-comfy"
_CARD = "https://huggingface.co/black-forest-labs/flux-3-image-klein"
_REF = "https://github.com/black-forest-labs/flux-3-image-klein"
_MODULES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")   # model.py Block, every Linear

FLUX3 = FamilyDescription(
    key="flux3",
    arch_id="flux3klein",
    display_name="Flux 3 Image Klein",
    gui_label="Flux 3 Image Klein",
    lora_name_suffix="f3k",
    aliases=("flux-3-image-klein", "flux3-klein"),

    model_files=(
        ModelFile("flux3_dit", "Flux 3 Image Klein Base DiT", True, _COMFY,
                  "diffusion_models/flux3_image_klein_bf16.safetensors", 11.08,
                  "bf16 Base (undistilled) for training. Files with _sd_ in the name are the distilled model.",
                  role="dit", gated=True),
        ModelFile("flux3_vae", "Flux 3 autoencoder", True, _COMFY, "vae/flux2-vae.safetensors", 0.34,
                  "The FLUX.2 autoencoder, as Klein 9B's.", role="vae", gated=True),
        ModelFile("flux3_text_encoder", "Qwen3-VL-4B text encoder", True, _COMFY,
                  "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88,
                  "Used for caching only, then unloaded.", role="text_encoder", gated=True),
        ModelFile("flux3_distilled_dit", "Flux 3 Image Klein (distilled, previews)", False, _COMFY,
                  "diffusion_models/flux3_image_klein_sd_bf16.safetensors", 11.10,
                  "16 steps, no CFG: fast previews and the workbench.", role="preview_dit", gated=True),
    ),
    text_encoder_label="Qwen3-VL-4B",
    vae_label="FLUX.2 autoencoder",

    latent_channels=128,              # autoencoder.py LATENT_CHANNELS: 32 channels 2x2-packed
    spatial_factor=16,                # autoencoder.py DOWNSAMPLING_RATIO
    bucket_step=16,                   # pipeline.py latent_shape: sides must be multiples of 16
    native_megapixels=1.0,            # pipeline.py default_size 1024x1024 (references up to 4 MP)

    n_blocks=28,                      # the single blocks; the driver's block_map adds the 3 x 5 per-mode blocks
    block_prefix="single_blocks",
    block_note="Five blocks each for the image, the reference images and the text run before the streams meet, "
               "then 28 shared single blocks. Reference blocks only run on edits.",

    lora=LoRAFormat(
        key_template="lora_unet_single_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        alpha_key="{prefix}.alpha",
        kohya=True,
        lokr_kohya_stems=True,         # LoKR under the same lora_unet_ names, so ComfyUI's Flux3 q/k/v map applies
        file_prefix="lora_unet_",
        note="kohya keys over the reference module names; ComfyUI maps lora_unet_<path>_q_proj / _k_proj / _v_proj "
             "onto thirds of its fused qkv.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet, Flux3 branch (master, 15 Oct 2026)",
    ),

    driver="fizgig.flux3.driver:Flux3Driver",
    modelspec_arch="Flux.3-image-klein",
    implementation=_REF,
    precisions=("bf16", "int8", "nf4"),
    # training peaks, LoRA rank 16, adamw (15 Oct 2026, RTX 5090). 1.0 MP measured for each precision; 0.25 MP from
    # the measured swap run (9.7 GB = 11.08 weights - 10 x 0.245 swapped + 1.07 overhead) minus the 0.4 GB bf16 grew
    # from 0.25 to 1.0 MP. Per swapped block: a single block's weights (0.245 GB bf16, 0.123 INT8). The preview round
    # sizes itself to the VRAM left (load_preview_checkpoint), so it is not the planning figure.
    train_memory={"bf16": (((0.25, 12.3), (1.0, 12.7)), 0.245), "int8": (((0.25, 8.7), (1.0, 9.1)), 0.12),
                  "nf4": (((0.25, 5.8), (1.0, 6.2)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    ema_default="0.98",
    compiles=True,
    compile_boundary="outside",       # as Qwen 2.1's: the checkpoint wraps the compiled block
    # 28 single blocks compiled: 0.31 -> 0.26 s/step (1.19x) at 0.25 MP bf16, ~10 s warm-up per shape (15 Oct 2026)
    compile_payback_steps={"bf16": 600, "int8": 600},
    int8_attention=True,              # workbench renders: 4.55 -> 4.25 s at 1024 (1.07x), 30.4 dB to the bf16 render
    edit_training=True,               # up to 10 references (pipeline.py MAX_REFERENCES), each its own l index
    reference_strength=True,          # the workbench's reference strength + previous-frame latent chain, as Klein's
    edit_note=("Pairs of an original and its edited version, the same crop and shape - about 40 pairs is a good "
               "start (20 at least). Each photo 1 MP or larger; Fizgig resizes them."),
    train_preview_checkpoint=True,    # training previews on the distilled (16 steps, no CFG), as Klein's Distilled
    workbench_follows_samples=True,   # Base CFG needs the Samples tab's negative (empty = the reference's uncond)

    sampling=(
        SamplingSettings("Reference Base", steps=50, cfg=3.5, sampler="euler", scheduler="simple",
                         options=(("alpha", 6.93),), negative_prompt=True,
                         note="Undistilled: Euler, shifted schedule alpha 6.93, CFG against the empty prompt.",
                         source=f"{_CARD}-base model card; {_REF} src/flux3/pipeline.py MODELS"),
    ),
    preview_checkpoint_sampling=SamplingSettings(
        "Reference distilled", steps=16, cfg=1.0, sampler="euler", scheduler="simple", options=(("alpha", 6.93),),
        note="Flow map: each step jumps to the next time, no CFG.",
        source=f"{_CARD} model card; {_REF} src/flux3/sampling.py denoise_flow_map"),
    preview_steps=50,
    preview_cfg=3.5,
    preview_negative="",              # the reference's unconditional is the empty prompt
    preview_width=1024,
    preview_height=1024,
    workbench=("repair", "profiler", "extract", "royale"),

    notes=(
        ("Text conditioning keeps the padding: prompts are padded to a multiple of 80 tokens and the DiT attends "
         "every token, so the cache stores the padded sequence.", f"{_REF} src/flux3/text_encoder.py"),
        ("Training timestep distribution is not published; the driver samples logit-normal shifted by the "
         "sampler's alpha 6.93 until measured otherwise.", "assumption, Flux3Driver README"),
    ),
)
