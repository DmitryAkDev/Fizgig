"""ACE-Step 1.5 XL through the standard layer: text-to-music (songs with vocals and lyrics, instrumentals), 48 kHz
stereo. A 4B flow DiT (32 layers, 2560 wide) over an Oobleck VAE, Qwen3-Embedding-0.6B text and lyric conditioning.
Driver: acestep/driver.py.

Facts from the official repository (github.com/ace-step/ACE-Step-1.5: the handler, trainer and docs; the XL model
repos' code), Side-Step's guides and ComfyUI's implementation (comfy/ldm/ace/ace_step15.py, text_encoders/ace15.py,
lora.py), Oct 2026.
"""
from fizgig.families.description import FamilyDescription, FamilyOption, LoRAFormat, ModelFile, SamplingSettings

_COMFY = "Comfy-Org/ace_step_1.5_ComfyUI_files"
_REPO = "https://github.com/ace-step/ACE-Step-1.5"
_TURBO_T = (1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3)


def _preset(rank, alpha, lr, epochs, kind="LoRA (standard)"):
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": alpha, "NETWORK_TYPE": kind, "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 50, "SEED": 42, "ADAPTIVE_LR": False,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 4, "MAX_GRAD_NORM": 1.0,
        "BLOCKS_SWAP": "Auto (detect from GPU)", "FAMILY_PRECISION": "Auto (fits your free VRAM)",
        "FAMILY_EMA": "Off", "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": False, "FAMILY_SLIDER": False,
    }


ACESTEP15 = FamilyDescription(
    key="acestep15xl",
    arch_id="acestep15xl",
    display_name="ACE-Step 1.5 XL",
    gui_label="ACE-Step 1.5 XL (music)",
    lora_name_suffix="acestep15xl",
    aliases=("ace-step", "acestep", "ace-step-1.5-xl", "acestep15"),
    experimental=True,

    model_files=(
        ModelFile("acestep15xl_dit", "ACE-Step 1.5 XL Turbo DiT", True, _COMFY,
                  "split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors", 9.97, role="dit",
                  hint="The XL Turbo model (8 steps): what LoRAs train on by default, and what the workbench and "
                       "previews render with."),
        ModelFile("acestep15xl_sft_dit", "ACE-Step 1.5 XL SFT DiT", False, _COMFY,
                  "split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors", 9.97,
                  hint="Optional: the non-distilled XL model, for the SFT Training Base (30-step previews with "
                       "guidance - slower, the most detail)."),
        ModelFile("acestep15xl_vae", "ACE-Step 1.5 VAE", True, _COMFY, "split_files/vae/ace_1.5_vae.safetensors", 0.34,
                  role="vae", hint="The Oobleck audio VAE: 48 kHz stereo to 64 channels at 25 per second."),
        ModelFile("acestep15xl_model_eq", "Model EQ LoRA (Fizgig)", False, "ShootTheSound/Fizgig-ACE-Step-1.5-XL-Model-EQ",
                  "acestep15xl_turbo_model_eq_r128.safetensors", 0.34, role="model_eq", fetch_optional=True,
                  hint="Optional: ACE-Step XL Turbo's own weights as a LoRA. In Repair Studio its block sliders "
                       "become an EQ of the model itself - 1.0 is the stock model - and a save bakes your EQ into "
                       "an ordinary LoRA. Load it as the primary or the donor with the Model EQ buttons."),
        ModelFile("acestep15xl_text_encoder", "Qwen3 0.6B text encoder", True, _COMFY,
                  "split_files/text_encoders/qwen_0.6b_ace15.safetensors", 1.19, role="text_encoder",
                  hint="Encodes the captions and lyrics for caching, then unloads before training."),
    ),
    prefs_title="Model Paths (ACE-Step 1.5 XL)",
    text_encoder_label="Qwen3 0.6B",
    vae_label="ACE-Step 1.5 VAE",

    media=("voice",),
    latent_channels=64,
    spatial_factor=1,
    bucket_step=1,
    native_megapixels=1.0,

    n_blocks=32,                      # config.json num_hidden_layers
    block_prefix="layers",
    block_note="32 DiT layers, sliding-window (even) and full (odd) self-attention, each with cross-attention to "
               "the caption, lyrics and timbre. The LoRA wraps q/k/v/o of both attentions (the official trainer's "
               "targets).",
    workbench=("repair", "extract"),
    # I'm feeling lucky's held bands, from Peter's by-ear block map (10 Oct 2026, fzgfunk): 0-5 structure (groove,
    # arrangement) and 28-31 the tidy-up of detail stay near 1 while Hold is ticked; 6-27 roll the full Min-Max range
    lucky_zones=((0, 5, 0.8, 1.2), (28, 31, 0.9, 1.1)),
    repair_size=30,                   # an audio family's Res is the song length in seconds
    ema_default="Off",                # available, off until an A/B says otherwise (as Klein)
    ema_short_run=True,               # music sets are small: 10 clips x accumulation 4 is ~3 steps an epoch
    ema_hint="Saves and previews the running average of the LoRA instead of its latest step. A song set gives few "
             "steps an epoch (10 clips at accumulation 4 is about 3), so 0.98 averages over most of a 20-epoch run "
             "and lags it; Short run sizes the window to a quarter of the run.",

    lora=LoRAFormat(
        key_template="base_model.model.layers.{block}.{module}.{ab}.weight",
        down="lora_A", up="lora_B",
        block_modules=tuple(f"{a}.{p}" for a in ("self_attn", "cross_attn")
                            for p in ("q_proj", "k_proj", "v_proj", "o_proj")),
        alpha_key="{prefix}.alpha",
        file_prefix="base_model.model.",
        note="PEFT keys on the decoder (the official trainer's LoRA files), plus .alpha; ComfyUI maps "
             "base_model.model.<path> to diffusion_model.decoder.<path>. LoKR loads under the same names.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet (ACEStep15)",
    ),

    driver="fizgig.acestep.driver:AceStepDriver",
    modelspec_arch="ACE-Step-1.5-XL",
    implementation=_REPO,
    precisions=("bf16",),
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    adaptive_lr=False,
    loss_watch=False,
    helper_files=(("ACE-Step/Ace-Step1.5", ("Qwen3-Embedding-0.6B/*.json", "Qwen3-Embedding-0.6B/*.txt",
                                            "Qwen3-Embedding-0.6B/*.jinja", "acestep-v15-turbo/silence_latent.pt")),),

    sampling=(
        SamplingSettings("ACE-Step 1.5 XL Turbo", steps=8, cfg=1.0, sigmas=_TURBO_T + (0.0,),
                         note="The official Turbo schedule (shift 3, 8 steps), no CFG.",
                         source=f"{_REPO} acestep/models/turbo SHIFT_TIMESTEPS"),
    ),
    samples_cfg_free=True,
    preview_steps=8,
    preview_cfg=1.0,
    samples_text=(
        ("banner", "Each sample prompt is a song: \"caption | lyrics\", lyric lines separated by \" / \" (e.g. "
                   "\"[verse] / first line / second line\"). Leave out the lyrics for an instrumental. Songs train "
                   "for hundreds of epochs, so a preview every 25-50 epochs is plenty."),
    ),
    options=(
        FamilyOption("ACESTEP_TRAIN_BASE", "Training Base", tab="model", choices=(
            ("XL Turbo (8 steps) — standard", ""),
            ("XL SFT (30 steps + guidance)", "--dit=pref:acestep15xl_sft_dit base=sft")),
            hint="Turbo is what most people generate with. SFT is the non-distilled model: its previews take 30 "
                 "guided steps (slower) and it needs the SFT DiT in Preferences."),
        FamilyOption("ACESTEP_PREVIEW_SECONDS", "Sample length", tab="samples", choices=(
            ("30 seconds", "preview_seconds=30"), ("60 seconds", "preview_seconds=60"),
            ("10 seconds", "preview_seconds=10"), ("120 seconds", "preview_seconds=120")),
            hint="How long each preview song is."),
    ),

    presets=(
        # the official LoRA defaults: rank 64 / alpha 128, 1e-4, batch 1 x accumulation 4; the tutorial's epochs:
        # 10-20 songs -> 800, ~100 songs -> 500 (LoRA_Training_Tutorial.md). Not yet measured in Fizgig.
        ("✨ ACE-Step Official LoRA (rank 64, 1e-4)", _preset(64, 128, 1e-4, 800)),
    ),

    notes=(
        ("Licence: MIT (code and weights); generated music may be used commercially.", _REPO),
        ("A training song is an audio file with a caption .txt; its lyrics go in <name>.lyrics.txt (none = "
         "instrumental) and bpm / timesignature / keyscale / language in an optional <name>.json. Songs over 240 s "
         "are cut to the first 240 s (the official trainer's max_duration).", _REPO),
        ("Conditioning follows the official handler's text-to-music: the 'Fill the audio semantic mask' prompt with "
         "the metadata block, lyrics as '# Languages / # Lyric' token embeddings, the silence latent as context and "
         "as the 30 s timbre reference. The official trainer feeds lyrics raw and a zero timbre reference; Fizgig "
         "trains the way the model is prompted.", f"{_REPO} acestep/core/generation/handler"),
        ("XL's timbre encoder reads a learned CLS token placed before the reference frames (the XL code); ComfyUI "
         "reads the first frame without it, which moves the velocity ~2% with the silence reference (measured). "
         "Fizgig follows the official XL code.", "ACE-Step/acestep-v15-xl-turbo modeling_acestep_v15_xl_turbo.py"),
        ("Turbo trains on its own 8 timesteps (shift 3) with no CFG dropout, as the official trainer does. SFT trains "
         "with continuous logit-normal timesteps (mu -0.4, sigma 1) and 15% null-condition dropout (Side-Step's "
         "corrected mode) and previews with 30 steps, shift 1, APG guidance 7 (the XL SFT code's defaults).",
         "Side-Step docs: Model Management.md; ACE-Step/acestep-v15-xl-sft"),
    ),
)
