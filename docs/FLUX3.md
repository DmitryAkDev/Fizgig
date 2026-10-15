# Flux 3 Image Klein

[← Back to the README](../README.md)

Flux 3 Image Klein is Black Forest Labs' 5.6B model for text-to-image and editing, released 15 October 2026. Fizgig trains it from day one: LoRAs, LoKRs, edit LoRAs, slider LoRAs and full fine-tunes, from an 8 GB card. Training runs on the **Base** model; the **distilled** model (16 steps, no CFG) renders the training previews and the workbench. Every workbench tool works on it: Studio with Turbo Preview, LoRA Royale (epochs, seed and prompt travel, edit chains), the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader.

Flux 3 support is new and marked experimental: the presets start from Krea 2's proven settings and haven't yet been tuned on Flux 3 itself.

## Getting set up

1. Accept the licence on the [model page](https://huggingface.co/black-forest-labs/flux-3-image-klein) (the files are gated) and make a free Hugging Face token.
2. On the **Preferences** tab, press **Download models for me** in the Flux 3 Image Klein section and paste the token. It fetches the Base model, the distilled model, the FLUX.2 autoencoder and the Qwen3-VL-4B text encoder (about 31 GB), plus Krea 2's Qwen3-VL-4B captioner for the Captions tab (about 5 GB) if you don't have it.
3. On the **Training** tab, pick **Flux 3 Image Klein** as the Base Model and load a preset.

From the command line (needs a token):

```bash
python -m fizgig.scripts.fetch_models --family flux3 --include-optional
```

| Model | File | Size |
|---|---|---|
| Base DiT (training) | `flux3_image_klein_bf16.safetensors` | 11.1 GB |
| Distilled DiT (previews, workbench) | `flux3_image_klein_sd_bf16.safetensors` | 11.1 GB |
| Autoencoder | `flux2-vae.safetensors` | 0.3 GB |
| Text encoder | `qwen3vl_4b_bf16.safetensors` | 8.9 GB |

All four come from [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy). Files with `_sd_` in the name are the distilled model; train on the one without.

## Presets

Every LoRA preset trains at 0.25 MP with adamw8bit, EMA 0.98 and the per-image loss watch, saving every epoch:

| Preset | Rank | Learning rate | Epochs |
|---|---|---|---|
| **Flux 3 Ultra Fast** | 8 | Adaptive LR 2e-4 to 4e-4 | 30 |
| **Flux 3 Standard** | 32 | Flat 1e-4 | 64 |
| **Flux 3 Style** | 16 | Adaptive LR 5e-5 to 2e-4 | 64 |
| **Flux 3 Edit** | 8 | Adaptive LR 2e-4 to 4e-4 | 12 |
| **Flux 3 Edit Strong** (trickier edits) | 32 | Flat 1e-4 | 12 |
| **Flux 3 Slider** | 8 | Flat 2e-4, at 0.5 MP | 20 |
| **Flux 3 Fine-tune** | — | 5e-5 | 30 |

Scrub the epochs in LoRA Royale to pick the best one.

## Memory: 8 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 from your free VRAM and sizes Block Swap to match. Measured peaks for a rank-16 LoRA at 1 MP on an RTX 5090:

| Base precision | Peak |
|---|---|
| bf16 | 12.7 GB |
| INT8 | 9.1 GB |
| NF4 | 6.2 GB |

On an 8 GB card the NF4 model trains in about 6 GB, and steps off the GPU while each round of previews renders on the distilled model, so the previews fit too. **Compile Blocks** speeds training up by about 1.2×.

A **full fine-tune** peaked at 12.6 GB at 0.25 MP and 14.0 GB at 1 MP, training the whole model at once on a 32 GB card. Cards with less room train it a part at a time. See [Full fine-tuning](FINETUNE.md).

## Previews

Training previews render on the distilled model: 16 steps, no CFG, quick. Without it they render on the Base model at 50 steps with CFG 3.5 against an empty negative prompt, as BFL's reference does, which is much slower. Studio and LoRA Royale preview on the distilled model and follow the Samples tab's settings.

**Negative prompts** work on the Base model, which samples with real CFG: the Samples tab's negative is used, and an empty one is the empty prompt, as in BFL's reference. The distilled model has no CFG, so it ignores the negative; untick **Use the distilled model for samples** for previews that use it.

## Edit LoRAs

Flux 3 is one model for text-to-image and editing, so a LoRA can learn an edit: a grade, a look, a relight, a retouch. Train it on pairs of the same photo, the original and your edited version.

**What you need:** about 40 pairs (20 at least). Each photo 1 MP or larger; Fizgig resizes them. An original and its edited version must have the same crop and shape.

1. Pick **Flux 3 Edit** (or **Edit Strong** for trickier edits); it ticks **Edit LoRA** under Training Parameters.
2. **Originals folder:** the unedited photos, no captions needed.
3. **Edited folder:** the same photos after your edit, with the same file names. This is the Start tab's folder.
4. **Captions:** type what the edit is, e.g. "Apply my concert grade.", and press **Write captions**.
5. **Test photo for previews (optional):** an original in neither folder; the previews show the edit applied to it.

In LoRA Royale, an edit LoRA's travels can chain frames, each one editing the last, with a reference strength for the photo.

## Slider LoRAs

A slider is a dial between two looks: strength +1 moves toward one, -1 toward the other. Pick **Kind of LoRA: Slider** or load **Flux 3 Slider**, then train it from photo pairs (two folders of the same shots, one per end) or from three prompts. The steps are the same as for Qwen Image 2.1: see [Slider LoRAs](QWEN_IMAGE.md#slider-loras).

## Using your LoRAs in ComfyUI

Load the LoRA with the standard LoRA loader on ComfyUI's Flux 3 Image Klein workflow. Fizgig saves it under the names ComfyUI expects, so no conversion is needed; a ComfyUI version from 15 October 2026 or later is needed for Flux 3 itself.

## Not charted yet

Flux 3's block map (which blocks carry identity, style and detail) isn't measured yet, so **Model Area to Train**, Extract's block presets and Studio's colour-coded categories aren't offered on Flux 3. Every block's slider works in Studio, and Extract's All Blocks and Custom work. Profiler reports on Flux 3 LoRAs are what will chart it.

## Licence

Flux 3 Image Klein is released under the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md): non-commercial use, plus commercial use of outputs by users under US$5M in yearly revenue, subject to its terms. Read it before publishing or selling anything made with it. The Qwen3-VL-4B text encoder is Apache-2.0.

## Command line

Flux 3 trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family flux3`. See [the CLI guide](CLI.md#flux-3-image-klein-training).
