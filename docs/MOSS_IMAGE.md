# MOSS-Image

[← Back to the README](../README.md)

MOSS-Image is OpenMOSS's 5B single-stream model for text-to-image, released 16 October 2026 under Apache-2.0. It renders text well (English and Chinese), takes prompts of up to 4,096 tokens and JSON layout prompts that place text and objects. Fizgig trains it on the **Base** model: LoRAs, LoKRs, slider LoRAs and full fine-tunes, with every workbench tool (Studio with Turbo Preview, LoRA Royale with seed and prompt travel, the Profiler and Extract). Saved LoRAs load in ComfyUI's standard LoRA loader.

MOSS-Image support is new and not yet in the Base Model picker: it runs from the command line until its GPU checks are done. The presets start from Flux 3's settings and haven't yet been tuned on MOSS-Image.

**No edit LoRAs:** the released model is text-to-image only. Its technical report says it "does not directly condition on reference images or support image editing"; the "layout editing" it shows is regenerating from an edited layout prompt.

## Getting set up

None of the files are gated.

```bash
python -m fizgig.scripts.fetch_models --family moss_image
```

| Model | File | Size |
|---|---|---|
| Base DiT (training) | `moss_image_base_bf16.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 10.2 GB |
| Autoencoder | `flux2-vae_small.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image): FLUX.2's encoder with the small decoder MOSS-Image ships with | 0.25 GB |
| Turbo LoRA (optional) | `lora.safetensors` from [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd): 8-step previews, softer on Base | 1.2 GB |
| SFT DiT (previews) | `moss_image_sft_int8_convrot.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 5.4 GB |
| Text encoder | `qwen3.5_9b_bf16.safetensors` from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5) | 19.3 GB |

Flux 3's or Klein 9B's full FLUX.2 autoencoder also works (the same latents), but the small decoder avoids spots in the pictures. Training samples render on the SFT model, OpenMOSS's everyday checkpoint, by default (a tick on the Samples tab): it shows a LoRA where most people use it, though a Base LoRA carries over only partly. Untick it to preview on the Base model being trained. On cards without room for the text encoder (about 15 GB), it streams from system memory one layer at a time, still exact. Comfy-Org's `qwen3.5_9b_int8_convrot.safetensors` (9.7 GB, the one ComfyUI's MOSS-Image workflow uses) also works as the text encoder. The official diffusers folders (OpenMOSS-Team/MOSS-Image-Base `transformer/` and `text_encoder/`) load too: give the path of any shard.

Train on Base. MOSS-Image-SFT is the fine-tuned model for use.

## Presets

Every LoRA preset trains at 0.5 MP with fused AdamW, EMA 0.98 and the per-image loss watch, saving every epoch:

| Preset | Rank | Learning rate | Epochs |
|---|---|---|---|
| **MOSS-Image Fast** | 16 | Adaptive LR 1e-4 to 2e-4 | 30 |
| **MOSS-Image Standard** | 32 | Flat 1e-4 | 30 |
| **MOSS-Image Style** | 16 | Adaptive LR 5e-5 to 2e-4 | 30 |
| **MOSS-Image Slider** | 8 | Flat 2e-4 | 20 |
| **MOSS-Image Fine-tune** | — | 1e-5 | 30 |

Scrub the epochs in LoRA Royale to pick the best one.

## Captions

MOSS-Image was trained on long, detailed captions (short, medium and long variants of each), and its text encoder reads up to 4,096 tokens, so describe the picture fully, including any text in it exactly as written. Keep the text encoder frozen, as OpenMOSS advises; Fizgig never trains it.

## Previews

Previews run res_multistep for 30 steps at CFG 4 on MOSS-Image's shift-5 schedule, with block 22 left out of the negative branch (OpenMOSS's own recipe is Euler at 50 steps, CFG 3). Training leaves block 22 out of the LoRA, so CFG's contrast there stays the base model's. The Samples tab's default negative is `worst quality, bad anatomy, watermark, jpeg artifact` (OpenMOSS's recommended negative without `nsfw`).

In ComfyUI the same settings are Euler / sgm_uniform after ModelSamplingSD3 shift 5, with SkipLayerGuidanceDiTSimple on layer 22. ComfyUI's schedule isn't snapped to the training grid, so a seed renders slightly differently there.
