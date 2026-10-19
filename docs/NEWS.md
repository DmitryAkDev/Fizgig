# Fizgig news

## 19 Oct 2026: MOSS-Image training, fixed

**MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to epoch.** There's nothing to set: update Fizgig and it's the default.

**What was wrong.** A diffusion model trains at every noise level, from nearly clean pictures to pure noise. The clean end is where faces and fine detail are learned; the noisy end is layout and composition. MOSS-Image's own training schedule puts only about 11% of steps at the clean end. That's why MOSS LoRAs were slow to pick up likeness, and why the picture jumped about between epochs.

**What Fizgig does now.** A new Training-tab setting, **Clean-end share**, puts 30% of training steps at the clean end, close to where Krea 2 and Qwen Image sit, the models that pick up likeness fastest in my own training. It changes training only; how MOSS-Image generates pictures is unchanged.

ArcFace likeness against 3 reference photos, same dataset:

| Epoch | 3 | 6 | 9 | 12 | 15 | 20 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Before (the model's own schedule) | 40% | 44% | 52% | 58% | 63% | 60% |
| Now (Clean-end 30%) | 42% | 59% | 63% | 66% | 69% | 69% |

The old settings peaked at 67% after 30 epochs. On the Standard preset (rank 32, 1 MP) a second dataset reached 72% by epoch 32, above the 71% that the dataset's own photos score against the same reference photos.

**By eye the difference is bigger still: the pictures are far sharper and more accurate, with no body horror, and the composition holds steady instead of hopping between epochs.**

**Fine-tunes too.** A MOSS-Image fine-tune uses the same Clean-end share, and it fits a 16 GB card, the whole model at once: a full Base fine-tune peaked at 12.6 GB at 1 MP.

30 is still a figure to explore; share what you find in the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1x9khtm/a_potential_fix_for_moss_training_issues/). Empty the box for the model's own schedule.

## 19 Oct 2026: Kromoss and its 8-step Turbo LoRA

**[Kromoss](https://huggingface.co/lodestones/Kromoss)**, from lodestones, is a full fine-tune of MOSS-Image Base made to fix Base's blurriness. Fizgig now offers it as your MOSS Base, with its own 8-step Turbo LoRA for fast previews.

**Get it.** In **Preferences → MOSS-Image**:

1. **Kromoss:** pick **Kromoss** beside **Download models for me** and press it. Fizgig downloads Kromoss (10.2 GB) and sets it as your Base DiT. (Or use **Download Kromoss** on the Base DiT row and point the row at the file.)
2. **Its Turbo LoRA (optional):** **Download** on the Turbo LoRA row (1.2 GB), then point the row at the file. It's distilled from MOSS-Image Base, so it speeds up previews on Base as well as on Kromoss. **It's very good compared with the earlier MOSS-Image speed LoRAs. We suggest 12 steps rather than its card's 8**, and 12 is Fizgig's default for it.

**What happens then.**

- **Training is unchanged:** LoRAs and fine-tunes train on Kromoss exactly as on Base, with the same settings and memory.
- **Previews render on Kromoss itself**, the model you'll use the LoRA on, instead of on SFT. The Samples tab's **Use the SFT model for samples** tick turns off by itself when you set Kromoss; tick it back on if you want SFT previews.
- **Fast previews:** with the Turbo LoRA set, put **Turbo LoRA strength** to 1 on the Samples tab. Steps move to 12 and CFG to 1.
- Switch back to MOSS-Image Base and SFT previews come back on.

**In ComfyUI:** load Kromoss in place of MOSS-Image Base. With its Turbo LoRA at strength 1: CFG 1, euler / simple, and 12 steps (its card says 8; 12 looked much better in our tests). Without it, its model card recommends Euler, 50 steps, CFG 3, shift 5.
