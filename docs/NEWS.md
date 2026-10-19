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

**[Kromoss](https://huggingface.co/lodestones/Kromoss)**, from lodestones, is a fine-tune of MOSS-Image Base made to fix Base's blurriness, presented as a drop-in replacement for SFT. Fizgig now offers it as your **preview model** in place of SFT: you train on Base as before, and your previews show the LoRA on Kromoss. Its 8-step Turbo LoRA works on Kromoss previews too.

**We recommend training on MOSS-Image Base, not on Kromoss:** in our test, a LoRA trained on Kromoss began to overbake after about 750 steps (epoch 12 on a 62-image dataset), where the same training on Base held up.

**Set it up.** In **Preferences → MOSS-Image**:

1. **Kromoss as the preview model (optional):** pick **Kromoss** beside **Download models for me** and press it. Fizgig downloads Kromoss (int8, 5.4 GB) into the **Preview model (SFT or Kromoss)** row, which used to hold only SFT. The row also has a **Download Kromoss** link.
2. **Turbo LoRA (optional):** **Download** on the Turbo LoRA row (1.2 GB), then point the row at the file. This is lodestones' 8-step Turbo LoRA from Kromoss, distilled from MOSS-Image Base. **It's very good compared with the earlier MOSS-Image speed LoRAs. We suggest 12 steps rather than its card's 8**, and 12 is Fizgig's default for it. It replaces the earlier SFT Turbo LoRA in that row.

**What happens then.**

- **Training is unchanged:** your LoRAs train on Base.
- **Previews render on Kromoss** with your LoRA on top, while **Use the preview model for samples** is ticked on the Samples tab (it is by default). Repair Studio and LoRA Royale render on it too.
- **The Turbo LoRA stays off unless you turn it on.** For fast previews, set **Turbo LoRA strength** to 1 on the Samples tab: Steps move to 12 and CFG to 1. It works with Kromoss as the preview model, and on previews of your Base, but not with SFT: with SFT as the preview model the strength box is greyed out, so set Kromoss as the preview model to use it.

**In ComfyUI:** use Kromoss in place of SFT with your Base-trained LoRA. With its Turbo LoRA at strength 1: 12 steps (its card says 8; 12 looked much better in our tests), CFG 1, euler / simple. Without it, its model card recommends Euler, 50 steps, CFG 3, shift 5.
