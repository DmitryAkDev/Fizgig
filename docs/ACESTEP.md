# ACE-Step 1.5 XL (music)

Train a LoRA on your own music and ACE-Step writes new songs in that sound: instrumentals or songs with lyrics, 48 kHz stereo. Fizgig trains on the XL Turbo model by default (8-step previews) or the XL SFT model (30 guided steps, slower, the most detail), and LoRAs load in ComfyUI as they are.

## Your songs

Put the songs in one folder, each with a caption file beside it:

| File | What it holds |
|---|---|
| `song.wav` (or .mp3, .flac, .ogg) | The audio. Songs over 4 minutes are cut to the first 4. |
| `song.txt` | The caption: a trigger word, then the style, e.g. `fzgchip, chiptune, 8-bit, instrumental, square wave lead, arpeggios` |
| `song.lyrics.txt` (optional) | The lyrics. Leave it out for an instrumental. |
| `song.json` (optional) | `bpm`, `timesignature`, `keyscale`, `language`. |

Pick songs that share one sound: one artist, one album or one tight genre. The model learns what the songs have in common.

## A tested recipe

**10 songs of 2 to 3.5 minutes each, one album, all instrumental, the same caption on every song.** Train with the **✨ ACE-Step Official LoRA (rank 64, 800 epochs)** preset: rank 64, alpha 128, learning rate 1e-4, a checkpoint and a preview every 40 epochs.

The dataset we tested it on, and the LoRA it made, are free to download: the [Fizgig ACE-Step Chiptune Starter](https://huggingface.co/datasets/ShootTheSound/Fizgig-ACE-Step-Chiptune-Starter) (Eric Skiff's *Resistor Anthems*, CC BY 3.0). Train it for your first run, or load its LoRA at strength 0.5 to hear where you're heading.

What to expect:

- **The first epochs sound worse, not better.** Ten songs give only a few training steps an epoch, so for the first 30 or so epochs the previews are quieter and hissier while the model moves away from its own sound. The new sound arrives after that.
- **The late epochs hold the detail.** The timbre, the playing style and the production keep sharpening for hundreds of epochs.
- **Play the final LoRA at about half strength.** After hundreds of epochs a LoRA at strength 1 starts copying the training songs: the same song whatever the seed. At around 0.5 you keep the learned detail and get new music. In our test, epoch 740 at strength 0.5 beat every earlier epoch at strength 1.

The Training tab shows this as a hint once Max Epochs is 400 or more.

## Previews and the workbench

Sample prompts are songs: `caption | lyrics`, with lyric lines separated by ` / `, e.g.

```
fzgchip, chiptune, 8-bit, upbeat, robotic vocals | [verse] / Jump jump, coins in the air / [chorus] / Power up, press play
```

Leave out the lyrics for an instrumental. The Samples tab sets the preview length (60 seconds by default).

Repair Studio plays songs in place of pictures: a baseline, the tweaked version and the song with no LoRA, side by side, keeping your place when you switch. Its built-in mixes (**✨Demo Slider Adjustment**, **✨Early Epoch Improver**, **✨Make it more full**) are good starting points. Extract reduces a LoRA's rank as it does for every model.

## Sliders

A slider is a LoRA whose strength is a dial between two kinds of music: +1 one way, -1 the other, 0 the song as it was. Pick **Slider** as the kind of training (or the **✨ ACE-Step Slider (rank 8, 20 epochs)** preset) and fill in three lines:

1. **What the music is**, e.g. `instrumental, synth pop,`
2. **The +1 end adds**, e.g. `energetic, fast tempo, loud drums`
3. **The -1 end adds**, e.g. `calm, slow tempo, soft`

No songs are needed: Fizgig teaches the dial from the model's own songs. Each checkpoint's preview plays -1, 0 and +1 one after another. Change one thing per slider: words that also change the genre or the instruments turn the dial into a genre switch.

## Full fine-tuning

The **✨ ACE-Step Fine-tune** preset trains the whole model (both attentions and the MLP of all 32 layers) a part at a time on a 4-bit base, at 1e-5. Each checkpoint is a complete model file of about 10 GB that ComfyUI loads like the original.
