# Fizgig on Vast.ai

[← Back to the RunPod guide](../docker/README.md)

Vast.ai rents GPUs from a marketplace of hosts, often for less than RunPod. Fizgig's pod image runs there as it does on RunPod: you open Fizgig in your browser, and training runs on the rented card. The app's **Pods** tab deploys RunPod pods for you; on Vast you start the instance on Vast's site and open it by its address.

**[Fizgig's template on Vast.ai](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning)**: new to Vast.ai? Signing up through this link supports Fizgig: Vast.ai pays a small referral share from its own side. It costs you nothing, and you pay the same rates.

## How do I…?

### …start Fizgig on Vast.ai?

1. Open [Fizgig's template](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning) and press **Rent** on a card with 24 GB or more (32 GB trains Krea 2 with no block swap).
2. To choose your own password, add **FIZGIG_PASSWORD** to the template's environment before you rent. Otherwise the instance makes one up at its first start and keeps it: open the instance's log and look for `password:`.
3. Wait until the instance is running (the first start downloads the image, a few minutes).

### …open it?

Press **Open** on the instance, or open its **IP & port info**: Fizgig is the port mapped to **8081**, at `http://<its IP>:<that port>`. Log in with your password.

### …get the models?

In Fizgig, **Preferences** > **Download models for me**. Vast's hosts download fast (over 100 MB/s in testing).

### …get my training pictures there?

Start tab > **Upload a training folder**, or drag them into the file manager: the port mapped to **8080**, user `admin`, the same password.

### …stop paying for it?

Destroy the instance on Vast.ai once your LoRAs are home (Runs & outputs downloads them). A stopped instance still costs a little for its disk; a destroyed one costs nothing, and its files are gone.

---

## Good to know

- **Plain http.** Vast serves an instance's ports as plain http at its public address, not through an HTTPS proxy as RunPod does. Use a password you don't use anywhere else.
- **Disk.** The template asks for 300 GB, as RunPod's does: one model family's files are 20 to 40 GB, and an instance's disk can't grow after it's made. Vast bills the disk until the instance is destroyed, stopped or not. Destroying the instance deletes everything on it.
- **Coming back to the same machine often?** Add a volume at `/workspace` when you rent: it keeps your models and files between instances on that machine. A Vast volume works only on the machine it was made on, and it bills until you delete it yourself, even after the instance is destroyed.
- **CUDA.** The image needs a host whose driver supports CUDA 12.8 or newer; the template's filter asks for that.
- **Stop when done** (Preferences > RunPod) is RunPod's only: on Vast, destroy the instance yourself when you've finished.

Measured on a Vast RTX 4090 (22 Oct 2026): MOSS-Image trained at about 1.75 it/s, the same as a RunPod 4090.
