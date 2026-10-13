# Fizgig web UI

Fizgig runs as a web page as well as a desktop app: on a pod, on a machine with no screen, or to set up and watch training from your phone or another computer. The page is built from the same family descriptions as the desktop app, so the same models, presets and settings appear in both.

## On a RunPod pod

The pod starts Fizgig by itself. In the pod's **Connect** menu, open **HTTP port 8081** and log in with the pod's password: the `FIZGIG_PASSWORD` you set in the template, or, if you didn't set one, the password printed in the pod's logs (generated on its first start and kept after that). There's no user name. On a pod from the Classic template it's the same page and the same `VNC_PASSWORD` as the remote desktop on port 6080. [The pod guide](../docker/README.md) has both.

From there everything works in the browser: download the models (Preferences), upload your training data (Start tab), train, watch the previews and download the results.

## Pods: a rented GPU from your own Fizgig

The **Pods** tab in Fizgig on your own computer opens your RunPod pods. Add a pod by pasting its ID (the code under its name on RunPod's Pods page) or its address; the address of a Fizgig on another computer on your network works too. Each pod shows whether it's ready, still starting, or stopped. **Open** takes the Fizgig window to the pod (in a browser, it opens a tab): it's the same app, running on the pod's GPU, with a **← This computer** button at the top right to come back. Close the window on a pod and it opens on that pod next time, if the pod is still running.

With your RunPod API key (RunPod > Settings > API Keys, pasted into the tab), your Fizgig pods are listed without adding them, and **Start and open** / **Stop** work from the tab: a stopped pod starts and opens by itself once its Fizgig answers, usually within a minute. The key is saved on your computer only. No pod yet? The tab's **Deploy Fizgig on RunPod** button opens the template.

Logging in on a pod, tick **Keep me logged in on this device** and it remembers you for 30 days, through pod restarts.

## On your own computer

In the desktop app: Preferences → Web UI → **Start web UI**. Tick **Start the web UI whenever Fizgig opens** to have it come up with the app. It stays up while the desktop app is open, and closing the desktop app closes it too. The card has links to it: one for this computer, and one with its network address for your phone or other computers.

To run the web UI on its own, without the desktop app, open a terminal in the Fizgig folder and run:

```
venv\Scripts\python lora_trainer_gui.py --web      (Windows)
venv/bin/python lora_trainer_gui.py --web          (Linux)
```

That copy keeps running until you close its terminal. It needs no screen and no Tkinter, so it runs on a headless server too (see [On a server](#on-a-server)).

## As a desktop app window

**Fizgig App (new look).bat** on Windows, `run_fizgig_app.sh` on Linux (or `python lora_trainer_gui.py --app`) opens the web UI in its own window, with no browser around it. It uses the web UI that is already running if there is one; otherwise it starts one on this computer and stops it when the window closes. With a web password set, this window logs itself in (the password guards the network, not your own screen). The window remembers its size and place. It runs on pywebview: WebView2 on Windows (built into Windows 11), GTK/WebKit or Qt on Linux. A Linux machine without either gets the web UI in its browser instead, with a note on installing one (your distro's WebKitGTK, or `pip install pywebview[qt]` in Fizgig's venv). The classic app (`run_fizgig.bat`) is unchanged; the two share settings, presets, run history and the queue.

## The queue

**Add to queue** (beside Start Training) checks the settings as Start does, then lines the run up: it starts when the runs before it finish, so you can set up several from your phone and leave them. The Queue card lists them; move one earlier or later, or remove it. The queue moves on only after a run finishes: one that fails or is stopped holds the queue (it says why) until you press **Start queue**, and so does restarting the web UI - nothing launches by surprise. A paused run is waited for: **Resume** it and the queue carries on after it finishes. The Resume button is still there after a restart, and a fine-tune resumes from its last checkpoint for the rotations it has left. Two queued runs can't save into the same files: give each its own LoRA name.

The web UI's queue is its own; the desktop app keeps its queue, shown below it read-only (change it in the desktop app). One run trains at a time across both: each queue waits while the other app is training or making captions, and says so.

## Runs started in the desktop app

While the desktop app is open, a run started there shows on the web UI's Training tab: its progress, log, previews and checkpoints, and the desktop app's queue below it. Pause, Stop and Resume on that page work on it, exactly as the desktop app's own buttons would, so you can start a run at your desk and check on it from your phone. Change the queue itself in the desktop app.

A run started in the web UI shows only in the web UI. One run trains at a time: the web UI won't start a run while the desktop app is training, and the desktop app won't start one while the web UI is. If the desktop app started the web UI, closing the desktop app asks before it stops a run started in the web UI.

## The password

Without a password the web UI only opens on the machine it runs on. With one (8 characters or more; a pod's own VNC password works at any length), it listens on every address and asks for it, so a phone, tablet or another computer on your network can open it.

- **Desktop app:** Preferences → Web UI → Password. Leave it empty to keep the web UI on this machine only. If the web UI is running, it restarts with the new password when you press Enter or leave the box, and anyone logged in logs in again.
- **By hand:** set `"web_password"` in `prefs.json` (see the example below).
- **For one launch:** the `FIZGIG_PASSWORD` environment variable, or `--password` (an environment variable keeps it out of the process list).

When more than one is set, `--password` wins, then `FIZGIG_PASSWORD`, then on a pod its `VNC_PASSWORD` (the Classic template's), then the one in Preferences.

**Whoever has the password can use Fizgig and open files on that machine**, so treat it like a remote-desktop password: use one you don't use anywhere else (it is saved in `prefs.json` as plain text). The web UI speaks plain HTTP, which is fine on your home network but not across the internet: to reach it from outside, use a VPN such as Tailscale, or a pod (its proxy adds HTTPS), rather than forwarding the port on your router.

## The port

The web UI uses port **8081** unless that port is taken. When another program already uses it, Fizgig takes the nearest free port above (8082, 8083, …) and saves it, so every later launch uses that same port. The desktop app checks this when it starts, so its Preferences tab already shows the port the web UI will get.

To change the port:

- **Desktop app:** Preferences → Web UI → Web UI port. A running web UI restarts on the new port when you press Enter or leave the box.
- **Web UI:** Preferences → Web UI → Port. The new port is used the next time the web UI starts.
- **By hand:** set `"web_port"` in `prefs.json` (see the example below), then start the web UI again.
- **For one launch:** `--port 8090`. This uses that port only, stops with a message if it is taken, and does not change the saved one.

On a RunPod pod the web UI always uses 8081, the port the pod's proxy forwards.

## Model downloads

Preferences lists each model's files. **⬇ Download models for me** at the foot of a model's section downloads every file it needs, plus the Captions tab's Qwen3-VL captioner and the small helper models, and fills in the paths when they land. A progress bar shows each file; Cancel stops it, and downloading again resumes where it left off. Files go into the `models` folder in Fizgig (on a pod, `/workspace/models` on the volume).

Some models are gated: their makers require each user to accept the licence on Hugging Face. Their section shows the licence pages and a token box: accept each licence, create a read token, and paste it in. The token is used for that download only and never saved. On a machine or pod with `HF_TOKEN` set, leave the token box empty.

## Uploading training data

The training folder on the Start tab is a folder on the machine running Fizgig. When your training data is on the computer you're browsing from, use **Upload a training folder** on the Start tab: drop your folder onto it (or click **Upload a folder…**), and it uploads everything inside, subfolders included, into a new folder with the same name. You can also drop or choose single files, after typing a name for the new folder. When the upload finishes, the new folder becomes the training folder.

It takes pictures, `.mp4` clips, songs (`.wav`, `.mp3`, `.flac`, `.m4a`), caption `.txt` files, ACE-Step's `.json` song details, and `.zip` files, which are unpacked as they arrive. Anything else is left out and listed. Convert other video formats (`.mov`, `.webm`) to `.mp4` first.

Uploads go into the desktop app's Preferences → Input Directories folder, in a folder with the name you typed. On a pod that is `/workspace/datasets`, on the volume. When it isn't set, they go into `training_data` inside the Fizgig folder. Uploading again with the same name adds to that folder. Files go up in 8 MB pieces: a dropped connection carries on where it stopped, and uploading the same folder again skips the files already there (same name and size), so only what's missing is sent. The progress line shows the file and the whole batch. For a big dataset, uploading the files themselves is still gentler than one huge zip, which has to arrive in full before it is unpacked.

### Other files and folders

Every file and folder box has **Choose…**, which picks from what is already on the machine running Fizgig: a LoRA you trained earlier, data you uploaded before or copied over another way. Its shortcuts jump to the training data folder, your trained LoRAs, uploaded files and the Fizgig folder.

Boxes that take a file or folder from your own data also have **Upload** beside it: the context LoRA, an edit run's test photo and edited-photos folder, a slider's second-look folder, and a fine-tune's regularisation images. **Upload…** sends one file from the device you're using into the `uploads` folder inside Fizgig; **Upload a folder…** sends a whole folder into the training data folder, as above. Either way the box is filled in for you. On the computer running Fizgig itself there's no need to upload: Choose… reaches the same files.

## MiniMax H3 RefMod

RefMod is in the web UI's model list, right after MiniMax H3, with the desktop app's RefMod card: Grid, References (with the token readout), Steps, Target MP, Clips, Description, Concept, the clip token cap and Audio support, its five presets, and H3's Training Base (Reference when its DiT is set). It runs on MiniMax H3's files and builds exactly the run the desktop app builds. A RefMod makes no previews and can't pause: it is made in one go. Judge a mod in the desktop app's RefMod Studio.

## Image Prep

The desktop app's Image Prep tab, in the browser: resize and convert to PNG, face close-ups, the What will happen summary, Test face detection on one photo, and the Look Consistency Filter (pick 3 baselines from the folder's pictures, scan, Auto-Suggest, mark, move the marked ones to `excluded_by_look`). It all runs on the Fizgig machine, on its CPU, with the same code the desktop app runs. Face tools need insightface there (`install_fizgig.py` adds it). Gizmo, for cutting video clips and voice segments, opens in its own window, so it stays in the desktop app.

## Captions

The Captions tab has the desktop app's captioning tools, run on the machine running Fizgig:

- **Caption All (AI)** with Qwen3-VL (when the Krea 2 text encoder is set in Preferences) or Florence-2, with the same tasks, trigger word, length and Overwrite choice as the desktop. **Edit instructions…** changes what Qwen3-VL is told for a task; the edit is shared with the desktop app and the trainer's auto-recaption. **Static Caption All** gives every file just the trigger word.
- The model stays loaded between batches, so **Re-caption** on a single file is quick; **Unload Model** frees its memory, and it unloads itself before any run starts (here or in the desktop app).
- **Caption every song** (ACE-Step), **Bilingual** (adds a Chinese translation to each caption) and **Find & Replace** with a preview.
- The captions below can be searched and are shown 40 at a time; edit any of them and press Save.

With a run training, captioning only starts if the card has room for it beside the run.

## Runs & outputs

**Runs** lists every training run started in the web UI or the desktop app, newest first: how it ended (finished, failed, stopped, paused), the model and training folder, when it ran and for how long, and its last figures. **Previews & checkpoints** opens that run's results; **Start again with these settings** loads its model, training folder and settings into the form (nothing starts until you press Start). The history lives in `.fizgig_runs.json` in the Fizgig folder.

Below that, **Saved files** lists every LoRA and fine-tune saved in Fizgig's output folders (the ones set per model in the web UI and the desktop app, plus `output_loras`), grouped by run, newest first. Each run shows its final file with the earlier checkpoints folded under it, a **Download** button on each, and **Download the previews** as one zip. Search by name or folder, or show only what was saved in the last 24 hours, 7 or 30 days, or between two dates. Only Fizgig's own outputs show at first; untick **Only Fizgig's outputs** to see the other LoRAs kept in the same folders too.

## Choosing the GPU

On a machine with more than one card, the web UI's Preferences has a **GPU** card: the card runs started from the web UI train on, and the one its memory meter shows. It is the same setting as the desktop app's GPU choice (`cuda_device` in `prefs.json`). A `CUDA_VISIBLE_DEVICES` set where Fizgig starts wins over it.

## On a server

On a headless machine, run the web UI as a service so it survives logging out. A systemd example (adjust the paths and user):

```ini
# /etc/systemd/system/fizgig-web.service
[Unit]
Description=Fizgig web UI
After=network-online.target

[Service]
User=fizgig
WorkingDirectory=/opt/Fizgig
EnvironmentFile=/opt/Fizgig/fizgig-web.env
ExecStart=/opt/Fizgig/venv/bin/python lora_trainer_gui.py --web --port 8090
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

with `/opt/Fizgig/fizgig-web.env` holding `FIZGIG_PASSWORD=your-password` (readable only by that user; it wins over a `web_password` in `prefs.json`, so a service needs only this one). Then `sudo systemctl enable --now fizgig-web`. A fixed `--port` suits a service: it never moves to another port, and the web UI's Preferences show it as set by `--port`.

**Stopping the service stops any run it started, without saving**: everything since the run's last saved epoch is lost. Before `systemctl stop`, a restart or a reboot, press **Pause** in the web UI and wait for the run to save and stop; **Resume** carries on from there.

## Behind nginx or Caddy

A reverse proxy in front adds HTTPS and a proper domain name (`https://fizgig.example.com`). Two things matter:

- **Set a password.** Without one the web UI only answers requests addressed to `localhost`, so a proxy's request for your domain is refused ("bad host"). With one it answers any address and asks for the password. Keep Fizgig itself off the network with `--host 127.0.0.1`: it then listens only for the proxy on the same machine.
- **Don't buffer the log.** The live log is a stream (server-sent events), so tell the proxy not to hold it back.

nginx:

```nginx
server {
    listen 443 ssl;
    server_name fizgig.example.com;
    # ssl_certificate / ssl_certificate_key as usual
    client_max_body_size 64m;              # upload pieces are 8 MB
    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_set_header Host $host;
        proxy_buffering off;               # the live log
        proxy_read_timeout 1h;
    }
}
```

Caddy (it fetches the certificate itself):

```
fizgig.example.com {
    reverse_proxy 127.0.0.1:8090 {
        flush_interval -1
    }
}
```

with Fizgig started as `lora_trainer_gui.py --web --host 127.0.0.1 --port 8090` and its password in `FIZGIG_PASSWORD`. Behind your own proxy every visitor reaches Fizgig from the proxy's address, so the wrong-password lockout counts them together (on a RunPod pod it tells visitors apart).

## Logging out

With a password set, Preferences → Web UI has **Log out** (this device) and **Log out everywhere** (every device's login ends, this one too: a phone left logged in, say). Each device then asks for the password again. If someone else knows the password, change it too, or they can simply log in again. Restarting the web UI also logs everyone out.

## prefs.json

All of these settings live in `prefs.json` in the Fizgig folder (next to `lora_trainer_gui.py`). The desktop app writes it; on a machine without it, create it yourself. A minimal example:

```json
{
  "web_port": "8090",
  "web_password": "a long password",
  "cuda_device": "1",
  "input_dataset_dir": "/data/fizgig/datasets"
}
```

- `web_port`: the web UI's port.
- `web_password`: its password (8 characters or more); leave it out to keep the web UI on this machine only.
- `cuda_device`: the GPU, by its number in `nvidia-smi` (`"0"`, `"1"`, …) or its UUID (`"GPU-…"`, what the GPU pickers save, which never changes meaning); leave it out for the machine's default.
- `input_dataset_dir`: where uploads go.

Model paths are added by the downloads in Preferences, so you don't need to type them.
