# Fizgig web UI

Fizgig runs as a web page as well as a desktop app: on a pod, on a machine with no screen, or to set up and watch training from your phone or another computer. The page is built from the same family descriptions as the desktop app, so the same models, presets and settings appear in both.

## On a RunPod pod

The pod starts the web UI by itself. In the pod's **Connect** menu, open **HTTP port 8081** and log in with the pod's password: the `VNC_PASSWORD` you set in the template, or, if you didn't set one, the password printed in the pod's logs ("generated one for this pod"). It is the same password as the remote desktop on port 6080; the web UI needs no user name.

From there everything works in the browser: download the models (Preferences), upload your training data (Start tab), train, watch the previews and download the results.

## On your own computer

In the desktop app: Preferences → Web UI → **Start web UI**. Tick **Start the web UI whenever Fizgig opens** to have it come up with the app. It stays up while the desktop app is open, and closing the desktop app closes it too. The card has links to it: one for this computer, and one with its network address for your phone or other computers.

To run the web UI on its own, without the desktop app, open a terminal in the Fizgig folder and run:

```
venv\Scripts\python lora_trainer_gui.py --web      (Windows)
venv/bin/python lora_trainer_gui.py --web          (Linux)
```

That copy keeps running until you close its terminal. It needs no screen and no Tkinter, so it runs on a headless server too (see [On a server](#on-a-server)).

## Runs started in the desktop app

While the desktop app is open, a run started there shows on the web UI's Training tab: its progress, log, previews and checkpoints, and the desktop app's queue below it. Pause, Stop and Resume on that page work on it, exactly as the desktop app's own buttons would, so you can start a run at your desk and check on it from your phone. Change the queue itself in the desktop app.

A run started in the web UI shows only in the web UI. One run trains at a time: the web UI won't start a run while the desktop app is training, and the desktop app won't start one while the web UI is. If the desktop app started the web UI, closing the desktop app asks before it stops a run started in the web UI.

## The password

Without a password the web UI only opens on the machine it runs on. With one (8 characters or more; a pod's own VNC password works at any length), it listens on every address and asks for it, so a phone, tablet or another computer on your network can open it.

- **Desktop app:** Preferences → Web UI → Password. Leave it empty to keep the web UI on this machine only. If the web UI is running, it restarts with the new password when you press Enter or leave the box, and anyone logged in logs in again.
- **By hand:** set `"web_password"` in `prefs.json` (see the example below).
- **For one launch:** the `FIZGIG_PASSWORD` environment variable, or `--password` (an environment variable keeps it out of the process list).

When more than one is set, `--password` wins, then `FIZGIG_PASSWORD`, then on a pod its `VNC_PASSWORD`, then the one in Preferences.

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

The training folder on the Start tab is a folder on the machine running Fizgig. When your training data is on the computer you're browsing from, use **Upload a training folder** on the Start tab: drop your folder onto it (or click **Choose a folder…**), and it uploads everything inside, subfolders included, into a new folder with the same name. You can also drop or choose single files, after typing a name for the new folder. When the upload finishes, the new folder becomes the training folder.

It takes pictures, `.mp4` clips, songs (`.wav`, `.mp3`, `.flac`, `.m4a`), caption `.txt` files, ACE-Step's `.json` song details, and `.zip` files, which are unpacked as they arrive. Anything else is left out and listed. Convert other video formats (`.mov`, `.webm`) to `.mp4` first.

Uploads go into the desktop app's Preferences → Input Directories folder, in a folder with the name you typed. On a pod that is `/workspace/datasets`, on the volume. When it isn't set, they go into `training_data` inside the Fizgig folder. Uploading again with the same name adds to that folder. For a big dataset, uploading the files themselves is gentler than one huge zip: each file is its own upload, so a dropped connection only costs that file.

### Other files and folders

Boxes in the training settings that take a file or folder from your own data have an **Upload** button beside **Browse…**: the context LoRA, an edit run's test photo and edited-photos folder, a slider's second-look folder, and a fine-tune's regularisation images. **Upload…** sends one file from the computer you're using into the `uploads` folder inside Fizgig; **Upload a folder…** sends a whole folder into the training data folder, as above. Either way the box is filled in for you.

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
