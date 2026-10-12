# Fizgig web UI

Fizgig can run as a web page as well as a desktop app: on a pod, on a machine with no screen, or to start and watch training from another device. The page is built from the same family descriptions as the desktop app, so the same families, presets and settings appear in both.

## Starting it

In the desktop app: Preferences → Web UI → **Start web UI**. Tick **Start the web UI whenever Fizgig opens** to have it come up with the app. It stays up while the desktop app is open, and closing the desktop app closes it too.

To run it on its own, without the desktop app, open a terminal in the Fizgig folder and run:

```
venv\Scripts\python lora_trainer_gui.py --web      (Windows)
venv/bin/python lora_trainer_gui.py --web          (Linux)
```

That copy keeps running until you close its terminal.

## Runs started in the desktop app

While the desktop app is open, a run started there shows on the web UI's Training tab: its progress, its log, its previews and checkpoints, and the desktop app's queue below it. Pause, Stop and Resume on that page work on it, exactly as the desktop app's own buttons would, so you can start a run at your desk and check on it from your phone. Change the queue itself in the desktop app.

A run started in the web UI shows only in the web UI. One run trains at a time: the web UI won't start a run while the desktop app is training, and the desktop app won't start one while the web UI is. If the desktop app started the web UI, closing the desktop app asks before it stops a run started in the web UI.

The console prints the address, for example `Fizgig web UI on http://localhost:8081/`. The desktop app's Preferences → Web UI card has links to it: one for this machine, and one with this machine's network address for other devices.

## The password

Without a password the web UI only opens on the machine it runs on. With one, it listens on every address and asks for the password, so a phone, tablet or another computer on your network can open it.

- **Desktop app:** Preferences → Web UI → Password. Leave it empty to keep the web UI on this machine only.
- **By hand:** set `"web_password"` in `prefs.json`, e.g. `"web_password": "something long"`.
- **For one launch:** `--password something`, or the `FIZGIG_PASSWORD` environment variable.

The password is saved in `prefs.json` as plain text, so use one you don't use anywhere else. When more than one is set, `--password` wins, then `FIZGIG_PASSWORD`, then the one in Preferences, then `VNC_PASSWORD` (the pod template's login password, so a pod has one with no setup).

## The port

The web UI uses port **8081** unless that port is taken. When another program already uses it, Fizgig takes the nearest free port above (8082, 8083, …) and saves it, so every later launch uses that same port. The desktop app checks this when it starts, so its Preferences tab already shows the port the web UI will get.

To change the port:

- **Desktop app:** Preferences → Web UI → Web UI port.
- **Web UI:** Preferences → Web UI → Port. The new port is used the next time the web UI starts.
- **By hand**, when neither is available: open `prefs.json` in the Fizgig folder (next to `lora_trainer_gui.py`) and set `"web_port"`, e.g. `"web_port": "8090"` (`"web_password"` sits beside it, see above). Add the line if it is not there, keeping the file valid JSON (a comma between entries). Then start the web UI again.
- **For one launch:** `python lora_trainer_gui.py --web --port 8090`. This uses that port only and does not change the saved one.

On a RunPod pod the web UI always uses 8081, the port the pod's proxy forwards, whatever is saved.

## Model downloads

Preferences lists each model's files. **⬇ Download models for me** at the foot of a model's section downloads every file it needs, plus the Captions tab's Qwen3-VL captioner and the small helper models, and fills in the paths when they land. A progress bar shows each file; Cancel stops it, and downloading again resumes where it left off. Files go into the `models` folder in Fizgig (on a pod, `/workspace/models` on the volume).

Some models are gated: their makers require each user to accept the licence on Hugging Face. Their section shows the licence pages and a token box: accept each licence, create a read token, and paste it in. The token is used for that download only and never saved. On a machine or pod with `HF_TOKEN` set, leave the token box empty.

## Uploading training data

The training folder on the Start tab is a folder on the machine running Fizgig. When your training data (pictures, clips or songs, with their caption `.txt` files) is on the computer you're browsing from, use **Upload a training folder** on the Start tab: type a name for the new folder, then drop the files in or choose them. A `.zip` is unpacked as it arrives. When the upload finishes, the new folder becomes the training folder.

Uploads go into the desktop app's Preferences → Input Directories folder, in a folder with the name you typed. On a pod that is `/workspace/datasets`, on the volume. When it isn't set, they go into `training_data` inside the Fizgig folder. Uploading again with the same name adds to that folder. Only pictures, clips, songs, captions and zips are accepted; anything else is left out and listed.
