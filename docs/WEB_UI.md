# Fizgig web UI

Fizgig can run as a web page as well as a desktop app: on a pod, on a machine with no screen, or to start and watch training from another device. The page is built from the same family descriptions as the desktop app, so the same families, presets and settings appear in both.

## Starting it

```
python lora_trainer_gui.py --web
```

The console prints the address, for example `Fizgig web UI on http://localhost:8081/`. On its own the web UI listens only on this machine. To reach it from other devices, give it a password (`--password`, or the `FIZGIG_PASSWORD` or `VNC_PASSWORD` environment variable); it then listens on every address and asks for that password.

## The port

The web UI uses port **8081** unless that port is taken. When another program already uses it, Fizgig takes the nearest free port above (8082, 8083, …) and saves it, so every later launch uses that same port. The desktop app checks this when it starts, so its Preferences tab already shows the port the web UI will get.

To change the port:

- **Desktop app:** Preferences → Web UI → Web UI port.
- **Web UI:** Preferences → Web UI → Port. The new port is used the next time the web UI starts.
- **By hand**, when neither is available: open `prefs.json` in the Fizgig folder (next to `lora_trainer_gui.py`) and set `"web_port"`, e.g. `"web_port": "8090"`. Add the line if it is not there, keeping the file valid JSON (a comma between entries). Then start the web UI again.
- **For one launch:** `python lora_trainer_gui.py --web --port 8090`. This uses that port only and does not change the saved one.

On a RunPod pod the web UI always uses 8081, the port the pod's proxy forwards, whatever is saved.

## Model downloads

Preferences lists each model's files. **⬇ Download models for me** at the foot of a model's section downloads every file it needs, plus the Captions tab's Qwen3-VL captioner and the small helper models, and fills in the paths when they land. A progress bar shows each file; Cancel stops it, and downloading again resumes where it left off. Files go into the `models` folder in Fizgig (on a pod, `/workspace/models` on the volume).

Some models are gated: their makers require each user to accept the licence on Hugging Face. Their section shows the licence pages and a token box: accept each licence, create a read token, and paste it in. The token is used for that download only and never saved. On a machine or pod with `HF_TOKEN` set, leave the token box empty.
