"""Fizgig as a desktop app window around the web UI (`lora_trainer_gui.py --app`), through pywebview: Windows'
WebView2, GTK/WebKit or Qt on Linux, WebKit on macOS - no browser, no address bar.

It uses the web UI already running on the saved port when there is one (the desktop app's, say), else starts its own
on this computer and stops it when the window closes. Its own server logs the window in with a one-time key, so a web
password guards the network, not this window. Size and place are remembered (prefs.json "app_window"). Without a
pywebview engine on the machine it opens the web UI in the default browser instead and says how to get the window.
"""
import json
import os
import secrets
import subprocess
import sys
import time
import webbrowser

from fizgig.web.port import free_from, is_fizgig, listening, saved_port

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PREFS_FILE = os.path.join(BASE_DIR, "prefs.json")


def _prefs():
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _remember(geom):
    """The window's size and place, written into prefs.json beside what is there (the desktop app may have it open)."""
    p = _prefs()
    p["app_window"] = geom
    tmp = PREFS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PREFS_FILE)
    except OSError:
        pass


def _icon(profile):
    """Fizgig's icon for the window: icon.png, made an .ico for Windows (which takes nothing else)."""
    png = os.path.join(BASE_DIR, "icon.png")
    if not os.path.isfile(png):
        return None
    if os.name != "nt":
        return png
    ico = os.path.join(profile, "fizgig.ico")
    try:
        if not os.path.isfile(ico) or os.path.getmtime(ico) < os.path.getmtime(png):
            from PIL import Image
            os.makedirs(profile, exist_ok=True)
            Image.open(png).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        return ico
    except Exception:
        return None


def _start_server(port, key):
    """The web UI on this computer only (as `--web` does with no password), the window's key in its environment."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", FIZGIG_APP_KEY=key,
               FIZGIG_APP_PARENT=str(os.getpid()))            # the server stops if this window's process goes
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
          if os.name == "nt" else {"start_new_session": True})
    return subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "lora_trainer_gui.py"), "--web", "--port",
                             str(port)], cwd=BASE_DIR, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kw)


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    port = saved_port(_prefs())
    own, key = None, ""
    if not is_fizgig(port):
        if listening(port):                      # another program has it: the next free one
            port = free_from(port + 1) or port
        key = secrets.token_urlsafe(24)
        own = _start_server(port, key)
        for _ in range(240):                     # the server's first start imports a lot: up to a minute
            if is_fizgig(port) or own.poll() is not None:
                break
            time.sleep(0.25)
        if not is_fizgig(port):
            _stop(own)
            sys.exit("Fizgig's web UI did not start. Run `python lora_trainer_gui.py --web` to see why.")
    url = f"http://127.0.0.1:{port}/" + (f"?app_key={key}" if key else "")
    try:
        import webview
        geom = _prefs().get("app_window") or {}
        window = webview.create_window(
            "Fizgig", url, width=int(geom.get("width") or 1440), height=int(geom.get("height") or 920),
            x=geom.get("x"), y=geom.get("y"), min_size=(380, 600), text_select=True, confirm_close=False)

        def keep():
            try:
                _remember({"width": window.width, "height": window.height, "x": window.x, "y": window.y})
            except Exception:
                pass
        window.events.closing += keep
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True       # Runs & outputs' Download saves where you choose
            webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True   # Buy me a coffee, RunPod, the docs
        except Exception:
            pass
        profile = os.path.join(BASE_DIR, ".fizgig_app_profile")
        webview.start(private_mode=False, storage_path=profile, icon=_icon(profile))
    except Exception as e:
        # no engine here (a Linux machine without GTK/WebKit or Qt): the browser, and how to get the window
        print(f"[app] no app window ({type(e).__name__}: {e}) - opening the web UI in your browser instead. For the "
              "window on Linux: install your distro's WebKitGTK (e.g. gir1.2-webkit2-4.1 and python3-gi) or "
              "`pip install pywebview[qt]` in Fizgig's venv.")
        webbrowser.open(url)
        if own is not None:
            try:
                own.wait()                               # keep the server up until it is stopped (Ctrl+C)
            except KeyboardInterrupt:
                pass
    finally:
        _stop(own)


if __name__ == "__main__":
    main()
