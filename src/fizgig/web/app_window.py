"""Fizgig as a desktop app window around the web UI (`lora_trainer_gui.py --app`), through pywebview: Windows'
WebView2, GTK/WebKit or Qt on Linux, WebKit on macOS - no browser, no address bar.

It uses the web UI already running on the saved port when there is one (the desktop app's, say), else starts its own
on this computer and stops it when the window closes. Its own server logs the window in with a one-time key, so a web
password guards the network, not this window. Size and place are remembered (prefs.json "app_window"). Without a
pywebview engine on the machine it opens the web UI in the default browser instead and says how to get the window.

The window can also show a pod (the Pods tab sends it there): on any page but this computer's it adds a bar with
"Switch to this computer", whatever version of Fizgig the pod runs, and names the pod in the title; back here, a bar
offers "Switch to the pod" (✕ hides it). A pod's page can ask the window to come back here, and read this computer's
training folders to send one up (local_bridge: read-only, those folders only). Closed on a pod, it opens on that pod
next time if it's up.
"""
import json
import os
import secrets
import subprocess
import sys
import threading
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


def _remember_last(url):
    p = _prefs()
    p["app_last_url"] = url
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


# the bar the window adds to a page (so a pod on older Fizgig gets it too): on a pod, where you are and the way back
# here; back here, the pod you left and the way back to it (with a ✕ to put it away)
_BAR_JS = r"""
(function (text, button, title, call, dismiss) {
  var old = document.getElementById('fizgigAppBar');
  if (old) old.remove();
  var b = document.createElement('div');
  b.id = 'fizgigAppBar';
  // the top middle: clear of the page's own top-right buttons (a full-screen picture's close)
  b.style.cssText = 'position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:2147483647;display:flex;' +
    'align-items:center;gap:10px;' +
    'max-width:60vw;padding:5px 6px 5px 14px;border-radius:999px;background:#5B21B6;color:#F0F4F8;' +
    'font:13px Segoe UI,system-ui,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,.45)';
  var t = document.createElement('span');
  t.textContent = text;
  t.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
  var h = document.createElement('button');
  h.textContent = button;
  h.title = title;
  h.style.cssText = 'background:#F0F4F8;color:#1A2029;border:0;border-radius:999px;padding:4px 12px;cursor:pointer;' +
    'font:600 13px Segoe UI,system-ui,sans-serif';
  h.onclick = function () { if (window.pywebview && window.pywebview.api) window.pywebview.api[call](); };
  b.appendChild(t); b.appendChild(h);
  if (dismiss) {
    var x = document.createElement('button');
    x.textContent = '✕';
    x.title = 'Hide this';
    x.style.cssText = 'background:none;color:#F0F4F8;border:0;cursor:pointer;font:13px Segoe UI,system-ui,sans-serif;padding:2px 6px';
    x.onclick = function () { b.remove(); if (window.pywebview && window.pywebview.api) window.pywebview.api.forget_pod(); };
    b.appendChild(x);
  }
  document.body.appendChild(b);
})(__ARGS__);
"""


def _bar(text, button, title, call, dismiss=False):
    return _BAR_JS.replace("__ARGS__", ", ".join(json.dumps(v) for v in (text, button, title, call, dismiss)))


def _where(url):
    """'pod <id>' for a pod's address, else its host:port."""
    from urllib.parse import urlparse
    u = urlparse(url or "")
    host = u.hostname or ""
    if host.endswith(".proxy.runpod.net"):
        return "pod " + host.split("-")[0]
    return u.netloc or url


def _pod_name(url):
    """The name the Pods tab saved it under, else where it is."""
    for p in _prefs().get("app_pods") or []:
        if isinstance(p, dict) and p.get("url") == url and p.get("name"):
            return f"{p['name']} ({_where(url)})"
    return _where(url)


def _is_local(url, port):
    from urllib.parse import urlparse
    u = urlparse(url or "")
    return u.hostname in ("127.0.0.1", "localhost") and (u.port or 80) == port


def _last_pod(port):
    """The pod the window was closed on, if Fizgig on it answers now (a stopped pod: this computer instead)."""
    url = str(_prefs().get("app_last_url") or "")
    if not url or _is_local(url, port):
        return ""
    try:
        from fizgig.web.pods import probe
        return url if probe(url, timeout=5) == "ready" else ""
    except Exception:
        return ""


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
    home = f"http://127.0.0.1:{port}/"
    try:
        import webview

        class Api:
            """What a page in the window - a pod's page included - can ask of it: to come back to this computer, and
            to read this computer's training folders (read-only, those folders only: local_bridge) so a pod's page
            can send one up with its own upload."""
            def home(self):
                # after this call has returned to the page: pywebview hands the page a reply, and the page must
                # still be there for it
                threading.Timer(0.05, lambda: window.load_url(home)).start()

            def to_pod(self):
                if left[0]:
                    threading.Timer(0.05, lambda: window.load_url(left[0])).start()

            def forget_pod(self):
                left[0] = ""

            def local_datasets(self):
                from fizgig.web import local_bridge
                return local_bridge.local_datasets()

            def local_files(self, folder):
                from fizgig.web import local_bridge
                return local_bridge.local_files(folder)

            def read_local(self, folder, name, offset, n):
                from fizgig.web import local_bridge
                return local_bridge.read_local(folder, name, offset, n)

        geom = _prefs().get("app_window") or {}
        first = _last_pod(port) or url
        window = webview.create_window(
            "Fizgig", first, width=int(geom.get("width") or 1440), height=int(geom.get("height") or 920),
            x=geom.get("x"), y=geom.get("y"), min_size=(380, 600), text_select=True, confirm_close=False,
            js_api=Api())
        if first != url:
            # its own server's one-time login, for when it comes back here: the key's cookie first, then the pod
            home = url

        left = [""]                       # the pod this window was last on, for Switch to the pod

        def loaded():
            try:
                cur = window.get_current_url() or ""
                if _is_local(cur, port):
                    window.set_title("Fizgig")
                    if left[0]:
                        window.evaluate_js(_bar(f"☁ {_pod_name(left[0])} is still running", "Switch to the pod",
                                                "Back to the pod's Fizgig", "to_pod", dismiss=True))
                    return
                left[0] = cur.split("#")[0]
                name = _pod_name(cur)
                window.set_title(f"Fizgig — {name}")
                window.evaluate_js(_bar(f"☁ You're on {name}", "Switch to this computer",
                                        "Back to Fizgig on this computer (the pod keeps running)", "home"))
            except Exception:
                pass
        window.events.loaded += loaded

        def keep():
            try:
                _remember({"width": window.width, "height": window.height, "x": window.x, "y": window.y})
            except Exception:
                pass
            try:
                cur = window.get_current_url() or ""
                _remember_last("" if _is_local(cur, port) else cur.split("#")[0])
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
