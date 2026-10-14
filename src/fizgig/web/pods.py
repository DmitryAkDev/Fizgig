"""Your RunPod pods, from the Fizgig on your own computer (the Pods tab): the ones you added by ID or address and,
with a RunPod API key, every Fizgig pod on your account, each with whether its Fizgig is answering yet. Start and
Stop go through RunPod's API. Opening a pod is the page's job: the app window goes there, a browser opens a tab.

Torch-free, like the rest of the server; every call to RunPod or a pod has a short timeout, so a slow pod never
holds the page up."""

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.runpod.io/graphql"
DEPLOY = "https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep"
UA = "Fizgig"                       # RunPod's proxy refuses Python's default User-Agent
POD_ID = re.compile(r"^[a-z0-9]{8,24}$")
PROXY = re.compile(r"^https?://([a-z0-9]+)-\d+\.proxy\.runpod\.net", re.I)
MAX_SAVED = 20

# runpodctl's own fields; FIELDS_MIN when RunPod rejects one of them
FIELDS = "id name desiredStatus imageName costPerHr gpuCount ports machine { gpuDisplayName }"
FIELDS_MIN = "id name desiredStatus imageName"


def pod_url(pod_id: str) -> str:
    return f"https://{pod_id}-8081.proxy.runpod.net/"


def parse(text: str) -> dict:
    """What someone pasted: a pod ID, any of the pod's proxy addresses (another port's is fine), RunPod's own page
    for the pod (…?id=…), or the address of a Fizgig somewhere else (another computer on the network) -> {id, url}."""
    t = (text or "").strip()
    if not t:
        raise ValueError("Paste the pod's ID or its address.")
    if POD_ID.match(t.lower()):
        return {"id": t.lower(), "url": pod_url(t.lower())}
    m = PROXY.match(t)
    if m:
        return {"id": m.group(1).lower(), "url": pod_url(m.group(1).lower())}
    if not re.match(r"^https?://", t, re.I) and re.match(r"^[\w.-]+(:\d+)?(/.*)?$", t):
        t = "http://" + t                                   # 192.168.0.20:8081
    u = urllib.parse.urlparse(t)
    if u.scheme in ("http", "https") and u.netloc:
        q = urllib.parse.parse_qs(u.query)
        pid = (q.get("id") or q.get("podId") or [""])[0].strip().lower()
        if "runpod.io" in u.netloc.lower():                 # the console's page for the pod
            if POD_ID.match(pid):
                return {"id": pid, "url": pod_url(pid)}
            raise ValueError("That's a RunPod page, not a pod. Paste the pod's ID (under its name on the Pods page).")
        return {"id": "", "url": f"{u.scheme}://{u.netloc}/"}
    raise ValueError("That isn't a pod ID or an address. The ID is the code under the pod's name on RunPod's Pods page.")


def saved(prefs) -> list:
    out = []
    for p in prefs.get("app_pods") or []:
        if isinstance(p, dict) and p.get("url"):
            out.append({"id": str(p.get("id") or ""), "url": str(p["url"]), "name": str(p.get("name") or ""),
                        "last": float(p.get("last") or 0), "lan_id": str(p.get("lan_id") or "")})
    return out


def add(prefs, text, name="") -> list:
    p = parse(text)
    rows = [r for r in saved(prefs) if r["url"] != p["url"]]
    rows.insert(0, {**p, "name": (name or "").strip()[:60], "last": time.time()})
    return rows[:MAX_SAVED]


def forget(prefs, url) -> list:
    return [r for r in saved(prefs) if r["url"] != url]


def touch(prefs, url, name="", lan_id="") -> list:
    """Opened just now: first in the list next time (and saved, if it came from the account's list or the
    network)."""
    rows = saved(prefs)
    hit = next((r for r in rows if r["url"] == url), None)
    if hit is None:
        p = parse(url)
        hit = {**p, "name": (name or "").strip()[:60]}
    rows = [r for r in rows if r["url"] != url]
    hit["last"] = time.time()
    if name and not hit.get("name"):
        hit["name"] = name.strip()[:60]
    if lan_id:
        hit["lan_id"] = lan_id[:32]
    return ([hit] + rows)[:MAX_SAVED]


def probe(url: str, timeout=6.0) -> str:
    """'ready' (Fizgig's page answers), 'waiting' (the address answers but Fizgig isn't up yet: still pulling the
    image or starting), or 'off' (nothing answers)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return "ready" if b"fizgig-token" in r.read(8192) else "waiting"
    except urllib.error.HTTPError:
        return "waiting"                                   # RunPod's proxy answers 404 / 502 until the port is up
    except Exception:
        return "off"


def _gql(key, query, variables=None, timeout=20):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "User-Agent": UA,
                                                         "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError("RunPod didn't accept that API key. Make one at RunPod > Settings > API Keys "
                             "(read and write).") from None
        raise ValueError(f"RunPod's API answered {e.code}. Try again in a moment.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ValueError(f"Couldn't reach RunPod ({getattr(e, 'reason', e)}).") from None
    if payload.get("errors"):
        raise ValueError(str(payload["errors"][0].get("message") or "RunPod refused that."))
    return payload.get("data") or {}


def is_fizgig(pod) -> bool:
    ports = str(pod.get("ports") or "")
    return "fizgig" in str(pod.get("imageName") or "").lower() or "8081" in ports


def account(key) -> list:
    """The account's Fizgig pods (made from a Fizgig template: its image, or port 8081 open), running ones first."""
    q = "query { myself { pods { %s } } }"
    try:
        data = _gql(key, q % FIELDS)
    except ValueError as e:
        if "Cannot query field" not in str(e):
            raise
        data = _gql(key, q % FIELDS_MIN)
    pods = ((data.get("myself") or {}).get("pods")) or []
    out = []
    for p in pods:
        if not is_fizgig(p):
            continue
        out.append({"id": p.get("id") or "", "name": p.get("name") or "", "url": pod_url(p.get("id") or ""),
                    "running": str(p.get("desiredStatus") or "").upper() == "RUNNING",
                    "status": str(p.get("desiredStatus") or "").upper(),
                    "gpu": ((p.get("machine") or {}).get("gpuDisplayName")) or "",
                    "gpus": int(p.get("gpuCount") or 1), "cost": float(p.get("costPerHr") or 0)})
    out.sort(key=lambda r: (not r["running"], r["name"].lower()))
    return out


def start(key, pod_id, gpus=1):
    """podResume - RunPod finds the pod's GPU on its own machine again; when that machine's GPUs are all rented, it
    says so and the pod stays stopped."""
    try:
        _gql(key, "mutation($id: String!, $n: Int!) { podResume(input: {podId: $id, gpuCount: $n}) "
                  "{ id desiredStatus } }", {"id": pod_id, "n": int(gpus or 1)})
    except ValueError as e:
        msg = str(e)
        if "not enough free" in msg.lower() or "no longer any instances" in msg.lower():
            raise ValueError("The machine this pod lives on has no free GPU right now. Try again in a while, or "
                             "deploy a new pod (this pod's files stay on its volume).") from None
        raise


def stop(key, pod_id):
    _gql(key, "mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }", {"id": pod_id})


def _lan_rows(rows, found, mine, save):
    """The Fizgigs that answered on the network, into the list: a saved computer whose address changed takes its new
    one (and keeps it), the rest are listed as found, ready to open."""
    moved = False
    for f in found:
        old = next((r for r in rows.values() if f.get("id") and r.get("lan_id") == f["id"]
                    and r["url"].rsplit(":", 1)[-1] == f["url"].rsplit(":", 1)[-1]), None)
        if old and old["url"] != f["url"]:
            rows.pop(old["url"])
            old["url"] = f["url"]
            moved = True
        base = old or rows.get(f["url"]) or {"id": "", "url": f["url"], "name": "", "last": 0, "saved": False,
                                            "account": False}
        gpu = str(f.get("gpu") or "").replace("NVIDIA GeForce ", "") + (f" ({f['vram_gb']} GB)" if f.get("vram_gb") else "")
        rows[f["url"]] = {**base, "name": base.get("name") or f.get("name") or "", "lan": True,
                          "lan_id": f.get("id") or "", "gpu": gpu, "busy": f.get("busy") or "",
                          "version": f.get("version") or "", "build": _build_word(mine, f.get("commit")),
                          "state": "ready"}
    if moved and save:
        save({"app_pods": [{k: r.get(k) for k in ("id", "url", "name", "last", "lan_id")}
                           for r in rows.values() if r.get("saved")]})


def _build_word(mine, theirs):
    """'' when both run the same build; else which one to update (git knows which build came first)."""
    theirs = str(theirs or "")
    if not mine or not theirs or mine == theirs:
        return ""
    import subprocess
    import os
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    try:
        known = subprocess.run(["git", "cat-file", "-e", theirs + "^{commit}"], cwd=base, capture_output=True,
                               timeout=5, **kw).returncode == 0
        older = known and subprocess.run(["git", "merge-base", "--is-ancestor", theirs, mine], cwd=base,
                                         capture_output=True, timeout=5, **kw).returncode == 0
    except Exception:
        return "other"
    return "theirs" if older else "mine"


def listing(prefs, key, save=None, mine=""):
    """Everything the Pods tab shows: saved pods and the account's, merged by address, each probed in parallel, and
    the Fizgigs that answer on this network."""
    from fizgig.web import lan
    finding = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(lan.discover, 1.0)
    rows, err = {}, ""
    for r in saved(prefs):
        rows[r["url"]] = {**r, "saved": True, "account": False}
    if key:
        try:
            for a in account(key):
                hit = rows.get(a["url"])
                rows[a["url"]] = {**(hit or {"last": 0, "saved": False}), **a,
                                  "name": (hit or {}).get("name") or a["name"], "account": True}
        except ValueError as e:
            err = str(e)
    try:
        _lan_rows(rows, finding.result(timeout=5), mine, save)
    except Exception:
        pass
    items = list(rows.values())
    live = [r for r in items if not r.get("account") or r.get("running")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        states = dict(zip([r["url"] for r in live],
                          ex.map(lambda r: probe(r["url"], 3.0 if r.get("lan") else 6.0), live)))
    for r in items:
        st = states.get(r["url"], "stopped" if r.get("account") and not r.get("running") else "off")
        # found on the network but its page doesn't answer: its firewall lets the question in, not the web UI
        r["state"] = ("ready" if st == "ready" else "blocked") if r.get("lan") else st
    rank = {"ready": 0, "waiting": 1, "stopped": 2, "off": 3}
    items.sort(key=lambda r: (rank.get(r["state"], 3), -(r.get("last") or 0), (r.get("name") or "").lower()))
    return {"pods": items, "error": err, "deploy": DEPLOY}
