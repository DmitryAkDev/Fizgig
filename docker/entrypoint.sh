#!/usr/bin/env bash
# Fizgig on a pod: update the app from git, set up /workspace, then serve the web UI on port 8081 (and the file
# manager on 8080). You use Fizgig in your own browser - nothing here draws a screen.
set -euo pipefail

log() { echo "[fizgig] $*"; }

REPO_URL="${FIZGIG_REPO:-https://github.com/shootthesound/Fizgig.git}"
REPO_REF="${FIZGIG_REF:-master}"
APP_DIR="/workspace/Fizgig"

mkdir -p /workspace/.cache /workspace/.insightface

# ---------------------------------------------------------------- the password
# The web UI on a public URL is a shell on someone else's GPU. FIZGIG_PASSWORD (or VNC_PASSWORD, the older
# templates' name) chooses it; with neither we generate one and print it, rather than leaving it open - a random
# password in the logs is recoverable, an open port is not fixable after the fact.
PW="${FIZGIG_PASSWORD:-${VNC_PASSWORD:-}}"
if [ -z "$PW" ]; then
  # NOT `tr -dc ... </dev/urandom | head -c 12`: head exits at 12 bytes, tr takes SIGPIPE and returns 141, which
  # pipefail + set -e turn into an instant exit before the first log line. Reading a FIXED number of bytes breaks
  # no pipe.
  _rand="$(head -c 512 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z0-9')"
  PW="${_rand:0:12}"
  unset _rand
  log "No FIZGIG_PASSWORD set — generated one for this pod: ${PW}"
  log "Set FIZGIG_PASSWORD in the template to choose your own."
fi
# The web server takes a pod's VNC_PASSWORD as its login at any length (fizgig/web/server.py main); the file
# manager below uses it too, so there is one password for everything on the pod.
export VNC_PASSWORD="$PW"
unset FIZGIG_PASSWORD

# ---------------------------------------------------------------- optional SSH
# Off by default — the file manager + runpodctl are the transfer path this image is built
# around, and an image that opens port 22 for everyone regardless of whether they asked is a
# worse default than a debugging feature nobody but us used. PUBLIC_KEY is RunPod's OWN
# convention (their "SSH Terminal Access" toggle sets it for you), so opting in there wires
# this up for free; on any other host, set PUBLIC_KEY to your public key yourself.
if [ -n "${PUBLIC_KEY:-}" ]; then
  # Self-installing so this also works against an already-published image that predates
  # openssh-server being baked in (see Dockerfile) — a no-op once a newer image has it.
  if ! command -v sshd >/dev/null 2>&1; then
    log "PUBLIC_KEY set — installing openssh-server"
    apt-get update -qq && apt-get install -y --no-install-recommends openssh-server -qq
  fi
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  # PasswordAuthentication stays default (no) — a key was just handed to us, a password wasn't,
  # and this is root on a box reachable from the whole internet.
  mkdir -p /run/sshd
  /usr/sbin/sshd
  log "SSH enabled on :22 (PUBLIC_KEY was set)"
fi

# ---------------------------------------------------------------- Fizgig source
# Pulled rather than baked, so a new release needs no image rebuild. A pod restart is an update.
if [ -d "$APP_DIR/.git" ]; then
  log "Updating Fizgig in $APP_DIR"
  # A persistent volume can carry a checkout from a DIFFERENT FIZGIG_REPO than the one
  # configured now (switched forks, fixed a typo) — origin is whatever the ORIGINAL clone
  # set it to, so point it at the current URL before fetching, or the ref lookup silently
  # targets the wrong repo.
  git -C "$APP_DIR" remote set-url origin "$REPO_URL"
  # NOT `fetch && reset` as one statement: under `set -e`, a failing command is only fatal
  # as the LAST command in an && chain — bash's own documented errexit exemption for
  # everything before it. A bad FIZGIG_REF used to fail the fetch, skip the reset, and
  # carry on running whatever was already checked out, with nothing but a bare `fatal:`
  # line (no [fizgig] prefix, nothing that reads as an error) to show for it.
  if ! git -C "$APP_DIR" fetch --depth 1 origin "$REPO_REF" \
      || ! git -C "$APP_DIR" reset --hard FETCH_HEAD; then
    # A checkout on a persistent volume can wedge permanently — a hard pod stop corrupts
    # git's object store, and from then on EVERY pod that mounts this volume fails this
    # update and silently runs old code, whatever image it booted from (observed in the
    # field: three different images, one stale volume). Limping on was the wrong policy:
    # quarantine the broken clone (rescuing a models dir if one ever landed inside) and
    # re-clone fresh. One slow boot instead of weeks of mystery-old pods.
    log "ERROR: could not update to '$REPO_REF' from $REPO_URL."
    if git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR.fresh"; then
      log "Re-cloning: the old checkout is quarantined at $APP_DIR.broken (delete it when convenient)."
      if [ -d "$APP_DIR/models" ]; then
        log "Rescuing $APP_DIR/models into the fresh checkout."
        mv "$APP_DIR/models" "$APP_DIR.fresh/models"
      fi
      rm -rf "$APP_DIR.broken" 2>/dev/null || true
      mv "$APP_DIR" "$APP_DIR.broken"
      mv "$APP_DIR.fresh" "$APP_DIR"
    else
      log "ERROR: re-clone failed too — check FIZGIG_REPO/FIZGIG_REF for a typo, or network access."
      log "       Continuing on whatever was already checked out ($(git -C "$APP_DIR" rev-parse --short HEAD 2>/dev/null))."
      rm -rf "$APP_DIR.fresh" 2>/dev/null || true
    fi
  fi
else
  log "Cloning Fizgig ($REPO_REF)"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
fi

# Top up dependencies in case the checkout is newer than the image. Almost always a no-op, and
# a few seconds when it isn't — far cheaper than making users re-pull 10 GB for a new pin.
# uv_install_deps.py comes from the fresh checkout above, not the image, so a pin/index-strategy
# fix to it reaches running pods on their next restart same as any other source change.
log "Checking dependencies"
python3 "$APP_DIR/uv_install_deps.py" "$APP_DIR/requirements.txt" "$VIRTUAL_ENV" \
   2>&1 | tail -5 || log "dependency top-up skipped"

# ---------------------------------------------------------------- persistent paths
# Point the portable output dirs at the volume. Model paths are left alone: fetch_models writes
# those itself, and overwriting them here would stomp a path the user set by hand.
python3 - <<'PY'
import json, os
p = "/workspace/Fizgig/prefs.json"
prefs = {}
if os.path.exists(p):
    try:
        prefs = json.load(open(p, encoding="utf-8"))
    except Exception:
        prefs = {}
for key, path in (("lora_output_dir", "/workspace/output_loras"),
                  ("profiles_dir",    "/workspace/profiles"),
                  ("cache_dir",       "/workspace/cache")):
    prefs.setdefault(key, path)
    os.makedirs(prefs[key], exist_ok=True)
# Where the Start tab's Browse opens. setdefault, not assignment: if the user picked their own
# folder we leave it alone.
prefs.setdefault("input_dataset_dir", "/workspace/datasets")
os.makedirs("/workspace/models", exist_ok=True)
json.dump(prefs, open(p, "w", encoding="utf-8"), indent=2)
print("[fizgig] output dirs on /workspace")
PY

# An obvious place to put training images. Created here rather than committed to the repo: in the
# repo it would land inside the git clone (the one directory that gets reset --hard on every boot)
# and every desktop user would get a stray pod-only folder they have no use for. Top level puts it
# beside models/ and output_loras/ in the file manager, which is where someone looks first.
mkdir -p /workspace/datasets
if [ ! -f /workspace/datasets/README.txt ]; then
    cat > /workspace/datasets/README.txt <<'NOTE'
Put your training images here.

Make one folder per LoRA, e.g.  /workspace/datasets/my_subject/
and put the images and their .txt caption files in it together.

Upload by dragging a folder into the file manager on port 8080, then on Fizgig's
Start tab click Browse and pick your folder under /workspace/datasets. Or use
the web UI on port 8081: Start tab -> Upload a training folder puts it here.

Everything under /workspace lives on your persistent volume, so it survives
stopping and restarting the pod. Anything outside /workspace does not.
NOTE
fi

# fetch_models defaults to <repo>/models, which puts 32 GB of weights INSIDE the git clone that
# this script runs `git reset --hard` on every boot. Point it at /workspace/models instead, next
# to the datasets and outputs, so it survives a re-clone and is where anyone would look for it.
if [ ! -L "$APP_DIR/models" ]; then
    if [ -d "$APP_DIR/models" ] && [ -n "$(ls -A "$APP_DIR/models" 2>/dev/null)" ]; then
        # Models already downloaded to the old location — move rather than orphan them.
        log "Moving existing models to /workspace/models"
        mv "$APP_DIR/models"/* /workspace/models/ 2>/dev/null || true
    fi
    rm -rf "$APP_DIR/models"
    ln -s /workspace/models "$APP_DIR/models"
fi

# Report the disk the models actually land on. Getting this wrong is expensive and silent: if the
# volume is not mounted, /workspace is the CONTAINER disk — which RunPod erases when the pod stops,
# so a 32 GB download evaporates on every restart. The failure mode is a "no space left on device"
# part-way through a download, which reads like a Fizgig bug rather than a template setting.
_avail_kb=$(df -Pk /workspace | awk 'NR==2{print $4}')
_avail_gb=$(( _avail_kb / 1024 / 1024 ))
_total_gb=$(( $(df -Pk /workspace | awk 'NR==2{print $2}') / 1024 / 1024 ))
if [ "$_total_gb" -gt 10000 ]; then
    # A network volume does not expose its quota in here — the kernel reports the host's whole
    # backing pool, so a 100 GB volume reads as ~1.4 PB. Printing that looks broken.
    log "Storage: /workspace is a network volume (size is set in RunPod, not visible from here)"
else
    log "Storage: /workspace has ${_avail_gb} GB free of ${_total_gb} GB"
fi
if [ "$_total_gb" -lt 60 ]; then
    log "  WARN: that looks like the CONTAINER disk, not a volume. Krea 2 alone needs ~32 GB,"
    log "        and container disk is wiped when the pod stops. Check the template has a"
    log "        Volume Disk of 100 GB+ AND a mount path of /workspace."
fi

# ---------------------------------------------------------------- optional model prefetch
# Off by default. Downloading tens of GB unasked spends the user's money and may fetch the
# family they didn't want — the Preferences button does this on demand, with a progress bar.
if [ -n "${FETCH_MODELS:-}" ]; then
  log "FETCH_MODELS=${FETCH_MODELS} — downloading before launch"
  IFS=',' read -ra FAMS <<< "$FETCH_MODELS"
  ARGS=()
  for f in "${FAMS[@]}"; do ARGS+=(--family "$(echo "$f" | xargs)"); done
  ( cd "$APP_DIR" && PYTHONPATH="$APP_DIR/src" python3 -m fizgig.scripts.fetch_models \
      "${ARGS[@]}" ${FETCH_MODELS_EXTRA:-} ) || log "model fetch incomplete — use the Preferences button"
fi

# ---------------------------------------------------------------- the file manager
# Drag-and-drop file transfer on its own port: datasets in, trained LoRAs out, no terminal.
# Shares the web UI's password rather than inventing a second credential — it exposes the same /workspace
# the web UI already does, so a separate password would be security theatre, and one fewer thing to lose. Non-fatal: file transfer failing must not stop training.
if command -v filebrowser >/dev/null 2>&1; then
    FB_DB=/workspace/.filebrowser.db
    # filebrowser rejects passwords under 12 characters. A shorter VNC_PASSWORD therefore creates
    # NO user at all, and the login just 403s — so pad rather than fail, and print what the
    # padded password actually is. (Found the hard way: the first version wrapped these commands
    # in `|| true`, so the log said "File manager on :8080" while the account did not exist.)
    FB_PW="$VNC_PASSWORD"
    while [ ${#FB_PW} -lt 12 ]; do FB_PW="${FB_PW}0"; done

    filebrowser config init --database "$FB_DB" >/dev/null 2>&1 || true
    filebrowser config set --database "$FB_DB" \
        --address 0.0.0.0 --port 8080 --root /workspace >/dev/null 2>&1 || true
    # Errors are logged, not swallowed: a file manager nobody can log into is worse than one
    # that says why.
    if ! _fb_out=$(filebrowser users add admin "$FB_PW" --perm.admin --database "$FB_DB" 2>&1); then
        if ! _fb_out=$(filebrowser users update admin --password "$FB_PW" \
                          --database "$FB_DB" 2>&1); then
            log "WARN: file manager login not set up: $(echo "$_fb_out" | tail -1)"
        fi
    fi
    filebrowser --database "$FB_DB" >/dev/null 2>&1 &
    if [ "$FB_PW" = "$VNC_PASSWORD" ]; then
        log "File manager on :8080 — user 'admin', same password as the web UI"
    else
        log "File manager on :8080 — user 'admin', password '${FB_PW}'"
        log "  (padded: filebrowser needs 12+ characters and the password is shorter)"
    fi
fi

cat <<EOF
[fizgig] ------------------------------------------------------------
[fizgig]  Ready. Open the pod's HTTP port 8081 for Fizgig.
[fizgig]      password: ${VNC_PASSWORD}
[fizgig]
[fizgig]  Models: Preferences -> "Download models for me"
[fizgig]  Files:  port 8080 is a file manager for /workspace (user 'admin')
[fizgig]
[fizgig]  Closing the browser tab doesn't stop training: Fizgig runs here.
[fizgig] ------------------------------------------------------------
EOF

# ---------------------------------------------------------------- the web UI
# Kept running: with no desktop, a stopped web server is a pod nobody can reach, so it starts again after a crash.
# A container stop (SIGTERM) passes to the server, which ends a run the way Stop does, and then the loop ends too.
cd "$APP_DIR"
export PYTHONPATH="$APP_DIR/src"
child=0
stopping=0
trap 'stopping=1; [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0' TERM INT
while true; do
  python3 lora_trainer_gui.py --web &
  child=$!
  set +e
  wait "$child"
  code=$?
  set -e
  [ "$stopping" = 1 ] && exit 0
  log "the web UI stopped (exit ${code}) — starting it again in 3 s"
  sleep 3
done
