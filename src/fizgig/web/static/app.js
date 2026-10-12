// Fizgig web UI. Every form on the page is built from /api/schema: each family's description and its form
// (fizgig/families/form.py). Nothing here names a family, a setting or a preset - a new one appears by itself.
"use strict";

const TOKEN = document.querySelector('meta[name="fizgig-token"]').content;
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
};

const S = {
  schema: null, fam: null, family: "", values: {}, all: {}, imageFolder: "", prefs: {},
  state: {hidden: [], disabled: [], notes: {}}, run: {}, logSince: 0, inputs: {},
};

// ---- server ------------------------------------------------------------------------------------------------
async function api(path, body) {
  const opt = {headers: {"X-Fizgig-Token": TOKEN}};
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && data.login) { showLogin(); throw new Error("login"); }
  if (!r.ok && !data.problems) throw new Error(data.error || r.statusText);
  return data;
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

// ---- start-up ----------------------------------------------------------------------------------------------
async function boot() {
  try { S.schema = await api("/api/schema"); } catch (e) { return; }
  const st = await api("/api/state");
  const pr = await api("/api/prefs");
  S.prefs = pr.models || {};
  S.webPort = pr.web_port;
  S.portNow = pr.port_now;
  S.onPod = pr.on_pod;
  S.portFixed = pr.port_fixed;
  S.gpu = {chosen: pr.cuda_device || "", list: pr.gpus || [], envLocked: pr.gpu_env_locked};
  S.all = st.values || {};
  S.imageFolder = st.image_folder || "";
  $("imageFolder").value = S.imageFolder;
  const sel = $("familySelect");
  sel.innerHTML = "";
  for (const f of S.schema.families) sel.append(el("option", {value: f.key, text: f.gui_label || f.display_name}));
  await chooseFamily(st.family && S.schema.order.includes(st.family) ? st.family : S.schema.order[0]);
  if (st.tab) showTab(st.tab);
  wire();
  pollRun(); pollStats(); followLog();
  setInterval(pollRun, 2000); setInterval(pollStats, 2000); setInterval(loadGallery, 15000);
  setInterval(loadCheckpoints, 15000);
}

async function chooseFamily(key) {
  S.family = key;
  S.fam = S.schema.families.find((f) => f.key === key);
  $("familySelect").value = key;
  if (S.all[key]) {
    S.values = {...defaults(), ...S.all[key]};
  } else {                              // a first visit takes the family's first preset, as the desktop does
    S.values = defaults();
    const first = S.fam.form.presets[0];
    if (first) S.values = (await api("/api/preset", {family: key, name: first.name, values: S.values})).values;
  }
  renderAll();
  await refreshState();
  saveState();
}

function defaults() {
  const v = {};
  for (const s of S.fam.form.sections) for (const r of s.rows) if (r.kind !== "note") v[r.key] = r.default;
  return v;
}

function renderAll() {
  $("familyNotes").textContent = S.fam.model_note || "";
  renderSections("samples", $("samplesForm"));
  renderSections("training", $("trainingForm"));
  renderPresets();
  renderPrefs();
  watchFetch();
  checkModels();
  loadGallery();
  loadCheckpoints();
}

// ---- the generated forms -----------------------------------------------------------------------------------
function renderSections(tab, host) {
  host.innerHTML = "";
  for (const sec of S.fam.form.sections.filter((s) => s.tab === tab)) {
    const body = el("div", {class: "card-content" + (sec.collapsed ? " collapsed" : "")});
    const head = el("div", {class: "card-title collapsible-header", onclick: () => body.classList.toggle("collapsed")},
      el("span", {text: (sec.collapsed ? "▶ " : "▼ ") + sec.title}));
    for (const r of sec.rows) body.append(renderRow(r));
    host.append(el("div", {class: "section-card", id: "sec-" + sec.id, "data-section": sec.id}, head, body));
  }
}

function renderRow(r) {
  const wrap = el("div", {class: "form-row", "data-key": r.key});
  if (r.kind === "note") { wrap.append(el("div", {class: "field-hint", text: r.label})); return wrap; }
  const v = S.values[r.key];
  let input;
  const set = (val) => onValue(r, val);
  if (r.kind === "check") {
    input = el("input", {type: "checkbox", id: "f_" + r.key, onchange: (e) => set(e.target.checked)});
    input.checked = !!v;
    wrap.append(el("label", {class: "checkbox-label"}, input, " " + r.label));
  } else if (r.kind === "radio") {
    const box = el("div", {class: "radio-row"});
    for (const c of r.choices) {
      const rb = el("input", {type: "radio", name: "f_" + r.key, value: c, onchange: () => set(c)});
      rb.checked = String(v) === c;
      box.append(el("label", {class: "checkbox-label"}, rb, " " + c));
    }
    wrap.append(el("span", {class: "field-label", text: r.label + ":"}), box);
    input = box;
  } else {
    if (r.kind === "choice") {
      input = el("select", {id: "f_" + r.key, onchange: (e) => set(e.target.value)});
      const opts = r.choices.includes(String(v)) || !v ? r.choices : [String(v), ...r.choices];
      for (const c of opts) input.append(el("option", {value: c, text: c}));
      input.value = String(v ?? "");
    } else if (r.kind === "text") {
      input = el("textarea", {id: "f_" + r.key, rows: 4, oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
    } else {
      const listId = r.kind === "combo" ? "dl_" + r.key : null;
      input = el("input", {type: "text", id: "f_" + r.key, list: listId, oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
      if (listId) wrap.append(el("datalist", {id: listId}, r.choices.map((c) => el("option", {value: c}))));
    }
    const row = el("div", {class: "field-row"}, el("span", {class: "field-label", text: r.label + ":"}), input);
    if (r.kind === "path" || r.kind === "folder") {
      row.append(el("button", {class: "btn", text: "Browse…", onclick: () =>
        pick(input.value || S.imageFolder, r.kind === "folder", (p) => { input.value = p; set(p); })}));
    }
    if (r.suffix) row.append(el("span", {class: "field-hint", text: r.suffix}));
    wrap.append(row);
  }
  wrap.append(el("div", {class: "field-hint row-note", id: "note_" + r.key}));
  if (r.hint) wrap.append(el("div", {class: "field-hint", text: r.hint}));
  return wrap;
}

function onValue(r, val) {
  S.values[r.key] = val;
  const extra = r.sets && r.sets[val];            // a kind of training that also sets others (Fine-tune's LR)
  if (extra) { Object.assign(S.values, extra); syncInputs(Object.keys(extra)); }
  if (r.timesteps && r.timesteps[val]) {           // a Model Area with its own timestep window
    [S.values.MIN_TIMESTEP, S.values.MAX_TIMESTEP] = r.timesteps[val];
    syncInputs(["MIN_TIMESTEP", "MAX_TIMESTEP"]);
  }
  refreshStateSoon();
  saveStateSoon();
}

function syncInputs(keys) {
  for (const k of keys) {
    const i = $("f_" + k);
    if (!i) continue;
    if (i.type === "checkbox") i.checked = !!S.values[k]; else i.value = S.values[k] ?? "";
  }
}

async function refreshState() {
  S.state = await api("/api/form/state", {family: S.family, values: S.values, image_folder: S.imageFolder});
  const hidden = new Set(S.state.hidden), disabled = new Set(S.state.disabled);
  document.querySelectorAll(".form-row[data-key]").forEach((row) => {
    const k = row.dataset.key;
    row.style.display = hidden.has(k) ? "none" : "";
    row.querySelectorAll("input,select,textarea,button").forEach((i) => { i.disabled = disabled.has(k); });
    const n = $("note_" + k);
    if (n) n.textContent = S.state.notes[k] || "";
  });
  document.querySelectorAll(".section-card[data-section]").forEach((c) => {
    c.style.display = hidden.has(c.dataset.section) ? "none" : "";
  });
}
const refreshStateSoon = debounce(refreshState, 250);

function renderPresets() {
  const host = $("presets");
  host.innerHTML = "";
  for (const p of S.fam.form.presets) {
    host.append(el("button", {class: "preset-chip", text: p.name, onclick: async () => {
      S.values = (await api("/api/preset", {family: S.family, name: p.name, values: S.values})).values;
      renderSections("samples", $("samplesForm"));
      renderSections("training", $("trainingForm"));
      await refreshState();
      saveState();
    }}));
  }
}

// ---- Preferences: every family's model files, from its description -----------------------------------------
function renderPrefs() {
  const host = $("prefsForm");
  host.innerHTML = "";
  host.append(webPortCard());
  host.append(dataFolderCard());
  const gpuCard = gpuPickCard();
  if (gpuCard) host.append(gpuCard);
  S.fetchRows = {};
  for (const f of S.schema.families) {
    const body = el("div", {class: "card-content" + (f.key === S.family ? "" : " collapsed")});
    const files = new Map(f.model_files.map((m) => [m.pref_key, m]));
    for (const key of f.model_keys) {
      const m = files.get(key) || {label: key, note: "", hint: ""};
      const input = el("input", {type: "text", class: "w-path", value: S.prefs[key] || ""});
      input.addEventListener("change", async () => {
        S.prefs[key] = input.value.trim();
        await api("/api/prefs", {changes: {[key]: S.prefs[key]}});
        checkModels();
      });
      const row = el("div", {class: "field-row"},
        el("span", {class: "field-label", text: (m.label || key) + (m.required ? " *" : "") + ":"}), input,
        el("button", {class: "btn", text: "Browse…", onclick: () =>
          pick(input.value, false, (p) => { input.value = p; input.dispatchEvent(new Event("change")); })}));
      if (m.repo && m.path) {
        row.append(el("a", {class: "btn", target: "_blank", rel: "noopener", text: "Download",
          href: `https://huggingface.co/${m.repo}/blob/main/${m.path}`}));
      }
      body.append(row);
      const hint = m.hint || m.note;
      if (hint) body.append(el("div", {class: "field-hint", text: hint}));
    }
    const fetchBox = fetchRow(f);
    if (fetchBox) body.append(fetchBox);
    host.append(el("div", {class: "section-card"},
      el("div", {class: "card-title collapsible-header", onclick: () => body.classList.toggle("collapsed")},
        el("span", {text: f.prefs_title || ("Model Paths (" + f.display_name + ")")})), body));
  }
  paintFetch();
}

// "Download models for me": the family's files + the helper models, paths filled in when they land
function fetchRow(f) {
  const files = f.model_files.filter((m) => m.repo && m.path);
  if (!files.length) return null;
  const total = files.reduce((a, m) => a + (m.size_gb || 0), 0);
  const gated = files.filter((m) => m.gated);
  const box = el("div", {class: "fetch-row"});
  const opt = f.fetch_optional_label ? el("input", {type: "checkbox"}) : null;
  const token = gated.length ? el("input", {type: "password", class: "w-path", placeholder: "hf_…"}) : null;
  const btn = el("button", {class: "btn btn-accent", text: "⬇ Download models for me", onclick: async () => {
    try {
      await api("/api/fetch", {family: f.key, token: token ? token.value.trim() : "", optional: opt ? opt.checked : false});
      if (token) token.value = "";
      S.fetchError = null;
      watchFetch();
    } catch (e) { S.fetchError = {family: f.key, text: e.message}; paintFetch(); }
  }});
  const line = el("div", {class: "field-row"}, btn);
  if (opt) line.append(el("label", {class: "fetch-opt"}, opt, " " + f.fetch_optional_label));
  box.append(line);
  if (token) {
    const seen = new Set();
    const links = gated.filter((m) => !seen.has(m.repo) && seen.add(m.repo)).map((m) =>
      el("a", {target: "_blank", rel: "noopener", href: `https://huggingface.co/${m.repo}`, text: m.label + " licence"}));
    box.append(el("div", {class: "fetch-token"},
      el("div", {class: "field-hint", text: "These files are gated: sign in to Hugging Face, accept the licence on " +
        "each page below, then paste a read token. It is used for this download only, never saved. " +
        "Leave it empty when the machine or pod has HF_TOKEN set."}),
      el("div", {class: "fetch-links"}, links,
        el("a", {target: "_blank", rel: "noopener", href: "https://huggingface.co/settings/tokens", text: "Create a token"})),
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Token:"}), token)));
  }
  box.append(el("div", {class: "field-hint", text: f.fetch_note ||
    `Fetches every file above (~${Math.round(total)} GB) and fills in these paths, plus the Qwen3-VL captioner ` +
    "the Captions tab uses and the small helper models. Interrupted downloads resume where they left off."}));
  const status = el("div", {class: "fetch-status"});
  box.append(status);
  S.fetchRows[f.key] = {status, btn};
  return box;
}

function paintFetch() {
  const st = S.fetch || {};
  const busy = st.state === "running";
  const gb = (n) => (n / 1024 ** 3).toFixed(1);
  for (const [key, r] of Object.entries(S.fetchRows || {})) {
    r.btn.disabled = busy;
    r.status.innerHTML = "";
    if (S.fetchError && S.fetchError.family === key) {
      r.status.append(el("div", {class: "fetch-msg bad", text: S.fetchError.text}));
    }
    if (st.family !== key || !st.state || st.state === "idle") continue;
    if (busy) {
      const pct = st.total ? Math.min(100, (100 * st.done) / st.total) : 0;
      r.status.append(
        el("div", {class: "fetch-msg", text: st.file ? `${st.file}: ${gb(st.done)} of ${gb(st.total)} GB` :
          (st.lines[st.lines.length - 1] || "Starting…")}),
        el("div", {class: "fetch-bar"}, el("div", {style: `width:${pct.toFixed(1)}%`})),
        el("button", {class: "btn", text: "Cancel", onclick: () => api("/api/fetch/cancel", {}).then(watchFetch)}));
    } else {
      const text = {done: "Done: the paths above are filled in.", cancelled: "Cancelled. Downloading again resumes it.",
        failed: "Some files did not download; press the button again to retry just those."}[st.state] || "";
      r.status.append(el("div", {class: "fetch-msg" + (st.state === "done" ? "" : " bad"), text}));
      if (st.state === "failed") r.status.append(el("pre", {class: "fetch-log", text: st.lines.slice(-12).join("\n")}));
    }
  }
}

async function watchFetch() {
  clearTimeout(S.fetchTimer);
  const was = S.fetch && S.fetch.state;
  try { S.fetch = await api("/api/fetch"); } catch (e) { return; }
  if (was === "running" && S.fetch.state !== "running") {
    S.prefs = (await api("/api/prefs")).models || {};    // the fetcher wrote the new paths
    renderPrefs();
    checkModels();
  }
  paintFetch();
  if (S.fetch.state === "running") S.fetchTimer = setTimeout(watchFetch, 1500);
}

// which GPU runs use (Preferences cuda_device, the desktop's GPU setting): only shown with more than one card
function gpuPickCard() {
  const g = S.gpu || {list: []};
  if (g.list.length < 2) return null;
  const sel = el("select", {class: "w-combo-lg"},
    el("option", {value: "", text: "The machine's default"}),
    g.list.map((c) => el("option", {value: c.uuid, text: `${c.index}: ${c.name} (${Math.round(c.total_gb)} GB)`})));
  const known = g.list.find((c) => c.uuid === g.chosen || String(c.index) === g.chosen);
  sel.value = known ? known.uuid : "";
  sel.disabled = !!g.envLocked;
  const hint = el("div", {class: "field-hint", text: g.envLocked ?
    "CUDA_VISIBLE_DEVICES is set where Fizgig was started, and it wins over this setting." :
    "The card every run started from here trains on, and the one the memory meter shows. The desktop app uses " +
    "the same setting. A run already going keeps its card."});
  sel.addEventListener("change", async () => {
    try { await api("/api/prefs", {changes: {cuda_device: sel.value}}); g.chosen = sel.value; }
    catch (e) { hint.textContent = e.message; }
  });
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "GPU"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Train on:"}), sel), hint));
}

// where uploads go: Preferences' training data folder (the desktop's Input Directories > Training images), else
// training_data inside Fizgig
function showUploadRoot() {
  api("/api/upload/root").then((d) => {
    S.uploadRoot = d;
    const box = $("dataFolder");
    if (box && document.activeElement !== box) box.value = d.set ? d.root : "";
    $("uploadWhere").textContent = `It goes into ${d.root}, in a folder with the name above. ` +
      (d.set ? "That folder is set in Preferences (Training data folder). " :
        "That is Fizgig's own training_data folder; choose another in Preferences (Training data folder). ") +
      "Uploading again with the same name adds to that folder.";
  }).catch(() => {});
}

function dataFolderCard() {
  const input = el("input", {type: "text", class: "w-path", id: "dataFolder", value: (S.uploadRoot && S.uploadRoot.set) ?
    S.uploadRoot.root : "", placeholder: "Fizgig's own training_data folder"});
  const save = async () => {
    try { await api("/api/prefs", {changes: {input_dataset_dir: input.value.trim()}}); showUploadRoot(); }
    catch (e) { hint.textContent = e.message; }
  };
  input.addEventListener("change", save);
  const hint = el("div", {class: "field-hint", text: "Where uploaded training data goes, each upload in its own folder " +
    "inside it, and where the folder pickers start. Leave it empty for Fizgig's own training_data folder. The " +
    "desktop app's Preferences > Input Directories > Training images is the same setting."});
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Training data folder"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Folder:"}), input,
        el("button", {class: "btn", text: "Browse…", onclick: () =>
          pick(input.value, true, (p) => { input.value = p; save(); })})), hint));
}

// the web UI's port: saved by the first launch (8081, or the next free port if 8081 was taken)
function webPortCard() {
  const input = el("input", {type: "number", min: "1024", max: "65535", class: "w-port", value: S.webPort || 8081});
  const hint = el("div", {class: "field-hint"});
  if (S.onPod || S.portFixed) {
    input.disabled = true;
    input.value = S.portNow;
    hint.textContent = S.onPod ? "A pod always uses 8081, the port its proxy forwards." :
      `Set by --port where Fizgig was started (${S.portNow}). Change it there.`;
    return el("div", {class: "section-card"},
      el("div", {class: "card-title"}, el("span", {text: "Web UI"})),
      el("div", {class: "card-content"},
        el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint));
  }
  const say = () => {
    const p = Number(input.value);
    hint.textContent = p === S.portNow ? "The port this page is on. Fizgig uses it every time it starts." :
      `Fizgig moves to port ${p} next time it starts (this page is on ${S.portNow} until then). ` +
      "If that port is taken, it uses the next free one above it.";
  };
  input.addEventListener("change", async () => {
    const p = Number(input.value);
    if (!Number.isInteger(p) || p < 1024 || p > 65535) { input.value = S.webPort; return say(); }
    await api("/api/prefs", {changes: {web_port: p}});
    S.webPort = p;
    say();
  });
  say();
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Web UI"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint));
}

function checkModels() {
  const missing = S.fam.model_files.filter((m) => m.required && !S.prefs[m.pref_key]).map((m) => m.label);
  $("modelWarning").style.display = missing.length ? "" : "none";
  $("modelWarningText").textContent = missing.length ?
    `${S.fam.display_name} needs: ${missing.join(", ")}. Set them on the Preferences tab.` : "";
}

// ---- saved state -------------------------------------------------------------------------------------------
function saveState() {
  S.all[S.family] = S.values;
  api("/api/state", {family: S.family, image_folder: S.imageFolder, values: S.all}).catch(() => {});
}
const saveStateSoon = debounce(saveState, 600);

// ---- the run -----------------------------------------------------------------------------------------------
const runBody = () => ({family: S.family, values: S.values, image_folder: S.imageFolder});

function showProblems(list) {
  $("problems").textContent = (list || []).length ? "Can't start yet:\n• " + list.join("\n• ") : "";
}

async function check() {
  const p = await api("/api/plan", runBody());
  showProblems(p.problems);
  if (!p.problems.length) $("runStatus").textContent = "Ready: " + p.stages.map((s) => s.name).join(" → ");
}

async function start(path) {
  const cmd = {"/api/pause": "pause", "/api/stop": "stop", "/api/resume": "resume"}[path];
  if (S.mode === "desktop" && cmd) {              // the desktop app's run: its own buttons do the work
    S.ackBefore = (S.run.desktop && S.run.desktop.ack && S.run.desktop.ack.at) || 0;   // the PC's clock, not ours
    try { await api("/api/desktop", {cmd}); $("runStatus").textContent = "Sent to the desktop app…"; }
    catch (e) { $("runStatus").textContent = e.message; }
    return setTimeout(pollRun, 1200);
  }
  try {
    const r = await api(path, path === "/api/start" ? runBody() : {});
    showProblems(r.problems);
    if (r.error) $("runStatus").textContent = r.error;
    pollRun();
  } catch (e) { $("runStatus").textContent = e.message; }
}

const RUN_LABELS = {idle: "Idle", starting: "Starting", running: "Running", pausing: "Pausing", paused: "Paused",
                    completed: "Finished", failed: "Failed", stopped: "Stopped"};

async function pollRun() {
  try { S.run = await api("/api/run"); } catch (e) { return; }
  const r = S.run, live = r.running || r.state === "starting";
  const d = r.desktop;
  // the desktop app's run shows here whenever this page has none of its own going
  const deskBusy = d && (d.running || ["running", "pausing", "paused"].includes(d.state));
  // ...and a run it just finished stays on show (Finished / Failed / Stopped) until either side starts another
  const deskDone = d && !deskBusy && d.last_result && d.last_result.at > (r.ended_at || 0);
  const mode = !live && (deskBusy || deskDone) ? "desktop" : "web";
  if (mode !== S.mode) switchConsole(mode);
  if (mode === "desktop") {
    const res = deskDone ? {finished: "completed", failed: "failed", stopped: "stopped"}[d.last_result.result] : null;
    const state = res || d.state;
    const label = RUN_LABELS[state] || state;
    S.run.output_dir = d.output_dir;          // previews and checkpoints follow the desktop's run
    const when = deskDone ? " · " + new Date(d.last_result.at * 1000).toLocaleTimeString() : "";
    $("runStatus").textContent = `${label}: ${(deskDone ? d.last_result.lora : d.lora) || "(unnamed)"}` +
      (deskDone ? when : d.stage ? " · " + d.stage : "");
    const ack = d.ack && S.ackBefore !== undefined && d.ack.at > S.ackBefore ? " " + d.ack.note : "";
    $("runSource").textContent = (deskDone ?
      `The last run started in the desktop app (${d.family}) ended. Its previews and checkpoints are below; ` +
      "Start begins a new run here." :
      `This run was started in the desktop app (${d.family}). Pause, Stop and Resume here work on it. Start is ` +
      "off until it finishes, so the two don't share the GPU." +
      (d.unresponsive ? " The desktop app isn't answering right now (busy for a moment?), so Pause, Stop and " +
        "Resume wait until it does." : "")) + ack;
    $("runBadge").textContent = label;
    $("runBadge").dataset.state = state;
    $("btnStart").disabled = !deskDone;
    $("btnPause").disabled = !(d.running && d.state === "running");
    $("btnStop").disabled = !d.running;
    $("btnResume").style.display = !d.running && d.state === "paused" ? "" : "none";
  } else {
    const label = RUN_LABELS[r.state] || r.state;
    $("runStatus").textContent = r.lora ? `${label}: ${r.lora}${r.stage ? " · " + r.stage : ""}` : label;
    $("runSource").textContent = d ? "The desktop app is open too. A run started there shows here." : "";
    $("runBadge").textContent = label;
    $("runBadge").dataset.state = r.state;
    $("btnStart").disabled = live;
    $("btnPause").disabled = !r.running;
    $("btnStop").disabled = !live;
    $("btnResume").style.display = !live && r.resume ? "" : "none";
  }
  renderQueue(r.queue || []);
}

function renderQueue(items) {
  $("queueCard").style.display = items.length ? "" : "none";
  const host = $("queueList");
  host.innerHTML = "";
  items.forEach((q, i) => host.append(el("div", {class: "checkpoint-row"},
    el("span", {class: "checkpoint-name", text: `${i + 1}. ${q.name}`}),
    el("span", {class: "field-hint", text: `${q.family} · ${q.folder}${q.queued_at ? " · queued " + q.queued_at : ""}`}))));
}

// the console shows this page's run (streamed) or the desktop app's (its console copy, polled)
function switchConsole(mode) {
  S.mode = mode;
  $("console").textContent = "";
  clearTimeout(S.deskLogTimer);
  if (mode === "desktop") {
    S.deskOffset = 0;
    pollDesktopLog();
  } else if (S.logSrc) {
    S.logSrc.close();
    S.logSince = 0;
    followLog();
  }
  loadGallery(); loadCheckpoints();
}

async function pollDesktopLog() {
  if (S.mode !== "desktop") return;
  try {
    const first = !S.deskOffset;
    const r = await api("/api/desktop/log?offset=" + (S.deskOffset || 0) + (first ? "&tail=1" : ""));
    const out = $("console");
    if (r.offset < (S.deskOffset || 0)) out.textContent = "";        // a new run started the copy again
    if (r.text) {
      const atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
      out.textContent = (out.textContent + r.text).slice(-200000);
      if (atEnd) out.scrollTop = out.scrollHeight;
    }
    S.deskOffset = r.offset;
  } catch (e) { /* the server restarts */ }
  S.deskLogTimer = setTimeout(pollDesktopLog, 1500);
}

function followLog() {
  const src = S.logSrc = new EventSource(`/api/log?since=${S.logSince}&t=${encodeURIComponent(TOKEN)}`);
  const out = $("console");
  src.onmessage = (e) => {
    if (S.mode === "desktop") return;
    S.logSince = Number(e.lastEventId) || S.logSince;
    const line = JSON.parse(e.data);
    const atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    // a progress bar redraws itself with \r: keep only its latest state
    const text = out.textContent;
    if (line.includes("\r") && !line.endsWith("\n")) {
      out.textContent = text.slice(0, text.lastIndexOf("\n") + 1) + line.split("\r").pop();
    } else {
      out.textContent = (text + line.split("\r").pop()).slice(-200000);
    }
    if (atEnd) out.scrollTop = out.scrollHeight;
  };
  src.onerror = () => { src.close(); if (S.logSrc === src) setTimeout(followLog, 3000); };
}

async function pollStats() {
  try {
    const s = await api("/api/stats");
    const gb = (b) => (b / 2 ** 30).toFixed(1);
    $("statusBar").textContent = `${s.device_name} · VRAM ${gb(s.vram[0])} / ${gb(s.vram[1])} GB ` +
      `(peak ${gb(s.vram_peak)}) · RAM ${gb(s.ram[0])} / ${gb(s.ram[1])} GB`;
  } catch (e) { /* the server restarts */ }
}

async function loadGallery() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR;
  const host = $("gallery");
  if (!out) { host.innerHTML = ""; return; }
  const sep = out.includes("\\") ? "\\" : "/";
  const g = await api("/api/samples?dir=" + encodeURIComponent(out + sep + "sample")).catch(() => ({items: []}));
  host.innerHTML = "";
  for (const it of g.items.slice(0, 60)) {
    const src = `/api/media?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    const ext = it.name.split(".").pop().toLowerCase();
    const media = ["wav", "mp3", "flac", "m4a"].includes(ext) ? el("audio", {controls: true, preload: "none", src}) :
      ["mp4", "webm"].includes(ext) ? el("video", {controls: true, preload: "none", src}) :
      el("img", {loading: "lazy", src, alt: it.name});
    host.append(el("figure", {class: "gallery-item"}, media, el("figcaption", {text: it.name})));
  }
  $("galleryInfo").textContent = g.items.length ? `${g.items.length} previews in ${out}${sep}sample` :
    "No previews yet. They appear here as the run makes them.";
}

async function loadCheckpoints() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR;
  const host = $("checkpoints");
  if (!out) { host.innerHTML = ""; return; }
  const r = await api("/api/checkpoints?dir=" + encodeURIComponent(out)).catch(() => ({items: []}));
  host.innerHTML = "";
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  for (const it of r.items) {
    const href = `/api/download?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: it.name}),
      el("span", {class: "field-hint", text: `${it.kind === "finetune" ? "Fine-tune (whole model)" : "LoRA"} · ` +
        `${size(it.size)} · ${new Date(it.time * 1000).toLocaleString()}`}),
      el("a", {class: "btn btn-sm btn-primary", href, download: it.name, text: "Download"})));
  }
  $("checkpointsInfo").textContent = r.items.length ? `${r.items.length} saved in ${out}, newest first.` :
    "Nothing saved yet. Checkpoints appear here as the run saves them.";
}

// ---- Captions tab ------------------------------------------------------------------------------------------
async function loadCaptions() {
  const host = $("captionsGrid");
  host.innerHTML = "";
  if (!S.imageFolder) { $("captionsSummary").textContent = "Choose the training folder on the Start tab."; return; }
  const ext = $("captionExt").value.trim() || ".txt";
  const r = await api(`/api/captions?folder=${encodeURIComponent(S.imageFolder)}&ext=${encodeURIComponent(ext)}`);
  const missing = r.items.filter((i) => i.caption === null).length;
  $("captionsSummary").textContent = `${r.items.length} files${missing ? `, ${missing} without a caption` : ""}`;
  for (const it of r.items) {
    const src = `/api/media?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    const ta = el("textarea", {rows: 4});
    ta.value = it.caption ?? "";
    const status = el("span", {class: "field-hint"});
    const save = el("button", {class: "btn btn-sm", text: "Save", onclick: async () => {
      await api("/api/captions/save", {folder: S.imageFolder, name: it.name, text: ta.value, ext});
      status.textContent = "Saved";
    }});
    const ext2 = it.name.split(".").pop().toLowerCase();
    const media = ["wav", "mp3", "flac", "m4a"].includes(ext2) ? el("audio", {controls: true, preload: "none", src}) :
      ["mp4", "webm"].includes(ext2) ? el("video", {controls: true, preload: "none", src}) :
      el("img", {loading: "lazy", src, alt: it.name});
    host.append(el("div", {class: "caption-card"}, media,
      el("div", {class: "caption-card-body"}, el("div", {class: "caption-card-title", text: it.name}), ta,
        el("div", {class: "field-row"}, save, status))));
  }
}

// ---- file picker -------------------------------------------------------------------------------------------
let pickDone = null, pickDir = false;
async function pick(start, dirOnly, done) {
  pickDone = done; pickDir = dirOnly;
  $("pickerTitle").textContent = dirOnly ? "Choose a folder" : "Choose a file";
  $("pickerChoose").style.display = dirOnly ? "" : "none";
  $("pickerModal").classList.add("active");
  await browse(start || "");
}
async function browse(path) {
  const r = await api("/api/browse?path=" + encodeURIComponent(path));
  $("pickerPath").value = r.path;
  $("pickerUp").onclick = () => r.parent && browse(r.parent);
  const list = $("pickerList");
  list.innerHTML = "";
  for (const d of r.dirs) list.append(el("div", {class: "picker-item dir", text: "📁 " + d.name, onclick: () => browse(d.path)}));
  if (!pickDir) {
    for (const f of r.files) {
      list.append(el("div", {class: "picker-item", text: f.name, onclick: () => {
        $("pickerModal").classList.remove("active"); pickDone(f.path);
      }}));
    }
  }
}

// ---- login -------------------------------------------------------------------------------------------------
function showLogin() { $("loginModal").classList.add("active"); $("loginPassword").focus(); }
async function login() {
  const r = await fetch("/api/login", {method: "POST", headers: {"X-Fizgig-Token": TOKEN, "Content-Type": "application/json"},
    body: JSON.stringify({password: $("loginPassword").value})});
  if (r.ok) { $("loginModal").classList.remove("active"); boot(); }
  else $("loginError").textContent = (await r.json().catch(() => ({}))).error || "Wrong password.";
}

// ---- tabs and wiring ---------------------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll(".tab-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "captions") loadCaptions();
  if (name === "samples") loadGallery();
  if (name === "training") loadCheckpoints();
  api("/api/state", {tab: name}).catch(() => {});
}

// a dropped / chosen folder: every file inside it, subfolders included (each still goes up on its own)
function collect(entry, out) {
  if (entry.isFile) return new Promise((res) => entry.file((f) => { out.push(f); res(); }, () => res()));
  const reader = entry.createReader();
  return new Promise((res) => {
    const batch = () => reader.readEntries(async (list) => {          // readEntries returns at most ~100 a call
      if (!list.length) return res();
      for (const en of list) await collect(en, out);
      batch();
    }, () => res());
    batch();
  });
}
// skip what no family trains on and the system's own files, so a folder upload doesn't list them as failures
const UPLOAD_OK = /\.(png|jpe?g|webp|bmp|avif|mp4|wav|mp3|flac|m4a|txt|caption|json|zip)$/i;
const SYSTEM_FILE = (n) => n.startsWith(".") || ["thumbs.db", "desktop.ini"].includes(n.toLowerCase());
const OTHER_VIDEO = /\.(mov|webm|mkv|avi|m4v|wmv)$/i;
// what goes up: the system's own files are skipped quietly; other files are skipped and named, with a convert-to
// hint for video Fizgig doesn't read
function sortUpload(files) {
  const keep = [], video = [], other = [];
  for (const f of files) {
    if (SYSTEM_FILE(f.name)) continue;
    if (UPLOAD_OK.test(f.name)) keep.push(f);
    else (OTHER_VIDEO.test(f.name) ? video : other).push(f.name);
  }
  return {keep, video, other};
}
// what a file can't be called on every system, replaced the same way the server does for a zip (Act 1: Intro.wav ->
// Act 1_ Intro.wav), so a song and its caption files still pair up
const tidyName = (n) => n.replace(/[<>:"|?*\x00-\x1f]/g, "_").trim().replace(/[. ]+$/, "");
// a dropped folder names the new one, unless a name is typed already
function nameFromFolder(name) {
  const box = $("uploadName");
  if (name && !box.value.trim()) box.value = name.replace(/[\/\\<>:"|?*]/g, "_").replace(/^[.\s]+|[.\s]+$/g, "");
}

// one file as the request body, reporting bytes sent (fetch can't report upload progress)
function sendFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.setRequestHeader("X-Fizgig-Token", TOKEN);
    x.upload.onprogress = (e) => onProgress(e.loaded);
    x.onload = () => { let data = {}; try { data = JSON.parse(x.responseText); } catch (e) { /* not JSON */ }
      resolve({ok: x.status >= 200 && x.status < 300, data}); };
    x.onerror = () => reject(new Error("the connection dropped"));
    x.send(file);
  });
}

// ---- upload a training folder from this computer ------------------------------------------------------------
async function uploadFiles(picked) {
  const {keep: files, video, other} = sortUpload(picked);
  const notes = [];
  if (video.length) notes.push(`Left out ${video.length} video file${video.length === 1 ? "" : "s"} Fizgig can't ` +
    `train on (${video.slice(0, 3).join(", ")}${video.length > 3 ? ", …" : ""}): convert them to .mp4 first.`);
  if (other.length) notes.push(`Left out (not pictures, clips, songs, captions or zips): ` +
    `${other.slice(0, 5).join(", ")}${other.length > 5 ? ", …" : ""}.`);
  const seen = new Map();
  for (const f of files) seen.set(f.name, (seen.get(f.name) || 0) + 1);
  const dupes = [...seen].filter(([, n]) => n > 1).map(([k]) => k);
  if (dupes.length) notes.push(`More than one file is named ${dupes.slice(0, 3).join(", ")}` +
    `${dupes.length > 3 ? ", …" : ""} (from different subfolders); they all go into one folder, so the last one is kept.`);
  const name = $("uploadName").value.trim();
  const say = (t, bad) => { $("uploadStatus").textContent = t; $("uploadStatus").classList.toggle("bad", !!bad); };
  if (!name) return say("Type a name for the new folder first.", true);
  if (!files.length) return say(["Nothing to upload: no pictures, .mp4 clips, songs, captions or zips in there."]
    .concat(notes).join(" "), true);
  let folder = "", saved = 0;
  const skipped = [], failed = [], replaced = [], badname = [];
  for (const [i, f] of [...files].entries()) {
    const mb = (n) => (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0);
    say(`Uploading ${i + 1} of ${files.length}: ${f.name}`);
    try {
      const r = await sendFile(`/api/upload?folder=${encodeURIComponent(name)}&name=${encodeURIComponent(tidyName(f.name))}`,
        f, (sent) => say(`Uploading ${i + 1} of ${files.length}: ${f.name}, ${mb(sent)} of ${mb(f.size)} MB` +
          (sent >= f.size && f.name.toLowerCase().endsWith(".zip") ? " (unpacking…)" : "")));
      const d = r.data;
      if (!r.ok) { failed.push(d.error || f.name); continue; }
      folder = d.folder; saved += d.saved.length; skipped.push(...d.skipped); replaced.push(...(d.replaced || []));
      badname.push(...(d.badname || []));
    } catch (e) { failed.push(`${f.name}: ${e.message}`); }
  }
  let msg = saved ? `${saved} file${saved === 1 ? "" : "s"} in ${folder}. That is now the training folder.` : "Nothing was saved.";
  if (skipped.length) msg += ` Left out (not pictures, clips, songs or captions): ${skipped.join(", ")}.`;
  if (replaced.length) msg += ` Replaced files that were already there with the same name: ` +
    `${replaced.slice(0, 5).join(", ")}${replaced.length > 5 ? ", …" : ""}.`;
  if (notes.length) msg += " " + notes.join(" ");
  if (badname.length) msg += ` Left out of the zip, as their names can't be stored here: ${badname.slice(0, 5).join(", ")}` +
    `${badname.length > 5 ? ", …" : ""}. Rename them (no leading dot, no slashes) and upload them again.`;
  if (failed.length) msg += ` Not uploaded: ${failed.join("; ")}`;
  say(msg, !saved || failed.length);
  if (saved) {
    $("imageFolder").value = folder;
    $("imageFolder").dispatchEvent(new Event("change"));
  }
}

let wired = false;
function wire() {
  if (wired) return;
  wired = true;
  document.querySelectorAll(".tab-item").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
  document.querySelectorAll("[data-close]").forEach((b) =>
    b.addEventListener("click", () => $(b.dataset.close).classList.remove("active")));
  $("familySelect").addEventListener("change", (e) => chooseFamily(e.target.value));
  const folder = $("imageFolder");
  const setFolder = () => { S.imageFolder = folder.value.trim(); refreshStateSoon(); saveStateSoon(); };
  folder.addEventListener("change", setFolder);
  $("uploadFiles").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
  $("uploadDir").addEventListener("change", (e) => {
    const files = [...e.target.files];
    nameFromFolder(files.length ? (files[0].webkitRelativePath || "").split("/")[0] : "");
    uploadFiles(files);
    e.target.value = "";
  });
  const drop = $("uploadDrop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const entries = [...e.dataTransfer.items].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
    if (!entries.some((en) => en.isDirectory)) return uploadFiles([...e.dataTransfer.files]);
    const dirs = entries.filter((en) => en.isDirectory);
    if (dirs.length === 1) nameFromFolder(dirs[0].name);
    $("uploadStatus").textContent = "Reading the folder…";
    const files = [];
    for (const en of entries) await collect(en, files);
    uploadFiles(files);
  });
  showUploadRoot();
  document.querySelector('[data-pick="imageFolder"]').addEventListener("click", () =>
    pick(folder.value, true, (p) => { folder.value = p; setFolder(); }));
  $("pickerChoose").addEventListener("click", () => { $("pickerModal").classList.remove("active"); pickDone($("pickerPath").value); });
  $("pickerPath").addEventListener("change", (e) => browse(e.target.value));
  $("captionsReload").addEventListener("click", loadCaptions);
  $("btnCheck").addEventListener("click", check);
  $("btnStart").addEventListener("click", () => start("/api/start"));
  $("btnPause").addEventListener("click", () => start("/api/pause"));
  $("btnStop").addEventListener("click", () => start("/api/stop"));
  $("btnResume").addEventListener("click", () => start("/api/resume"));
  $("loginButton").addEventListener("click", login);
  $("loginPassword").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
}

boot();
