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
  S.prefs = (await api("/api/prefs")).models || {};
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
    host.append(el("div", {class: "section-card"},
      el("div", {class: "card-title collapsible-header", onclick: () => body.classList.toggle("collapsed")},
        el("span", {text: f.prefs_title || ("Model Paths (" + f.display_name + ")")})), body));
  }
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
  try {
    const r = await api(path, path === "/api/start" ? runBody() : {});
    showProblems(r.problems);
    if (r.error) $("runStatus").textContent = r.error;
    pollRun();
  } catch (e) { $("runStatus").textContent = e.message; }
}

async function pollRun() {
  try { S.run = await api("/api/run"); } catch (e) { return; }
  const r = S.run, live = r.running || r.state === "starting";
  const label = {idle: "Idle", starting: "Starting", running: "Running", paused: "Paused", completed: "Finished",
                 failed: "Failed", stopped: "Stopped"}[r.state] || r.state;
  $("runStatus").textContent = r.lora ? `${label}: ${r.lora}${r.stage ? " · " + r.stage : ""}` : label;
  $("runBadge").textContent = label;
  $("runBadge").dataset.state = r.state;
  $("btnStart").disabled = live;
  $("btnPause").disabled = !r.running;
  $("btnStop").disabled = !live;
  $("btnResume").style.display = !live && r.resume ? "" : "none";
}

function followLog() {
  const src = new EventSource(`/api/log?since=${S.logSince}&t=${encodeURIComponent(TOKEN)}`);
  const out = $("console");
  src.onmessage = (e) => {
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
  src.onerror = () => { src.close(); setTimeout(followLog, 3000); };
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
    const media = ["wav", "mp3", "flac"].includes(ext) ? el("audio", {controls: true, preload: "none", src}) :
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
    const media = ["wav", "mp3", "flac"].includes(ext2) ? el("audio", {controls: true, preload: "none", src}) :
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
