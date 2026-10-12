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
  S.uploadRoot = await api("/api/upload/root").catch(() => null);   // before the forms: where Choose… starts
  const pr = await api("/api/prefs");
  S.prefs = pr.models || {};
  S.webPort = pr.web_port;
  S.portNow = pr.port_now;
  S.onPod = pr.on_pod;
  S.passwordOn = pr.password_on;
  S.portFixed = pr.port_fixed;
  S.outputDefault = pr.output_default || "";
  S.outputPref = pr.lora_output_dir || "";
  S.gpu = {chosen: pr.cuda_device || "", list: pr.gpus || [], envLocked: pr.gpu_env_locked};
  S.all = st.values || {};
  S.imageFolder = st.image_folder || "";
  S.capSaved = st.captions || {};
  S.prepSaved = st.prep || {};
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
  try {                                     // the latest saved, in case another open page changed it
    const st = await api("/api/state");
    if (st && st.values && st.values[key]) S.all[key] = st.values[key];
  } catch (e) { /* keep what this page has */ }
  S.family = key;
  S.fam = S.schema.families.find((f) => f.key === key);
  $("familySelect").value = key;
  if (S.all[key]) {
    S.values = {...defaults(), ...S.all[key]};
  } else {                              // a first visit takes the family's first preset, as the desktop does
    S.values = defaults();
    const first = S.fam.form.presets[0];
    if (first) {
      S.values = (await api("/api/preset", {family: key, name: first.name, values: S.values})).values;
      S.values.__preset = first.name;
    }
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
  // its kind sets its layout (style.css): boxes in the grid, switches as tiles, a choice of pills, files and notes
  // across the card - from the schema's row kind, so every model's form lays out the same way
  const kind = {check: "check", radio: "radio", note: "note", text: "wide", path: "wide", folder: "wide"}[r.kind] || "field";
  const wrap = el("div", {class: "form-row kind-" + kind, "data-key": r.key});
  if (r.kind === "note") { wrap.append(el("div", {class: "field-hint", text: r.label})); return wrap; }
  const v = S.values[r.key];
  let input;
  const set = (val) => onValue(r, val);
  if (r.kind === "blocks") return blockPicker(r, wrap);
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
      // a saved value in another case is the same choice ("auto" saved before the box read "Auto")
      const same = v != null && r.choices.find((c) => c.toLowerCase() === String(v).toLowerCase());
      const val = same || v;
      if (same && same !== v) S.values[r.key] = same;
      const opts = r.choices.includes(String(val)) || !val ? r.choices : [String(val), ...r.choices];
      for (const c of opts) input.append(el("option", {value: c, text: c}));
      input.value = String(val ?? "");
    } else if (r.kind === "text") {
      input = el("textarea", {id: "f_" + r.key, rows: 4, oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
    } else if (r.kind === "combo") {
      // a dropdown of the usual values, as the desktop's, with Other… for a value of your own (a browser's
      // suggestion list only shows what matches what's typed, so with 1.0 in the box it showed nothing)
      const other = el("input", {type: "text", class: "combo-other", placeholder: "Your value",
        oninput: (e) => set(e.target.value)});
      input = el("select", {id: "f_" + r.key, onchange: (e) => {
        const custom = e.target.value === "__other";
        other.style.display = custom ? "" : "none";
        if (custom) { other.value = ""; other.focus(); } else set(e.target.value);
      }});
      fillCombo(input, r.choices, v, r.labels);
      other.style.display = "none";
      input._other = other;
    } else {
      input = el("input", {type: "text", id: "f_" + r.key, oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
    }
    const row = el("div", {class: "field-row"}, el("span", {class: "field-label", text: r.label + ":"}), input);
    if (input._other) row.append(input._other);
    let upStatus = null;
    if (r.kind === "path" || r.kind === "folder") {
      const startAt = r.start === "output" ? (S.values.LORA_OUTPUT_DIR || "") :
        r.start === "data" ? ((S.uploadRoot && S.uploadRoot.root) || S.imageFolder) : S.imageFolder;
      row.append(el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
        pick(input.value || startAt, r.kind === "folder", (p) => { input.value = p; set(p); })}));
      if (r.upload) {
        const up = uploadButton(r, input, set);
        row.append(up.btn);
        upStatus = up.status;
      }
    }
    if (r.key === "LORA_OUTPUT_DIR") {
      // empty = the default from Preferences; a folder picked here is this model's own (kept per model)
      input.placeholder = "Default: " + (S.outputDefault || "output_loras inside Fizgig");
      row.append(el("button", {class: "btn", text: "Reset to default",
        title: "Back to the LoRA output folder in Preferences", onclick: () => { input.value = ""; set(""); }}));
    }
    if (r.suffix) row.append(el("span", {class: "field-hint", text: r.suffix}));
    wrap.append(row);
    if (upStatus) wrap.append(upStatus);
  }
  wrap.append(el("div", {class: "field-hint row-note", id: "note_" + r.key}));
  if (r.hint) wrap.append(el("div", {class: "field-hint", text: r.hint}));
  return wrap;
}

// preview settings the two apps share per model (fizgig/web/live.py SHARED_SAMPLE_KEYS): a change here goes to the
// desktop app too - its boxes if it's open, its saved settings if not
const SHARED_SAMPLE_KEYS = ["SAMPLE_STEPS", "SAMPLE_CFG_SCALE", "SAMPLE_EVERY_N_EPOCHS", "FAMILY_TURBO_STRENGTH",
  "FAMILY_TURBO_STEPS", "FAMILY_TURBO_PACE"];
S.toDesktop = {};
const sendDesktopSoon = debounce(() => {
  for (const [family, values] of Object.entries(S.toDesktop))
    api("/api/desktop/samples", {family, values}).catch(() => {});
  S.toDesktop = {};
}, 800);
function shareSamples(keys) {
  const k = keys.filter((x) => SHARED_SAMPLE_KEYS.includes(x));
  if (!k.length) return;
  const m = S.toDesktop[S.family] = S.toDesktop[S.family] || {};
  for (const x of k) m[x] = S.values[x];
  sendDesktopSoon();
}

// Model Area to Train > Custom: the model's blocks as ticks (All / None / an area's blocks), as the desktop's
function blockPicker(r, wrap) {
  const sel = () => ({...(S.values[r.key] || {})});
  const ticks = {};
  const put = (m) => { onValue(r, m); for (const [id, t] of Object.entries(ticks)) t.checked = !!m[id]; count(); };
  const n = el("span", {class: "field-hint"});
  const count = () => { const m = S.values[r.key] || {}; n.textContent = `${Object.values(m).filter(Boolean).length} chosen`; };
  const all = (on) => { const m = {}; for (const g of r.groups) for (const b of g.blocks) m[b.id] = on; put(m); };
  wrap.append(el("div", {class: "field-row block-head"}, el("b", {text: "Select blocks to train:"}),
    el("button", {class: "btn btn-sm", text: "All", onclick: () => all(true)}),
    el("button", {class: "btn btn-sm", text: "None", onclick: () => all(false)}),
    ...Object.entries(r.areas || {}).map(([name, ids]) => el("button", {class: "btn btn-sm", text: name, onclick: () => {
      const m = {}; for (const g of r.groups) for (const b of g.blocks) m[b.id] = ids.includes(b.id); put(m); }})), n));
  for (const g of r.groups) {
    const row = el("div", {class: "block-row"}, el("span", {class: "block-group", text: g.label + ":"}));
    for (const b of g.blocks) {
      const t = ticks[b.id] = el("input", {type: "checkbox", onchange: (e) => { const m = sel(); m[b.id] = e.target.checked; onValue(r, m); count(); }});
      t.checked = !!(S.values[r.key] || {})[b.id];
      row.append(el("label", {class: "block-tick"}, t, el("span", {text: b.label})));
    }
    wrap.append(row);
  }
  count();
  return wrap;
}

function onValue(r, val) {
  S.values[r.key] = val;
  shareSamples([r.key]);
  if (r.key === "FAMILY_TURBO_STRENGTH") S.turboByUser = true;   // the steps / CFG it moves go to the desktop too
  const extra = r.sets && r.sets[val];            // a kind of training that also sets others (Fine-tune's LR)
  if (extra) { Object.assign(S.values, extra); syncInputs(Object.keys(extra)); }
  if (r.timesteps && r.timesteps[val]) {           // a Model Area with its own timestep window
    [S.values.MIN_TIMESTEP, S.values.MAX_TIMESTEP] = r.timesteps[val];
    syncInputs(["MIN_TIMESTEP", "MAX_TIMESTEP"]);
  }
  refreshStateSoon();
  saveStateSoon();
}

// a combo row's dropdown: the usual values, a value of your own kept as an option, and Other…
function fillCombo(sel, choices, v, labels) {
  sel.innerHTML = "";
  const val = v == null ? "" : String(v);
  const opts = val && !choices.includes(val) ? [...choices, val] : choices;
  opts.forEach((c, i) => sel.append(el("option", {value: c, text: (labels && labels[i]) || c})));
  sel.append(el("option", {value: "__other", text: "Other…"}));
  sel.value = val || (choices[0] ?? "");
}

function syncInputs(keys) {
  for (const k of keys) {
    const i = $("f_" + k);
    if (!i) continue;
    if (i.tagName === "SELECT" && [...i.options].some((o) => o.value === "__other")) {
      const row = (S.fam.form.sections.flatMap((s) => s.rows).find((r) => r && r.key === k)) || {choices: []};
      fillCombo(i, row.choices, S.values[k], row.labels);
      continue;
    }
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
  turboFollow();
}

// Steps and CFG follow the Turbo LoRA, as the desktop's Samples tab: its own (Qwen: 8 steps, CFG 1) while it renders -
// file set in Preferences AND strength above 0 - else the plain model's (Qwen: 25, CFG 3). They switch on a first visit
// or when the turbo turned on or off since last seen (the file set, or the strength moved across 0), and only a box
// still on the other side's default, so a value you typed is never replaced.
function turboFollow() {
  let row = null;
  for (const s of S.fam.form.sections) for (const r of s.rows) if (r.turbo) row = r;
  if (!row) return;
  const raw = S.values.FAMILY_TURBO_STRENGTH;
  const strength = parseFloat(raw === undefined || raw === "" ? row.default : raw);
  const live = !!S.state.turbo_file && strength > 0;
  if (S.values.__turbo_live === live) return;
  const want = live ? row.turbo.on : row.turbo.off, other = live ? row.turbo.off : row.turbo.on;
  const moved = Object.keys(want).filter((k) => S.values[k] === undefined || S.values[k] === "" ||
    parseFloat(S.values[k]) === parseFloat(other[k]));          // "3.0" is the default 3
  for (const k of moved) S.values[k] = want[k];
  S.values.__turbo_live = live;
  syncInputs(moved);
  if (S.turboByUser) shareSamples(moved);
  S.turboByUser = false;
  saveStateSoon();
}
const refreshStateSoon = debounce(refreshState, 250);

async function renderPresets() {
  const sel = $("presets");
  const fam = S.family;
  const list = await api("/api/presets?family=" + encodeURIComponent(fam)).catch(() =>
    ({builtin: S.fam.form.presets.map((p) => p.name), user: []}));
  if (fam !== S.family) return;
  sel.innerHTML = "";
  // the preset last picked shows (a first visit applies the first, as the desktop); "Choose a preset…" before any
  sel.append(el("option", {value: "", text: "Choose a preset…"}));
  for (const n of list.builtin) sel.append(el("option", {value: n, text: n}));
  if (list.user.length) sel.append(el("optgroup", {label: "Your presets"}, ...list.user.map((n) => el("option", {value: n, text: n}))));
  const names = [...list.builtin, ...list.user];
  sel.value = names.includes(S.values.__preset) ? S.values.__preset : "";
  sel.onchange = async () => {
    const name = sel.value;
    if (!name) return;
    S.values = (await api("/api/preset", {family: S.family, name, values: S.values})).values;
    S.values.__preset = name;
    renderSections("samples", $("samplesForm"));
    renderSections("training", $("trainingForm"));
    await refreshState();
    saveState();
  };
}

// ---- Preferences: every family's model files, from its description -----------------------------------------
function renderPrefs() {
  const host = $("prefsForm");
  host.innerHTML = "";
  host.append(webPortCard());
  host.append(dataFolderCard());
  host.append(outputFolderCard());
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
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
          pick(input.value, false, (p) => { input.value = p; input.dispatchEvent(new Event("change")); })}));
      if (m.repo && m.path) {
        row.append(el("a", {class: "btn", target: "_blank", rel: "noopener", text: "Download",
          href: `https://huggingface.co/${m.repo}/blob/main/${m.path}`}));
      }
      const hint = m.hint || m.note;
      body.append(el("div", {class: "pref-file"}, row, hint ? el("div", {class: "field-hint", text: hint}) : null));
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
    el("option", {value: "", text: "System default (GPU 0)"}),
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
    if (!S.onPod) $("imageFolder").placeholder = d.root + (d.root.includes("\\") ? "\\" : "/") + "my_subject";
    const box = $("dataFolder");
    if (box && document.activeElement !== box) box.value = d.set ? d.root : "";
    $("uploadWhere").textContent = `It goes into ${d.root}, in a folder with the name above. ` +
      (d.set ? "That folder is set in Preferences (Training data folder). " :
        "That is Fizgig's own training_data folder; choose another in Preferences (Training data folder). ") +
      "Uploading again with the same name adds to that folder.";
  }).catch(() => {});
}

// the default LoRA output folder: where a model's LoRAs go until you pick its own on the Training tab
function outputFolderCard() {
  const input = el("input", {type: "text", class: "w-path", value: S.outputPref || "",
    placeholder: "output_loras inside Fizgig"});
  const hint = el("div", {class: "field-hint", text: "Where a model's LoRAs and checkpoints go until you pick a " +
    "folder for it on the Training tab (that one is kept for that model; Reset to default comes back here). Leave " +
    "it empty for output_loras inside Fizgig. The desktop app's Preferences > Output Directories > LoRA output is " +
    "the same setting."});
  const save = async () => {
    try {
      await api("/api/prefs", {changes: {lora_output_dir: input.value.trim()}});
      const pr = await api("/api/prefs");
      S.outputDefault = pr.output_default; S.outputPref = pr.lora_output_dir;
      const f = $("f_LORA_OUTPUT_DIR");
      if (f) f.placeholder = "Default: " + S.outputDefault;
      hint.dataset.saved = "1";
    } catch (e) { hint.textContent = e.message; }
  };
  input.addEventListener("change", save);
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "LoRA output folder"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Folder:"}), input,
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
          pick(input.value || S.outputDefault, true, (p) => { input.value = p; save(); })})), hint));
}

function dataFolderCard() {
  const input = el("input", {type: "text", class: "w-path", id: "dataFolder", value: (S.uploadRoot && S.uploadRoot.set) ?
    S.uploadRoot.root : "", placeholder: "Fizgig's own training_data folder"});
  const save = async () => {
    try { await api("/api/prefs", {changes: {input_dataset_dir: input.value.trim()}}); showUploadRoot(); loadDatasets(); }
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
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
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
        el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint, logoutRow()));
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
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint,
      logoutRow()));
}

// with a password: log this device out, or every device (a phone left logged in, a password you shared)
function logoutRow() {
  if (!S.passwordOn) return null;
  const done = () => { showLogin(); };
  return el("div", {class: "logout-row"},
    el("div", {class: "field-row"},
      el("button", {class: "btn", text: "Log out", onclick: () => api("/api/logout", {}).then(done)}),
      el("button", {class: "btn btn-warning", text: "Log out everywhere", onclick: () =>
        api("/api/logout_all", {}).then(done)})),
    el("div", {class: "field-hint", text: "Log out everywhere ends every device's login, this one too: a phone left " +
      "logged in, say. Each one asks for the password again. If someone else knows the password, change it too " +
      "(the desktop app's Preferences > Web UI), or they can just log in again."}));
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
  // only the model on show: another open page (a phone, another tab) keeps the models it changed
  api("/api/state", {family: S.family, image_folder: S.imageFolder,
    model_values: {family: S.family, values: S.values}}).catch(() => {});
}
const saveStateSoonInner = debounce(() => { S.saveWaiting = false; saveState(); }, 600);
const saveStateSoon = () => { S.saveWaiting = true; saveStateSoonInner(); };

// ---- the run -----------------------------------------------------------------------------------------------
const runBody = () => ({family: S.family, values: S.values, image_folder: S.imageFolder});

function showProblems(list, okText) {
  const box = $("problems"), bad = (list || []).length;
  box.classList.toggle("ok", !bad && !!okText);
  box.textContent = bad ? "Can't start yet:\n• " + list.join("\n• ") : (okText || "");
}

async function check() {
  try {
    const p = await api("/api/plan", runBody());
    showProblems(p.problems, "✓ All good - nothing missing, Start Training will work. It will run: " +
      p.stages.map((s) => s.name).join(" → ") + ".");
  } catch (e) { showProblems([e.message]); }
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
  const deskBusy = d && (d.running || ["running", "pausing"].includes(d.state));
  // a run paused there is on show too, but the GPU is free: Start still begins a run here
  const deskPaused = d && !deskBusy && d.state === "paused";
  // ...and a run it just finished stays on show (Finished / Failed / Stopped) until either side starts another
  const deskDone = d && !deskBusy && !deskPaused && d.last_result && d.last_result.at > (r.ended_at || 0);
  const mode = !live && (deskBusy || deskPaused || deskDone) ? "desktop" : "web";
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
    $("runSource").textContent = (deskPaused ?
      `This is a run paused in the desktop app (${d.family}), not your settings above: ${d.lora} in ` +
      `${d.output_dir}. Resume carries it on; Start Training begins your run here instead - nothing is ` +
      "using the GPU while it's paused." : deskDone ?
      `The last run started in the desktop app (${d.family}) ended. Its previews and checkpoints are below; ` +
      "Start begins a new run here." :
      `This run was started in the desktop app (${d.family}). Pause, Stop and Resume here work on it. Start is ` +
      "off until it finishes, so the two don't share the GPU." +
      (d.unresponsive ? " The desktop app isn't answering right now (busy for a moment?), so Pause, Stop and " +
        "Resume wait until it does." : "")) + ack;
    $("runBadge").textContent = label;
    $("runBadge").dataset.state = state;
    $("runCardBadge").textContent = label;
    $("runCardBadge").dataset.state = state;
    showProgress(deskDone ? d.last_result.progress : d.progress, state, deskDone ? d.last_result.lora : d.lora,
      deskDone ? "" : d.stage);
    $("btnStart").disabled = !(deskDone || deskPaused);
    $("btnPause").disabled = !(d.running && d.state === "running");
    $("btnStop").disabled = !d.running;
    $("btnResume").style.display = !d.running && d.state === "paused" ? "" : "none";
  } else {
    const label = RUN_LABELS[r.state] || r.state;
    $("runStatus").textContent = r.lora ? `${label}: ${r.lora}${r.stage ? " · " + r.stage : ""}` : label;
    $("runSource").textContent = d ? "The desktop app is open too. A run started there shows here." : "";
    $("runBadge").textContent = label;
    $("runBadge").dataset.state = r.state;
    $("runCardBadge").textContent = label;
    $("runCardBadge").dataset.state = r.state;
    showProgress(r.progress, r.state, r.lora, r.stage);
    $("btnStart").disabled = live;
    $("btnPause").disabled = !r.running;
    $("btnStop").disabled = !live;
    $("btnResume").style.display = !live && r.resume ? "" : "none";
  }
  renderQueue(r.queue || []);
  renderWebQueue(r.web_queue || {items: []});
}

// the web UI's own queue: reorder, remove, start or hold
function renderWebQueue(q) {
  const items = q.items || [];
  $("webQueueCard").style.display = items.length ? "" : "none";
  const badge = $("webQueueBadge");
  badge.textContent = !items.length ? "" : q.held ? "Held" : q.waiting ? "Waiting" : `${items.length} queued`;
  badge.dataset.state = q.held ? "paused" : q.waiting ? "pausing" : "running";
  $("webQueueState").textContent = q.held ? q.reason : q.waiting ? q.waiting :
    items.length ? "The first starts as soon as nothing else is training." : "";
  $("webQueueStart").style.display = q.held ? "" : "none";
  $("webQueueHold").style.display = q.held || !items.length ? "none" : "";
  const famName = (k) => ((S.schema.families.find((f) => f.key === k) || {}).display_name) || k;
  const host = $("webQueueList");
  host.innerHTML = "";
  items.forEach((it, i) => host.append(el("div", {class: "checkpoint-row queue-row"},
    el("span", {class: "checkpoint-name", text: `${i + 1}. ${it.lora}`}),
    el("span", {class: "field-hint", text: `${famName(it.family)} · ${it.image_folder} · queued ${it.queued_at}`}),
    el("span", {class: "queue-actions"},
      el("button", {class: "btn btn-sm", text: "↑", title: "Earlier", disabled: i === 0,
        onclick: () => api("/api/queue/move", {id: it.id, delta: -1}).then(pollRun)}),
      el("button", {class: "btn btn-sm", text: "↓", title: "Later", disabled: i === items.length - 1,
        onclick: () => api("/api/queue/move", {id: it.id, delta: 1}).then(pollRun)}),
      el("button", {class: "btn btn-sm", text: "Remove", onclick: () => api("/api/queue/remove", {id: it.id}).then(pollRun)})))));
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
  let s;
  try { s = await api("/api/stats"); } catch (e) { return; }     // the server restarts
  const gb = (b) => (b / 2 ** 30).toFixed(1);
  const meter = (label, value, frac, warn) => el("div", {class: "meter" + (warn ? " warn" : "")},
    el("div", {class: "meter-head"}, el("span", {text: label}), el("span", {text: value})),
    el("div", {class: "meter-bar"}, el("span", {style: `width:${Math.max(0, Math.min(100, frac * 100)).toFixed(1)}%`})));
  const host = $("statusBar");
  host.innerHTML = "";
  host.append(el("div", {class: "gpu-name", text: s.device_name}));
  if (s.util !== undefined && s.util !== null) host.append(meter("GPU", `${Math.round(s.util)}%` +
    (s.temp ? ` · ${Math.round(s.temp)}°C` : ""), s.util / 100, s.temp >= 85));
  host.append(meter("VRAM", `${gb(s.vram[0])} / ${gb(s.vram[1])} GB`, s.vram[1] ? s.vram[0] / s.vram[1] : 0,
    s.vram[1] && s.vram[0] / s.vram[1] > 0.95));
  host.append(meter("RAM", `${gb(s.ram[0])} / ${gb(s.ram[1])} GB`, s.ram[1] ? s.ram[0] / s.ram[1] : 0,
    s.ram[1] && s.ram[0] / s.ram[1] > 0.95));
}

// the run's figures (the same from a web run and from a desktop run): the bar, the tiles, the sidebar
function showProgress(p, state, name, extraNote) {
  const pct = p ? p.percent : (["completed"].includes(state) ? 100 : 0);
  for (const id of ["runBar", "sideRunBar"]) $(id).style.width = pct.toFixed(1) + "%";
  for (const id of ["runBar", "sideRunBar"]) $(id).parentElement.dataset.state = state;
  $("runName").textContent = name || "";
  $("sideRunName").textContent = name || "No run yet";
  const tiles = $("runTiles");
  tiles.innerHTML = "";
  const tile = (k, v) => v ? tiles.append(el("div", {class: "run-tile"}, el("span", {text: k}), el("b", {text: v}))) : null;
  if (p) {
    tile("Epoch", p.epoch ? `${p.epoch} / ${p.total_epochs}` : "");
    tile("Step", p.total_steps ? `${p.step} / ${p.total_steps}` : "");
    tile("Speed", p.speed);
    tile("Loss", p.loss);
    tile("Time left", p.eta);
  }
  const live = ["running", "starting", "pausing"].includes(state);
  $("sideRunText").textContent = (p && p.note) || extraNote ||
    (p && live && p.total_steps ? `Epoch ${p.epoch || "?"}/${p.total_epochs || "?"} · ${Math.round(p.percent)}%` +
      (p.eta ? ` · ${p.eta} left` : "") : "");
  $("runWhen").textContent = (p && p.note) || "";
}

// the run on show: the one training (here or in the desktop app), else the LoRA name in the form
const currentRunName = () => (S.mode === "desktop" && S.run.desktop ? S.run.desktop.lora : S.run.lora) ||
  S.values.LORA_NAME || "";

// what a preview is, from its name (the server reads it): "Epoch 4 · Prompt 1 · strength 0.5"
function previewLabel(it) {
  if (it.epoch == null) return it.name;
  return [`Epoch ${it.epoch}`, `Prompt ${it.idx + 1}`, it.strength != null && `strength ${it.strength}`]
    .filter(Boolean).join(" · ");
}
function previewFigure(it, prompts) {
  const prompt = it.idx != null && prompts ? prompts[it.idx] : "";
  return el("figure", {class: "gallery-item", title: it.name}, mediaFor(it),
    el("figcaption", {}, el("b", {text: previewLabel(it)}), prompt ? el("span", {class: "preview-prompt", text: prompt}) : null,
      el("span", {class: "preview-file", text: it.name})));
}

// prompts as rows, epochs as columns (full-strength pictures; a slider's strip is its own picture)
function previewGrid(items, prompts) {
  const full = items.filter((it) => it.epoch != null && it.strength == null);
  const epochs = [...new Set(full.map((it) => it.epoch))].sort((a, b) => a - b);
  const idxs = [...new Set(full.map((it) => it.idx))].sort((a, b) => a - b);
  const at = new Map(full.map((it) => [it.epoch + ":" + it.idx, it]));
  const table = el("table", {class: "prompt-grid"});
  table.append(el("tr", {}, el("th", {}), ...epochs.map((e) => el("th", {text: `Epoch ${e}`}))));
  for (const i of idxs) {
    table.append(el("tr", {},
      el("th", {class: "prompt-head"}, el("b", {text: `Prompt ${i + 1}`}),
        prompts[i] ? el("span", {class: "preview-prompt", text: prompts[i]}) : null),
      ...epochs.map((e) => { const it = at.get(e + ":" + i);
        return el("td", {}, it ? el("figure", {class: "gallery-item", title: it.name}, mediaFor(it),
          el("figcaption", {class: "visually-hidden"}, el("b", {text: previewLabel(it)}),
            prompts[i] ? el("span", {text: " · " + prompts[i]}) : null)) : null); })));
  }
  return el("div", {class: "gallery prompt-grid-wrap"}, table);
}

async function loadGallery() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR;
  const host = $("gallery");
  if (!out) { host.innerHTML = ""; $("galleryControls").style.display = "none"; return; }
  const sep = out.includes("\\") ? "\\" : "/";
  const dir = out + sep + "sample";
  // the run picked here stays picked until another run starts
  const cur = currentRunName();
  if (S.galleryFor !== cur) { S.galleryRun = ""; S.galleryFor = cur; }
  const want = S.galleryRun || cur;
  const get = (run) => api(`/api/samples?dir=${encodeURIComponent(dir)}&run=${encodeURIComponent(run)}`)
    .catch(() => ({items: [], runs: [], prompts: []}));
  let g = await get(want);
  let run = want;
  if (!g.items.length && g.runs.length && !g.runs.includes(want)) { run = g.runs[0]; g = await get(run); }
  const sel = $("galleryRun");
  sel.innerHTML = "";
  for (const r of g.runs) sel.append(el("option", {value: r, text: r + (r === cur ? " (this run)" : "")}));
  sel.value = run;
  $("galleryControls").style.display = g.runs.length ? "" : "none";
  const view = S.galleryView || "list";
  document.querySelectorAll("#galleryView button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  host.innerHTML = "";
  host.classList.toggle("gallery-as-grid", view === "grid");
  if (view === "grid" && g.items.length) host.append(previewGrid(g.items, g.prompts || []));
  else for (const it of g.items.slice(0, 120)) host.append(previewFigure(it, g.prompts || []));
  const others = g.runs.filter((r) => r !== run).length;
  $("galleryInfo").textContent = g.items.length ?
    `${g.items.length} preview${g.items.length === 1 ? "" : "s"} of ${run} in ${dir}` +
    (others ? ` · ${others} other run${others === 1 ? "" : "s"} have previews here too (pick one above)` : "") :
    "No previews yet. They appear here as the run makes them.";
}

async function loadCheckpoints() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR;
  const host = $("checkpoints");
  if (!out) { host.innerHTML = ""; return; }
  const r = await api("/api/checkpoints?dir=" + encodeURIComponent(out)).catch(() => ({items: []}));
  host.innerHTML = "";
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const run = currentRunName();
  const mine = r.items.filter((it) => it.run === run);
  const shown = S.ckptAll ? r.items : mine;
  for (const it of shown) {
    const href = `/api/download?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: it.name}),
      el("span", {class: "field-hint", text: `${it.kind === "finetune" ? "Fine-tune (whole model)" : "LoRA"} · ` +
        `${size(it.size)} · ${new Date(it.time * 1000).toLocaleString()}`}),
      el("a", {class: "btn btn-sm btn-primary", href, download: it.name, text: "Download"})));
  }
  const others = r.items.length - mine.length;
  $("checkpointsInfo").replaceChildren(
    document.createTextNode(S.ckptAll ? `All ${r.items.length} saved in ${out}, newest first. ` :
      mine.length ? `${mine.length} saved by ${run} in ${out}, newest first. ` :
      `Nothing saved by ${run || "this run"} yet: checkpoints appear here as the run saves them. `),
    others ? el("a", {href: "#", class: "inline-link", text: S.ckptAll ? `Only ${run}'s` :
      `Show the ${others} from other runs in this folder`,
      onclick: (e) => { e.preventDefault(); S.ckptAll = !S.ckptAll; loadCheckpoints(); }}) : document.createTextNode(""));
}

// ---- Start tab: your training data (the folders in the training data folder) ---------------------------------
async function loadDatasets() {
  try { S.datasets = await api("/api/datasets"); } catch (e) { return; }
  $("datasetsDesc").textContent = `The folders in ${S.datasets.root}. Click one to train on it; ` +
    "upload a new one below, or pick any folder with Choose… above.";
  $("datasetsSearchRow").style.display = S.datasets.total > 8 ? "" : "none";
  renderDatasets();
}

// a dataset's pictures in the viewer, stepping through them all
async function datasetPictures(d) {
  const r = await api(`/api/captions?folder=${encodeURIComponent(d.path)}&ext=${encodeURIComponent(capExt())}`)
    .catch(() => ({items: []}));
  const q = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
  const pics = r.items.filter((it) => /\.(png|jpe?g|webp|bmp|avif|gif)$/i.test(it.name))
    .map((it) => ({src: q(it.path), name: it.name + (it.caption ? " · " + it.caption : "")}));
  if (!pics.length) return;
  S.lb.list = pics;
  lightboxShow(0);
}

function renderDatasets() {
  const r = S.datasets;
  if (!r) return;
  const q = $("datasetsSearch").value.trim().toLowerCase();
  const host = $("datasetsGrid");
  host.innerHTML = "";
  const norm = (p) => (p || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  const list = r.datasets.filter((d) => !q || d.name.toLowerCase().includes(q));
  const shown = q || S.datasetsAll ? list.slice(0, 300) : list.slice(0, 12);     // the newest twelve, unless asked
  for (const d of shown) {
    const what = [d.pictures && `${d.pictures} picture${d.pictures === 1 ? "" : "s"}`,
      d.clips && `${d.clips} clip${d.clips === 1 ? "" : "s"}`, d.songs && `${d.songs} song${d.songs === 1 ? "" : "s"}`]
      .filter(Boolean).join(" · ");
    const total = d.pictures + d.clips + d.songs;
    const caps = d.captioned >= total ? "all captioned" : d.captioned ? `${d.captioned} of ${total} captioned` : "no captions yet";
    const mosaic = el("div", {class: "dataset-mosaic n" + Math.min(4, d.thumbs.length)},
      d.thumbs.length ? d.thumbs.map((p) => el("img", {loading: "lazy", alt: "",
        src: `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`})) :
        el("span", {class: "dataset-kind", text: d.songs ? "♪" : d.clips ? "▶" : ""}));
    const choose = () => {
      if (norm(d.path) !== norm(S.imageFolder)) {
        $("imageFolder").value = d.path;
        $("imageFolder").dispatchEvent(new Event("change"));
      }
      renderDatasets();
    };
    const act = (text, title, fn) => el("button", {class: "btn btn-sm", text, title, onclick: (e) => {
      e.stopPropagation(); fn(); }});
    // the tile chooses the folder; its buttons also open it in Captions, or show its pictures big
    const card = el("div", {class: "dataset-card" + (norm(d.path) === norm(S.imageFolder) ? " active" : ""),
      title: d.path, role: "button", tabindex: "0", onclick: choose,
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } }}, mosaic,
      el("div", {class: "dataset-meta"}, el("b", {text: d.name}), el("span", {text: what}),
        el("span", {class: d.captioned >= total ? "ok" : "warn", text: caps}),
        el("span", {class: "dataset-when", text: d.time ? new Date(d.time * 1000).toLocaleDateString() : ""}),
        el("div", {class: "dataset-actions"},
          act("Captions", "Train on this folder and open it on the Captions tab", () => { choose(); showTab("captions"); }),
          d.pictures ? act("Pictures", "Look through its pictures, big", () => datasetPictures(d)) : null)));
    host.append(card);
  }
  if (shown.length < list.length) host.append(el("button", {class: "btn dataset-more", text: `Show all ${list.length}`,
    onclick: () => { S.datasetsAll = true; renderDatasets(); }}));
  if (!list.length) host.append(el("div", {class: "field-hint", text: r.total ?
    "No folder matches that." : "No training data here yet. Upload a folder below, or pick one with Choose… above."}));
}

// ---- Runs & outputs: the run history (fizgig/web/history.py) -----------------------------------------------
const RESULT_STATE = {running: "running", finished: "completed", failed: "failed", stopped: "stopped",
                      paused: "paused", unknown: "idle"};
const RESULT_LABEL = {running: "Running", finished: "Finished", failed: "Failed", stopped: "Stopped", paused: "Paused",
                      unknown: "Ended"};

async function loadRuns() {
  let r;
  try { r = await api("/api/runs"); } catch (e) { return; }
  const host = $("runsList");
  host.innerHTML = "";
  const famName = (k) => ((S.schema.families.find((f) => f.key === k) || {}).display_name) || k;
  const dur = (a, b) => { const m = Math.round(((b || Date.now() / 1000) - a) / 60);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`; };
  for (const run of r.runs) {
    const p = run.progress;
    const figs = p ? [p.epoch && `epoch ${p.epoch}/${p.total_epochs}`, p.loss && `loss ${p.loss}`].filter(Boolean).join(" · ") : "";
    const files = el("div", {class: "run-files", style: "display:none"});
    const showFiles = el("button", {class: "btn btn-sm", text: "Previews & checkpoints", onclick: async () => {
      if (files.style.display === "none") {
        files.style.display = "";
        files.textContent = "Loading…";
        files.replaceChildren(...await runFiles(run));
      } else files.style.display = "none";
    }});
    const again = el("button", {class: "btn btn-sm btn-primary", text: "Start again with these settings",
      title: "Loads this run's model, training folder and settings into the form (nothing starts until you press Start)",
      onclick: async () => {
        try {
          const s = await api("/api/runs/settings", {id: run.id});
          if (s.family !== S.family) await chooseFamily(s.family);
          const kept = Object.fromEntries(Object.entries(s.values).filter(([, v]) => v != null));
          S.values = {...S.values, ...kept};
          // a run recorded before it had a Turbo value: its steps and CFG follow the Turbo on now (refreshState)
          if (!("FAMILY_TURBO_STRENGTH" in kept)) delete S.values.__turbo_live;
          S.all[S.family] = S.values;
          if (s.image_folder) { $("imageFolder").value = s.image_folder; S.imageFolder = s.image_folder; }
          renderSections("samples", $("samplesForm"));
          renderSections("training", $("trainingForm"));
          await refreshState();
          saveState();
          showTab("training");
          $("runStatus").textContent = `Loaded the settings of ${run.lora || "that run"}: check them, then Start.`;
        } catch (e) { alertRow.textContent = e.message; }
      }});
    // Delete run: says what goes first (checkpoints, saved states, previews - never the training folder), then deletes
    const delBox = el("div", {class: "run-delete", style: "display:none"});
    const del = el("button", {class: "btn btn-sm btn-danger", text: "Delete run…", onclick: async () => {
      if (delBox.style.display !== "none") { delBox.style.display = "none"; return; }
      delBox.style.display = "";
      delBox.textContent = "Checking…";
      try {
        const w = await api("/api/runs/delete", {id: run.id});
        const gb = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
        const parts = [w.checkpoints && `${w.checkpoints} checkpoint${w.checkpoints > 1 ? "s" : ""}`,
          w.states && `${w.states} saved state${w.states > 1 ? "s" : ""} (for resuming)`,
          w.previews && `${w.previews} preview${w.previews > 1 ? "s" : ""}`].filter(Boolean);
        delBox.replaceChildren(
          el("div", {class: "field-hint", text: (parts.length ?
            `Deletes ${parts.join(", ")} - ${gb(w.bytes)} from ${w.folder} - for good, and takes the run off this ` +
            "list." : "Nothing of this run is left on disk: this takes it off the list.") +
            " Your training folder and its captions are not touched."}),
          el("div", {class: "field-row"},
            el("button", {class: "btn btn-sm btn-danger", text: "Delete", onclick: async () => {
              try { await api("/api/runs/delete", {id: run.id, confirm: true}); loadRuns(); loadOutputs(); }
              catch (e) { delBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
            }}),
            el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { delBox.style.display = "none"; }})));
      } catch (e) { delBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
    }});
    const alertRow = el("div", {class: "field-hint"});
    host.append(el("div", {class: "run-row"},
      el("div", {class: "run-row-head"},
        el("span", {class: "run-badge", "data-state": RESULT_STATE[run.result] || "idle", text: RESULT_LABEL[run.result] || run.result}),
        el("b", {text: run.lora || "(unnamed)"}),
        el("span", {class: "field-hint", text: `${famName(run.family)} · ${run.source === "desktop" ? "desktop app" : "web UI"}`})),
      el("div", {class: "field-hint", text: [run.image_folder && `Trained on ${run.image_folder}`,
        `${new Date(run.started_at * 1000).toLocaleString()} · ${dur(run.started_at, run.ended_at)}`, figs]
        .filter(Boolean).join("  ·  ")}),
      lossChart(run.curve),
      el("div", {class: "field-row"}, showFiles, again, del), delBox, alertRow, files));
  }
  if (!r.runs.length) host.append(el("div", {class: "field-hint", text: "No runs yet. Every run you start, here or in " +
    "the desktop app, is listed here with its results."}));
}

// a run's loss, epoch by epoch, as a small chart (the history keeps each epoch's last average loss)
function lossChart(curve) {
  const pts = (curve || []).filter((p) => Array.isArray(p) && isFinite(p[1]));
  if (pts.length < 2) return null;
  const W = 220, Hh = 44, pad = 3;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = (x) => pad + (W - 2 * pad) * (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0));
  const Y = (y) => pad + (Hh - 2 * pad) * (y1 === y0 ? 0.5 : 1 - (y - y0) / (y1 - y0));
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
  svg.setAttribute("class", "loss-chart");
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "));
  svg.append(line);
  const best = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
  const dot = document.createElementNS(ns, "circle");
  dot.setAttribute("cx", X(best[0])); dot.setAttribute("cy", Y(best[1])); dot.setAttribute("r", 2.6);
  svg.append(dot);
  const t = document.createElementNS(ns, "title");
  t.textContent = pts.map((p) => `epoch ${p[0]}: ${p[1]}`).join("\n");
  svg.append(t);
  return el("div", {class: "loss-chart-box"}, svg, el("span", {class: "field-hint",
    text: `Loss by epoch: ${ys[0]} → ${ys[ys.length - 1]} · lowest ${best[1]} at epoch ${best[0]}`}));
}

// a run's previews and checkpoints, from its output folder
async function runFiles(run) {
  const out = [];
  const sep = (run.output_dir || "").includes("\\") ? "\\" : "/";
  const [g, c] = await Promise.all([
    api(`/api/samples?dir=${encodeURIComponent(run.output_dir + sep + "sample")}&run=${encodeURIComponent(run.lora)}`)
      .catch(() => ({items: [], prompts: []})),
    api("/api/checkpoints?dir=" + encodeURIComponent(run.output_dir)).catch(() => ({items: []}))]);
  const shots = g.items.slice(0, 12);
  if (shots.length) out.push(el("div", {class: "gallery run-gallery"}, shots.map((it) => previewFigure(it, g.prompts))));
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const ckpts = c.items.filter((it) => it.run === run.lora);
  if (ckpts.length) out.push(el("div", {class: "checkpoint-list"}, ckpts.map((it) => el("div", {class: "checkpoint-row"},
    el("span", {class: "checkpoint-name", text: it.name}),
    el("span", {class: "field-hint", text: `${it.kind === "finetune" ? "Fine-tune" : "LoRA"} · ${size(it.size)}`}),
    el("a", {class: "btn btn-sm btn-primary", download: it.name, text: "Download",
      href: `/api/download?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`})))));
  if (!out.length) out.push(el("div", {class: "field-hint", text: "Nothing saved by this run in " + run.output_dir + "."}));
  return out;
}

// ---- Download your outputs tab ------------------------------------------------------------------------------
async function loadOutputs() {
  const all = !$("outFizgigOnly").checked;
  let r;
  try { r = await api("/api/outputs" + (all ? "?all=1" : "")); } catch (e) { $("outSummary").textContent = e.message; return; }
  S.outputs = r;
  renderOutputs();
}

function renderOutputs() {
  const r = S.outputs;
  if (!r) return;
  const q = $("outSearch").value.trim().toLowerCase();
  // saved when: the last N days, or between two dates (the end date counts in full); a run keeps its files in range
  const when$ = $("outWhen").value;
  let lo = 0, hi = Infinity;
  if (when$ === "custom") {
    if ($("outFrom").value) lo = new Date($("outFrom").value + "T00:00:00").getTime() / 1000;
    if ($("outTo").value) hi = new Date($("outTo").value + "T23:59:59").getTime() / 1000;
  } else if (when$ !== "any") lo = Date.now() / 1000 - Number(when$) * 86400;
  const inRange = (f) => f.time >= lo && f.time <= hi;
  const runs = r.runs
    .filter((x) => !q || (x.name + " " + x.folder).toLowerCase().includes(q))
    .map((x) => ({...x, files: x.files.filter(inRange)}))
    .filter((x) => x.files.length);
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const when = (t) => new Date(t * 1000).toLocaleString();
  $("outSummary").textContent = `${r.runs.length} run${r.runs.length === 1 ? "" : "s"} in ${r.folders.length} ` +
    `folder${r.folders.length === 1 ? "" : "s"}` + (q || when$ !== "any" ? ` · ${runs.length} shown` : "") +
    (r.hidden ? ` · ${r.hidden} other LoRA${r.hidden === 1 ? "" : "s"} in these folders hidden` : "") + ".";
  const host = $("outRuns");
  host.innerHTML = "";
  const dl = (f) => el("a", {class: "btn btn-sm btn-primary", download: f.name, text: "Download",
    href: `/api/download?path=${encodeURIComponent(f.path)}&t=${encodeURIComponent(TOKEN)}`});
  const row = (f) => el("div", {class: "checkpoint-row"},
    el("span", {class: "checkpoint-name", text: f.name}),
    el("span", {class: "field-hint", text: `${f.kind === "finetune" ? "Fine-tune (whole model)" : "LoRA"} · ` +
      `${size(f.size)} · ${when(f.time)}`}), dl(f));
  for (const run of runs.slice(0, 200)) {
    const [top, ...rest] = run.files;
    const more = el("div", {class: "checkpoint-list", style: "display:none"}, rest.map(row));
    const body = el("div", {class: "card-content"}, el("div", {class: "checkpoint-list"}, row(top)));
    if (rest.length) body.append(el("button", {class: "btn btn-sm", text: `Show ${rest.length} earlier checkpoint` +
      (rest.length === 1 ? "" : "s"), onclick: (e) => {
        const open = more.style.display === "none";
        more.style.display = open ? "" : "none";
        e.target.textContent = open ? "Hide the earlier checkpoints" : `Show ${rest.length} earlier checkpoint` +
          (rest.length === 1 ? "" : "s");
      }}), more);
    if (run.samples) body.append(el("div", {class: "field-row"}, el("a", {class: "btn btn-sm",
      text: `Download the previews (${run.samples} files, ${size(run.samples_bytes || 0)}, .zip)`,
      href: `/api/outputs/previews?dir=${encodeURIComponent(run.folder)}&name=${encodeURIComponent(run.name)}` +
        `&t=${encodeURIComponent(TOKEN)}`})));
    host.append(el("div", {class: "section-card"},
      el("div", {class: "card-title"}, el("span", {text: run.name})),
      el("div", {class: "card-desc", text: `${run.folder} · last saved ${when(run.time)}`}), body));
  }
  if (!runs.length) host.append(el("div", {class: "section-card"}, el("div", {class: "card-content"},
    el("div", {class: "field-hint", text: r.runs.length ? "Nothing matches that search and date." :
      "Nothing saved yet. A run's LoRAs and checkpoints appear here as it saves them."}))));
}

// a song's waveform (the picture beside it with the same name) above its player, and its length; any other media
// as before
const fmtSecs = (s) => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
function mediaFor(it) {
  const q = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
  const kind = it.name.split(".").pop().toLowerCase();
  if (["wav", "mp3", "flac", "m4a"].includes(kind)) {
    // nothing downloads until it plays (a page of songs is hundreds of MB); the length comes from the server
    const audio = el("audio", {controls: true, preload: "none", src: q(it.path)});
    const wave = el("div", {class: "song-wave-box", title: "Play / pause",
      onclick: () => (audio.paused ? audio.play() : audio.pause())},
      it.image ? el("img", {loading: "lazy", src: q(it.image), alt: "", class: "song-wave"}) :
        el("div", {class: "song-wave song-wave-none", text: "♪"}),
      it.duration != null ? el("span", {class: "song-len", text: fmtSecs(it.duration)}) : null);
    return el("div", {class: "song-media"}, wave, audio);
  }
  if (["mp4", "webm"].includes(kind)) return el("video", {controls: true, preload: "none", src: q(it.path)});
  return el("img", {loading: "lazy", src: q(it.path), alt: it.name});
}
const mediaLabel = (it) => it.name;

// ---- Image Prep tab: resize / PNG / face close-ups, and the Look Filter (fizgig/web/prep_jobs.py) -------------
S.prep = {opts: null, logSize: 0, timer: null, look: null, marked: new Set(), bases: [], page: 0, picking: false};
const thumbOf = (p) => `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
const prepSettings = () => ({
  mode: (document.querySelector('input[name="prepMode"]:checked') || {}).value || "Auto Prep (Face Crops)",
  mp: $("prepMp").value || "1.0", face: $("prepFace").value || "Largest Face", padding: $("prepPadding").value || "20",
  replace: (document.querySelector('input[name="prepReplace"]:checked') || {}).value === "1"});
const savePrepSoon = debounce(() => { S.prepSaved = prepSettings(); api("/api/state", {prep: S.prepSaved}).catch(() => {}); }, 600);

async function loadPrep() {
  if (!S.prep.opts) {
    S.prep.opts = await api("/api/prep/options");
    const o = S.prep.opts, saved = S.prepSaved || {};
    $("prepMp").innerHTML = ""; for (const m of o.megapixels) $("prepMp").append(el("option", {value: m, text: m}));
    $("prepFace").innerHTML = ""; for (const f of o.faces) $("prepFace").append(el("option", {value: f, text: f}));
    $("prepMp").value = saved.mp || "1.0";
    $("prepFace").value = saved.face || "Largest Face";
    if (saved.padding) $("prepPadding").value = saved.padding;
    const mode = saved.mode || "Auto Prep (Face Crops)";
    document.querySelectorAll('input[name="prepMode"]').forEach((r) => { r.checked = r.value === mode; });
    document.querySelectorAll('input[name="prepReplace"]').forEach((r) => { r.checked = r.value === (saved.replace ? "1" : "0"); });
    $("prepNoFaces").style.display = o.face_tools ? "none" : "";
  }
  $("prepFolder").textContent = S.imageFolder ? `Folder: ${S.imageFolder}  (set on the Start tab)` :
    "Choose the training folder on the Start tab first.";
  await prepSummary();
  pollPrep();
  lookPoll();
}

async function prepSummary() {
  const p = prepSettings(), faceOn = p.mode !== "Resize Only";
  document.querySelectorAll(".prep-face select, .prep-face input").forEach((i) => { i.disabled = !faceOn || !S.prep.opts.face_tools; });
  $("prepTest").disabled = !faceOn || !S.prep.opts.face_tools;
  if (!S.imageFolder) { $("prepSummary").textContent = ""; return; }
  const r = await api(`/api/prep/summary?folder=${encodeURIComponent(S.imageFolder)}&mode=${encodeURIComponent(p.mode)}` +
    `&mp=${p.mp}&replace=${p.replace ? 1 : 0}&train_mp=${encodeURIComponent(S.values.DATASET_MEGAPIXELS || "")}`).catch(() => null);
  if (!r) return;
  $("prepAudioOnly").style.display = r.audio_only ? "" : "none";
  $("prepMain").style.display = r.audio_only ? "none" : "";
  $("prepSummary").replaceChildren(...r.lines.map((t) => el("div", {text: t})));
  $("prepMpWarn").textContent = r.warn || "";
  if (!S.prep.running) $("prepRun").textContent = r.count ? `✨ Prepare ${r.count} Image${r.count === 1 ? "" : "s"} Now` :
    "✨ Prepare Images Now";
}

async function pollPrep() {
  clearTimeout(S.prep.timer);
  const r = await api("/api/prep/status?since=" + S.prep.logSize).catch(() => null);
  if (r) {
    if (r.size < S.prep.logSize) { $("prepLog").textContent = ""; S.prep.logSize = 0; }
    if (r.log) { $("prepLog").textContent += r.log; $("prepLog").scrollTop = $("prepLog").scrollHeight; }
    S.prep.logSize = r.size;
    const was = S.prep.running;
    S.prep.running = r.state === "running";
    $("prepRun").disabled = S.prep.running;
    if (S.prep.running) $("prepRun").textContent = "Preparing…";
    $("prepState").textContent = S.prep.running ? "Working on the Fizgig computer: you can leave this page." :
      r.state === "done" ? "Done." : r.state === "failed" ? "It stopped with an error: see the log." : "";
    if (was && !S.prep.running) { prepSummary(); loadDatasets(); }
  }
  if (S.prep.running) S.prep.timer = setTimeout(pollPrep, 1000);
}

async function prepRun() {
  if (!S.imageFolder) { $("prepState").textContent = "Choose the training folder on the Start tab first."; return; }
  const p = prepSettings();
  try {
    await api("/api/prep/run", {folder: S.imageFolder, ...p});
    $("prepLog").textContent = ""; S.prep.logSize = 0;
    pollPrep();
  } catch (e) { $("prepState").textContent = e.message; }
}

function prepTest() {
  pick(S.imageFolder, false, async (path) => {
    const host = $("prepTestResult");
    host.replaceChildren(el("div", {class: "field-hint", text: "Finding faces… (the first time loads the face model)"}));
    try {
      const r = await api("/api/prep/test", {path, face: $("prepFace").value});
      if (!r.faces.length) { host.replaceChildren(el("div", {class: "field-hint", text: "No faces found in that photo."})); return; }
      host.replaceChildren(
        el("div", {class: "field-hint", text: `${r.faces.length} face${r.faces.length === 1 ? "" : "s"} found: ` +
          r.faces.map((f, i) => `face ${i + 1}, ${f.gender === "unknown" ? "?" : f.gender}, ${f.area.toLocaleString()} px` +
            (f.selected ? " (chosen for the crop)" : "")).join("; ") + ". Green = chosen, yellow = the others."}),
        el("img", {class: "prep-test-img", src: r.image, alt: "faces found"}));
    } catch (e) { host.replaceChildren(el("div", {class: "problems", text: e.message})); }
  });
}

// ---- the Look Filter ----
async function lookPickOpen() {
  const host = $("lookPicker");
  if (S.prep.picking) { S.prep.picking = false; host.style.display = "none"; return; }
  if (!S.imageFolder) { $("lookStatus").textContent = "Choose the training folder on the Start tab first."; return; }
  S.prep.picking = true;
  host.style.display = "";
  host.replaceChildren(el("div", {class: "field-hint", text: "Loading the folder's pictures…"}));
  const r = await api(`/api/captions?folder=${encodeURIComponent(S.imageFolder)}&ext=${encodeURIComponent(capExt())}`).catch(() => ({items: []}));
  const pics = r.items.filter((it) => /\.(png|jpe?g|webp|bmp)$/i.test(it.name));
  host.replaceChildren(el("div", {class: "field-hint look-pick-hint", text: "Click the 3 pictures that best nail the look you want."}),
    ...pics.map((it) => el("button", {class: "look-tile pick" + (S.prep.bases.includes(it.path) ? " chosen" : ""),
      title: it.name, onclick: (e) => {
        const i = S.prep.bases.indexOf(it.path);
        if (i >= 0) S.prep.bases.splice(i, 1);
        else if (S.prep.bases.length < 3) S.prep.bases.push(it.path);
        else return;
        e.currentTarget.classList.toggle("chosen", S.prep.bases.includes(it.path));
        lookBasesShow();
      }}, el("img", {loading: "lazy", src: thumbOf(it.path), alt: ""}), el("span", {text: it.name}))));
}

function lookBasesShow() {
  $("lookBases").replaceChildren(...S.prep.bases.map((p) => el("img", {src: thumbOf(p), title: p, alt: ""})),
    el("span", {class: "field-hint", text: `${S.prep.bases.length} of 3 chosen`}));
  $("lookScan").disabled = S.prep.bases.length !== 3;
}

async function lookScan() {
  try {
    await api("/api/look/scan", {folder: S.imageFolder, baselines: S.prep.bases});
    S.prep.picking = false; $("lookPicker").style.display = "none";
    S.prep.marked = new Set(); S.prep.page = 0;
    lookPoll();
  } catch (e) { $("lookStatus").textContent = e.message; }
}

async function lookPoll() {
  clearTimeout(S.prep.lookTimer);
  const r = await api("/api/look/status").catch(() => null);
  if (!r) return;
  S.prep.look = r;
  if (r.state === "running") {
    $("lookStatus").textContent = r.progress[0] ? `Scoring faces… ${r.progress[0]}/${r.progress[1]}` :
      "Loading the face model (the first time downloads about 300 MB)…";
    S.prep.lookTimer = setTimeout(lookPoll, 1000);
  } else if (r.state === "failed") $("lookStatus").textContent = r.error;
  else if (r.state === "done" && r.rows.length) {
    if (!S.prep.bases.length) { S.prep.bases = r.baselines; lookBasesShow(); }
    const faces = r.rows.filter((x) => x.score != null).length;
    $("lookStatus").textContent = `${r.rows.length} picture(s) scored: ${faces} with faces, ${r.rows.length - faces} ` +
      "without. Worst matches first. Click a picture to mark it, or use Auto-Suggest. Scores are saved with the " +
      "dataset: low scorers you keep can ease in with “Warm up look outliers” on the Training tab.";
  }
  lookRender();
}

const LOOK_PAGE = 200;
function lookRender() {
  const r = S.prep.look, host = $("lookRows");
  const rows = r && r.state === "done" && r.folder === S.imageFolder ? r.rows : [];
  $("lookTools").style.display = rows.length ? "" : "none";
  host.innerHTML = "";
  const pages = Math.max(1, Math.ceil(rows.length / LOOK_PAGE));
  S.prep.page = Math.min(S.prep.page, pages - 1);
  $("lookPages").style.display = pages > 1 ? "" : "none";
  $("lookPageInfo").textContent = `Page ${S.prep.page + 1} of ${pages} · ${rows.length} pictures · ${S.prep.marked.size} marked`;
  $("lookPrev").disabled = S.prep.page === 0; $("lookNext").disabled = S.prep.page >= pages - 1;
  $("lookCount").textContent = S.prep.marked.size ? `${S.prep.marked.size} marked` : "";
  $("lookMove").disabled = !S.prep.marked.size;
  for (const x of rows.slice(S.prep.page * LOOK_PAGE, (S.prep.page + 1) * LOOK_PAGE)) {
    const marked = S.prep.marked.has(x.path);
    const tile = el("button", {class: "look-tile" + (marked ? " marked" : ""), style: `--verdict:${x.colour}`,
      title: x.blurb, onclick: () => { S.prep.marked.has(x.path) ? S.prep.marked.delete(x.path) : S.prep.marked.add(x.path); lookRender(); }},
      el("img", {loading: "lazy", src: thumbOf(x.path), alt: ""}),
      el("span", {class: "look-name", text: x.name}),
      el("span", {class: "look-verdict", text: (x.score != null ? `${Math.round(x.score * 100)}% · ` : "") + x.verdict.toUpperCase() +
        (x.baseline ? " · ★ baseline" : "")}),
      el("span", {class: "look-mark", text: marked ? "✕ marked to move out" : ""}));
    host.append(tile);
  }
}

function lookSuggest() {
  const r = S.prep.look;
  if (!r || r.cutoff == null) { $("lookStatus").textContent = "Not enough scored faces for statistics: mark by eye."; return; }
  const newly = r.rows.filter((x) => x.suggest && !S.prep.marked.has(x.path));
  newly.forEach((x) => S.prep.marked.add(x.path));
  $("lookStatus").textContent = `Auto-suggest marked ${newly.length} picture(s) (dataset median ${Math.round(r.median * 100)}%, ` +
    `cutoff ${Math.round(r.cutoff * 100)}%). Review before moving: it flags statistical drift, not certainty.`;
  lookRender();
}

function lookMoveAsk() {
  const n = S.prep.marked.size;
  if (!n) return;
  const sep = S.imageFolder.includes("\\") ? "\\" : "/";
  $("lookConfirm").replaceChildren(el("div", {class: "run-delete"},
    el("div", {class: "field-hint", text: `Move ${n} picture(s), with their captions, out of the dataset to ` +
      `${S.imageFolder}${sep}excluded_by_look? Nothing is deleted: move them back to put them in again.`}),
    el("div", {class: "field-row"},
      el("button", {class: "btn btn-sm btn-danger", text: "Move", onclick: async () => {
        try {
          const r = await api("/api/look/move", {folder: S.imageFolder, paths: [...S.prep.marked], ext: capExt()});
          S.prep.marked = new Set();
          $("lookConfirm").innerHTML = "";
          $("lookStatus").textContent = `Moved ${r.moved} picture(s) to ${r.dest}. ${r.left} remain in the dataset.` +
            (r.errors.length ? " " + r.errors.join(" ") : "");
          await lookPoll(); loadDatasets();
        } catch (e) { $("lookConfirm").replaceChildren(el("div", {class: "problems", text: e.message})); }
      }}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { $("lookConfirm").innerHTML = ""; }}))));
}

// Save preset: a name, then saved for this model in the desktop's format (asks before replacing one)
function presetSaveOpen() {
  const host = $("presetSaveRow");
  if (host.childNodes.length) { host.innerHTML = ""; return; }
  const name = el("input", {type: "text", class: "w-path", placeholder: "A name for this preset",
    value: S.values.__preset && !S.fam.form.presets.some((p) => p.name === S.values.__preset) ? S.values.__preset : ""});
  const msg = el("span", {class: "field-hint"});
  const save = async (overwrite) => {
    const n = name.value.trim();
    if (!n) { msg.textContent = "Type a name first."; return; }
    try {
      const r = await api("/api/preset/save", {family: S.family, name: n, values: S.values, overwrite: !!overwrite});
      if (r.exists) {
        msg.replaceChildren(document.createTextNode(`There's a preset called ${n} already. `),
          el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => save(true)}));
        return;
      }
      S.values.__preset = n;
      saveState();
      host.innerHTML = "";
      await renderPresets();
      $("runStatus").textContent = `Saved the preset ${n}: it's in both apps' preset list for this model.`;
    } catch (e) { msg.textContent = e.message; }
  };
  name.addEventListener("keydown", (e) => { if (e.key === "Enter") save(false); });
  host.append(el("div", {class: "field-row"}, name, el("button", {class: "btn btn-primary", text: "Save", onclick: () => save(false)}),
    el("button", {class: "btn", text: "Cancel", onclick: () => { host.innerHTML = ""; }}), msg));
  name.focus();
}

function wirePrep() {
  document.querySelectorAll('input[name="prepMode"], input[name="prepReplace"]').forEach((r) =>
    r.addEventListener("change", () => { prepSummary(); savePrepSoon(); }));
  for (const id of ["prepMp", "prepFace", "prepPadding"]) $(id).addEventListener("change", () => { prepSummary(); savePrepSoon(); });
  $("prepRun").addEventListener("click", prepRun);
  $("prepTest").addEventListener("click", prepTest);
  $("lookPick").addEventListener("click", lookPickOpen);
  $("lookScan").addEventListener("click", lookScan);
  $("lookSuggest").addEventListener("click", lookSuggest);
  $("lookMove").addEventListener("click", lookMoveAsk);
  $("lookPrev").addEventListener("click", () => { S.prep.page--; lookRender(); });
  $("lookNext").addEventListener("click", () => { S.prep.page++; lookRender(); });
}

// ---- Captions tab ----------------------------------------------------------------------------------------
// The desktop Captions tab's tools, run on the Fizgig machine (fizgig/web/captions.py). Settings are remembered.
let CAP_PAGE = 12;                         // Per page (the desktop's sizes), remembered with the tab's settings
S.cap = {page: 0, items: [], opts: null, lastState: ""};

function capSettings() {
  return {trigger: $("capTrigger").value, model: $("capModel").value, task: $("capTask").value,
    tokens: $("capTokens").value, overwrite: $("capOverwrite").checked, songWords: $("capSongWords").value,
    skipCjk: $("capSkipCjk").checked, perPage: $("capPerPage").value};
}
const saveCapSoon = debounce(() => { S.capSaved = capSettings(); api("/api/state", {captions: S.capSaved}).catch(() => {}); }, 600);
const qwenOn = () => !!S.cap.opts && S.cap.opts.qwen_available && $("capModel").value === S.cap.opts.qwen_model;
const qwenTask = () => (S.cap.opts.qwen_tasks || []).find((t) => t.label === $("capTask").value);

async function loadCaptionOptions() {
  const o = S.cap.opts = await api("/api/captions/options?family=" + encodeURIComponent(S.family));
  const saved = S.capSaved || {};
  const model = $("capModel");
  model.innerHTML = "";
  for (const m of o.models) model.append(el("option", {value: m, text: m}));
  model.value = o.models.includes(saved.model) ? saved.model : o.default_model;
  $("capModelHint").textContent = o.qwen_available ?
    "Qwen3-VL (the Krea 2 text encoder from Preferences) follows an instruction you can edit and writes the best " +
    "training captions. Florence-2 downloads itself the first time." :
    "Set the Krea 2 text encoder in Preferences to caption with Qwen3-VL, which writes better training captions. " +
    "Florence-2 downloads itself the first time.";
  if (saved.trigger !== undefined) $("capTrigger").value = saved.trigger;
  if (saved.overwrite !== undefined) $("capOverwrite").checked = saved.overwrite;
  if (saved.skipCjk !== undefined) $("capSkipCjk").checked = saved.skipCjk;
  if (saved.perPage) { $("capPerPage").value = saved.perPage; CAP_PAGE = Number(saved.perPage) || 12; }
  fillTasks(saved.task);
  if (saved.tokens) $("capTokens").value = saved.tokens;
  $("capSongCard").style.display = o.song_words ? "" : "none";
  // a model that trains on no pictures or clips (ACE-Step): the AI card has nothing to caption, the song card does
  const media = (S.fam && S.fam.media) || ["photo"];
  const pictures = media.includes("photo") || media.includes("clip");
  document.querySelector("#capAll").closest(".section-card").style.display = pictures ? "" : "none";
  for (const id of ["capModel", "capTask", "capTokens"]) $(id).closest(".field-row").style.display = pictures ? "" : "none";
  $("capModelHint").style.display = pictures ? "" : "none";
  if (!pictures) $("capInstrBox").style.display = "none";
  if (o.song_words) $("capSongWords").value = saved.songWords || o.song_words;
}

function fillTasks(want) {
  const o = S.cap.opts, task = $("capTask");
  task.innerHTML = "";
  const labels = qwenOn() ? o.qwen_tasks.map((t) => t.label) : o.florence_tasks;
  for (const l of labels) task.append(el("option", {value: l, text: l}));
  task.value = labels.includes(want) ? want : (qwenOn() ? o.default_qwen_task : "<DETAILED_CAPTION>");
  $("capEditInstr").style.display = qwenOn() ? "" : "none";
  $("capInstrBox").style.display = "none";
}

function onCapTask() {
  const t = qwenOn() && qwenTask();
  if (t) $("capTokens").value = t.tokens;            // each Qwen task suggests its own length
  if ($("capInstrBox").style.display !== "none") openInstr();
  saveCapSoon();
}

function openInstr() {
  const t = qwenTask();
  if (!t) return;
  $("capInstrBox").style.display = "";
  $("capInstr").value = t.instruction;
  $("capInstrHint").textContent = `What Qwen3-VL is told for "${t.label}"` + (t.edited ? " (edited)" : " (as shipped)") +
    ". Saved here, it is used by the desktop app's Captions tab and by the trainer's auto-recaption too.";
  $("capInstrStatus").textContent = "";
}

async function saveInstr(text) {
  const t = qwenTask();
  await api("/api/captions/instruction", {key: t.key, text});
  const keep = $("capTask").value;
  await loadCaptionOptions();
  $("capTask").value = keep;
  openInstr();
  $("capInstrStatus").textContent = "Saved.";
}

const capExt = () => {                    // Training > Other Options > Caption Extension (.txt unless set)
  let e = String(S.values.DATASET_CAPTION_EXT || ".txt").trim();
  if (e && !e.startsWith(".")) e = "." + e;
  return /^\.[A-Za-z0-9_]{1,12}$/.test(e) ? e : ".txt";
};

function capJobBody(extra = {}) {
  const t = qwenOn() && qwenTask();
  return {family: S.family, folder: S.imageFolder, model: $("capModel").value, task: $("capTask").value, ext: capExt(),
    instruction: t ? t.instruction : "", max_tokens: Number($("capTokens").value) || 120,
    trigger: $("capTrigger").value, overwrite: $("capOverwrite").checked, ...extra};
}

async function capRun(path, body) {
  try {
    const r = await api(path, body);
    if (r.error) $("capStatus").textContent = r.error;
    pollCaptions();
    return r;
  } catch (e) { $("capStatus").textContent = e.message; }
}

async function pollCaptions() {
  clearTimeout(S.cap.timer);
  let st;
  try { st = await api("/api/captions/status"); } catch (e) { return; }
  const busy = ["loading", "running", "translating"].includes(st.state);
  const pct = st.total ? (100 * st.done) / st.total : 0;
  $("capBar").style.width = (busy ? pct : 0).toFixed(1) + "%";
  $("capStatus").textContent = {
    idle: "No model loaded.", loaded: `${st.model} is loaded, ready.`,
    loading: `Loading ${st.model}… (the first time it downloads)`,
    running: `Captioning ${st.done} of ${st.total}` + (st.fail ? ` · ${st.fail} failed` : ""),
    translating: st.total ? `Translating ${st.done} of ${st.total}` : "Loading the translator…",
  }[st.state] || st.state;
  $("capAll").disabled = busy; $("capStatic").disabled = busy; $("capTranslate").disabled = busy;
  $("capStop").disabled = !busy; $("capUnload").disabled = busy || !st.loaded;
  const log = $("capLog");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.textContent = st.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (S.cap.lastState && ["running", "translating"].includes(S.cap.lastState) && !busy) loadCaptions();
  S.cap.lastState = st.state;
  if (busy || document.querySelector("#tab-captions.active")) S.cap.timer = setTimeout(pollCaptions, busy ? 1500 : 5000);
}

async function loadCaptions() {
  if (!S.cap.opts || S.cap.optsFamily !== S.family) {        // a model change can bring the song card
    await loadCaptionOptions().catch(() => {});
    S.cap.optsFamily = S.family;
  }
  pollCaptions();
  $("captionExt").value = capExt();         // the files this tab writes and reads are the ones training reads
  if (!S.imageFolder) { $("captionsSummary").textContent = "Choose the training folder on the Start tab."; return; }
  const ext = $("captionExt").value.trim() || ".txt";
  const r = await api(`/api/captions?folder=${encodeURIComponent(S.imageFolder)}&ext=${encodeURIComponent(ext)}`);
  S.cap.items = r.items;
  renderCaptions();
}

function renderCaptions() {
  const host = $("captionsGrid");
  host.innerHTML = "";
  const ext = $("captionExt").value.trim() || ".txt";
  const q = $("capSearch").value.trim().toLowerCase();
  const all = S.cap.items;
  const items = q ? all.filter((i) => i.name.toLowerCase().includes(q) || (i.caption || "").toLowerCase().includes(q)) : all;
  const pages = Math.max(1, Math.ceil(items.length / CAP_PAGE));
  S.cap.page = Math.min(S.cap.page, pages - 1);
  const missing = all.filter((i) => i.caption === null).length;
  $("captionsSummary").textContent = `${all.length} files${missing ? `, ${missing} without a caption` : ""}` +
    (q ? ` · ${items.length} match` : "") + (pages > 1 ? ` · page ${S.cap.page + 1} of ${pages}` : "");
  $("capPrev").disabled = S.cap.page === 0; $("capNext").disabled = S.cap.page >= pages - 1;
  for (const it of items.slice(S.cap.page * CAP_PAGE, (S.cap.page + 1) * CAP_PAGE)) {
    const src = `/api/media?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    const ta = el("textarea", {rows: 4});
    ta.value = it.caption ?? "";
    const status = el("span", {class: "field-hint"});
    const save = el("button", {class: "btn btn-sm", text: "Save", onclick: async () => {
      await api("/api/captions/save", {folder: S.imageFolder, name: it.name, text: ta.value, ext});
      it.caption = ta.value;
      status.textContent = "Saved";
    }});
    const kind = it.name.split(".").pop().toLowerCase();
    const isAudio = ["wav", "mp3", "flac", "m4a"].includes(kind);
    const trainsPictures = !S.fam || !S.fam.media || S.fam.media.includes("photo") || kind === "mp4";
    const recap = isAudio || !trainsPictures ? null : el("button", {class: "btn btn-sm", text: "Re-caption",
      title: "Write this file's caption again with the model and task above", onclick: async () => {
        status.textContent = "Re-captioning…";
        const r = await capRun("/api/captions/ai", capJobBody({only: [it.path], overwrite: true}));
        if (r && r.error) status.textContent = r.error;
      }});
    const media = mediaFor(it);
    host.append(el("div", {class: "caption-card"}, media,
      el("div", {class: "caption-card-body"}, el("div", {class: "caption-card-title", text: mediaLabel(it)}), ta,
        el("div", {class: "field-row"}, save, recap, status))));
  }
}

async function findReplace(apply) {
  const host = $("frChanges");
  try {
    const r = await api("/api/captions/replace", {family: S.family, folder: S.imageFolder, find: $("frFind").value,
      replace: $("frReplace").value, apply, ext: capExt()});
    host.innerHTML = "";
    $("frStatus").textContent = r.changes.length ? (apply ? `Replaced in ${r.changes.length} captions.` :
      `${r.changes.length} captions would change:`) : "No caption contains that.";
    if (!apply) for (const c of r.changes.slice(0, 100)) host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: c.name}),
      el("span", {class: "field-hint", text: "Before: " + c.old}), el("span", {class: "field-hint", text: "After: " + c.new})));
    if (apply) loadCaptions();
  } catch (e) { $("frStatus").textContent = e.message; }
}

function wireCaptions() {
  $("captionsReload").addEventListener("click", loadCaptions);
  $("capModel").addEventListener("change", () => { fillTasks(); onCapTask(); });
  $("capPerPage").addEventListener("change", () => {
    CAP_PAGE = Number($("capPerPage").value) || 12; S.cap.page = 0; renderCaptions(); saveCapSoon(); });
  $("capTask").addEventListener("change", onCapTask);
  for (const id of ["capTrigger", "capTokens", "capOverwrite", "capSongWords", "capSkipCjk"])
    $(id).addEventListener("change", saveCapSoon);
  // a trigger word typed here fills auto-recaption's on the Training tab (an empty one leaves it alone)
  $("capTrigger").addEventListener("input", () => {
    const word = $("capTrigger").value.trim().replace(/,+$/, "").trim();
    if (!word) return;
    S.values.CAPTION_TRIGGER_WORD = word;
    syncInputs(["CAPTION_TRIGGER_WORD"]);
    saveStateSoon();
  });
  $("capEditInstr").addEventListener("click", () =>
    $("capInstrBox").style.display === "none" ? openInstr() : ($("capInstrBox").style.display = "none"));
  $("capInstrSave").addEventListener("click", () => saveInstr($("capInstr").value));
  $("capInstrReset").addEventListener("click", () => saveInstr(""));
  $("capAll").addEventListener("click", () => capRun("/api/captions/ai", capJobBody()));
  $("capStatic").addEventListener("click", async () => {
    const r = await capRun("/api/captions/static", capJobBody());
    if (r && !r.error) { $("capStatus").textContent = `${r.written} written, ${r.skipped} kept.`; loadCaptions(); }
  });
  $("capSongs").addEventListener("click", async () => {
    const r = await capRun("/api/captions/songs", capJobBody({words: $("capSongWords").value}));
    if (r && !r.error) { $("capStatus").textContent = `${r.written} songs captioned "${r.caption}", ${r.skipped} kept.`; loadCaptions(); }
  });
  $("capTranslate").addEventListener("click", () =>
    capRun("/api/captions/translate", capJobBody({skip_existing: $("capSkipCjk").checked})));
  $("capStop").addEventListener("click", () => capRun("/api/captions/stop", {}));
  $("capUnload").addEventListener("click", () => capRun("/api/captions/unload", {}));
  $("frPreview").addEventListener("click", () => findReplace(false));
  $("frApply").addEventListener("click", () => findReplace(true));
  $("capSearch").addEventListener("input", debounce(() => { S.cap.page = 0; renderCaptions(); }, 250));
  $("capPrev").addEventListener("click", () => { S.cap.page--; renderCaptions(); window.scrollTo(0, 0); });
  $("capNext").addEventListener("click", () => { S.cap.page++; renderCaptions(); });
}

// ---- file picker -------------------------------------------------------------------------------------------
let pickDone = null, pickDir = false;
async function pick(start, dirOnly, done) {
  pickDone = done; pickDir = dirOnly;
  $("pickerTitle").textContent = dirOnly ? "Choose a folder" : "Choose a file";
  $("pickerChoose").style.display = dirOnly ? "" : "none";
  $("pickerNew").style.display = dirOnly ? "" : "none";           // a folder can be made where one is chosen
  $("pickerNewRow").style.display = "none";
  $("pickerNewMsg").textContent = "";
  $("pickerModal").classList.add("active");
  await browse(start || "");
}
async function browse(path) {
  const r = await api("/api/browse?path=" + encodeURIComponent(path));
  $("pickerPath").value = r.path;
  $("pickerUp").onclick = () => r.parent && browse(r.parent);
  const places = $("pickerPlaces");
  places.innerHTML = "";
  for (const p of r.places || []) places.append(el("button", {class: "btn btn-sm", text: p.name, title: p.path,
    onclick: () => browse(p.path)}));
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
  document.querySelectorAll(".tab-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    if (b.dataset.tab === name) b.scrollIntoView({block: "nearest", inline: "nearest"});
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "captions") loadCaptions();
  if (name === "prep") loadPrep();
  if (name === "samples") loadGallery();
  if (name === "training") loadCheckpoints();
  if (name === "outputs") { loadRuns(); if (S.outputsOpen) loadOutputs(); }
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

// a file in 8 MB pieces: asks what is already there first (an earlier, dropped upload resumes; a file already there
// is skipped), retries a piece after a drop, and reports the bytes sent. Returns {ok, data, skipped}
const PIECE = 8 * 2 ** 20;
async function uploadResumable(params, file, onProgress) {
  params = {...params, mtime: (file.lastModified || 0) / 1000};       // same name, size AND time = already there
  const q = (o) => Object.entries(o).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
  let st;
  try { st = await api(`/api/upload/status?${q({...params, size: file.size})}`); }
  catch (e) { return {ok: false, data: {error: e.message}}; }
  if (st.done) {
    onProgress(file.size);
    return {ok: true, skipped: true, data: {saved: [], skipped: [], replaced: [], path: st.path, folder: st.folder}};
  }
  let offset = st.have || 0, tries = 0, last = null;
  while (true) {
    const piece = file.slice(offset, Math.min(file.size, offset + PIECE));
    try {
      const r = await sendFile(`/api/upload/chunk?${q({...params, offset, total: file.size})}`, piece,
        (sent) => onProgress(offset + sent));
      if (!r.ok) return r;
      last = r.data;
      if (last.resync !== undefined) { offset = last.resync; continue; }      // the server says where it is
      tries = 0;
      if (last.have === undefined) return {ok: true, data: last};             // the last piece: finished
      offset = last.have;
    } catch (e) {                                                             // the connection dropped: wait, carry on
      if (++tries > 6) return {ok: false, data: {error: `${file.name}: the connection kept dropping (` +
        `${Math.round(100 * offset / file.size)}% sent). Upload again to carry on from there.`}};
      onProgress(offset, `connection dropped, retrying (${tries})…`);
      await new Promise((r) => setTimeout(r, 1500 * tries));
      try { offset = (await api(`/api/upload/status?${q({...params, size: file.size})}`)).have || 0; } catch (e2) { /* still down */ }
    }
  }
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
// opts: name (the new folder's), say(text, bad), done(folder) - by default the Start tab's boxes and training folder
async function uploadFiles(picked, opts = {}) {
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
  const name = opts.name !== undefined ? opts.name : $("uploadName").value.trim();
  const say = opts.say || ((t, bad) => { $("uploadStatus").textContent = t; $("uploadStatus").classList.toggle("bad", !!bad); });
  if (!name) return say("Type a name for the new folder first.", true);
  if (!files.length) return say(["Nothing to upload: no pictures, .mp4 clips, songs, captions or zips in there."]
    .concat(notes).join(" "), true);
  let folder = "", saved = 0, already = 0, doneBytes = 0;
  const skipped = [], failed = [], replaced = [], badname = [];
  const allBytes = files.reduce((a, f) => a + f.size, 0);
  const mb = (n) => n >= 2 ** 30 ? (n / 2 ** 30).toFixed(2) + " GB" : (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0) + " MB";
  for (const [i, f] of [...files].entries()) {
    say(`Uploading ${i + 1} of ${files.length}: ${f.name}`);
    try {
      const r = await uploadResumable({folder: name, name: tidyName(f.name)}, f, (sent, note) => say(
        `Uploading ${i + 1} of ${files.length}: ${f.name}, ${mb(sent)} of ${mb(f.size)}` +
        (note ? ` (${note})` : sent >= f.size && f.name.toLowerCase().endsWith(".zip") ? " (unpacking…)" : "") +
        ` · all files: ${Math.round(100 * (doneBytes + sent) / Math.max(1, allBytes))}% of ${mb(allBytes)}`));
      doneBytes += f.size;
      const d = r.data;
      if (r.skipped) { already++; folder = folder || d.folder; continue; }
      if (!r.ok) { failed.push(d.error || f.name); continue; }
      folder = d.folder; saved += d.saved.length; skipped.push(...d.skipped); replaced.push(...(d.replaced || []));
      badname.push(...(d.badname || []));
    } catch (e) { failed.push(`${f.name}: ${e.message}`); }
  }

  let msg = saved || already ? `${saved} file${saved === 1 ? "" : "s"} uploaded` +
    (already ? `, ${already} already there (same size, skipped)` : "") + ` in ${folder}.` +
    (opts.done ? " It is filled in above." : " That is now the training folder.") : "Nothing was saved.";
  if (!saved && already) saved = already;           // all already there: still the folder to use
  if (skipped.length) msg += ` Left out (not pictures, clips, songs or captions): ${skipped.join(", ")}.`;
  if (replaced.length) msg += ` Replaced files that were already there with the same name: ` +
    `${replaced.slice(0, 5).join(", ")}${replaced.length > 5 ? ", …" : ""}.`;
  if (notes.length) msg += " " + notes.join(" ");
  if (badname.length) msg += ` Left out of the zip, as their names can't be stored here: ${badname.slice(0, 5).join(", ")}` +
    `${badname.length > 5 ? ", …" : ""}. Rename them (no leading dot, no slashes) and upload them again.`;
  if (failed.length) msg += ` Not uploaded: ${failed.join("; ")}`;
  say(msg, !saved || failed.length);
  if (saved) loadDatasets();
  if (saved && opts.done) opts.done(folder);
  else if (saved) {
    $("imageFolder").value = folder;
    $("imageFolder").dispatchEvent(new Event("change"));
  }
}

// an Upload button beside a form box whose file or folder may be on this computer, not the one training
function uploadButton(r, input, set) {
  const status = el("div", {class: "field-hint upload-status"});
  const say = (t, bad) => { status.textContent = t; status.classList.toggle("bad", !!bad); };
  const picker = r.kind === "folder" ? el("input", {type: "file", webkitdirectory: true, hidden: true}) :
    el("input", {type: "file", hidden: true, accept: ".safetensors,image/*,.mp4,audio/*,.m4a"});
  picker.addEventListener("change", async () => {
    const files = [...picker.files];
    picker.value = "";
    if (!files.length) return;
    if (r.kind === "folder") {
      const top = (files[0].webkitRelativePath || "").split("/")[0];
      const name = tidyName(top).replace(/^[.\s]+/, "") || r.key.toLowerCase();
      return uploadFiles(files, {name, say, done: (folder) => { input.value = folder; set(folder); }});
    }
    const f = files[0];
    const mb = (n) => (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0);
    try {
      const res = await uploadResumable({single: 1, folder: "", name: tidyName(f.name)}, f,
        (sent, note) => say(`Uploading ${f.name}: ${mb(sent)} of ${mb(f.size)} MB` + (note ? ` (${note})` : "")));
      if (!res.ok) return say(res.data.error || "Upload failed.", true);
      if (res.skipped) {                     // the same file is already there: use it, nothing to send
        input.value = res.data.path;
        set(res.data.path);
        return say(`Already uploaded (same size): ${res.data.path}`);
      }
      input.value = res.data.path;
      set(res.data.path);
      say(`Uploaded to ${res.data.path}` + ((res.data.replaced || []).length ? " (replacing the one there)." : "."));
    } catch (e) { say(`${f.name}: ${e.message}`, true); }
  });
  const btn = el("label", {class: "btn", title: "From the device you're using now, sent to the machine running Fizgig. On the computer running Fizgig itself there's no need: Choose… picks it directly."},
    r.kind === "folder" ? "Upload a folder…" : "Upload…", picker);
  return {btn, status};
}

let wired = false;
// ---- the picture viewer: any preview or Captions picture, big, with the others in its list a step away --------
const LIGHTBOX_PICS = ".gallery-item > img, .caption-card > img";
S.lb = {list: [], at: 0};
function lightboxShow(i) {
  const pic = S.lb.list[i];
  if (!pic) return;
  S.lb.at = i;
  $("lightboxImg").src = pic.src;
  $("lightboxName").textContent = pic.name;
  $("lightboxPrev").style.visibility = i > 0 ? "" : "hidden";
  $("lightboxNext").style.visibility = i < S.lb.list.length - 1 ? "" : "hidden";
  $("lightbox").style.display = "";
}
function lightboxClose() { $("lightbox").style.display = "none"; $("lightboxImg").src = ""; }
function wireLightbox() {
  document.addEventListener("click", (e) => {
    const img = e.target.closest && e.target.closest(LIGHTBOX_PICS);
    if (!img) return;
    const host = img.closest(".gallery, .captions-grid") || document;   // step through the pictures beside it
    const imgs = [...host.querySelectorAll(LIGHTBOX_PICS)];
    S.lb.list = imgs.map((i) => {
      const fig = i.closest(".gallery-item, .caption-card");
      const cap = fig && fig.querySelector("figcaption, .caption-card-title");
      const label = cap ? [...cap.childNodes].map((n) => n.textContent).filter(Boolean).join(" · ") : "";
      return {src: i.currentSrc || i.src, name: label || i.alt};
    });
    lightboxShow(imgs.indexOf(img));
  });
  $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox" || e.target.id === "lightboxImg") lightboxClose(); });
  $("lightboxClose").addEventListener("click", lightboxClose);
  $("lightboxPrev").addEventListener("click", () => lightboxShow(S.lb.at - 1));
  $("lightboxNext").addEventListener("click", () => lightboxShow(S.lb.at + 1));
  document.addEventListener("keydown", (e) => {
    if ($("lightbox").style.display === "none") return;
    if (e.key === "Escape") lightboxClose();
    else if (e.key === "ArrowLeft") lightboxShow(S.lb.at - 1);
    else if (e.key === "ArrowRight") lightboxShow(S.lb.at + 1);
  });
}

// back on this page after a while (another tab or a phone may have changed things): take the saved settings of the
// model on show, unless this page has a change of its own still waiting to save
async function catchUp() {
  if (document.visibilityState !== "visible" || S.saveWaiting) return;
  let st;
  try { st = await api("/api/state"); } catch (e) { return; }
  const saved = st && st.values && st.values[S.family];
  if (!saved || JSON.stringify(saved) === JSON.stringify(S.values)) return;
  S.all[S.family] = saved;
  S.values = {...defaults(), ...saved};
  renderSections("samples", $("samplesForm"));
  renderSections("training", $("trainingForm"));
  renderPresets();
  await refreshState();
}

function wire() {
  if (wired) return;
  wired = true;
  document.addEventListener("visibilitychange", catchUp);
  window.addEventListener("focus", catchUp);
  wireLightbox();
  wirePrep();
  $("presetSave").addEventListener("click", presetSaveOpen);
  document.querySelectorAll(".tab-item").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
  // the sidebar's run opens the run card itself (it sits under the Training form)
  $("sideRun").addEventListener("click", () => setTimeout(() =>
    document.querySelector(".run-card").scrollIntoView({behavior: "smooth", block: "start"}), 50));
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
  // New folder: made in the folder on show, then opened (Use this folder picks it)
  const newRow = (on) => { $("pickerNewRow").style.display = on ? "" : "none"; if (on) $("pickerNewName").focus(); };
  $("pickerNew").addEventListener("click", () => { $("pickerNewMsg").textContent = ""; $("pickerNewName").value = ""; newRow(true); });
  $("pickerNewCancel").addEventListener("click", () => newRow(false));
  const makeFolder = async () => {
    try {
      const r = await api("/api/mkdir", {parent: $("pickerPath").value, name: $("pickerNewName").value});
      newRow(false);
      await browse(r.path);
      $("pickerNewMsg").textContent = "Made it. Press Use this folder to choose it.";
    } catch (e) { $("pickerNewMsg").textContent = e.message; }
  };
  $("pickerNewMake").addEventListener("click", makeFolder);
  $("pickerNewName").addEventListener("keydown", (e) => { if (e.key === "Enter") makeFolder(); });
  wireCaptions();
  $("outReload").addEventListener("click", loadOutputs);
  $("outToggle").addEventListener("click", () => {     // Saved files: folded until asked for, scanned only then
    S.outputsOpen = !S.outputsOpen;
    for (const id of ["outBody", "outRuns"]) $(id).style.display = S.outputsOpen ? "" : "none";
    $("outFolded").style.display = S.outputsOpen ? "none" : "";
    $("outToggle").textContent = S.outputsOpen ? "Hide" : "Show";
    if (S.outputsOpen) loadOutputs();
  });
  $("runsReload").addEventListener("click", loadRuns);
  $("datasetsReload").addEventListener("click", loadDatasets);
  $("aboutBtn").addEventListener("click", async () => {
    $("aboutModal").classList.add("active");
    try {
      const a = await api("/api/about");
      $("aboutVersion").textContent = "Version " + a.version;
      $("aboutUpdate").textContent = a.update;
    } catch (e) { /* the box still reads well without them */ }
  });
  $("datasetsSearch").addEventListener("input", debounce(renderDatasets, 200));
  loadDatasets();
  $("outFizgigOnly").addEventListener("change", loadOutputs);
  $("outSearch").addEventListener("input", debounce(renderOutputs, 200));
  $("outWhen").addEventListener("change", () => {
    $("outRange").style.display = $("outWhen").value === "custom" ? "" : "none";
    renderOutputs();
  });
  $("outFrom").addEventListener("change", renderOutputs);
  $("outTo").addEventListener("change", renderOutputs);
  $("btnCheck").addEventListener("click", check);
  $("btnStart").addEventListener("click", () => start("/api/start"));
  $("galleryRun").addEventListener("change", () => { S.galleryRun = $("galleryRun").value; loadGallery(); });
  document.querySelectorAll("#galleryView button").forEach((b) => b.addEventListener("click", () => {
    S.galleryView = b.dataset.view; loadGallery(); }));
  $("btnViewPreviews").addEventListener("click", async () => {   // the Samples tab, at its previews
    showTab("samples");
    const go = () => $("galleryCard").scrollIntoView({block: "start"});
    requestAnimationFrame(go);
    try { await loadGallery(); } catch (e) { /* the previews show when they can */ }
    go();                                    // again with the previews in (their tiles keep a fixed shape)
  });
  $("btnQueue").addEventListener("click", async () => {
    try {
      const r = await api("/api/queue/add", runBody());
      showProblems(r.problems);
      if (r.ok) $("runStatus").textContent = `Added to the queue (number ${r.position}).`;
      pollRun();
    } catch (e) { $("runStatus").textContent = e.message; }
  });
  $("webQueueStart").addEventListener("click", () => api("/api/queue/start", {}).then(pollRun));
  $("webQueueHold").addEventListener("click", () => api("/api/queue/hold", {}).then(pollRun));
  $("btnPause").addEventListener("click", () => start("/api/pause"));
  $("btnStop").addEventListener("click", () => start("/api/stop"));
  $("btnResume").addEventListener("click", () => start("/api/resume"));
  $("loginButton").addEventListener("click", login);
  $("loginPassword").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
}

boot();
