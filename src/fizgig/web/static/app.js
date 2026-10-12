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
  // Fizgig was updated while this web UI kept running the old server: some new parts won't work until it restarts
  $("outdatedBanner").style.display = S.schema.outdated ? "" : "none";
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
  if (r.kind === "note") {                        // its own text, and any the form's state writes (RefMod's tokens)
    if (r.label) wrap.append(el("div", {class: "field-hint", text: r.label}));
    wrap.append(el("div", {class: "field-hint row-note", id: "note_" + r.key}));
    return wrap;
  }
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
    if (f.prefs_skip) continue;              // RefMod: MiniMax H3's files, on H3's card
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
    // a model that can't pause (a RefMod is made in one go): no Pause button for its runs
    const runFam = S.schema.families.find((f) => f.key === (r.family || S.family)) || {};
    $("btnPause").style.display = runFam.no_pause ? "none" : "";
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
    // per-image loss (a family with the loss watch, when the run kept its log)
    const ilBox = el("div", {class: "il-box", style: "display:none"});
    const ilBtn = run.image_loss ? el("button", {class: "btn btn-sm", text: "Per-image loss",
      title: "Each training image's loss by epoch, against the rest of the dataset - which images are hard, which are being learned, and what the loss watch decided about each.",
      onclick: async () => {
        if (ilBox.style.display !== "none") { ilBox.style.display = "none"; return; }
        ilBox.style.display = ""; ilBox.textContent = "Loading…";
        try { imageLossPanel(ilBox, await api("/api/runs/image_loss?id=" + encodeURIComponent(run.id))); }
        catch (e) { ilBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
      }}) : null;
    host.append(el("div", {class: "run-row"},
      el("div", {class: "run-row-head"},
        el("span", {class: "run-badge", "data-state": RESULT_STATE[run.result] || "idle", text: RESULT_LABEL[run.result] || run.result}),
        el("b", {text: run.lora || "(unnamed)"}),
        el("span", {class: "field-hint", text: `${famName(run.family)} · ${run.source === "desktop" ? "desktop app" : "web UI"}`})),
      el("div", {class: "field-hint", text: [run.image_folder && `Trained on ${run.image_folder}`,
        `${new Date(run.started_at * 1000).toLocaleString()} · ${dur(run.started_at, run.ended_at)}`, figs]
        .filter(Boolean).join("  ·  ")}),
      lossChart(run.curve),
      el("div", {class: "field-row"}, showFiles, ilBtn, again, del), delBox, alertRow, files, ilBox));
  }
  if (!r.runs.length) host.append(el("div", {class: "field-hint", text: "No runs yet. Every run you start, here or in " +
    "the desktop app, is listed here with its results."}));
}

// ---- per-image loss: each image's residual by epoch (its loss minus the dataset's mean at the same noise level, the
// loss watch's own measure) with the watch's verdict - the desktop's Problem Images colours and words
const IL_VERDICTS = {
  excluded: ["#7F8C8D", "Excluded from training - two AI captions couldn't fix it. Edit the caption to re-admit it, or remove it from the dataset."],
  stuck: ["#E74C3C", "Stuck - persistently hard, not improving. Review this image and its caption."],
  suspect: ["#D35400", "Suspect - extremely hard from the start; slowed while the trend confirms. Worth a caption check now."],
  watch: ["#E67E22", "Watching - looked stuck this epoch; needs more epochs to confirm."],
  warmup: ["#8E7CC3", "Look-filter outlier easing in on an LR ramp; releases early once it starts improving."],
  exhausted: ["#16A085", "Fully mined - improved a lot, then plateaued. Caption is fine; LR eased to prevent overbake."],
  learning: ["#5B9BD5", "Learning - hard but improving. Leave it alone."],
  mid: ["#95A5A6", "Normal."],
  easy: ["#70AD47", "Learned - consistently easy. Gets a gentle x1.1 boost."]};
const IL_ORDER = ["excluded", "stuck", "suspect", "watch", "warmup", "exhausted", "learning", "mid", "easy"];
const IL_PROBLEMS = new Set(["excluded", "stuck", "suspect", "watch"]);
const IL_PAGE = 48;

function imageLossPanel(box, d) {
  const imgs = d.images || [], rep = d.report || {};
  // one scale for every tile, so tiles compare: room for every image's recent level with a margin (a one-epoch
  // spike past it is clipped; a consistently hard image keeps its shape)
  const ymax = Math.max(1e-6, 1.25 * Math.max(...imgs.map((im) => Math.abs(im.recent)), 0));
  const counts = {};
  for (const im of imgs) counts[im.verdict || "mid"] = (counts[im.verdict || "mid"] || 0) + 1;
  const tally = IL_ORDER.filter((v) => counts[v]).map((v) => `${counts[v]} ${v}`).join(" · ");
  const head = [`${imgs.length} images · ${d.epochs.length} epochs`, rep.epoch ? `the watch's verdicts as of epoch ${rep.epoch}` : "",
    tally].filter(Boolean).join(" · ");
  const sort = el("select", {class: "w-combo"}, ...[["hard", "Hardest now first"], ["easy", "Easiest now first"],
    ["verdict", "By verdict"], ["name", "By name"]].map(([v, t]) => el("option", {value: v, text: t})));
  const show = el("select", {class: "w-combo"}, el("option", {value: "all", text: "All images"}),
    el("option", {value: "problems", text: "Problems only (excluded, stuck, suspect, watch)"}));
  const grid = el("div", {class: "il-grid"}), pager = el("div", {class: "field-row"});
  let page = 0;
  const draw = () => {
    let list = show.value === "problems" ? imgs.filter((im) => IL_PROBLEMS.has(im.verdict)) : [...imgs];
    const o = (v) => { const i = IL_ORDER.indexOf(v || "mid"); return i < 0 ? 2 : i; };
    list.sort(sort.value === "name" ? (a, b) => a.key.localeCompare(b.key) :
      sort.value === "easy" ? (a, b) => a.recent - b.recent :
      sort.value === "verdict" ? (a, b) => o(a.verdict) - o(b.verdict) || b.recent - a.recent : (a, b) => b.recent - a.recent);
    const pages = Math.max(1, Math.ceil(list.length / IL_PAGE));
    page = Math.min(page, pages - 1);
    grid.replaceChildren(...list.slice(page * IL_PAGE, (page + 1) * IL_PAGE).map((im) => imageLossTile(im, ymax)));
    if (!list.length) grid.append(el("div", {class: "field-hint", text: "No image has a problem verdict."}));
    pager.replaceChildren(...(pages > 1 ? [
      el("button", {class: "btn btn-sm", text: "‹ Prev", disabled: page === 0, onclick: () => { page--; draw(); }}),
      el("span", {class: "field-hint", text: `page ${page + 1} of ${pages}`}),
      el("button", {class: "btn btn-sm", text: "Next ›", disabled: page >= pages - 1, onclick: () => { page++; draw(); }})] : []));
  };
  sort.addEventListener("change", () => { page = 0; draw(); });
  show.addEventListener("change", () => { page = 0; draw(); });
  box.replaceChildren(
    el("div", {class: "field-hint", text: head}),
    rep.plateaued && rep.best_epoch_estimate ? el("div", {class: "il-plateau",
      text: `The run plateaued: the best checkpoint is likely around epoch ${rep.best_epoch_estimate} - compare a few epochs either side.`}) : null,
    el("div", {class: "field-hint", text: "Each line is one image's loss minus the dataset's average at the same noise level, " +
      "epoch by epoch. Above the dotted line it is harder than the rest; a line falling is the image being learned. " +
      "All tiles share one scale."}),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Sort:"}), sort, show), grid, pager);
  draw();
}

function imageLossTile(im, ymax) {
  const [col, say] = IL_VERDICTS[im.verdict] || IL_VERDICTS.mid;
  const W = 200, H = 56, pad = 3, pts = im.series;
  const e0 = pts[0][0], e1 = pts[pts.length - 1][0];
  const X = (e) => pad + (W - 2 * pad) * (e1 === e0 ? 0.5 : (e - e0) / (e1 - e0));
  const Y = (v) => H / 2 - (H / 2 - pad) * Math.max(-1, Math.min(1, v / ymax));
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "il-chart");
  const zero = document.createElementNS(ns, "line");
  zero.setAttribute("x1", 0); zero.setAttribute("x2", W); zero.setAttribute("y1", H / 2); zero.setAttribute("y2", H / 2);
  zero.setAttribute("class", "il-zero");
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "));
  line.setAttribute("stroke", col);
  const t = document.createElementNS(ns, "title");
  t.textContent = pts.map((p) => `epoch ${p[0]}: ${p[1] >= 0 ? "+" : ""}${p[1].toFixed(4)}`).join("\n");
  svg.append(zero, line, t);
  const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(4);
  return el("figure", {class: "gallery-item il-tile"},
    im.path ? el("img", {loading: "lazy", alt: im.key, src: `/api/thumb?path=${encodeURIComponent(im.path)}&t=${encodeURIComponent(TOKEN)}`}) :
      el("div", {class: "il-nopic", text: "(not in the training folder now)"}),
    el("figcaption", {},
      el("div", {class: "il-name", title: im.key, text: im.key}),
      el("span", {class: "il-verdict", style: `background:${col}`, title: say,
        text: (im.verdict || "mid") + (im.multiplier != null && im.multiplier !== 1 ? ` ×${im.multiplier}` : "")}),
      el("span", {class: "field-hint", text: ` now ${sign(im.recent)}`, title: "Mean over the last 5 epochs (the watch's window)"})),
    svg);
}

// ---- LoRA Royale: every epoch on one seed, crossfaded (fizgig/web/royale.py; renders on the GPU worker) ----
S.ry = {wired: false, info: null, items: [], gen: 0, shown: -1, scores: {}, best: null, poll: null};
const RY_KEY = "fizgig_royale";
const rySaved = () => { try { return JSON.parse(localStorage.getItem(RY_KEY) || "{}"); } catch (e) { return {}; } };
function ryRemember() {
  const v = (id) => $(id).value;
  try { localStorage.setItem(RY_KEY, JSON.stringify({source: rySource(), single: v("rySingle"), cmp: v("ryCPrompts"),
    family: v("ryFamily"), folder: v("ryFolder"), prompt: v("ryPrompt"),
    seed: v("rySeed"), ref: v("ryRef"), refStr: v("ryRefStr"), w: v("ryW"), h: v("ryH"), max: v("ryMax"),
    like: v("ryLikeRef"), out: v("ryOut"), fmt: v("ryFmt"), speed: v("rySpeed")})); } catch (e) { /* private */ }
}
const ryLabel = (l) => (String(l).match(/^\d+$/) ? `Epoch ${l}` : String(l));

async function royaleOpen() {
  if (!S.ry.wired) {
    S.ry.wired = true;
    const sv = rySaved();
    for (const [id, k] of [["ryFolder", "folder"], ["ryPrompt", "prompt"], ["rySeed", "seed"], ["ryRef", "ref"],
      ["ryRefStr", "refStr"], ["ryLikeRef", "like"], ["ryOut", "out"]]) if (sv[k] != null) $(id).value = sv[k];
    for (const [id, k] of [["ryW", "w"], ["ryH", "h"], ["ryMax", "max"], ["ryFmt", "fmt"], ["rySpeed", "speed"]])
      if (sv[k]) $(id).value = sv[k]; else if (id === "ryW" || id === "ryH") $(id).value = "512";
    await royaleFamily(sv.family);
    wireRoyale();
    await wireRoyaleTravel();
    royaleScan();
  }
  royalePoll();
  ryTravelPoll();
}

// ---- Royale's travel modes and comparison sheet ----
const RY_MODES = {seed: "ryS", strength: "ryL", prompt: "ryP"};
S.ry.tr = {};

function rySource() { return (document.querySelector('input[name="rySource"]:checked') || {}).value || "folder"; }
function rySourceChanged() { $("tab-royale").classList.toggle("single", rySource() === "single"); ryRemember(); }

async function ryPtRefresh(applyPreset) {
  // the prompt-travel card: a preset fills it (subject, prompt, knobs); the travel's waypoints fill Start / End
  const preset = $("ryPPreset").value, subj = $("ryPSubject").value || "Woman";
  const dim = preset && preset !== "Custom words" ? null : (preset === "Custom words" ? "Custom" : null);
  const r = await api(`/api/royale/pt?preset=${encodeURIComponent(applyPreset ? preset : "")}&subject=${encodeURIComponent(subj)}` +
    `&dim=${encodeURIComponent(S.ry.ptDim || dim || "")}&custom=${encodeURIComponent($("ryPCustom").value)}`).catch(() => null);
  if (!r) return;
  if (applyPreset === "subject" && r.preset) {      // a new subject: the prompt re-composed, the knobs left alone
    $("ryPPrompt").value = r.preset.prompt; S.ry.ptDim = r.preset.dim;
    return ryPtRefresh(false);
  }
  if (applyPreset && r.preset) {
    const p = r.preset;
    if (p.subject && p.subject !== $("ryPSubject").value && applyPreset === "preset") { $("ryPSubject").value = p.subject; return ryPtRefresh("preset2"); }
    S.ry.ptDim = p.dim; $("ryPPrompt").value = p.prompt; $("ryPInterp").value = p.interp; $("ryPDrift").value = p.drift;
    $("ryPVary").checked = !!p.vary_seed; $("ryPSeq").checked = !!p.sequential; $("ryPAnchor").checked = !!p.anchor;
    $("ryPAnchorStr").value = p.anchor_str; $("ryPRefStr").value = p.ref_strength; $("ryPMp").value = p.ref_mp;
    return ryPtRefresh(false);
  }
  const words = r.words || [];
  for (const id of ["ryPStart", "ryPEnd"]) {
    const cur = $(id).value;
    $(id).replaceChildren(...words.map((w) => el("option", {value: w, text: w})));
    $(id).value = words.includes(cur) ? cur : (id === "ryPStart" ? words[0] : words[words.length - 1]) || "";
  }
  S.ry.ptWords = words;
  ryPtWords();
}

function ryPtRange() {
  const w = S.ry.ptWords || [];
  if (w.length < 2) return w;
  const si = Math.max(0, w.indexOf($("ryPStart").value)), ei = w.indexOf($("ryPEnd").value) < 0 ? w.length - 1 : w.indexOf($("ryPEnd").value);
  return si <= ei ? w.slice(si, ei + 1) : w.slice(ei, si + 1).reverse();
}
function ryPtWords() {
  const w = ryPtRange();
  $("ryPWords").textContent = w.length >= 2 ? "Waypoints:  " + w.join("  →  ") : w.length ?
    "Pick a Start/End that span at least two waypoints." : "Add at least two comma-separated custom words to travel between.";
  $("ryPDrift").disabled = $("ryPVary").checked;
}

async function ryTravel(mode) {
  const p = RY_MODES[mode], v = (id) => $(p + id).value;
  const body = {mode, family: $("ryFamily").value, source: rySource(), single: $("rySingle").value.trim(),
    i: royaleCurrent(), prompt: $("ryPrompt").value.trim(), seed: parseInt($("rySeed").value, 10) || 42,
    setup_ref: $("ryRef").value.trim(), width: Number(v("W")), height: Number(v("H")), frames: Number(v("Frames"))};
  if (mode !== "strength") Object.assign(body, {ref: v("Ref").trim(), use_epoch: $(p + "UseEpoch").checked, ref_strength: v("RefStr"),
    ref_mp: v("Mp"), sequential: $(p + "Seq").checked});
  if (mode === "seed") Object.assign(body, {seed_a: v("A"), seed_b: v("B"), waypoints: $("rySWay").value});
  if (mode === "strength") Object.assign(body, {s_start: $("ryLA").value, s_end: $("ryLB").value});
  if (mode === "prompt") Object.assign(body, {base: $("ryPPrompt").value.trim(), words: ryPtRange(), interp: $("ryPInterp").value,
    drift: $("ryPDrift").value, vary_seed: $("ryPVary").checked, anchor: $("ryPAnchor").checked, anchor_str: $("ryPAnchorStr").value});
  try {
    const r = await api("/api/royale/travel", body);
    if (r.family !== $("ryFamily").value) await royaleFamily(r.family);
    $(p + "Sc").style.display = "none";
    ryRemember(); ryTravelPoll();
  } catch (e) { $(p + "Status").textContent = e.message; }
}

async function rySheet() {
  try {
    const r = await api("/api/royale/travel", {mode: "sheet", family: $("ryFamily").value, source: rySource(),
      single: $("rySingle").value.trim(), i: royaleCurrent(), prompts: $("ryCPrompts").value, sheet_mode: $("ryCMode").value,
      epochs: $("ryCEpochs").value, trigger: $("ryCTrigger").value.trim(), seed: parseInt($("ryCSeed").value, 10) || 42,
      width: Number($("ryCW").value), height: Number($("ryCH").value), row_labels: $("ryCRows").checked, brand: $("ryCTag").checked});
    if (r.family !== $("ryFamily").value) await royaleFamily(r.family);
    $("ryCSheet").style.display = "none";
    ryRemember(); ryTravelPoll();
  } catch (e) { $("ryCStatus").textContent = e.message; }
}

async function ryTravelPoll() {
  clearTimeout(S.ry.tpoll);
  let st;
  try { st = await api("/api/royale/travel/status"); } catch (e) { return; }
  let running = false;
  for (const [mode, t] of Object.entries(st)) {
    running = running || t.state === "rendering";
    if (mode === "sheet") {
      $("ryCGo").disabled = t.state === "rendering";
      if (t.status) $("ryCStatus").textContent = t.status;
      if (t.state === "done" && t.sheet && S.ry.tr.sheet !== t.gen) {
        S.ry.tr.sheet = t.gen;
        $("ryCImg").src = `/api/royale/travel/image?mode=sheet&g=${t.gen}&t=${encodeURIComponent(TOKEN)}`;
        $("ryCSheet").style.display = "";
      }
      continue;
    }
    const p = RY_MODES[mode], sc = $(p + "Sc");
    $(p + "Go").disabled = t.state === "rendering";
    if (t.status) $(p + "Status").textContent = t.status;
    const key = `${t.gen}:${t.frames.length}`;
    if (t.frames.length && S.ry.tr[mode] !== key) {
      S.ry.tr[mode] = key;
      sc._frames = t.frames.slice().sort((a, b) => a.i - b.i); sc._gen = t.gen;
      const range = sc.querySelector(".ry-scrub");
      range.max = sc._frames.length - 1;
      if (t.state === "done" || Number(range.value) > range.max) range.value = t.state === "done" ? 0 : range.max;
      sc.style.display = "";
      ryScScrub(sc);
    }
  }
  if (running && $("tab-royale").classList.contains("active")) S.ry.tpoll = setTimeout(ryTravelPoll, 1000);
}

function ryScScrub(sc) {
  const f = sc._frames || [];
  if (!f.length) return;
  const i = Math.max(0, Math.min(Number(sc.querySelector(".ry-scrub").value), f.length - 1));
  sc.querySelector("img").src = `/api/royale/travel/image?mode=${sc.dataset.mode}&i=${f[i].i}&g=${sc._gen}&t=${encodeURIComponent(TOKEN)}`;
  sc.querySelector(".ry-label").textContent = `Frame ${i + 1}/${f.length}    ${f[i].label}`;
}

async function ryScSave(sc, fmt) {
  const mode = sc.dataset.mode, p = RY_MODES[mode], msg = sc.querySelector(".ry-sc-msg");
  const f = sc._frames || [], i = Math.max(0, Math.min(Number(sc.querySelector(".ry-scrub").value), f.length - 1));
  msg.textContent = "Saving…";
  try {
    const r = fmt === "frame" ? await api("/api/royale/travel/frame", {mode, i: f[i].i, out_dir: $("ryOut").value.trim()}) :
      await api("/api/royale/travel/save", {mode, format: fmt, out_dir: $("ryOut").value.trim(), speed: $(p + "Speed").value,
        loop: $(p + "Loop").checked, badge: $(p + "Badge").checked, tag: $(p + "Tag").checked, deflicker: $(p + "Deflicker").value});
    msg.replaceChildren(el("span", {text: `Saved ${r.path} `}), el("a", {class: "btn btn-sm", download: r.path.split(/[\\/]/).pop(),
      text: "Download", href: `/api/media?path=${encodeURIComponent(r.path)}&t=${encodeURIComponent(TOKEN)}`}));
  } catch (e) { msg.textContent = e.message; }
}

async function wireRoyaleTravel() {
  document.querySelectorAll('input[name="rySource"]').forEach((r) => r.addEventListener("change", rySourceChanged));
  const setSingle = (p) => { $("rySingle").value = p; ryRemember(); };
  const up = uploadButton({kind: "path", key: "ROYALE_SINGLE"}, $("rySingle"), setSingle);
  $("rySingleBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick($("rySingle").value || S.outputDefault, false, setSingle)}),
    up.btn, up.status);
  for (const [mode, p] of Object.entries(RY_MODES)) {
    $(p + "Go").addEventListener("click", () => ryTravel(mode));
    const sc = $(p + "Sc");
    sc.querySelector(".ry-scrub").addEventListener("input", () => ryScScrub(sc));
    sc.querySelectorAll(".ry-sc-save button").forEach((b) => b.addEventListener("click", () => ryScSave(sc, b.dataset.fmt)));
    if (mode !== "strength") {
      $(p + "RefPick").addEventListener("click", () => pick($(p + "Ref").value || S.imageFolder, false, (x) => { $(p + "Ref").value = x; ryRemember(); }));
      $(p + "RefClear").addEventListener("click", () => { $(p + "Ref").value = ""; ryRemember(); });
      $(p + "UseEpoch").addEventListener("change", () => { for (const id of ["Ref", "RefPick", "RefClear"]) $(p + id).disabled = $(p + "UseEpoch").checked; });
    }
  }
  $("ryCGo").addEventListener("click", rySheet);
  $("ryCSave").addEventListener("click", async () => {
    try { const r = await api("/api/royale/travel/save", {mode: "sheet", out_dir: $("ryOut").value.trim()});
      $("ryCMsg").textContent = `Saved ${r.path}`; } catch (e) { $("ryCMsg").textContent = e.message; }
  });
  $("ryCImg").addEventListener("click", () => window.open($("ryCImg").src, "_blank"));
  $("rySDice").addEventListener("click", () => { $("rySA").value = Math.floor(Math.random() * 1e6); $("rySB").value = Math.floor(Math.random() * 1e6); });
  const info = await api("/api/royale/pt").catch(() => ({presets: [], subjects: [], seed_presets: {}}));
  S.ry.seedPresets = info.seed_presets;
  $("rySPreset").append(...Object.keys(info.seed_presets).map((n) => el("option", {value: n, text: n})));
  $("rySPreset").addEventListener("change", () => {
    const p = S.ry.seedPresets[$("rySPreset").value]; if (!p) return;
    $("rySRefStr").value = p.ref_strength; $("rySMp").value = p.ref_mp; $("rySSeq").checked = !!p.sequential; $("rySWay").value = p.waypoints;
  });
  $("ryPPreset").append(...info.presets.map((n) => el("option", {value: n, text: n})));
  $("ryPSubject").replaceChildren(...info.subjects.map((n) => el("option", {value: n, text: n})));
  $("ryPPreset").addEventListener("change", () => { if ($("ryPPreset").value) ryPtRefresh("preset"); });
  $("ryPSubject").addEventListener("change", () => ryPtRefresh($("ryPPreset").value ? "subject" : false));
  $("ryPCustom").addEventListener("change", () => ryPtRefresh(false));
  for (const id of ["ryPStart", "ryPEnd", "ryPVary"]) $(id).addEventListener("change", ryPtWords);
  $("ryPSlot").addEventListener("click", () => { const e = $("ryPPrompt"), at = e.selectionStart ?? e.value.length;
    e.value = e.value.slice(0, at) + "{x}" + e.value.slice(at); e.focus(); });
  for (const p of Object.values(RY_MODES).concat(["ryC"])) for (const d of ["W", "H"]) if (!$(p + d).dataset.set) $(p + d).value = "512";
  const sv = rySaved();
  if (sv.source === "single") document.querySelector('input[name="rySource"][value="single"]').checked = true;
  if (sv.single) $("rySingle").value = sv.single;
  if (sv.cmp) $("ryCPrompts").value = sv.cmp;
  rySourceChanged();
  if (S.ry.ptDim === undefined) S.ry.ptDim = "Age";
  ryPtRefresh(false);
  ryTravelPoll();
}

async function royaleFamily(key) {
  const r = await api("/api/royale/info?family=" + encodeURIComponent(key || "")).catch(() => null);
  if (!r) return;
  S.ry.info = r;
  $("ryFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name + (f.video ? " (clip previews)" : "")})));
  $("ryFamily").value = r.family;
  $("ryRefRow").style.display = r.reference_kind ? "" : "none";
  $("ryRefStrBox").style.display = r.reference_strength ? "" : "none";
  $("tab-royale").classList.toggle("no-ref", !r.reference_kind);
  $("tab-royale").classList.toggle("no-refstr", !r.reference_strength);
  if (!$("ryOut").value) $("ryOut").value = r.out_dir;
}

async function royaleScan() {
  const f = $("ryFolder").value.trim();
  if (!f) { $("ryScan").textContent = ""; return; }
  const r = await api("/api/royale/scan?folder=" + encodeURIComponent(f)).catch(() => ({summary: ""}));
  $("ryScan").textContent = r.summary || "";
}

async function royaleRender() {
  try {
    const r = await api("/api/royale/render", {family: $("ryFamily").value, folder: $("ryFolder").value.trim(),
      prompt: $("ryPrompt").value.trim(), seed: parseInt($("rySeed").value, 10) || 42, ref: $("ryRef").value.trim(),
      ref_strength: Number($("ryRefStr").value) || 1, width: Number($("ryW").value), height: Number($("ryH").value),
      max: $("ryMax").value});
    if (r.family !== $("ryFamily").value) {         // the checkpoints are another model's: follow them (as the desktop)
      await royaleFamily(r.family);
      $("ryStatus").textContent = `Switched to ${$("ryFamily").selectedOptions[0].text} to match the checkpoints.`;
    }
    S.ry.items = []; S.ry.shown = -1; S.ry.scores = {}; S.ry.best = null;
    $("ryLikeStatus").textContent = ""; $("ryBest").disabled = true;
    ryRemember();
    royalePoll();
  } catch (e) { $("ryStatus").textContent = e.message; }
}

async function royalePoll() {
  clearTimeout(S.ry.poll);
  let st;
  try { st = await api("/api/royale/status"); } catch (e) { return; }
  const running = st.state === "rendering";
  $("ryRender").disabled = running; $("ryStop").disabled = !running;
  $("rySBadge").textContent = running ? "Rendering" : "";
  $("rySBadge").dataset.state = running ? "running" : "idle";
  const done = st.items.filter((x) => x.png).length;
  $("ryBar").style.width = running ? `${Math.round(100 * Math.max(done, 0.3) / Math.max(st.total, 1))}%` : "0";
  if (st.status) $("ryStatus").textContent = st.status + (st.error && st.state === "failed" ? ` ${st.error}` : "");
  const items = st.items.filter((x) => x.png).sort((a, b) => a.i - b.i);
  if (st.gen !== S.ry.gen || items.length !== S.ry.items.length) {
    const fresh = st.gen !== S.ry.gen;
    S.ry.gen = st.gen; S.ry.items = items;
    if (fresh) { S.ry.scores = {}; S.ry.best = null; }
    royaleGrid();
    $("ryScrub").max = Math.max(0, items.length - 1);
    if (items.length && (fresh || S.ry.shown < 0)) { $("ryScrub").value = 0; S.ry.shown = -1; }
    royaleScrub();
  }
  if (running && $("tab-royale").classList.contains("active")) S.ry.poll = setTimeout(royalePoll, 1000);
}

const ryImg = (i) => `/api/royale/image?i=${S.ry.items[i].i}&gen=${S.ry.gen}&t=${encodeURIComponent(TOKEN)}`;

function royaleScrub() {
  // the crossfade: the two epochs either side of the slider, the upper one faded in by how far along it is
  const items = S.ry.items;
  if (!items.length) { $("ryLabel").textContent = ""; return; }
  const p = Number($("ryScrub").value), lo = Math.max(0, Math.min(Math.floor(p), items.length - 1));
  const hi = Math.min(lo + 1, items.length - 1), a = p - lo;
  if ($("ryA").dataset.i !== String(lo)) { $("ryA").src = ryImg(lo); $("ryA").dataset.i = String(lo); }
  if ($("ryB").dataset.i !== String(hi)) { $("ryB").src = ryImg(hi); $("ryB").dataset.i = String(hi); }
  $("ryB").style.opacity = lo === hi ? 0 : a;
  $("ryLabel").textContent = a <= 0.01 || lo === hi ? ryLabel(items[lo].label) :
    `${ryLabel(items[lo].label)}  →  ${ryLabel(items[hi].label)}    (${Math.round(a * 100)}%)`;
  const at = Math.round(p);
  document.querySelectorAll("#ryGrid .ry-cell").forEach((c, i) => c.classList.toggle("at", i === at));
}

function royaleGrid() {
  $("ryGrid").replaceChildren(...S.ry.items.map((it, i) => {
    const base = String(it.label).match(/^\d+$/) ? `e${it.label}` : String(it.label).slice(0, 16);
    const sc = S.ry.scores[String(it.label)];
    const cap = sc === undefined ? base : `${base}  ${sc == null ? "—" : sc.toFixed(2)}`;
    return el("div", {class: "ry-cell" + (S.ry.best != null && String(it.label) === String(S.ry.best) ? " best" : ""),
      title: it.path, onclick: () => { $("ryScrub").value = i; royaleScrub(); }},
      el("img", {src: ryImg(i), alt: base, loading: "lazy"}), el("div", {text: cap}));
  }));
}

function royaleCurrent() { return Math.max(0, Math.min(Math.round(Number($("ryScrub").value)), S.ry.items.length - 1)); }

async function royaleScore() {
  $("ryLikeStatus").textContent = "Scoring… (the first run loads the face model)";
  try {
    const r = await api("/api/royale/score", {ref: $("ryLikeRef").value.trim()});
    S.ry.scores = r.scores; S.ry.best = r.best;
    royaleGrid();
    if (r.best == null) { $("ryLikeStatus").textContent = "No faces detected in the renders or subject image."; $("ryBest").disabled = true; return; }
    $("ryBest").disabled = false;
    $("ryLikeStatus").textContent = `Best: ${ryLabel(r.best).toLowerCase()} (${r.scores[r.best].toFixed(3)})` +
      (r.no_face ? `  (${r.no_face} no-face)` : "");
    royaleBest();
  } catch (e) { $("ryLikeStatus").textContent = e.message; }
}

function royaleBest() {
  const i = S.ry.items.findIndex((it) => String(it.label) === String(S.ry.best));
  if (i >= 0) { $("ryScrub").value = i; royaleScrub(); }
}

async function royaleExport(likeness) {
  const say = likeness ? $("ryLikeStatus") : $("ryExportStatus");
  say.textContent = likeness ? "Building likeness clip…" : "Building frames…";
  try {
    const r = await api("/api/royale/export", {likeness, format: $("ryFmt").value, speed: $("rySpeed").value,
      loop: $("ryLoop").checked, ticker: $("ryTicker").checked, tag: $("ryTag").checked, out_dir: $("ryOut").value.trim()});
    say.replaceChildren(el("span", {text: `Saved ${r.frames} frames → ${r.path} `}),
      el("a", {class: "btn btn-sm", download: r.path.split(/[\\/]/).pop(), text: "Download",
        href: `/api/media?path=${encodeURIComponent(r.path)}&t=${encodeURIComponent(TOKEN)}`}));
  } catch (e) { say.textContent = e.message + (/codec|writer/i.test(e.message) ? " Try the GIF format instead." : ""); }
}

function wireRoyale() {
  const remember = () => ryRemember();
  for (const id of ["ryPrompt", "rySeed", "ryRefStr", "ryW", "ryH", "ryMax", "ryFmt", "rySpeed", "ryOut"])
    $(id).addEventListener("change", remember);
  $("ryFamily").addEventListener("change", () => { royaleFamily($("ryFamily").value); remember(); });
  $("ryFolder").addEventListener("change", () => { royaleScan(); remember(); });
  $("ryFolderPick").addEventListener("click", () => pick($("ryFolder").value || S.outputDefault, true, (p) => {
    $("ryFolder").value = p; royaleScan(); remember(); }));
  $("ryRefPick").addEventListener("click", () => pick($("ryRef").value || S.imageFolder, false, (p) => { $("ryRef").value = p; remember(); }));
  $("ryRefClear").addEventListener("click", () => { $("ryRef").value = ""; remember(); });
  $("ryLikePick").addEventListener("click", () => pick($("ryLikeRef").value || S.imageFolder, false, (p) => { $("ryLikeRef").value = p; remember(); }));
  $("ryLikeClear").addEventListener("click", () => { $("ryLikeRef").value = ""; remember(); });
  $("ryOutPick").addEventListener("click", () => pick($("ryOut").value, true, (p) => { $("ryOut").value = p; remember(); }));
  $("ryRender").addEventListener("click", royaleRender);
  $("ryStop").addEventListener("click", () => api("/api/royale/cancel", {}).then(royalePoll));
  $("tab-royale").addEventListener("click", (e) => { if (e.target.matches('[id$="Go"]')) setTimeout(ryTravelPoll, 400); });
  $("ryUnload").addEventListener("click", async () => {
    await api("/api/studio/reset", {});            // the shared GPU worker quits (Studio's Unload)
    $("ryStatus").textContent = "Model unloaded - the renders stay here until the next render.";
  });
  $("ryScrub").addEventListener("input", royaleScrub);
  $("ryScore").addEventListener("click", royaleScore);
  $("ryBest").addEventListener("click", royaleBest);
  $("ryExport").addEventListener("click", () => royaleExport(false));
  $("ryLikeExport").addEventListener("click", () => royaleExport(true));
  $("ryStills").addEventListener("click", async () => {
    $("ryExportStatus").textContent = "Saving stills…";
    try { const r = await api("/api/royale/stills", {out_dir: $("ryOut").value.trim()});
      $("ryExportStatus").textContent = `Saved ${r.saved.length} stills → ${$("ryOut").value.trim()}`; }
    catch (e) { $("ryExportStatus").textContent = e.message; }
  });
  $("ryPromote").addEventListener("click", async () => {
    try { const r = await api("/api/royale/promote", {i: royaleCurrent(), out_dir: $("ryOut").value.trim(),
      name: $("ryPromoteName").value.trim()});
      $("ryPromoteStatus").textContent = `Saved ${ryLabel(S.ry.items[royaleCurrent()].label).toLowerCase()} → ${r.path}`; }
    catch (e) { $("ryPromoteStatus").textContent = e.message; }
  });
}

// ---- Profiler: what a LoRA's blocks do (fizgig/web/tools.py runs scripts/profile_run.py, the desktop Profiler's job) ----
S.pf = {wired: false, photos: [], poll: null, last: null};
const PF_KEY = "fizgig_profiler";
const pfSaved = () => { try { return JSON.parse(localStorage.getItem(PF_KEY) || "{}"); } catch (e) { return {}; } };
function pfRemember() {
  try { localStorage.setItem(PF_KEY, JSON.stringify({family: $("pfFamily").value, lora: $("pfLora").value, mode: pfMode(),
    size: $("pfSize").value, prompt: $("pfPrompt").value, cls: $("pfClass").value, photos: S.pf.photos})); } catch (e) { /* private */ }
}
const pfMode = () => (document.querySelector('input[name="pfMode"]:checked') || {}).value || "quick";

async function profilerOpen() {
  if (!S.pf.wired) {
    S.pf.wired = true;
    const r = await api("/api/profile/info").catch(() => ({families: []}));
    $("pfFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name})));
    const sv = pfSaved();
    if (sv.family && r.families.some((f) => f.key === sv.family)) $("pfFamily").value = sv.family;
    $("pfLora").value = sv.lora || ""; $("pfPrompt").value = sv.prompt || ""; $("pfClass").value = sv.cls || "";
    if (sv.size) $("pfSize").value = sv.size;
    const m = document.querySelector(`input[name="pfMode"][value="${sv.mode || "quick"}"]`); if (m) m.checked = true;
    S.pf.photos = sv.photos || [];
    wireProfiler();
    pfPhotosDraw(); pfModeChanged();
  }
  profilerPoll();
}

function pfModeChanged() { $("pfRender").classList.toggle("off", pfMode() === "weights"); pfRemember(); }

function pfPhotosDraw() {
  $("pfPhotos").replaceChildren(...S.pf.photos.map((p, i) => el("span", {class: "pf-photo", title: p},
    el("img", {src: `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`, alt: ""}),
    el("button", {class: "btn btn-sm", text: "×", onclick: () => { S.pf.photos.splice(i, 1); pfPhotosDraw(); pfRemember(); }}))));
  if (!S.pf.photos.length) $("pfPhotos").append(el("span", {class: "field-hint", text: "None (picture change only)"}));
  $("pfPhotoBtns").style.display = S.pf.photos.length >= 3 ? "none" : "";
}

async function pfLora(p, fresh) {
  // a LoRA chosen: its model, then (a new one) its trigger prompts and its run's subject photos, as the desktop does
  $("pfLora").value = p;
  try {
    const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
    if (f.key && [...$("pfFamily").options].some((o) => o.value === f.key)) $("pfFamily").value = f.key;
  } catch (e) { /* unreadable: the run says so */ }
  if (fresh) {
    const r = await api(`/api/profile/prefill?path=${encodeURIComponent(p)}`).catch(() => ({}));
    $("pfPrompt").value = r.prompt || ""; $("pfClass").value = r.class_prompt || "";
    S.pf.photos = r.baselines || []; pfPhotosDraw();
  }
  pfRemember();
}

async function profilerRun() {
  try {
    await api("/api/profile/run", {family: $("pfFamily").value, lora: $("pfLora").value.trim(), mode: pfMode(),
      prompt: $("pfPrompt").value.trim(), class_prompt: $("pfClass").value.trim(), baselines: S.pf.photos,
      size: Number($("pfSize").value)});
    $("pfReport").style.display = $("pfStudio").style.display = "none";
    profilerPoll();
  } catch (e) { $("pfStatus").textContent = e.message; }
}

async function profilerPoll() {
  clearTimeout(S.pf.poll);
  let st;
  try { st = await api("/api/profile/status"); } catch (e) { return; }
  const running = st.state === "running";
  $("pfRun").disabled = running; $("pfStop").disabled = !running;
  $("pfBadge").textContent = {running: "Running", done: "Done", failed: "Failed", stopped: "Stopped"}[st.state] || "";
  $("pfBadge").dataset.state = running ? "running" : st.state === "done" ? "completed" : st.state === "idle" ? "idle" : "failed";
  const p = st.progress;
  $("pfBar").style.width = running ? (p && p[1] ? `${Math.round(100 * p[0] / p[1])}%` : "100%") : st.state === "done" ? "100%" : "0";
  $("pfBar").parentElement.classList.toggle("indeterminate", running && !(p && p[1]));
  const res = st.result;
  $("pfStatus").textContent = running ? st.text : st.state === "done" ? "Done." : st.state === "stopped" ? "Stopped." :
    st.state === "failed" ? "Error - see below." : "";
  $("pfLog").textContent = [...st.log.filter((l) => !/^INFO:/.test(l)), ...(res ? ["", ...res.lines] : []), ...(st.state === "failed" ? ["", st.error] : [])].join("\n");
  if (res && res.report) {
    $("pfReport").href = `/api/profile/report?path=${encodeURIComponent(res.report)}&t=${encodeURIComponent(TOKEN)}`;
    $("pfReport").style.display = "";
  }
  S.pf.last = res && res.studio;
  $("pfStudio").style.display = S.pf.last ? "" : "none";
  if (running && $("tab-profiler").classList.contains("active")) S.pf.poll = setTimeout(profilerPoll, 800);
}

async function profilerToStudio() {
  // the desktop's "Open in Repair Studio": the LoRA on the same prompt, seed and size, the profile's suggestions on the
  // sliders (bleed-heavy groups turned down, quiet ones off), so the change shows side by side and saves as a new file
  const h = S.pf.last;
  if (!h) return;
  showTab("studio");                                 // opens Studio; wait for its model list and layout
  for (let i = 0; i < 100 && !(S.st.familiesLoaded && S.st.layout); i++) await new Promise((r) => setTimeout(r, 100));
  if ($("stFamily").value !== h.family) { $("stFamily").value = h.family; await studioFamily(); }
  $("stPrimary").value = h.lora; $("stPrompt").value = h.prompt; $("stSeed").value = String(h.seed);
  if ([...$("stRes").options].some((o) => o.value === String(h.size))) $("stRes").value = String(h.size);
  for (const [id, b] of Object.entries(S.st.blocks)) {
    const [on, strength] = h.blocks[id] || [true, 1.0];
    b.pe = !!on; b.ps = Number(strength);
  }
  studioSync(); studioCommit(); stRemember();
  let msg = h.notes.length ? "From the profile: " + h.notes.join("; ") + "." :
    "The profile found no group to turn down or switch off: the likeness and any bleed sit in the same blocks, so the sliders stay as trained.";
  if (h.class_prompt) msg += ` To see the bleed, set the prompt to "${h.class_prompt}".`;
  $("stSetupMsg").textContent = msg;                // stays put while the status line follows the load
  await studioStart();
}

function wireProfiler() {
  document.querySelectorAll('input[name="pfMode"]').forEach((r) => r.addEventListener("change", pfModeChanged));
  for (const id of ["pfFamily", "pfSize", "pfPrompt", "pfClass"]) $(id).addEventListener("change", pfRemember);
  $("pfLora").addEventListener("change", () => pfLora($("pfLora").value.trim(), true));
  const up = uploadButton({kind: "path", key: "PROFILER_LORA"}, $("pfLora"), (p) => pfLora(p, true));
  $("pfLoraBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () =>
    pick($("pfLora").value || S.outputDefault, false, (p) => pfLora(p, true))}), up.btn, up.status);
  const addPhoto = (p) => { if (S.pf.photos.length < 3 && !S.pf.photos.includes(p)) S.pf.photos.push(p); pfPhotosDraw(); pfRemember(); };
  const pup = uploadButton({kind: "path", key: "PROFILER_PHOTO"}, el("input", {type: "text"}), addPhoto);
  $("pfPhotoBtns").replaceChildren(el("button", {class: "btn btn-sm", text: "Add photo…", onclick: () => pick(S.imageFolder, false, addPhoto)}),
    pup.btn, pup.status);
  $("pfPhotosClear").addEventListener("click", () => { S.pf.photos = []; pfPhotosDraw(); pfRemember(); });
  $("pfRun").addEventListener("click", profilerRun);
  $("pfStop").addEventListener("click", () => api("/api/profile/stop", {}).then(profilerPoll));
  $("pfStudio").addEventListener("click", profilerToStudio);
}

// ---- Extract: a LoRA reduced to a lower rank (exact weight SVD from the file; fizgig/web/tools.py runs it) ----
S.ex = {info: null, poll: null, wired: false};
const EX_KEY = "fizgig_extract";
const exSaved = () => { try { return JSON.parse(localStorage.getItem(EX_KEY) || "{}"); } catch (e) { return {}; } };
function exRemember() {
  try { localStorage.setItem(EX_KEY, JSON.stringify({family: $("exFamily").value, source: $("exSource").value,
    rank: $("exRank").value, preset: $("exPreset").value})); } catch (e) { /* private window */ }
}

async function extractOpen() {
  if (!S.ex.wired) wireExtract();
  const sv = exSaved();
  if (!S.ex.info) {
    if (sv.source) $("exSource").value = sv.source;
    if (sv.rank) $("exRank").value = sv.rank;
    await extractFamily(sv.family, sv.preset);
  }
  extractPoll();
}

async function extractFamily(key, preset) {
  let r;
  try { r = await api("/api/extract/info?family=" + encodeURIComponent(key || "")); }
  catch (e) { $("exStatus").textContent = e.message; return; }
  S.ex.info = r;
  $("exFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name})));
  $("exFamily").value = r.family;
  $("exOutDir").value = r.out_dir;
  const names = r.presets.map((p) => p.name).concat(r.presets.length ? ["Custom"] : []);
  $("exPresetCard").style.display = names.length ? "" : "none";
  $("exPreset").replaceChildren(...names.map((n) => el("option", {value: n, text: n})));
  $("exPreset").value = names.includes(preset) ? preset : (names[0] || "");
  // the Custom picker: All / None, a quick pick per preset, the model's block groups
  const box = $("exCustom");
  box.replaceChildren();
  if (r.presets.length) {
    const tick = (on) => box.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = on; });
    box.append(el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Select blocks:"}),
      el("button", {class: "btn btn-sm", text: "All", onclick: () => tick(true)}),
      el("button", {class: "btn btn-sm", text: "None", onclick: () => tick(false)}),
      ...r.presets.filter((p) => p.blocks.length).map((p) => el("button", {class: "btn btn-sm", text: p.name, onclick: () =>
        box.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = p.blocks.includes(c.value); })}))));
    for (const g of r.groups)
      box.append(el("div", {class: "ex-group"}, el("span", {class: "field-label", text: g.label.split(" ")[0] + ":"}),
        ...g.blocks.map((b) => el("label", {class: "checkbox-label"}, el("input", {type: "checkbox", value: b.id}),
          " " + b.label.split(" ").pop()))));
  }
  extractPresetChanged();
}

function extractPresetChanged() {
  $("exCustom").style.display = $("exPreset").value === "Custom" ? "" : "none";
  extractName();
  exRemember();
}

function extractName() {
  // the desktop's suggestion: <source>_<preset>_r<rank>
  const src = $("exSource").value.trim();
  if (!src) return;
  const base = src.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
  const slug = ($("exPreset").value || "all_blocks").toLowerCase().replace(/\+/g, "_").replace(/ /g, "_");
  $("exName").value = `${base}_${slug}_r${$("exRank").value}.safetensors`;
}

async function extractSource(p) {
  $("exSource").value = p;
  try {                                           // a LoRA of another model: switch to it, as the desktop does
    const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
    if (f.key && f.key !== $("exFamily").value && [...$("exFamily").options].some((o) => o.value === f.key))
      await extractFamily(f.key, $("exPreset").value);
  } catch (e) { /* unreadable: the run says so */ }
  extractName(); exRemember();
}

async function extractRun() {
  const custom = $("exPreset").value === "Custom";
  const blocks = custom ? [...$("exCustom").querySelectorAll("input:checked")].map((c) => c.value) : [];
  if (custom && !blocks.length) { $("exStatus").textContent = "Custom is selected but no blocks are ticked."; return; }
  try {
    const r = await api("/api/extract/run", {family: $("exFamily").value, source: $("exSource").value.trim(),
      name: $("exName").value.trim(), out_dir: $("exOutDir").value.trim(), rank: Number($("exRank").value),
      preset: custom || !(S.ex.info.presets || []).length ? "" : $("exPreset").value, blocks});
    $("exStatus").textContent = `Extracting to ${r.output}…`;
    $("exDone").replaceChildren();
    extractPoll();
  } catch (e) { $("exStatus").textContent = e.message; }
}

async function extractPoll() {
  clearTimeout(S.ex.poll);
  let st;
  try { st = await api("/api/extract/status"); } catch (e) { return; }
  const running = st.state === "running";
  $("exRun").disabled = running;
  $("exBadge").textContent = {running: "Running", done: "Done", failed: "Failed"}[st.state] || "";
  $("exBadge").dataset.state = running ? "running" : st.state === "done" ? "completed" : st.state === "failed" ? "failed" : "idle";
  const p = st.progress;
  $("exBar").style.width = running ? (p && p[1] ? `${Math.round(100 * (p[0] + 1) / p[1])}%` : "100%") : st.state === "done" ? "100%" : "0";
  $("exBar").parentElement.classList.toggle("indeterminate", running && !(p && p[1]));
  $("exLog").textContent = st.log.join("\n");
  if (running && p) $("exStatus").textContent = `SVD: ${p[0] + 1} of ${p[1]} layers…`;
  if (st.state === "done") {
    $("exStatus").textContent = `Saved ${st.output}`;
    if (!$("exDone").children.length) $("exDone").append(el("a", {class: "btn btn-sm btn-primary", download: "", text: "Download",
      href: `/api/download?path=${encodeURIComponent(st.output)}&t=${encodeURIComponent(TOKEN)}`}));
  }
  if (st.state === "failed") $("exStatus").textContent = `The extraction failed: ${st.error}`;
  if (running && $("tab-extract").classList.contains("active")) S.ex.poll = setTimeout(extractPoll, 500);
}

function wireExtract() {
  S.ex.wired = true;
  $("exFamily").addEventListener("change", () => { extractFamily($("exFamily").value, $("exPreset").value); exRemember(); });
  $("exPreset").addEventListener("change", extractPresetChanged);
  $("exRank").addEventListener("change", () => { extractName(); exRemember(); });
  $("exSource").addEventListener("change", () => extractSource($("exSource").value.trim()));
  const up = uploadButton({kind: "path", key: "EXTRACT_SOURCE"}, $("exSource"), extractSource);
  $("exSourceBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () =>
    pick($("exSource").value || S.outputDefault, false, extractSource)}), up.btn, up.status);
  $("exOutPick").addEventListener("click", () => pick($("exOutDir").value, true, (p) => { $("exOutDir").value = p; }));
  $("exRun").addEventListener("click", extractRun);
}

// ---- Metadata: any .safetensors file's ModelSpec metadata, edited (the header rewritten; tensors untouched) ----
const MD_FIELDS = [["modelspec.title", "Title"], ["modelspec.author", "Author"], ["modelspec.license", "License"],
  ["modelspec.tags", "Tags"], ["modelspec.trigger_phrase", "Trigger phrase"], ["modelspec.usage_hint", "Usage hint"]];
S.md = {path: "", thumb: null, wired: false};

function mdWire() {
  if (S.md.wired) return;
  S.md.wired = true;
  $("mdStandard").replaceChildren(...MD_FIELDS.map(([k, label]) => el("div", {class: "field-row"},
    el("span", {class: "field-label", text: label + ":"}), el("input", {type: "text", class: "w-path", "data-key": k}))),
  el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Description:"}),
    el("textarea", {rows: 4, "data-key": "modelspec.description"})));
  const load = (p) => { $("mdPath").value = p; mdLoad(p); };
  const up = uploadButton({kind: "path", key: "METADATA_FILE"}, $("mdPath"), load);
  $("mdPathBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick($("mdPath").value || S.outputDefault, false, load)}),
    up.btn, up.status);
  $("mdPath").addEventListener("change", () => mdLoad($("mdPath").value.trim()));
  const setThumb = async (p) => {
    try { const r = await api("/api/metadata/thumbnail", {path: p}); S.md.thumb = r.uri; mdThumb(); }
    catch (e) { $("mdSaveMsg").textContent = e.message; }
  };
  const tup = uploadButton({kind: "path", key: "METADATA_THUMB"}, el("input", {type: "text"}), setThumb);
  $("mdThumbBtns").replaceChildren(el("button", {class: "btn", text: "Replace…", onclick: () => pick(S.imageFolder, false, setThumb)}),
    tup.btn, tup.status);
  $("mdThumbClear").addEventListener("click", () => { S.md.thumb = null; mdThumb(); });
  $("mdAdd").addEventListener("click", () => mdRow("", ""));
  $("mdSave").addEventListener("click", () => mdSave(""));
  $("mdSaveAs").addEventListener("click", () => mdSave($("mdDest").value.trim()));
}

function mdThumb() {
  $("mdThumb").replaceChildren(S.md.thumb && String(S.md.thumb).startsWith("data:image") ?
    el("img", {src: S.md.thumb, alt: "thumbnail"}) : document.createTextNode("(no thumbnail)"));
}

function mdRow(k, v) {
  const row = el("div", {class: "md-row"}, el("input", {type: "text", class: "md-key", value: k, placeholder: "key"}),
    el("input", {type: "text", class: "md-val", value: v, placeholder: "value"}),
    el("button", {class: "btn btn-sm", text: "✕", title: "Remove this field", onclick: () => row.remove()}));
  $("mdCustom").append(row);
}

async function mdLoad(p) {
  mdWire();
  if (!p) return;
  let r;
  try { r = await api(`/api/metadata?path=${encodeURIComponent(p)}`); }
  catch (e) { $("mdStatus").textContent = e.message; return; }
  S.md.path = p;
  const meta = r.meta || {};
  document.querySelectorAll("#mdStandard [data-key]").forEach((i) => { i.value = meta[i.dataset.key] || ""; });
  S.md.thumb = meta["modelspec.thumbnail"] || null;
  mdThumb();
  const std = new Set([...MD_FIELDS.map((f) => f[0]), "modelspec.description", "modelspec.thumbnail"]);
  $("mdCustom").replaceChildren();
  for (const k of Object.keys(meta).filter((k) => !std.has(k)).sort()) mdRow(k, meta[k]);
  const n = Object.keys(meta).length;
  $("mdStatus").textContent = `Loaded - ${n} metadata key${n !== 1 ? "s" : ""} found.`;
  $("mdSaveMsg").textContent = "";
  $("mdDest").value = "";
}

async function mdSave(dest, overwrite) {
  if (!S.md.path) { $("mdSaveMsg").textContent = "Load a .safetensors file first."; return; }
  const meta = {};
  $("mdCustom").querySelectorAll(".md-row").forEach((r) => {
    const k = r.querySelector(".md-key").value.trim();
    if (k) meta[k] = r.querySelector(".md-val").value;
  });
  document.querySelectorAll("#mdStandard [data-key]").forEach((i) => { if (i.value.trim()) meta[i.dataset.key] = i.value.trim(); });
  if (S.md.thumb) meta["modelspec.thumbnail"] = S.md.thumb;
  if (!dest && !overwrite) {                      // the file itself: say so first, as the desktop asks
    $("mdSaveMsg").replaceChildren(el("span", {text: `Save the changes into ${S.md.path}? `}),
      el("button", {class: "btn btn-sm btn-primary", text: "Save", onclick: () => mdSave("", true)}));
    return;
  }
  $("mdSaveMsg").textContent = "Saving…";
  try {
    const r = await api("/api/metadata/save", {path: S.md.path, dest, meta, overwrite: !!overwrite});
    if (r.exists) {
      $("mdSaveMsg").replaceChildren(el("span", {text: "That file is there already. "}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => mdSave(dest, true)}));
      return;
    }
    if (dest) { $("mdPath").value = r.dest; await mdLoad(r.dest); }       // the copy is the file being edited now
    $("mdSaveMsg").textContent = `Saved ${r.dest}`;
  } catch (e) { $("mdSaveMsg").textContent = e.message; }
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

// ---- Repair Studio (fizgig/web/studio.py runs the desktop's engine in a worker that keeps the model loaded) -----
// The page holds the sliders; every change renders after a short pause, a newer change replacing the render going
S.st = {layout: null, blocks: {}, locks: new Set(), primaryBlocks: new Set(), donorBlocks: new Set(), donor: false,
  shown: 0, hist: [], redo: [], loadedKey: "", timer: null, poll: null, ready: false};
const ST_KEY = "fizgig_studio";
const stSaved = () => { try { return JSON.parse(localStorage.getItem(ST_KEY) || "{}"); } catch (e) { return {}; } };
const stRemember = debounce(() => {
  const d = {family: $("stFamily").value, primary: $("stPrimary").value, donor: $("stDonor").value,
    prompt: $("stPrompt").value, seed: $("stSeed").value, res: $("stRes").value, turbo: $("stTurbo").checked,
    ref: $("stRef").value, refMp: $("stRefMp").value, refStr: $("stRefStr").value, luckyLo: $("stLuckyLo").value,
    luckyHi: $("stLuckyHi").value, neg: {...(stSaved().neg || {}), [$("stFamily").value]: $("stNeg").value},
    scales: {...(stSaved().scales || {}), [$("stFamily").value]: [$("stPScale").value, $("stDScale").value]},
    h3: stSaved().h3, ace: stSaved().ace, metricsRef: stSaved().metricsRef};                           // H3's clip settings and photos (studioH3Remember's)
  try { localStorage.setItem(ST_KEY, JSON.stringify(d)); } catch (e) { /* private window */ }
}, 500);

async function studioOpen() {
  if (!S.st.familiesLoaded) {
    let r;
    try { r = await api("/api/studio/families"); }
    catch (e) {                                 // a server started before Studio existed: the page is newer than it
      $("stStatus").textContent = "This web UI was started before Studio was added: restart it (Preferences > Web UI " +
        "in the desktop app, or close and reopen the Fizgig window), then reload this page.";
      return;
    }
    const sel = $("stFamily");
    sel.innerHTML = "";
    for (const f of r.families) sel.append(el("option", {value: f.key, text: f.name}));
    const sv = stSaved();
    if (sv.family && r.families.some((f) => f.key === sv.family)) sel.value = sv.family;
    for (const [id, k] of [["stPrimary", "primary"], ["stDonor", "donor"], ["stPrompt", "prompt"], ["stSeed", "seed"],
      ["stRef", "ref"], ["stRefMp", "refMp"], ["stRefStr", "refStr"], ["stLuckyLo", "luckyLo"], ["stLuckyHi", "luckyHi"]])
      if (sv[k] != null && sv[k] !== "") $(id).value = sv[k];
    if (sv.turbo != null) $("stTurbo").checked = sv.turbo;
    S.st.familiesLoaded = true;
    await studioFamily(sv.res);
  }
  studioPoll();
}

async function studioFamily(res) {
  const key = $("stFamily").value;
  const L = S.st.layout = await api("/api/studio/layout?family=" + encodeURIComponent(key));
  const sv = stSaved();
  studioKind(L, sv);
  $("stRes").innerHTML = "";
  for (const s of L.sizes) $("stRes").append(el("option", {value: s, text: s}));
  $("stRes").value = L.sizes.includes(String(res)) ? String(res) : L.size;
  const sc = (sv.scales || {})[key];
  $("stPScale").value = sc ? sc[0] : L.default_strength;
  $("stDScale").value = sc ? sc[1] : 1;
  $("stNeg").value = (sv.neg || {})[key] ?? (L.negative_default || "");
  $("stTurboRow").style.display = L.turbo_preview ? "" : "none";
  $("stRefRow").style.display = L.reference_kind ? "" : "none";
  $("stRefMp").disabled = L.reference_kind === "edit";
  $("stRefStr").style.display = $("stRefStrLabel").style.display = L.reference_strength ? "" : "none";
  // which model previews: the follows-the-Samples-tab families say so; others pick fast or full
  $("stDitRow").style.display = L.follows_samples || !L.dit_fast ? "none" : "";
  $("stDitFast").textContent = L.dit_fast; $("stDitFull").textContent = L.dit_full;
  $("stDitNote").textContent = L.follows_samples && L.follow_ck ? L.follow_ck : L.follows_samples ? `Previews ${L.follow_turbo ? "on the " + L.follow_turbo + ", " : ""}` +
    `${L.follow_steps} steps, CFG ${L.follow_cfg} - from this model's Samples tab.` :
    !L.dit_fast ? L.dit_full.replace(", slow", "") : "";
  studioNegRow();
  S.st.blocks = {};
  for (const g of L.groups) for (const b of g.blocks) S.st.blocks[b.id] = {pe: true, ps: 1, de: true, ds: 1};
  S.st.locks = new Set(); S.st.hist = []; S.st.redo = [];
  studioMasters();
  studioPresets();
  studioRows();
}

// H3 (clips) and ACE-Step (songs): their own controls, the desktop's
function studioKind(L, sv) {
  const video = L.kind === "video", audio = L.kind === "audio";
  $("stVideoRows").style.display = video ? "" : "none";
  $("stFrames").style.display = video ? "" : "none";
  $("stResLabel").textContent = audio ? "Length (s):" : "Res:";
  $("stRes").style.display = $("stResLabel").style.display = video ? "none" : "";
  document.querySelectorAll(".st-pane").forEach((p) => p.classList.toggle("audio", audio));
  $("stPreviewHint").textContent = video ? "Each clip plays with its sound; the picture is its middle frame." :
    audio ? "Play any song; the picture is its waveform." : "Click either picture to see both full size (← → switches between them).";
  $("stPlayBoth").style.display = video || audio ? "" : "none";
  studioAce(L, sv);
  studioEqButtons(L);
  studioMixShow(L);
  studioMetricsShow();
  if (!video) return;
  const V = L.video, h3 = sv.h3 || {};
  const fill = (id, opts, val) => { const s = $(id); s.innerHTML = ""; for (const o of opts) s.append(el("option", {value: o, text: o})); s.value = opts.includes(val) ? val : opts[0]; };
  fill("stH3Model", V.models, h3.model || V.models[0]);   // the desktop starts on First/last frame
  fill("stH3Base", V.bases, h3.base);
  fill("stH3Len", V.lengths, h3.len || "22 frames (~1s)");
  fill("stH3W", V.dims, h3.w || "768");
  fill("stH3H", V.dims, h3.h || "640");
  fill("stH3Scale", V.scales, h3.scale || "Full");
  if (h3.steps) $("stH3Steps").value = h3.steps;
  if (h3.turbo != null) $("stH3Turbo").value = h3.turbo;
  $("stH3Sound").checked = V.sound && h3.sound !== false;
  $("stH3Sound").disabled = !V.sound;
  $("stH3NoLora").checked = !!h3.nolora;
  S.st.kf = h3.kf || {};
  S.st.kfRect = h3.kfRect || {};
  studioSlots();
}

function studioSlots() {
  const ref = $("stH3Model").value.includes("ref2va");
  $("stFramesTitle").textContent = ref ? "Reference Images" : "First / Last Frame";
  $("stFramesDesc").textContent = ref ? "Optional: up to two photos the clip should look like (the ref2va model's " +
    "references), each cropped to the clip's shape - you place the crop when you pick one." :
    "Optional: photos the clip starts and ends on (a still uses the first only), each cropped to the clip's shape - " +
    "you place the crop when you pick one.";
  const host = $("stFrameSlots");
  host.innerHTML = "";
  const slots = ref ? [["ref1", "Reference 1"], ["ref2", "Reference 2"]] : [["first", "First frame"], ["last", "Last frame"]];
  for (const [slot, label] of slots) {
    const input = el("input", {type: "text", class: "w-path", value: (S.st.kf[slot] || ""), placeholder: "a photo (optional)"});
    const thumb = el("canvas", {width: 1, height: 1});
    const info = el("span", {class: "field-hint"});
    const show = () => {
      const p = S.st.kf[slot], r = S.st.kfRect[slot];
      input.value = p || "";
      info.textContent = p && r ? `crop ${r[2] - r[0]}×${r[3] - r[1]}` : "";
      studioSlotThumb(thumb, p, r);
    };
    // a new photo: place its crop (Cancel keeps what was there); Re-crop moves the box of the one there
    const set = (p) => {
      if (!p) { delete S.st.kf[slot]; delete S.st.kfRect[slot]; show(); studioH3Remember(); studioSchedule(); return; }
      studioCrop(p, p === S.st.kf[slot] ? S.st.kfRect[slot] : null, (rect) => {
        S.st.kf[slot] = p; S.st.kfRect[slot] = rect; show(); studioH3Remember(); studioSchedule();
      }, show);
    };
    input.addEventListener("change", () => set(input.value.trim()));
    const up = uploadButton({kind: "path", key: "STUDIO_" + slot}, input, set);
    host.append(el("div", {class: "st-slot"}, el("span", {class: "field-label", text: label + ":"}), thumb,
      el("div", {}, el("div", {class: "field-row"}, input,
        el("button", {class: "btn", text: "Choose…", onclick: () => pick(input.value || S.imageFolder, false, set)}), up.btn,
        el("button", {class: "btn", text: "Re-crop", onclick: () => S.st.kf[slot] && set(S.st.kf[slot])}),
        el("button", {class: "btn", text: "Clear", onclick: () => set("")}), info), up.status)));
    show();
  }
}

const mediaUrl = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;

function studioSlotThumb(cv, path, rect) {
  // the slot's thumbnail: the crop itself, as the desktop shows it
  const ctx = cv.getContext("2d");
  if (!path) { cv.width = cv.height = 1; return; }
  const img = new Image();
  img.onload = () => {
    const r = rect || [0, 0, img.naturalWidth, img.naturalHeight];
    const w = r[2] - r[0], h = r[3] - r[1], k = Math.min(160 / w, 96 / h);
    cv.width = Math.max(1, Math.round(w * k)); cv.height = Math.max(1, Math.round(h * k));
    ctx.drawImage(img, r[0], r[1], w, h, 0, 0, cv.width, cv.height);
  };
  img.src = mediaUrl(path);
}

function studioCrop(path, initial, ok, cancel) {
  // the desktop's crop window: the photo fitted, a box locked to the clip's shape (W x H) - drag to move, drag a
  // corner to resize. ok([x0, y0, x1, y1] in the photo's pixels); cancel() leaves things as they were.
  const W = Number($("stH3W").value) || 768, H = Number($("stH3H").value) || 640, aspect = W / H;
  const cv = $("cropCanvas"), ctx = cv.getContext("2d"), img = new Image();
  $("cropTitle").textContent = `Crop to the clip's shape (${W}×${H})`;
  img.onerror = () => { $("stStatus").textContent = "Couldn't open that photo."; if (cancel) cancel(); };
  img.onload = () => {
    const sw = img.naturalWidth, sh = img.naturalHeight;
    const scale = Math.min((window.innerWidth * 0.9 - 40) / sw, (window.innerHeight * 0.85 - 200) / sh, 1);
    const dw = Math.max(1, Math.round(sw * scale)), dh = Math.max(1, Math.round(sh * scale));
    cv.width = dw; cv.height = dh;
    const HANDLE = 9, MIN_W = 32 * scale * 4;
    let box;
    const centre = () => { const bw = dw / dh >= aspect ? dh * aspect : dw, bh = bw / aspect;
      box = {x0: (dw - bw) / 2, y0: (dh - bh) / 2, x1: (dw + bw) / 2, y1: (dh + bh) / 2}; };
    if (initial) box = {x0: initial[0] * scale, y0: initial[1] * scale, x1: initial[2] * scale, y1: initial[3] * scale};
    else centre();
    const src = () => [box.x0, box.y0, box.x1, box.y1].map((v, i) => Math.min(i % 2 ? sh : sw, Math.max(0, Math.round(v / scale))));
    const draw = () => {
      ctx.clearRect(0, 0, dw, dh); ctx.drawImage(img, 0, 0, dw, dh);
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, 0, dw, box.y0); ctx.fillRect(0, box.y1, dw, dh - box.y1);
      ctx.fillRect(0, box.y0, box.x0, box.y1 - box.y0); ctx.fillRect(box.x1, box.y0, dw - box.x1, box.y1 - box.y0);
      ctx.strokeStyle = "#3B9FD8"; ctx.lineWidth = 2; ctx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
      ctx.fillStyle = "#3B9FD8";
      for (const [hx, hy] of [[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1], [box.x1, box.y1]])
        ctx.fillRect(hx - HANDLE, hy - HANDLE, HANDLE * 2, HANDLE * 2);
      const r = src();
      $("cropSize").textContent = `crop ${r[2] - r[0]}×${r[3] - r[1]} of ${sw}×${sh} → ${W}×${H}`;
    };
    const at = (e) => { const b = cv.getBoundingClientRect(); return [(e.clientX - b.left) * dw / b.width, (e.clientY - b.top) * dh / b.height]; };
    let drag = null;
    cv.onpointerdown = (e) => {
      const [x, y] = at(e);
      const corner = [["nw", box.x0, box.y0], ["ne", box.x1, box.y0], ["sw", box.x0, box.y1], ["se", box.x1, box.y1]]
        .find(([, hx, hy]) => Math.abs(x - hx) <= HANDLE * 1.5 && Math.abs(y - hy) <= HANDLE * 1.5);
      if (corner) drag = {mode: "resize", c: corner[0]};
      else if (x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1) drag = {mode: "move", dx: x - box.x0, dy: y - box.y0};
      else return;
      cv.setPointerCapture(e.pointerId);
    };
    cv.onpointermove = (e) => {
      if (!drag) return;
      const [x, y] = at(e);
      if (drag.mode === "move") {
        const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
        const nx = Math.min(Math.max(0, x - drag.dx), dw - bw), ny = Math.min(Math.max(0, y - drag.dy), dh - bh);
        box = {x0: nx, y0: ny, x1: nx + bw, y1: ny + bh};
      } else {
        // the opposite corner anchors; the width follows the pointer, the height the shape; clamped to the photo
        const c = drag.c, east = c === "ne" || c === "se", south = c === "sw" || c === "se";
        const ax = east ? box.x0 : box.x1, ay = south ? box.y0 : box.y1;
        let w = Math.abs(Math.min(Math.max(x, 0), dw) - ax), h = Math.abs(Math.min(Math.max(y, 0), dh) - ay);
        w = Math.max(MIN_W, w, h * aspect); h = w / aspect;
        const maxW = east ? dw - ax : ax, maxH = south ? dh - ay : ay;
        if (w > maxW) { w = maxW; h = w / aspect; }
        if (h > maxH) { h = maxH; w = h * aspect; }
        box = {x0: east ? ax : ax - w, x1: east ? ax + w : ax, y0: south ? ay : ay - h, y1: south ? ay + h : ay};
      }
      draw();
    };
    cv.onpointerup = () => { drag = null; };
    const close = () => { $("cropModal").classList.remove("active"); cv.onpointerdown = cv.onpointermove = cv.onpointerup = null; };
    $("cropOk").onclick = () => { close(); ok(src()); };
    $("cropCancel").onclick = $("cropX").onclick = () => { close(); if (cancel) cancel(); };
    $("cropCentre").onclick = () => { centre(); draw(); };
    draw();
    $("cropModal").classList.add("active");
  };
  img.src = mediaUrl(path);
}

function studioH3Remember() {
  try {
    const d = stSaved();
    d.h3 = {model: $("stH3Model").value, base: $("stH3Base").value, len: $("stH3Len").value, w: $("stH3W").value,
      h: $("stH3H").value, scale: $("stH3Scale").value, steps: $("stH3Steps").value, turbo: $("stH3Turbo").value,
      sound: $("stH3Sound").checked, nolora: $("stH3NoLora").checked, kf: S.st.kf, kfRect: S.st.kfRect};
    localStorage.setItem(ST_KEY, JSON.stringify(d));
  } catch (e) { /* private window */ }
}

function studioExtra() {
  // what a clip or a song needs beyond the sliders
  const L = S.st.layout;
  if (L.kind === "audio") return {audio: studioAceExtra()};
  if (L.kind !== "video") return {};
  const V = L.video, f = V.scale_of[$("stH3Scale").value] || 1;
  const snap = (x) => f >= 0.999 ? x : Math.max(256, Math.floor(x * f / 32) * 32);
  const w = snap(Number($("stH3W").value) || 768), h = snap(Number($("stH3H").value) || 640);
  const ref = $("stH3Model").value.includes("ref2va");
  const slots = ref ? ["ref1", "ref2"] : ["first", "last"];
  return {video: {frames: V.length_frames[$("stH3Len").value] || 22, width: w, height: h,
    steps: Math.max(1, Math.min(40, parseInt($("stH3Steps").value, 10) || 4)),
    turbo: Math.max(0, Math.min(1.5, Number($("stH3Turbo").value) || 0)), sound: $("stH3Sound").checked,
    nolora: $("stH3NoLora").checked},
    keyframes: slots.filter((s) => S.st.kf[s]).map((s) => ({slot: s, path: S.st.kf[s], rect: S.st.kfRect[s] || null})),
    ref_mode: ref};
}

// ---- ACE-Step: Prompt B, audio to audio, the reference recording, the section on the waveform (the desktop's) ----
const snap05 = (v) => Math.round(Number(v) / 0.05) * 0.05;

function studioAce(L, sv) {
  const on = L.kind === "audio";
  $("stAudioRows").style.display = on && (L.prompt_blend || L.audio_source) ? "" : "none";
  $("stSongTools").style.display = on ? "" : "none";
  if (!on) { S.st.region = null; studioRegionDraw(); return; }
  const a = sv.ace || {};
  $("stPbOn").parentElement.style.display = L.prompt_blend ? "" : "none";
  $("stSrcOn").parentElement.style.display = $("stSndOn").parentElement.style.display = L.audio_source ? "" : "none";
  $("stPbOn").checked = !!a.pbOn && L.prompt_blend; $("stPromptB").value = a.pb || "";
  $("stPbBlend").value = a.blend ?? 0.5; $("stPbKeep").checked = a.keep !== false;
  $("stSrcOn").checked = !!a.srcOn && L.audio_source; $("stSrcPath").value = a.src || "";
  $("stSrcMode").value = a.mode || "song"; $("stSrcFollow").value = a.follow ?? 1; $("stSrcDenoise").value = a.denoise ?? 1;
  $("stSrcStrength").value = a.strength ?? 0.5;
  $("stSndOn").checked = !!a.sndOn && L.audio_source; $("stSndPath").value = a.snd || "";
  S.st.region = null;
  studioAceLayout();
}

function studioAceLayout() {
  $("stPbRows").style.display = $("stPbOn").checked ? "" : "none";
  $("stSrcRows").style.display = $("stSrcOn").checked ? "" : "none";
  $("stSndRows").style.display = $("stSndOn").checked ? "" : "none";
  const whole = $("stSrcMode").value === "song";
  $("stSrcWhole").style.display = whole ? "" : "none";
  $("stSrcRep").style.display = whole ? "none" : "";
  const w = snap05($("stPbBlend").value);
  $("stPbReadout").textContent = `${Math.round((1 - w) * 100)}% A · ${Math.round(w * 100)}% B`;
}

function studioAceRemember() {
  try {
    const d = stSaved();
    d.ace = {pbOn: $("stPbOn").checked, pb: $("stPromptB").value, blend: $("stPbBlend").value, keep: $("stPbKeep").checked,
      srcOn: $("stSrcOn").checked, src: $("stSrcPath").value.trim(), mode: $("stSrcMode").value,
      follow: $("stSrcFollow").value, denoise: $("stSrcDenoise").value, strength: $("stSrcStrength").value,
      sndOn: $("stSndOn").checked, snd: $("stSndPath").value.trim()};
    localStorage.setItem(ST_KEY, JSON.stringify(d));
  } catch (e) { /* private window */ }
}

function studioPromptB() {
  // the Prompt B the engine renders: B as typed, or (Keep Prompt A's lyrics) B's style with A's lyrics
  if (!$("stPbOn").checked) return null;
  const b = $("stPromptB").value.trim();
  if (!b) return null;
  if (!$("stPbKeep").checked) return b;
  const a = $("stPrompt").value, lyrics = a.includes("|") ? a.split("|").slice(1).join("|").trim() : "";
  const style = b.split("|")[0].trim();
  return lyrics ? `${style} | ${lyrics}` : style;
}

function studioAceExtra() {
  const pb = studioPromptB();
  return {seconds: Number($("stRes").value) || 30, prompt_b: pb, blend: pb ? snap05($("stPbBlend").value) : 0,
    source: $("stSrcOn").checked && $("stSrcPath").value.trim() ? {path: $("stSrcPath").value.trim(), mode: $("stSrcMode").value,
      follow: snap05($("stSrcFollow").value), denoise: snap05($("stSrcDenoise").value),
      strength: snap05($("stSrcStrength").value), region: S.st.region} : null,
    reference: $("stSndOn").checked && $("stSndPath").value.trim() ? $("stSndPath").value.trim() : null};
}

function studioRegionDraw() {
  const r = S.st.region, box = $("stRegion"), img = $("stTweak"), secs = Number($("stRes").value) || 30;
  if (!r || img.style.display === "none" || !img.naturalWidth) { box.style.display = "none"; }
  else {
    box.style.display = "block";
    box.style.top = img.offsetTop + "px"; box.style.height = img.offsetHeight + "px";
    box.style.left = (img.offsetLeft + img.offsetWidth * r[0] / secs) + "px";
    box.style.width = (img.offsetWidth * (r[1] - r[0]) / secs) + "px";
  }
  $("stRegionNote").textContent = r ? `Section ${r[0].toFixed(1)}–${r[1].toFixed(1)} s: the players loop it` +
    ($("stSrcOn").checked && $("stSrcMode").value === "repaint" ? "; Repaint remakes it." : ".") :
    "Drag across the Tweaked waveform to loop a section (and choose what Repaint remakes).";
}

function studioRecordButton(input, set) {
  // a take from this device's microphone (the browser's), the Length long, sent to the Fizgig machine
  const btn = el("button", {class: "btn", text: "🎙 Record", title: "Record from the microphone of the device you're using, for the Length setting, then use the take."});
  btn.addEventListener("click", async () => {
    if (btn.disabled) return;
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({audio: true}); }
    catch (e) { $("stStatus").textContent = `Could not start the microphone: ${e.message}`; return; }
    const secs = Math.max(1, Number($("stRes").value) || 30), rec = new MediaRecorder(stream), parts = [];
    rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(tick); btn.disabled = false; btn.textContent = "🎙 Record";
      const ext = (rec.mimeType || "audio/webm").includes("ogg") ? "ogg" : rec.mimeType.includes("mp4") ? "m4a" : "webm";
      const file = new File(parts, `take_${Date.now()}.${ext}`, {type: rec.mimeType || "audio/webm"});
      try {
        const res = await uploadResumable({single: 1, folder: "", name: file.name}, file, () => {});
        if (!res.ok) throw new Error(res.data.error || "upload failed");
        input.value = res.data.path; set(res.data.path);
        $("stStatus").textContent = `Recorded ${secs} s.`;
      } catch (e) { $("stStatus").textContent = `Recording failed: ${e.message}`; }
    };
    const t0 = Date.now();
    const tick = setInterval(() => { $("stStatus").textContent = `Recording: ${Math.max(0, secs - (Date.now() - t0) / 1000).toFixed(0)} s left…`; }, 250);
    btn.disabled = true; btn.textContent = "⏺ Recording…";
    rec.start(); setTimeout(() => rec.state !== "inactive" && rec.stop(), secs * 1000);
  });
  return btn;
}

function wireStudioAce() {
  const change = () => { studioAceLayout(); studioAceRemember(); studioRegionDraw(); studioSchedule(); };
  for (const id of ["stPbOn", "stSrcOn", "stSndOn", "stSrcMode", "stPbKeep", "stPbBlend", "stSrcFollow", "stSrcDenoise", "stSrcStrength"])
    $(id).addEventListener("change", change);
  for (const id of ["stPbBlend"]) $(id).addEventListener("input", studioAceLayout);
  // typed text and new files render on Update, as the prompt does
  const typed = () => { studioAceRemember(); if (S.st.ready) $("stStart").textContent = "Update"; };
  for (const id of ["stPromptB", "stSrcPath", "stSndPath"]) $(id).addEventListener("input", typed);
  for (const [id, host] of [["stSrcPath", "stSrcBtns"], ["stSndPath", "stSndBtns"]]) {
    const input = $(id), set = (p) => { input.value = p; studioAceRemember(); studioSchedule(); };
    const up = uploadButton({kind: "path", key: "STUDIO_" + id}, input, set);
    $(host).replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick(input.value || S.imageFolder, false, set)}),
      up.btn, studioRecordButton(input, set), up.status);
  }
  // the section: drag across the Tweaked waveform; the players loop it
  const img = $("stTweak");
  let a0 = null;
  const secAt = (e) => { const b = img.getBoundingClientRect(), secs = Number($("stRes").value) || 30;
    return Math.min(secs, Math.max(0, (e.clientX - b.left) / b.width * secs)); };
  img.addEventListener("click", (e) => { if (S.st.layout && S.st.layout.kind === "audio") e.stopPropagation(); });   // no viewer for a waveform
  img.addEventListener("pointerdown", (e) => { if (!(S.st.layout && S.st.layout.kind === "audio")) return;
    e.preventDefault(); a0 = secAt(e); img.setPointerCapture(e.pointerId); });
  img.addEventListener("pointermove", (e) => { if (a0 == null) return; const t = secAt(e);
    S.st.region = [Math.min(a0, t), Math.max(a0, t)]; studioRegionDraw(); });
  img.addEventListener("pointerup", () => {
    if (a0 == null) return;
    a0 = null;
    if (S.st.region && S.st.region[1] - S.st.region[0] < 0.25) S.st.region = null;   // a click, not a drag
    studioRegionDraw();
    if ($("stSrcOn").checked && $("stSrcMode").value === "repaint") studioSchedule();
  });
  $("stRegionClear").addEventListener("click", () => { S.st.region = null; studioRegionDraw();
    if ($("stSrcOn").checked && $("stSrcMode").value === "repaint") studioSchedule(); });
  for (const id of ["stNolAud", "stBaseAud", "stTweakAud"])
    $(id).addEventListener("timeupdate", (e) => { const r = S.st.region, m = e.target;
      if (r && S.st.layout && S.st.layout.kind === "audio" && (m.currentTime >= r[1] || m.currentTime < r[0] - 0.05)) m.currentTime = r[0]; });
  window.addEventListener("resize", studioRegionDraw);
  $("stTweak").addEventListener("load", studioRegionDraw);
  $("stKeepBtn").addEventListener("click", async () => {
    try {
      const r = await api("/api/studio/keep", {});
      $("stSrcOn").checked = true; $("stSrcPath").value = r.path;
      if ($("stSrcMode").value === "song") $("stSrcMode").value = "repaint";
      studioAceLayout(); studioAceRemember(); studioRegionDraw();
      $("stStatus").textContent = "The Tweaked song is now the source: drag across the waveform to choose the section to repaint" +
        (S.st.region ? " (or keep this one)." : ".");
      if (S.st.region) studioSchedule();
    } catch (e) { $("stStatus").textContent = e.message; }
  });
}

function studioAceNotes(n) {
  if (!n) return;
  $("stSrcDenoiseNote").textContent = n.denoise || ""; $("stSrcFollowNote").textContent = n.follow || "";
  $("stSrcRepNote").textContent = n.strength || "";
}

// ---- Model EQ Mode (a family with a Model EQ LoRA in Preferences): the row holds the model's own weights ----
const ST_EQ_ROWS = {primary: ["stPrimary", "stPrimaryPick", "stPScale", "stEqP"], donor: ["stDonor", "stDonorPick", "stDScale", "stEqD"]};

function studioEqButtons(L) {
  const on = !!(L && L.model_eq);
  for (const role of ["primary", "donor"]) {
    if (!on && S.st.eq[role]) studioEq(role, false, false);
    $(ST_EQ_ROWS[role][3]).style.display = on ? "" : "none";
  }
}

function studioEq(role, on, restore = true) {
  // on: the row loads the Model EQ LoRA at strength 1, its path, Choose and strength locked (the button lit);
  // off: unlocked and (restore) the LoRA and strength it had before come back - as the desktop's _repair_set_eq_mode
  const [pathId, pickId, scaleId, btnId] = ST_EQ_ROWS[role];
  const L = S.st.layout;
  if (on) {
    if (!L || !L.model_eq) { $("stStatus").textContent = "Set the Model EQ LoRA in Preferences first."; return; }
    S.st.eqSaved[role] = [$(pathId).value, $(scaleId).value];
    $(pathId).value = L.model_eq; $(scaleId).value = "1";
  }
  S.st.eq[role] = on;
  for (const id of [pathId, pickId, scaleId]) $(id).disabled = on;
  $(btnId).classList.toggle("btn-primary", on);
  if (on) {
    if (role === "primary") { $("stStatus").textContent = "Model EQ as the primary: press Start / Update to load it (1.0 = the stock model).";
      if (S.st.ready) $("stStart").textContent = "Update"; }
    else if (S.st.ready) api("/api/studio/donor", {path: L.model_eq}).then(studioPoll).catch((e) => { $("stStatus").textContent = e.message; });
    stRemember();
    return;
  }
  if (!restore) { delete S.st.eqSaved[role]; return; }
  const [prev, prevScale] = S.st.eqSaved[role] || ["", String((L && L.default_strength) || 1)];
  delete S.st.eqSaved[role];
  $(scaleId).value = prevScale || "1";
  if (role === "primary") {
    $(pathId).value = prev;
    $("stStatus").textContent = "Model EQ Mode off: the primary is your own LoRA again - press Update to load it.";
    if (S.st.ready) $("stStart").textContent = "Update";
  } else {
    $(pathId).value = prev;
    if (S.st.ready) api("/api/studio/donor", {path: prev}).then(() => { studioPoll(); studioSchedule(); });
  }
  stRemember();
}

function studioEqNote(st) {
  // after a load: what the sliders mean for a Model EQ row (the engine's gain for that file)
  const eq = st.eq || {}, bits = [];
  for (const [role, g] of Object.entries(eq)) if (g != null)
    bits.push(`${role === "primary" ? "Primary" : "Donor"} is the Model EQ: 1.0 = the stock model, 0 / 2 = -${Math.round(g * 100)}% / +${Math.round(g * 100)}% of each block's main directions.`);
  return bits.join(" ");
}

// ---- the mix: Epochs of this run / LoRA mix for the primary, the donor's epoch blend (the desktop's) ----
// S.st.mix = {mode: "off"|"epochs"|"mix", epochs: [n...], upto, hist, rows: [[path, amount]...]}; S.st.dmix the donor's
function studioMixReset(note) {
  S.st.mix = {mode: "off", epochs: [], rows: []}; S.st.dmix = {on: false, epochs: []};
  document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
  $("stDmixOn").checked = false;
  $("stMixInfo").textContent = note || ""; $("stDmixInfo").textContent = "";
  studioMixLayout();
}

function studioDmixOff() {
  if (!S.st.dmix) return;
  S.st.dmix = {on: false, epochs: []}; $("stDmixOn").checked = false; $("stDmixInfo").textContent = ""; studioMixLayout();
}

function studioMixShow(L) {
  const on = !!L && L.kind !== "video";                 // H3's engine has no mix (as the desktop)
  $("stMixRows").style.display = $("stDmixRows").style.display = on ? "" : "none";
  studioMixReset();
}

function studioMixLayout() {
  const m = S.st.mix || {mode: "off"};
  $("stMixEpochs").style.display = m.mode === "epochs" ? "" : "none";
  $("stMixList").style.display = m.mode === "mix" ? "" : "none";
  for (const [p, st, on] of [["stMix", m, m.mode === "epochs"], ["stDmix", S.st.dmix || {}, !!(S.st.dmix || {}).on]]) {
    const eps = st.epochs || [];
    for (const id of [p + "Upto", p + "Hist"]) $(id).disabled = !on || eps.length < 2;
    $(p + "Upto").max = Math.max(1, eps.length - 1);
    const i = Math.max(0, Math.min(eps.length - 1, parseInt($(p + "Upto").value, 10) || 0));
    $(p + "UptoTxt").textContent = eps.length ? `epoch ${eps[i]}` : "";
    const h = snap05($(p + "Hist").value);
    $(p + "HistTxt").textContent = !eps.length ? "" : h < 0.025 ? "0 (that epoch alone)" : `${h.toFixed(2)} (about ${Math.round(1 / (1 - h))} epochs)`;
  }
}

async function studioMixEpochs(role) {
  // the run beside the role's file; false when there is nothing to blend (the note says why)
  const path = (role === "primary" ? $("stPrimary") : $("stDonor")).value.trim();
  const info = role === "primary" ? "stMixInfo" : "stDmixInfo";
  if (!S.st.ready || (role === "donor" && !S.st.donor)) {
    $("stStatus").textContent = `Load the ${role} LoRA first, then ${role === "primary" ? "pick a mix" : "tick its epoch blend"}.`;
    return false;
  }
  let r;
  try { r = await api(`/api/studio/epochs?path=${encodeURIComponent(path)}`); }
  catch (e) { $("stStatus").textContent = `Could not read the ${role}'s epochs: ${e.message}`; return false; }
  if ((r.epochs || []).length < 2) {
    $(info).textContent = `Only one epoch of the ${role}'s run is beside its file, so there is nothing to blend.`;
    return false;
  }
  const st = role === "primary" ? S.st.mix : S.st.dmix, p = role === "primary" ? "stMix" : "stDmix";
  st.epochs = r.epochs;
  $(p + "Upto").max = r.epochs.length - 1;
  $(p + "Upto").value = r.epochs.indexOf(r.own) >= 0 ? r.epochs.indexOf(r.own) : r.epochs.length - 1;
  return true;
}

function studioMixRows() {
  const host = $("stMixListRows");
  host.replaceChildren();
  S.st.mix.rows.forEach(([path, amount], i) => {
    const name = path.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
    const val = el("span", {class: "field-hint", text: Number(amount).toFixed(2)});
    const range = el("input", {type: "range", min: "-1", max: "2", step: "0.05", value: String(amount)});
    range.addEventListener("input", () => { val.textContent = Number(range.value).toFixed(2); });
    range.addEventListener("change", () => { S.st.mix.rows[i][1] = snap05(range.value); studioSchedule(); });
    host.append(el("div", {class: "field-row"}, el("span", {class: "field-label"}), el("div", {class: "st-mixrow"},
      el("span", {class: "st-mixname", title: path, text: (i === 0 ? "★ " : "") + name}), range, val,
      i === 0 ? el("span", {class: "field-hint", text: "(the primary)"}) :
        el("button", {class: "btn btn-sm", text: "✕", title: "Take it out of the mix", onclick: () => {
          S.st.mix.rows.splice(i, 1); studioMixRows(); studioSchedule(); }}))));
  });
}

async function studioMixMode(mode) {
  const m = S.st.mix;
  if (mode !== "off" && !S.st.ready) {
    document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
    $("stStatus").textContent = "Load the primary LoRA first, then pick a mix.";
    return;
  }
  m.mode = mode;
  if (mode === "epochs" && !(await studioMixEpochs("primary"))) {
    m.mode = "off"; document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
  }
  if (mode === "mix") {
    const prim = $("stPrimary").value.trim();
    if (!m.rows.length || m.rows[0][0] !== prim) m.rows = [[prim, 1.0]];
    studioMixRows();
  }
  if (m.mode === "off") $("stMixInfo").textContent = "Off: the primary file alone.";
  studioMixLayout();
  studioSchedule();
}

function studioMixExtra() {
  // what the worker mixes before the render: {primary, donor}, each null for the file alone
  const L = S.st.layout;
  if (!L || L.kind === "video" || !S.st.mix) return {};
  const m = S.st.mix, d = S.st.dmix || {};
  const dial = (p, eps) => ({upto: eps[Math.max(0, Math.min(eps.length - 1, parseInt($(p + "Upto").value, 10) || 0))],
    hist: snap05($(p + "Hist").value)});
  return {mix: {
    primary: m.mode === "epochs" && m.epochs.length > 1 ? {mode: "epochs", ...dial("stMix", m.epochs)} :
      m.mode === "mix" && m.rows.length ? {mode: "mix", paths: m.rows.map((r) => r[0]), amounts: m.rows.map((r) => r[1])} : null,
    donor: d.on && d.epochs.length > 1 && S.st.donor ? {mode: "epochs", ...dial("stDmix", d.epochs)} : null}};
}

function wireStudioMix() {
  document.querySelectorAll('input[name="stMixMode"]').forEach((r) => r.addEventListener("change", () => studioMixMode(r.value)));
  for (const p of ["stMix", "stDmix"]) for (const id of [p + "Upto", p + "Hist"]) {
    $(id).addEventListener("input", studioMixLayout);
    $(id).addEventListener("change", () => { studioMixLayout(); studioSchedule(); });
  }
  $("stDmixOn").addEventListener("change", async () => {
    S.st.dmix.on = $("stDmixOn").checked;
    if (S.st.dmix.on && !(await studioMixEpochs("donor"))) { S.st.dmix.on = false; $("stDmixOn").checked = false; }
    if (!S.st.dmix.on) $("stDmixInfo").textContent = "Off: the donor file alone.";
    studioMixLayout(); studioSchedule();
  });
  $("stMixAdd").addEventListener("click", () => pick($("stPrimary").value.trim(), false, async (p) => {
    if (S.st.mix.rows.some((r) => r[0].toLowerCase() === p.toLowerCase())) return;
    try {
      const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
      if (f.key && S.st.layout && f.key !== S.st.layout.key) {
        $("stStatus").textContent = `Not added: ${p.split(/[\\/]/).pop()} is a LoRA for ${f.name}.`;
        return;
      }
    } catch (e) { /* unreadable: the render says so */ }
    S.st.mix.rows.push([p, 0.5]); studioMixRows(); studioSchedule();
  }));
}

function studioMixNotes(n) {
  if (!n) return;
  if (n.mix_primary != null && S.st.mix && S.st.mix.mode !== "off") $("stMixInfo").textContent = n.mix_primary;
  if (n.mix_donor != null && S.st.dmix && S.st.dmix.on) $("stDmixInfo").textContent = n.mix_donor;
}

// ---- the metrics strip (the desktop's pop-out strip): paired numbers for the renders on screen ----
function studioMetricsShow() {
  const L = S.st.layout, on = !!L && L.kind !== "audio";
  $("stMetrics").style.display = on ? "" : "none";
  const ref = (stSaved().metricsRef) || "";
  $("stMetRefName").textContent = ref ? ref.split(/[\\/]/).pop() : "(none)";
}

async function studioMetrics() {
  const L = S.st.layout;
  if (!L || L.kind === "audio" || !S.st.shown) return;
  const gen = (S.st.metGen = (S.st.metGen || 0) + 1), ref = stSaved().metricsRef || "";
  const chips = {like: "stMetLike", grid: "stMetGrid", tex: "stMetTex", clip: "stMetClip", sat: "stMetSat"};
  const put = (k, text, cls) => { const c = $(chips[k]); c.textContent = text; c.className = "st-chip" + (cls ? " " + cls : ""); };
  for (const k in chips) put(k, "…");
  let m;
  try { m = await api(`/api/studio/metrics?ref=${encodeURIComponent(ref)}`); } catch (e) { m = {error: e.message}; }
  if (gen !== S.st.metGen) return;                    // a newer pair is on screen: these numbers aren't its
  if (m.error) { put("like", `metrics failed: ${String(m.error).slice(0, 60)}`, "bad"); for (const k of ["grid", "tex", "clip", "sat"]) put(k, ""); return; }
  if (!m.ref_set) put("like", "Likeness: set a reference photo →");
  else if (!m.ref_face) put("like", "Likeness: no face in reference", "bad");
  else if (m.like_base == null || m.like_tweak == null) put("like", "Likeness: no face in render");
  else {
    const lb = m.like_base * 100, lt = m.like_tweak * 100, d = lt - lb;
    put("like", `Likeness ${lb.toFixed(0)}% ${d > 0.5 ? "▲" : d < -0.5 ? "▼" : "→"} ${lt.toFixed(0)}%`, d > 0.5 ? "good" : d < -0.5 ? "bad" : "");
  }
  put("grid", `Grid ${m.grid_base.toFixed(2)} → ${m.grid_tweak.toFixed(2)}`, m.grid_delta > 0.05 ? "bad" : m.grid_delta < -0.05 ? "good" : "");
  const td = (m.texture_tweak - m.texture_base) / Math.max(m.texture_base, 1e-6);
  put("tex", `Detail ${m.texture_base.toFixed(0)} → ${m.texture_tweak.toFixed(0)}`, Math.abs(td) > 0.10 ? "warm" : "");
  put("clip", `Clipped ${m.clip_base.toFixed(1)}% → ${m.clip_tweak.toFixed(1)}%`, m.clip_delta > 0.2 ? "bad" : m.clip_delta < -0.2 ? "good" : "");
  put("sat", `Sat ${m.sat_base.toFixed(0)} → ${m.sat_tweak.toFixed(0)}`, Math.abs(m.sat_delta) > 12 ? "warm" : "");
}

function wireStudioMetrics() {
  const setRef = (p) => {
    try { const d = stSaved(); d.metricsRef = p || ""; localStorage.setItem(ST_KEY, JSON.stringify(d)); } catch (e) { /* private window */ }
    studioMetricsShow(); studioMetrics();
  };
  $("stMetRefPick").addEventListener("click", () => pick(stSaved().metricsRef || S.imageFolder, false, setRef));
  $("stMetRefClear").addEventListener("click", () => setRef(""));
}

// ---- Save / Load Setup: the whole session in one file, in the desktop's format (its variable names), so either
// app loads the other's. What only the web has (H3's photos and crops, the mix, the section) rides along in "web".
const ST_SRC_MODES = [["song", "Whole song"], ["repaint", "Repaint the loop section"], ["extend", "Extend to the Length"]];
const ST_SETUP_MAP = [
  ["stPrimary", "repair_primary_var"], ["stDonor", "repair_donor_var"], ["stPScale", "repair_primary_scale_var"],
  ["stDScale", "repair_donor_scale_var"], ["stPrompt", "repair_prompt_var"], ["stSeed", "repair_seed_var"],
  ["stRes", "repair_res_var"], ["stNeg", "repair_negative_var"], ["stTurbo", "repair_turbo_var", "check"],
  ["stRef", "repair_ref_path_var"], ["stRefMp", "repair_ref_mp_var"], ["stRefStr", "repair_ref_strength_var"],
  ["stLuckyLo", "repair_lucky_min_var"], ["stLuckyHi", "repair_lucky_max_var"],
  ["stH3Model", "repair_h3_model_var"], ["stH3Base", "repair_h3_base_var"], ["stH3Len", "repair_h3_frames_var"],
  ["stH3W", "repair_h3_width_var"], ["stH3H", "repair_h3_height_var"], ["stH3Scale", "repair_h3_dial_scale_var"],
  ["stH3Steps", "repair_h3_steps_var"], ["stH3Turbo", "repair_h3_turbo_var"], ["stH3Sound", "repair_h3_sound_var", "check"],
  ["stH3NoLora", "repair_h3_nolora_var", "check"],
  ["stPbOn", "repair_pb_on_var", "check"], ["stPromptB", "repair_prompt_b_var"], ["stPbBlend", "repair_pb_blend_var", "num"],
  ["stPbKeep", "repair_pb_keep_lyrics_var", "check"], ["stSrcOn", "repair_src_on_var", "check"],
  ["stSrcPath", "repair_src_path_var"], ["stSrcMode", "repair_src_mode_var", "srcmode"],
  ["stSrcFollow", "repair_src_follow_var", "num"], ["stSrcDenoise", "repair_src_denoise_var", "num"],
  ["stSrcStrength", "repair_src_strength_var", "num"], ["stSndOn", "repair_sound_on_var", "check"],
  ["stSndPath", "repair_sound_path_var"]];

function studioSetupData() {
  const values = {};
  for (const [id, name, kind] of ST_SETUP_MAP) {
    const e = $(id);
    values[name] = kind === "check" ? e.checked : kind === "num" ? Number(e.value) :
      kind === "srcmode" ? (ST_SRC_MODES.find((m) => m[0] === e.value) || ST_SRC_MODES[0])[1] : e.value;
  }
  values.repair_dit_choice_var = (document.querySelector('input[name="stDit"]:checked') || {}).value === "full" ? "base" : "distilled";
  values.repair_metrics_ref_var = stSaved().metricsRef || "";
  const blocks = {};
  for (const [id, b] of Object.entries(S.st.blocks || {}))
    blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
  return {fizgig_repair_setup: 1, family: $("stFamily").value, values, blocks,
    web: {kf: S.st.kf || {}, kfRect: S.st.kfRect || {}, region: S.st.region || null,
      mix: S.st.mix ? {mode: S.st.mix.mode, rows: S.st.mix.rows, upto: $("stMixUpto").value, hist: $("stMixHist").value} : null,
      dmix: S.st.dmix ? {on: S.st.dmix.on, upto: $("stDmixUpto").value, hist: $("stDmixHist").value} : null}};
}

async function studioSetupList(select) {
  try {
    const r = await api("/api/studio/setups");
    const sel = $("stSetupList");
    sel.replaceChildren(el("option", {value: "", text: "Load a setup…"}),
      ...r.setups.map((x) => el("option", {value: x.name, text: x.family ? `${x.name}  (${x.family})` : x.name})));
    if (select) sel.value = select;
  } catch (e) { /* the list stays as it was */ }
}

async function studioSetupSave(overwrite) {
  const name = $("stSetupName").value.trim() || `${$("stFamily").value}_setup`;
  try {
    const r = await api("/api/studio/setup_save", {name, data: studioSetupData(), overwrite: !!overwrite});
    if (r.exists) {
      $("stSetupMsg").replaceChildren(el("span", {text: `"${r.name}" is there already. `}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => studioSetupSave(true)}));
      return;
    }
    $("stSetupMsg").textContent = `Setup saved: ${r.name}`;
    $("stSetupName").value = r.name;
    studioSetupList();
  } catch (e) { $("stSetupMsg").textContent = e.message; }
}

async function studioSetupLoad(name) {
  // its model first (waiting for the switch to land), then every control and slider, then Start - which loads its
  // LoRAs and renders (the desktop's _load_repair_setup)
  let d;
  try { d = await api(`/api/studio/setup?name=${encodeURIComponent(name)}`); }
  catch (e) { $("stSetupMsg").textContent = e.message; return; }
  const fam = d.family && [...$("stFamily").options].some((o) => o.value === d.family) ? d.family : $("stFamily").value;
  if (fam !== $("stFamily").value) { $("stFamily").value = fam; await studioFamily(); stRemember(); }
  const v = d.values || {};
  for (const [id, key, kind] of ST_SETUP_MAP) {
    if (!(key in v)) continue;
    const e = $(id), val = v[key];
    if (kind === "check") e.checked = !!val;
    else if (kind === "srcmode") e.value = (ST_SRC_MODES.find((m) => m[1] === val || m[0] === val) || ST_SRC_MODES[0])[0];
    else if (e.tagName === "SELECT" && ![...e.options].some((o) => o.value === String(val))) continue;
    else e.value = String(val);
  }
  if ("repair_dit_choice_var" in v) {
    const want = v.repair_dit_choice_var === "base" ? "full" : "fast";
    const r = document.querySelector(`input[name="stDit"][value="${want}"]`); if (r) r.checked = true;
  }
  try { const s = stSaved(); s.metricsRef = v.repair_metrics_ref_var || ""; localStorage.setItem(ST_KEY, JSON.stringify(s)); } catch (e) { /* private */ }
  for (const [id, b] of Object.entries(d.blocks || {})) {
    if (!S.st.blocks[id]) continue;
    S.st.blocks[id] = {pe: !!b.primary_enabled, ps: Number(b.primary_strength), de: !!b.donor_enabled, ds: Number(b.donor_strength)};
  }
  const w = d.web || {};
  if (w.kf) { S.st.kf = w.kf; S.st.kfRect = w.kfRect || {}; }
  if (S.st.layout && S.st.layout.kind === "video") studioSlots();
  S.st.region = w.region || null;
  studioEq("primary", false, false); studioEq("donor", false, false);
  studioAceLayout(); studioAceRemember(); studioH3Remember(); stRemember(); studioMetricsShow(); studioNegRow();
  studioSync(); studioCommit();
  studioMixReset();
  S.st.pendingMix = w.mix || w.dmix ? w : null;          // the mix follows once its LoRAs are loaded
  $("stSetupMsg").textContent = `Setup loaded: ${name}`;
  $("stSetupName").value = name;
  if ($("stPrimary").value.trim()) {
    await studioStart();                                // a new LoRA / model loads; the same one just renders
    if (S.st.ready && S.st.pendingMix) studioSetupMix();
  }
}

async function studioSetupMix() {
  // after a setup's load: its mix back (the run's epochs need the LoRA there)
  const w = S.st.pendingMix;
  S.st.pendingMix = null;
  if (!w) return;
  if (w.mix && w.mix.mode === "mix" && (w.mix.rows || []).length) {
    S.st.mix.rows = w.mix.rows; document.querySelector('input[name="stMixMode"][value="mix"]').checked = true;
    await studioMixMode("mix");
  } else if (w.mix && w.mix.mode === "epochs") {
    $("stMixHist").value = w.mix.hist;
    document.querySelector('input[name="stMixMode"][value="epochs"]').checked = true;
    await studioMixMode("epochs");
    $("stMixUpto").value = w.mix.upto; studioMixLayout();
  }
  if (w.dmix && w.dmix.on && S.st.donor) {
    $("stDmixOn").checked = true; S.st.dmix.on = true;
    if (await studioMixEpochs("donor")) { $("stDmixUpto").value = w.dmix.upto; $("stDmixHist").value = w.dmix.hist; }
    else { S.st.dmix.on = false; $("stDmixOn").checked = false; }
    studioMixLayout();
  }
  studioSchedule();
}

function wireStudioSetups() {
  $("stSetupSave").addEventListener("click", () => studioSetupSave(false));
  $("stSetupList").addEventListener("change", () => { const n = $("stSetupList").value; $("stSetupList").value = ""; if (n) studioSetupLoad(n); });
  studioSetupList();
}

function studioCanvas() {
  // H3: [width, height, frames, keyframes, references] of the first clip, so Auto loads the base with room for it
  if (S.st.layout.kind !== "video") return null;
  const x = studioExtra(), n = x.keyframes.length;
  return [x.video.width, x.video.height, x.video.frames, x.ref_mode ? 0 : n, x.ref_mode ? n : 0];
}

function studioNegRow() {
  const L = S.st.layout;
  if (!L) return;
  const fast = (document.querySelector('input[name="stDit"]:checked') || {}).value !== "full";
  const cfg = L.follows_samples ? (L.follow_ck ? L.cfg_fast : L.follow_cfg) : fast && L.dit_fast ? L.cfg_fast : L.cfg_full;   // Samples-tab models: theirs
  $("stNegRow").style.display = L.negative_default != null && cfg > 1 ? "" : "none";
}

// ---- the sliders ----
const stFmt = (v) => Math.abs(v - 1) < 1e-9 ? "1.00" : (v >= 0 ? "+" : "") + v.toFixed(2);
function studioRows() {
  const host = $("stBlocks");
  host.innerHTML = "";
  for (const g of S.st.layout.groups) {
    const half = Math.ceil(g.blocks.length / 2);
    const cols = el("div", {class: "st-cols"}, el("div", {}, ...g.blocks.slice(0, half).map(studioRow)),
      el("div", {}, ...g.blocks.slice(half).map(studioRow)));
    host.append(el("div", {class: "st-group"}, el("h3", {text: g.label}), cols));
  }
  studioActivity();
}

function studioRow(b) {
  const box = el("div", {class: "st-block", "data-id": b.id});
  const mk = (side) => {
    const st = () => S.st.blocks[b.id];
    const en = side === "p" ? "pe" : "de", sv = side === "p" ? "ps" : "ds";
    const tick = el("input", {type: "checkbox", title: side === "p" ? "This block on" : "The donor's block on",
      onchange: (e) => studioSet(b.id, en, e.target.checked)});
    const range = el("input", {type: "range", min: "-3", max: "3", step: "0.05",
      oninput: (e) => studioSet(b.id, sv, Number(e.target.value), true), onchange: () => studioCommit()});
    const val = el("span", {class: "st-val"});
    const btn = (t, title, f) => el("button", {class: "btn btn-sm", text: t, title, onclick: () => { f(); studioCommit(); }});
    const label = side === "p" ? el("span", {class: "st-label", style: b.color ? `color:${b.color}` : ""},
      b.label, b.cat ? el("small", {text: `[${b.cat}]`}) : null) : el("span", {class: "st-label", text: "donor"});
    const lock = side === "p" ? el("button", {class: "st-lock", title: "Lock this block (presets, Lucky and bulk leave it alone)",
      text: "🔓", onclick: (e) => { S.st.locks.has(b.id) ? S.st.locks.delete(b.id) : S.st.locks.add(b.id);
        e.currentTarget.textContent = S.st.locks.has(b.id) ? "🔒" : "🔓"; e.currentTarget.classList.toggle("on"); }}) :
      el("span", {});
    const row = el("div", {class: "st-row" + (side === "d" ? " donor" : "")}, lock, tick, label, range, val,
      el("span", {class: "st-btns"},
        btn("0", "Set to 0", () => studioSet(b.id, sv, 0)), btn("1", "Set to 1", () => studioSet(b.id, sv, 1)),
        btn("±", "Flip the sign", () => studioSet(b.id, sv, -st()[sv])),
        btn("−", "−0.1", () => studioSet(b.id, sv, Math.max(-3, Math.round((st()[sv] - 0.1) * 100) / 100))),
        btn("+", "+0.1", () => studioSet(b.id, sv, Math.min(3, Math.round((st()[sv] + 0.1) * 100) / 100))),
        btn("⇄", "Balance: the other side gets 1 − this (with a donor)", () => {
          const o = side === "p" ? "ds" : "ps", v = st()[sv];
          if (S.st.donor && v >= 0 && v <= 1) { studioSet(b.id, o, Math.round((1 - v) * 100) / 100); studioSet(b.id, "de", true); }
        })));
    row._sync = () => { tick.checked = st()[en]; range.value = st()[sv]; val.textContent = stFmt(st()[sv]); };
    row._sync();
    return row;
  };
  const p = mk("p"), d = mk("d");
  d.style.display = S.st.donor ? "" : "none";
  box.append(p, d);
  box._rows = [p, d];
  return box;
}

function studioSync(ids) {
  document.querySelectorAll("#stBlocks .st-block").forEach((box) => {
    if (!ids || ids.includes(box.dataset.id)) box._rows.forEach((r) => r._sync());
  });
}

function studioActivity() {
  // a block the LoRA doesn't touch is greyed; the donor's rows show only with a donor
  document.querySelectorAll("#stBlocks .st-block").forEach((box) => {
    const id = box.dataset.id;
    box._rows[0].classList.toggle("absent", S.st.ready && !S.st.primaryBlocks.has(id));
    box._rows[1].style.display = S.st.donor ? "" : "none";
    box._rows[1].classList.toggle("absent", S.st.donor && !S.st.donorBlocks.has(id));
  });
  document.querySelectorAll(".st-donor-only").forEach((x) => { x.style.display = S.st.donor ? "" : "none"; });
  $("stTargetDonor").disabled = !S.st.donor;
}

function studioSet(id, k, v, live) {
  if (S.st.locks.has(id)) { studioSync([id]); return; }
  S.st.blocks[id][k] = v;
  studioSync([id]);
  studioSchedule(live);
}

// undo / redo: a snapshot after each change settles
function studioSnap() { return JSON.stringify(S.st.blocks); }
function studioCommit() {
  const now = studioSnap();
  if (S.st.hist[S.st.hist.length - 1] !== now) { S.st.hist.push(now); if (S.st.hist.length > 200) S.st.hist.shift(); S.st.redo = []; }
}
function studioUndo(redo) {
  const from = redo ? S.st.redo : S.st.hist, to = redo ? S.st.hist : S.st.redo;
  if (!redo && from.length < 2) { $("stStatus").textContent = "Nothing to undo."; return; }
  if (redo && !from.length) { $("stStatus").textContent = "Nothing to redo."; return; }
  if (redo) { const s = from.pop(); to.push(s); S.st.blocks = JSON.parse(s); }
  else { to.push(from.pop()); S.st.blocks = JSON.parse(from[from.length - 1]); }
  studioSync(); studioSchedule();
  $("stStatus").textContent = `${redo ? "Redone" : "Undone"} (${Math.max(0, S.st.hist.length - 1)} steps to undo, ${S.st.redo.length} to redo).`;
}

// ---- masters (Klein): one slider per category, on the primary or the donor ----
function studioMasters() {
  const L = S.st.layout;
  $("stMasters").style.display = L.masters.length ? "" : "none";
  const host = $("stMasterRows");
  host.innerHTML = "";
  for (const m of L.masters) {
    const range = el("input", {type: "range", min: "-3", max: "3", step: "0.05", value: "1"});
    const val = el("span", {class: "st-val", text: "1.00"});
    const apply = (v) => {
      const donor = (document.querySelector('input[name="stTarget"]:checked') || {}).value === "donor";
      const has = donor ? S.st.donorBlocks : S.st.primaryBlocks;
      for (const [id, c] of Object.entries(L.category_of))
        if (c === m.cat && (!S.st.ready || has.has(id)) && !S.st.locks.has(id)) S.st.blocks[id][donor ? "ds" : "ps"] = v;
      range.value = v; val.textContent = stFmt(v);
      studioSync(); studioSchedule(true);
    };
    range.addEventListener("input", () => apply(Number(range.value)));
    range.addEventListener("change", studioCommit);
    const b = (t, f) => el("button", {class: "btn btn-sm", text: t, onclick: () => { f(); studioCommit(); }});
    host.append(el("div", {class: "st-row"}, el("span", {}), el("span", {}),
      el("span", {class: "st-label", style: `color:${m.color}`, text: m.label}), range, val,
      el("span", {class: "st-btns"}, b("0", () => apply(0)), b("1", () => apply(1)), b("±", () => apply(-Number(range.value))))));
    host.lastChild._cat = m.cat; host.lastChild._range = range; host.lastChild._val = val;
  }
}

function studioMastersFollow() {
  // flipping the target shows each category's average on that side
  const donor = (document.querySelector('input[name="stTarget"]:checked') || {}).value === "donor";
  for (const row of $("stMasterRows").children) {
    const ids = Object.entries(S.st.layout.category_of).filter(([, c]) => c === row._cat).map(([id]) => id);
    const vals = ids.map((id) => S.st.blocks[id][donor ? "ds" : "ps"]);
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 1;
    row._range.value = avg; row._val.textContent = stFmt(avg);
  }
}

// ---- presets and bulk ----
async function studioPresets() {
  const L = S.st.layout, sel = $("stPreset");
  const user = (await api("/api/studio/presets?family=" + encodeURIComponent(L.key)).catch(() => ({user: []}))).user;
  sel.innerHTML = "";
  sel.append(el("option", {value: "", text: "Presets…"}));
  for (const p of L.presets) sel.append(el("option", {value: "b:" + p.name, text: p.name}));
  if (user.length) sel.append(el("optgroup", {label: "Your presets"}, ...user.map((n) => el("option", {value: "u:" + n, text: n}))));
}

async function studioPresetApply(v) {
  const L = S.st.layout;
  if (v.startsWith("b:")) {
    const p = L.presets.find((x) => x.name === v.slice(2));
    for (const [id, b] of Object.entries(S.st.blocks)) {
      if (S.st.locks.has(id)) continue;
      Object.assign(b, {pe: true, ps: 1, de: true, ds: 1});
      if (p.keep) b.pe = p.keep.includes(L.category_of[id]);
      if (p.blocks && id in p.blocks) { b.pe = true; b.ps = Number(p.blocks[id]); }
    }
  } else {
    const r = await api("/api/studio/preset_load", {family: L.key, name: v.slice(2)});
    for (const [id, bs] of Object.entries(r.blocks || {})) if (S.st.blocks[id] && !S.st.locks.has(id))
      S.st.blocks[id] = {pe: !!bs.primary_enabled, ps: Number(bs.primary_strength), de: !!bs.donor_enabled, ds: Number(bs.donor_strength)};
  }
  studioSync(); studioCommit(); studioSchedule();
  $("stPreset").value = "";
}

function studioBulk(kind) {
  const L = S.st.layout;
  const lo = Number($("stLuckyLo").value) || 0, hi = Number($("stLuckyHi").value) || 1.5;
  let rolled = 0;
  for (const [id, b] of Object.entries(S.st.blocks)) {
    if (S.st.locks.has(id)) continue;
    if (kind === "reset") Object.assign(b, {pe: true, ps: 1, de: true, ds: 1});
    else if (kind === "off") b.pe = false;
    else if (kind === "on") b.pe = true;
    else if (kind === "invert") b.pe = !b.pe;
    else if (kind === "donor_on") Object.assign(b, {de: true, ds: 1});
    else if (kind === "donor_off") b.ds = 0;
    else if (kind === "lucky") {
      if ((S.st.ready && !S.st.primaryBlocks.has(id)) || !b.pe) continue;
      let a = lo, z = hi;
      for (const zone of L.lucky_zones || []) if (zone[0] && zone[0].includes && zone[0].includes(id)) { a = Math.max(a, zone[1]); z = Math.min(z, zone[2]); }
      b.ps = Math.round((a + Math.random() * (z - a)) * 20) / 20; rolled++;
    }
  }
  studioSync(); studioCommit(); studioSchedule();
  if (kind === "lucky") $("stStatus").textContent = `Lucky mix: ${rolled} block(s) rolled, ${lo} to ${hi} (Undo for the mix before; Save preset… to keep it).`;
  stRemember();
}

function studioPresetSaveOpen() {
  const host = $("stPresetSaveRow");
  if (host.childNodes.length) { host.innerHTML = ""; return; }
  const name = el("input", {type: "text", class: "w-path", placeholder: "A name for these block settings"});
  const msg = el("span", {class: "field-hint"});
  const save = async (overwrite) => {
    const blocks = {};
    for (const [id, b] of Object.entries(S.st.blocks)) blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
    try {
      const r = await api("/api/studio/preset_save", {family: S.st.layout.key, name: name.value, blocks, overwrite: !!overwrite});
      if (r.exists) { msg.replaceChildren(document.createTextNode("That name is taken. "), el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => save(true)})); return; }
      host.innerHTML = ""; studioPresets(); $("stStatus").textContent = "Preset saved (the desktop's Repair Studio lists it too).";
    } catch (e) { msg.textContent = e.message; }
  };
  host.append(el("div", {class: "field-row"}, name, el("button", {class: "btn btn-primary", text: "Save", onclick: () => save(false)}),
    el("button", {class: "btn", text: "Cancel", onclick: () => { host.innerHTML = ""; }}), msg));
  name.focus();
}

// ---- rendering ----
function studioState() {
  const blocks = {};
  for (const [id, b] of Object.entries(S.st.blocks)) blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
  const res = Number($("stRes").value) || 768;
  const num = (id, d) => { const v = Number($(id).value); return isFinite(v) ? v : d; };
  return {blocks, seed: parseInt($("stSeed").value, 10) || 42, prompt: $("stPrompt").value.trim(), preview_width: res,
    preview_height: res, ref_image_path: $("stRefRow").style.display === "none" ? "" : $("stRef").value.trim(),
    ref_megapixels: num("stRefMp", 1), ref_strength: num("stRefStr", 1), primary_scale: num("stPScale", 1),
    donor_scale: num("stDScale", 1)};
}

function studioSchedule(live) {
  if (!S.st.ready) return;
  clearTimeout(S.st.timer);
  S.st.timer = setTimeout(studioRender, live ? ($("stTurbo").checked ? 150 : 400) : 100);
}

async function studioRender() {
  if (!$("stPrompt").value.trim()) { $("stStatus").textContent = "Enter a prompt (include the LoRA's trigger word) to see previews."; return; }
  try { await api("/api/studio/render", {state: studioState(), ...studioExtra(), ...studioMixExtra()}); studioPoll(); }
  catch (e) { $("stStatus").textContent = e.message; }
}

async function studioStart() {
  stRemember();
  const L = S.st.layout;
  try {
    const st = await api("/api/studio/status");
    const fast = (document.querySelector('input[name="stDit"]:checked') || {}).value !== "full";
    const key = [L.key, $("stPrimary").value.trim(), fast, L.kind === "video" ? $("stH3Model").value + $("stH3Base").value : ""].join("|");
    if (st.state === "ready" && S.st.loadedKey === key) {     // the same LoRA: the donor may change, then a render
      if ($("stDonor").value.trim() !== (st.donor || "")) await api("/api/studio/donor", {path: $("stDonor").value.trim()});
      studioRender();
      return;
    }
    S.st.ready = false;
    await api("/api/studio/load", {family: L.key, primary: $("stPrimary").value.trim(), donor: $("stDonor").value.trim(),
      fast, turbo: $("stTurbo").checked, negative: $("stNeg").value.trim(),
      h3_model: L.kind === "video" ? $("stH3Model").value : "", base_mode: L.kind === "video" ? L.video.base_of[$("stH3Base").value] : "",
      canvas: studioCanvas()});
    if (S.st.mix && S.st.mix.mode !== "off" || (S.st.dmix && S.st.dmix.on)) studioMixReset();
    S.st.loadedKey = key;
    S.st.pendingFirst = true;
    studioPoll();
  } catch (e) { $("stStatus").textContent = e.message; }
}

async function studioPoll() {
  clearTimeout(S.st.poll);
  let st;
  try { st = await api("/api/studio/status"); } catch (e) { return; }
  const wasReady = S.st.ready;
  S.st.ready = st.state === "ready";
  S.st.donor = !!st.donor;
  S.st.primaryBlocks = new Set(st.primary_blocks || []); S.st.donorBlocks = new Set(st.donor_blocks || []);
  $("stStatus").textContent = (st.status || $("stStatus").textContent) +
    (st.status && /^(Loaded|Ready|Donor loaded)/.test(st.status) && studioEqNote(st) ? " " + studioEqNote(st) : "");
  $("stBadge").textContent = {loading: "Loading", ready: st.rendering ? "Rendering" : "Loaded", idle: ""}[st.state] || "";
  $("stBadge").dataset.state = st.state === "loading" || st.rendering ? "running" : st.state === "ready" ? "completed" : "idle";
  const p = st.progress;
  $("stBar").style.width = p && p[1] ? `${Math.round(100 * p[0] / p[1])}%` : st.rendering || st.state === "loading" ? "100%" : "0";
  $("stBar").parentElement.classList.toggle("indeterminate", !(p && p[1]) && (st.rendering || st.state === "loading"));
  $("stStart").textContent = S.st.ready ? "Update" : "Start";
  if (S.st.ready && !wasReady) {
    studioActivity();
    if (!$("stOut").value) { const pth = st.primary; $("stOut").value = pth.replace(/\.safetensors$/i, "") + "_repaired.safetensors"; }
    if (S.st.pendingFirst) { S.st.pendingFirst = false; studioCommit(); studioRender(); }
    if (S.st.pendingMix) studioSetupMix();
  }
  if (st.shown && st.shown !== S.st.shown) {
    S.st.shown = st.shown;
    const q = (w) => `/api/studio/image?which=${w}&gen=${st.shown}&t=${encodeURIComponent(TOKEN)}`;
    $("stBase").src = q("baseline"); $("stTweak").src = q("tweaked");
    // a clip plays in its pane (the picture its poster); a song gets its player under the waveform
    const kind = S.st.layout && S.st.layout.kind;
    for (const [w, img, vid, aud] of [["baseline", "stBase", "stBaseVid", "stBaseAud"], ["tweaked", "stTweak", "stTweakVid", "stTweakAud"]]) {
      const has = st.media && st.media[w];
      $(vid).style.display = kind === "video" && has ? "" : "none";
      $(aud).style.display = kind === "audio" && has ? "" : "none";
      $(img).style.display = kind === "video" && has ? "none" : "";
      if (kind === "video" && has) { $(vid).poster = q(w); $(vid).src = q(w + "_media"); }
      if (kind === "audio" && has) $(aud).src = q(w + "_media");
    }
    studioMetrics();
    const ps = Number($("stPScale").value) || 1;
    $("stBaseCap").textContent = ps === 1 ? "Baseline (LoRA at default 1.0)" : `Baseline (LoRA at ${ps})`;
  }
  studioNoLoraPane(st);
  studioAceNotes(st.notes);
  studioMixNotes(st.notes);
  const pr = st.profile;
  $("stProfile").style.display = pr ? "" : "none";
  if (pr) {
    $("stProfileLine").textContent = `${pr.lora_name} · profiled ${pr.created} · most active blocks:`;
    const col = {"Style+Comp": "#5B9BD5", "Style/ID": "#5BB3A6", Identity: "#70AD47", "ID/Detail": "#B8A547", Details: "#ED7D31"};
    $("stPills").replaceChildren(...pr.blocks.map((b) => el("span", {class: "st-pill",
      style: `background:${({style_composition: "#5B9BD5", style_ident_overlap: "#5BB3A6", identity: "#70AD47", ident_details_overlap: "#B8A547", details: "#ED7D31"})[b.category] || col[b.category] || "#8A96A8"}`,
      text: `${b.name}  ${Number(b.pct).toFixed(1)}%`})));
    $("stReport").style.display = pr.report ? "" : "none";
    $("stReport").href = `/api/studio/report?t=${encodeURIComponent(TOKEN)}`;
  }
  if (st.saved && st.saved.out !== S.st.savedShown) { S.st.savedShown = st.saved.out; studioSavedSay(st.saved); }
  const active = st.state === "loading" || st.rendering || st.pending;
  if ($("tab-studio").classList.contains("active")) S.st.poll = setTimeout(studioPoll, active ? 400 : 2000);
}

function studioNoLoraPane(st) {
  // the base model's own clip / song, left of the pair (the desktop's player has it there): H3 with the tick, ACE always
  const kind = S.st.layout && S.st.layout.kind;
  const on = kind === "audio" || (kind === "video" && $("stH3NoLora").checked);
  $("stNolPane").style.display = on ? "" : "none";
  $("stPreview").classList.toggle("three", on);
  if (!on) return;
  $("stNolCap").textContent = st.nolora_busy ? "No LoRA (rendering…)" : "No LoRA (base model, same seed)";
  if (st.nolora_gen && st.nolora_gen !== S.st.nolShown && st.media && st.media.nolora != null) {
    S.st.nolShown = st.nolora_gen;
    const q = (w) => `/api/studio/image?which=${w}&gen=${st.nolora_gen}&t=${encodeURIComponent(TOKEN)}`;
    const media = st.media.nolora;
    $("stNol").src = q("nolora");
    $("stNolVid").style.display = kind === "video" && media ? "" : "none";
    $("stNolAud").style.display = kind === "audio" && media ? "" : "none";
    $("stNol").style.display = kind === "video" && media ? "none" : "";
    if (kind === "video" && media) { $("stNolVid").poster = q("nolora"); $("stNolVid").src = q("nolora_media"); }
    if (kind === "audio" && media) $("stNolAud").src = q("nolora_media");
  }
}

function studioSavedSay(sv) {
  const s = sv.summary || {};
  const lines = [`Saved: ${sv.out}`];
  if (s.keys_in != null) lines.push(`Keys: ${s.keys_in} → ${s.keys_out}`);
  for (const [k, t] of [["dropped_blocks", "Dropped blocks"], ["rescaled_blocks", "Rescaled blocks"], ["blended_blocks", "Donor-blended blocks"]])
    if (s[k] && s[k].length) lines.push(`${t} (${s[k].length}): ${s[k].join(", ")}`);
  if (s.use_at != null) lines.push(`Use it at strength ${s.use_at}.`);
  $("stSaveMsg").replaceChildren(el("pre", {class: "field-hint st-saved", text: lines.join("\n")}));
}

async function studioSave(overwrite) {
  const st = studioState();
  const includeDonor = S.st.donor && Object.values(S.st.blocks).some((b) => b.de);
  try {
    const r = await api("/api/studio/save", {out: $("stOut").value.trim(), state: st, include_donor: includeDonor, overwrite: !!overwrite});
    if (r.exists) {
      $("stSaveMsg").replaceChildren(el("div", {class: "field-row"}, el("span", {class: "field-hint", text: "That file is there already."}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => studioSave(true)})));
      return;
    }
    $("stSaveMsg").replaceChildren(el("div", {class: "field-hint", text: "Saving…"}));
    studioPoll();
  } catch (e) { $("stSaveMsg").replaceChildren(el("div", {class: "problems", text: e.message})); }
}

function wireStudio() {
  $("stFamily").addEventListener("change", () => { studioFamily(); stRemember(); });
  // H3: a length / size / steps / sound change renders; Model and Base take effect at the next Start
  for (const id of ["stH3Len", "stH3W", "stH3H", "stH3Scale", "stH3Steps", "stH3Turbo", "stH3Sound", "stH3NoLora"])
    $(id).addEventListener("change", () => { studioH3Remember(); studioSchedule(); });
  $("stH3Model").addEventListener("change", () => { studioH3Remember(); studioSlots(); $("stStart").textContent = "Update"; });
  $("stH3Base").addEventListener("change", () => { studioH3Remember(); $("stStart").textContent = "Update"; });
  $("stPlayBothBtn").addEventListener("click", () => {
    const v = S.st.layout.kind === "video", ids = v ? ["stBaseVid", "stTweakVid"] : ["stBaseAud", "stTweakAud"];
    if ($("stNolPane").style.display !== "none") ids.unshift(v ? "stNolVid" : "stNolAud");
    for (const id of ids) { const m = $(id); m.currentTime = 0; m.play().catch(() => {}); }
  });
  $("stStart").addEventListener("click", studioStart);
  wireStudioAce();
  S.st.eq = {primary: false, donor: false}; S.st.eqSaved = {};
  $("stEqP").addEventListener("click", () => studioEq("primary", !S.st.eq.primary));
  wireStudioMix();
  wireStudioMetrics();
  wireStudioSetups();
  $("stEqD").addEventListener("click", () => studioEq("donor", !S.st.eq.donor));
  $("stUnload").addEventListener("click", async () => { await api("/api/studio/reset", {}); S.st.ready = false; S.st.loadedKey = ""; studioPoll(); });
  $("stPrimaryPick").addEventListener("click", () => pick($("stPrimary").value || S.outputDefault, false, (p) => { $("stPrimary").value = p; stRemember(); }));
  $("stDonorPick").addEventListener("click", () => pick($("stDonor").value || $("stPrimary").value, false, (p) => { $("stDonor").value = p; stRemember();
    studioDmixOff();
    if (S.st.ready) api("/api/studio/donor", {path: p}).then(studioPoll).catch((e) => { $("stStatus").textContent = e.message; }); }));
  $("stDonorUnload").addEventListener("click", () => { $("stDonor").value = ""; stRemember(); studioDmixOff();
    if (S.st.ready) api("/api/studio/donor", {path: ""}).then(() => { studioPoll(); studioSchedule(); }); });
  $("stRefPick").addEventListener("click", () => pick($("stRef").value || S.imageFolder, false, (p) => { $("stRef").value = p; stRemember(); studioSchedule(); }));
  $("stRefClear").addEventListener("click", () => { $("stRef").value = ""; stRemember(); studioSchedule(); });
  $("stSeedNew").addEventListener("click", () => { $("stSeed").value = 1 + Math.floor(Math.random() * 99999); stRemember(); studioSchedule(); });
  for (const id of ["stSeed", "stRes", "stRefMp", "stRefStr", "stPScale", "stDScale"])
    $(id).addEventListener("change", () => { stRemember(); studioSchedule(); });
  for (const id of ["stPrompt", "stPrimary", "stDonor", "stLuckyLo", "stLuckyHi"]) $(id).addEventListener("input", stRemember);
  $("stNeg").addEventListener("change", () => { stRemember();
    if (S.st.ready) api("/api/studio/settings", {family: S.st.layout.key, negative: $("stNeg").value.trim()}).then(() => studioSchedule()); });
  $("stTurbo").addEventListener("change", () => { stRemember(); if (S.st.ready) api("/api/studio/settings", {family: S.st.layout.key, turbo: $("stTurbo").checked}); });
  document.querySelectorAll('input[name="stDit"]').forEach((r) => r.addEventListener("change", studioNegRow));
  document.querySelectorAll('input[name="stTarget"]').forEach((r) => r.addEventListener("change", studioMastersFollow));
  document.querySelectorAll("[data-bulk]").forEach((b) => b.addEventListener("click", () => studioBulk(b.dataset.bulk)));
  $("stPreset").addEventListener("change", () => { if ($("stPreset").value) studioPresetApply($("stPreset").value); });
  $("stPresetSave").addEventListener("click", studioPresetSaveOpen);
  $("stUndo").addEventListener("click", () => studioUndo(false));
  $("stRedo").addEventListener("click", () => studioUndo(true));
  $("stSave").addEventListener("click", () => studioSave(false));
  document.addEventListener("keydown", (e) => {
    if (!$("tab-studio").classList.contains("active") || /INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || "")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); studioUndo(e.shiftKey); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); studioUndo(true); }
  });
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
  let items = q ? all.filter((i) => i.name.toLowerCase().includes(q) || (i.caption || "").toLowerCase().includes(q)) : all;
  const sort = $("capSort").value, px = (i) => (i.size ? i.size[0] * i.size[1] : null);
  if (sort !== "name") {                       // by pixel count; songs (no size) go last either way
    items = [...items].sort((a, b) => (px(a) == null) - (px(b) == null) ||
      (sort === "small" ? px(a) - px(b) : px(b) - px(a)) || a.name.localeCompare(b.name));
  }
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
    const card = el("div", {class: "caption-card"});
    const remove = el("button", {class: "btn btn-sm", text: "Remove",
      title: "Take it out of the training set: it and its caption move into the folder's removed_images subfolder (nothing is deleted).",
      onclick: () => capRemove(it, card, status)});
    const size = it.size ? `${it.size[0]}×${it.size[1]} · ${(it.size[0] * it.size[1] / 1e6).toFixed(1)} MP` : "";
    card.append(media, el("div", {class: "caption-card-body"},
      el("div", {class: "caption-card-title", text: mediaLabel(it)}),
      size ? el("div", {class: "field-hint caption-card-size", text: size}) : null, ta,
      el("div", {class: "field-row"}, save, recap, remove, status)));
    host.append(card);
  }
}

async function capRemove(it, card, status) {
  // out of the training set (into removed_images/), with an Undo on the spot
  const folder = S.imageFolder;
  try {
    const r = await api("/api/captions/remove", {folder, name: it.name, ext: $("captionExt").value.trim() || ".txt"});
    S.cap.items = S.cap.items.filter((x) => x !== it);
    const undo = el("button", {class: "btn btn-sm", text: "Undo", onclick: async () => {
      try {
        await api("/api/captions/restore", {folder, moved: r.moved});
        S.cap.items.push(it); S.cap.items.sort((a, b) => a.name.localeCompare(b.name)); renderCaptions();
      } catch (e) { note.textContent = e.message; }
    }});
    const note = el("span", {class: "field-hint", text: `${it.name} moved to removed_images.`});
    card.replaceChildren(el("div", {class: "caption-card-body caption-removed"}, note, el("div", {class: "field-row"}, undo)));
    $("captionsSummary").textContent = `${S.cap.items.length} files · ${it.name} removed`;
  } catch (e) { status.textContent = e.message; }
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
  $("capSort").addEventListener("change", () => { S.cap.page = 0; renderCaptions(); });
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
  if (name === "studio") studioOpen();
  if (name === "extract") extractOpen();
  if (name === "profiler") profilerOpen();
  if (name === "royale") royaleOpen();
  if (name === "metadata") mdWire();
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
const LIGHTBOX_PICS = ".gallery-item > img, .caption-card > img";      // (Repair Studio's panes are gallery items)
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
    const host = img.closest(".gallery, .captions-grid, .st-preview") || document;   // step through the pictures beside it
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
  wireStudio();
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
