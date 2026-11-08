// serve/web/app.js - the DACAN web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
// numbers in the page's language (web/i18n.js: 1,234.5 / 1 234,5)
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(locale(), {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("dacan." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("dacan." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", t("Copied to clipboard"), "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("dacan.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("dacan.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", t("API key saved"), t("Kept in this browser only.")); };

// ------------------------------------------------------------------ simultaneous users (the config's "slots", 06.11.2026)
// Same key as About -> Model settings, but in the header: the owner asked for the switch where he looks.  The
// value is the engine's --batch N; a change restarts the model, so the switch asks first.
async function loadSlotsQuick() {
  try {
    const r = await fetch("config", {headers: headers()});
    if (!r.ok) return;                                   // an older server, or no key yet
    const c = await r.json();
    const k = (c.keys || []).find((x) => x.key === "slots");
    if (!k) return;
    const sel = $("slots-quick");
    sel.innerHTML = (k.choices || [1, 2, 3, 4]).map((n) => `<option value="${n}">${n}</option>`).join("");
    if (k.value != null) sel.value = String(k.value);
    $("slots-box").hidden = false;
  } catch (e) { /* no switch, the card in Settings stays */ }
}
$("slots-quick").onchange = async () => {
  const n = Number($("slots-quick").value);
  if (!confirm(t("The model restarts with {n} simultaneous users (about 2 minutes). Continue?", {n}))) {
    loadSlotsQuick();
    return;
  }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set: {slots: n}})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    toast("success", t("Saved"), b.engine === "restarting" ? t("The model restarts; requests wait meanwhile.") : "", 6000);
  } catch (e) {
    toast("error", t("Not saved"), String(e.message || e), 6000);
  }
};
loadSlotsQuick();

let health = {model: "dacan", images: false, max_context: 0}, healthLoaded = false;
// the texts that come from /health: the attach button's tip, the empty chat's line with the model's name
function healthTexts() {
  $("attach-btn").title = t(health.images ? "Attach a text file or a picture (or drop it here)" : "Attach a text file (or drop it here)");
  if (healthLoaded) $("chat-empty-sub").textContent = t("{model} runs on this PC. Nothing leaves it.", {model: health.model});
}
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    healthLoaded = true;
    healthTexts();
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
// the eight cards, in the page's language (built again on a language switch)
function buildMetricCards() {
  $("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(t(m.label))}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">${esc(t("Decode"))}</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">${esc(t("Prefill"))}</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");
}
buildMetricCards();

// DACAN: a card per GPU (load, VRAM, temperature, power, PCIe) and per CPU socket (load, the RAM of its NUMA node)
const sparkSvg = (id, tone) => `<svg class="st-metric__spark" id="${id}" viewBox="0 0 100 32" preserveAspectRatio="none"${tone ? ` data-tone="${tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
let devicesKey = "";
function buildDevices(st) {
  const gpus = st.gpus || [], sockets = st.sockets || [];
  const key = `${gpus.map((g) => `${g.index}:${g.bus}`).join(",")}|${sockets.map((s) => s.node).join(",")}`;
  if (key === devicesKey) return;
  devicesKey = key;
  const box = $("devices");
  box.hidden = gpus.length < 2 && sockets.length < 2;
  const row = (id, label) => `<dt>${esc(t(label))}</dt><dd id="${id}">–</dd>`;
  box.innerHTML = gpus.map((g, i) => `
    <div class="st-card dev-card" data-kind="gpu">
      <div class="dev-card__head">${icon("gpu", "st-icon st-icon--sm")}<span>${esc(t("GPU {n}", {n: String(g.index)}))}</span>
        <span class="dev-card__tag">${g.numa != null ? `NUMA ${g.numa} · ` : ""}${esc((g.bus || "").replace(/^0000:/, ""))}</span></div>
      <div class="dev-card__name">${esc(g.name || "GPU")}</div>
      <div class="dev-card__value"><span class="st-metric__value" id="dv-gpu${i}">–</span><span class="muted small">${esc(t("load"))}</span></div>
      ${sparkSvg(`sp-gpu${i}`)}
      <dl class="dev-card__rows">${row(`dv-gpu${i}-mem`, "VRAM")}${row(`dv-gpu${i}-temp`, "Temperature")}
        ${row(`dv-gpu${i}-smclock`, "SM clock")}${row(`dv-gpu${i}-memclock`, "VRAM clock")}
        ${row(`dv-gpu${i}-hotspot`, "Hot spot")}${row(`dv-gpu${i}-vramtemp`, "VRAM temp")}
        ${row(`dv-gpu${i}-power`, "Power")}${row(`dv-gpu${i}-pcie`, "PCIe")}</dl>
    </div>`).join("") + sockets.map((s, i) => `
    <div class="st-card dev-card" data-kind="cpu">
      <div class="dev-card__head">${icon("cpu", "st-icon st-icon--sm")}<span>${esc(t("CPU socket {n}", {n: String(i)}))}</span>
        <span class="dev-card__tag">NUMA ${s.node} · ${esc(t("{n} threads", {n: s.threads}))}</span></div>
      <div class="dev-card__name">${esc(st.cpu_name || "CPU")}</div>
      <div class="dev-card__value"><span class="st-metric__value" id="dv-cpu${i}">–</span><span class="muted small">${esc(t("load"))}</span></div>
      ${sparkSvg(`sp-cpu${i}`, "info")}
      <dl class="dev-card__rows">${row(`dv-cpu${i}-ram`, "RAM of this node")}${row(`dv-cpu${i}-gpu`, "GPU on this node")}</dl>
    </div>`).join("");
}
function setText(id, text) { const el = $(id); if (el) el.textContent = text == null || text === "" ? "–" : text; }

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", t("API key needed"));
      if (!keyWarned) { keyWarned = true; toast("warn", t("API key needed"), t("This server needs a key: add it under About > Settings."), 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", t("Server not reachable"));
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    // 13.10.2026 (DACAN): the read speed in the pill too; before the first chunk is read, how long it has been reading
    const rate = live.prefill_tok_s_mean ? ` · ${fmt(live.prefill_tok_s_mean)} ${t("tok/s")}` : "";
    setPill("reading", pct != null ? t("Reading prompt · {pct}%", {pct}) + rate
                                   : t("Reading prompt") + (live.elapsed_s != null ? ` · ${fmt(live.elapsed_s)} ${t("s")}` : ""));
  } else if (live.state === "generating") {
    setPill("generating", t("Generating · {x} tok/s", {x: fmt(live.tok_s, 1)}));
  } else {
    setPill("idle", t("Idle"));
  }
  if (live.queued > 0) setPill("queued", t("{n} queued", {n: live.queued}));
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? tr("just now") : s < 3600 ? tr("{x} min ago", {x: fmt(s / 60)}) : tr("{x} h ago", {x: fmt(s / 3600, 1)});
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} ${t("GB")}`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? t("{a} of {b} requests reused part of their prompt", {a: c.requests_reused || 0, b: c.requests}) : "";
  const share = c.prompt_tokens ? t(" ({x}% of all prompt tokens)", {x: fmt((100 * c.reused_tokens) / c.prompt_tokens)}) : "";
  const event = c.last_event ? t(c.last_event === "parked" ? "Parked {n} tokens, {since}" : "Restored {n} tokens, {since}",
                                 {n: c.last_tokens || 0, since: since(c.last_at)}) : null;
  facts($("cc-facts"), [
    [t("Last request"), c.last_prompt != null ? t("{a} of {b} prompt tokens reused", {a: c.last_reused || 0, b: c.last_prompt}) : null],
    [t("Reused since start"), c.requests ? t("{n} tokens", {n: c.reused_tokens || 0}) + share : null],
    [t("Parked / restored"), c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? t(" · {n} evicted", {n: c.evictions}) : ""}` : null],
    [t("Last switch"), c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? t("A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full.")
    : t("The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).");
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString(locale(), {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? tr(" at {x} tok/s", {x: fmt(read / (t.prompt_ms / 1000))}) : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? tr(" at {x} tok/s", {x: fmt(t.output_tokens / (t.decode_ms / 1000), 1)}) : "";
  return tr("Since {since}: {n} requests · {read} prompt tokens read{pSpeed} ({reused} reused) · {out} written{oSpeed}",
            {since, n: t.requests, read, pSpeed, reused: t.reused, out: t.output_tokens, oSpeed});
}
// the engine's phase while it writes (server.py: thinking, answering, "writing a tool call: X", "tool call complete")
const PHASES = {thinking: ["Thinking", "Думает"], answering: ["Answering", "Отвечает"],
                "tool call complete": ["Tool call complete", "Вызов инструмента готов"]};
function phaseLabel(p) {
  if (!p) return t("Generating");
  const w = /^writing a tool call: (.*)$/.exec(p);
  if (w) return t("Writing a tool call: {name}", {name: w[1]});
  if (PHASES[p]) return PHASES[p][LANG === "ru" ? 1 : 0];
  return p[0].toUpperCase() + p.slice(1);
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = t("Waiting for a request"), detail = "", pct = 0;
  if (live.state === "reading") {
    label = t("Reading prompt");
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = t("{a} / {b} tokens · {pct}%", {a: live.prompt_read, b: live.prompt_total, pct}) +
               (live.prefill_tok_s_mean ? ` · ${fmt(live.prefill_tok_s_mean)} ${t("tok/s")}` : "");
    } else {
      // the engine reports the read per chunk (8192 tokens): until the first one, the tokens and the time so far
      detail = t("{n} tokens", {n: live.prompt_tokens}) + (live.elapsed_s != null ? t(" · reading {x} s", {x: fmt(live.elapsed_s)}) : "");
    }
  } else if (live.state === "generating") {
    label = phaseLabel(live.phase);
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = t("{n} tokens · {x} tok/s", {n: live.generated, x: fmt(live.tok_s, 1)});
  } else if (last) {
    delete prog.dataset.tone;
    detail = t("last: {n} tokens", {n: last.output_tokens}) + (last.decode_tok_s ? t(" at {x} tok/s", {x: fmt(last.decode_tok_s, 1)}) : "");
  }
  $("state-label").textContent = label;
  if (live.parallel > 1) {
    const busy = (live.slots || []).filter((s) => s.state !== "idle").length;
    const concurrency = t("Requests: {n} · busy slots: {a}/{b}", {n: live.running || 0, a: busy, b: live.parallel});
    detail = detail ? `${detail} · ${concurrency}` : concurrency;
  }
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  // the decode speed of the last answer long enough to measure it (MIN_DECODE_TOKENS in server.py): a reply of
  // a few tokens is mostly its first step and has none
  const lastDec = (requests || []).find((r) => r.decode_tok_s != null);
  const speed = live.state === "generating" ? live.tok_s : lastDec ? lastDec.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), t("t/s"),
            live.state === "generating" ? t(live.parallel > 1 ? "Decode now · total" : "Decode now")
            : lastDec ? t("Decode last answer · {n} tokens", {n: lastDec.output_tokens}) : t("Decode"));
  // the prompt read of the last request that read enough fresh tokens to measure it: a few tokens take the engine's
  // fixed start-up time, not reading time
  // 13.10.2026 (DACAN): 512 -> 4096 — a follow-up that reads ~1300 new tokens spends ~5 s mostly on that start-up time
  const MIN_PREFILL_TOKENS = 4096;
  const freshOf = (r) => Math.max(0, (r.prompt_tokens || 0) - (r.reused || 0));
  const lastRead = (requests || []).find((r) => r.prompt_ms > 0 && freshOf(r) >= MIN_PREFILL_TOKENS);
  // 13.10.2026 (DACAN): the live figure only while reading; while generating it may be the previous read's
  const prefill = live.state === "reading" ? live.prefill_tok_s_mean
                : lastRead ? freshOf(lastRead) / (lastRead.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), t("t/s"),
            live.state === "reading" ? t("Prefill now")
            : lastRead ? t("Prefill last read · {n} new tokens", {n: freshOf(lastRead)}) : t("Prefill"));
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? t("not available") : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} ${t("GB")}` : t("GB"),
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} ${t("GB")}`))
                  : eng.expert_slots ? t("{n} experts cached", {n: eng.expert_slots}) : (st.gpu_note ? t("not available on Windows AMD yet") : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), t("W"),
            hw.gpu_power_limit ? t("of {x} W limit", {x: fmt(hw.gpu_power_limit)}) : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  // Show each current link: a healthy first GPU must not hide a narrow second one.
  const links = (hw.gpus || []).length ? hw.gpus : [{pcie_gen: hw.gpu_pcie_gen, pcie_width: hw.gpu_pcie_width}];
  const linkText = (g) => g.pcie_gen ? `Gen${g.pcie_gen} x${g.pcie_width || "?"}` : "–";
  setMetric("pcie", links.map((g) => `x${g.pcie_width || "?"}`).join(" / "), "",
            links.map((g) => `${g.index == null ? "" : `GPU ${g.index}: `}${linkText(g)}`).join(" · ") +
            (hw.gpu_pcie_rx_mb == null ? "" : " · " + t("to GPU {x} MB/s", {x: fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)})));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", [(st.sockets || []).length > 1 ? t("{n} sockets", {n: st.sockets.length}) : "",
            st.threads ? `${st.cores ? `${t("{n} cores", {n: st.cores})} · ` : ""}${t("{n} threads", {n: st.threads})}` : "",
            hw.ram_total ? t("RAM {a} / {b} GB", {a: gb(hw.ram_used), b: gb(hw.ram_total, 0)}) : ""].filter(Boolean).join(" · "));
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : t("needs psutil (setup installs it)"));
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), t(big ? "GB/s" : "MB/s"),
              hw.disk_write_mb == null ? "" : t("write {x} MB/s", {x: fmt(hw.disk_write_mb, 1)}));
  }
  spark("sp-disk", h.disk_read_mb);

  // DACAN: a card per GPU and per socket
  buildDevices(st);
  const sg = st.gpus || [];
  (hw.gpus || []).forEach((g, i) => {
    setText(`dv-gpu${i}`, g.util == null ? null : `${fmt(g.util)}%`);
    if ($(`sp-gpu${i}`)) spark(`sp-gpu${i}`, h[`gpu${i}_util`], 100);
    setText(`dv-gpu${i}-mem`, g.mem_total ? `${gb(g.mem_used)} / ${gb(g.mem_total, 0)} ${t("GB")}` : null);
    setText(`dv-gpu${i}-temp`, g.temp == null ? null : `${fmt(g.temp)} °C`);
    const te = $(`dv-gpu${i}-temp`); if (te) te.dataset.hot = g.temp >= 80 ? "1" : "";
    // The clocks and the extra temperature sensors: a card that has no such sensor (no hot spot, no memory
    // temperature) says NA rather than a dash, so a missing reading is not mistaken for a zero.
    const na = (v) => v == null ? t("NA") : v;
    setText(`dv-gpu${i}-smclock`, na(g.sm_clock == null ? null : `${fmt(g.sm_clock)} MHz`));
    setText(`dv-gpu${i}-memclock`, na(g.mem_clock == null ? null : `${fmt(g.mem_clock)} MHz`));
    setText(`dv-gpu${i}-hotspot`, na(g.temp_hotspot == null ? null : `${fmt(g.temp_hotspot)} °C`));
    setText(`dv-gpu${i}-vramtemp`, na(g.temp_vram == null ? null : `${fmt(g.temp_vram)} °C`));
    setText(`dv-gpu${i}-power`, g.power == null ? null : `${fmt(g.power)} ${t("W")}${g.power_limit ? t(" of {x}", {x: fmt(g.power_limit)}) : ""}`);
    const gn = g.pcie_gen;
    setText(`dv-gpu${i}-pcie`, gn ? `Gen${gn} x${g.pcie_width || "?"}${g.pcie_rx_mb == null ? "" :
            ` · ↓${fmt(g.pcie_rx_mb, g.pcie_rx_mb < 10 ? 1 : 0)} ↑${fmt(g.pcie_tx_mb || 0, (g.pcie_tx_mb || 0) < 10 ? 1 : 0)} ${t("MB/s")}`}` : null);
  });
  (hw.sockets || []).forEach((s, i) => {
    setText(`dv-cpu${i}`, s.cpu == null ? null : `${fmt(s.cpu)}%`);
    if ($(`sp-cpu${i}`)) spark(`sp-cpu${i}`, h[`socket${i}_cpu`], 100);
    setText(`dv-cpu${i}-ram`, s.ram_total ? `${gb(s.ram_used || 0)} / ${gb(s.ram_total, 0)} ${t("GB")}` : null);
    const near = sg.filter((g) => g.numa === s.node).map((g) => `GPU ${g.index}`);
    setText(`dv-cpu${i}-gpu`, near.length ? near.join(", ") : t("none"));
  });

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} ${t("GB")}` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} ${t("GB")}` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">${esc(t("No requests yet"))}</td></tr>`;
  } else {
    const badge = {stop: ["", t("Done")], length: ["", t("Max tokens")], cancel: ["st-badge--queued", t("Stopped")],
                   disconnect: ["st-badge--queued", t("Closed")], error: ["st-badge--error", t("Error")]};
    const projTitle = {true: t("experimental speed projection on"), false: t("experimental speed projection off")};
    const pcieTitle = t("routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed");
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const when = new Date(r.time * 1000).toLocaleTimeString(locale(), {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="${esc(projTitle[!!r.projection])}">${r.projection ? "ESP" : esc(t("stock"))}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="${esc(pcieTitle)}">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh >= 512 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;   // MIN_PREFILL_TOKENS
      return `<tr><td>${esc(when)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} ${esc(t("s"))}</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? t("Show fewer") : t("Show all ({n})", {n: kept});
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="${esc(t("Copy"))}">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return t(mode === "project" ? "Projection control vector on layers {a}–{b}" : "Additive control vector on layers {a}–{b}",
           {a: a || "", b: b || ""}) +
         (single ? t(" (layer {n}'s direction)", {n: single.replace("single", "")}) : "") +
         t(". Per chat in Sampling. Its package describes the vector as a refusal-direction projection; measure the speed yourself");
}
function renderAbout(eng, hw, st) {
  // the K/V cache's format; the minimax_m3 path (INFO arch) keeps its 4-bit rows without the rotation
  const kv = {int8: t("8-bit"), q4_0: t(eng.arch === "minimax_m3" ? "4-bit" : "4-bit (Hadamard-rotated)"), fp16: t("16-bit"),
              k8v4: t("K 8-bit + V 4-bit"), f32: t("32-bit")}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    [t("Model"), eng.model],
    [t("Engine"), eng.version ? `v${eng.version}` : t("built from source")],
    [t("Context"), eng.max_context ? t("{n} tokens", {n: eng.max_context}) : null],
    [t("KV cache"), kv ? `${kv}${eng.kv_resident ? t(", streamed: {n} positions per layer in VRAM, the rest in RAM", {n: eng.kv_resident}) : t(", all in VRAM")}` : null],
    [t("Experts in VRAM"), eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} ${t("GB")})` : null],
    [t("Speculation"), eng.spec ? t("MTP drafts up to {n} tokens", {n: Math.max(0, (eng.mtp_max || eng.spec) - 1)}) + (eng.lookup ? t(", prompt lookup on") : "") : null],
    [t("Images"), eng.images ? t("on") : t("off")],
    [t("Experimental speed projection"), projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    [t("GPU"), st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} ${t("GB")}` : ""}` : (st.gpu_note || t("not readable (NVML)"))],
    [t("CPU"), st.cpu_name ? `${st.cpu_name}${st.threads ? t(", {n} threads", {n: st.threads}) : ""}` : null],
    [t("RAM"), hw.ram_total ? `${gb(hw.ram_total, 0)} ${t("GB")}` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    [t("OpenAI base URL"), `${base}/v1`, true],
    [t("Anthropic base URL"), base, true],
    [t("Model name"), eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "dacan_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? t("{n} tools · {a} of {b} servers connected", {n: mcpInfo.tools || 0, a: ready.length, b: servers.length}) : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? t("{n} tools from {names}; the model calls them when it decides to",
                                                   {n: mcpInfo.tools, names: ready.map((s) => s.name).join(", ")})
                                               : t("no server is connected yet (see the Monitor)");
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] ? [MCP_STATE[s.status][0], t(MCP_STATE[s.status][1])] : ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${esc(t("{n} tools", {n: s.tools.length}))}${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (dacan-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", t("on")], ["false", t("off")]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>${esc(t("default"))}</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="${esc(t("default"))}" autocomplete="off">`;
    }
    // the help text comes from the server (serve/runconfig.py EDITABLE) in English; the dictionary has each one
    const help = t(k.help || "");
    return `<label for="${id}" title="${esc(help)}">${esc(help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = t("Nothing changed."); return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? t("Saved ({list}); the earlier file is {file}.bak. Start the model again to use it.", {list: b.changed.join(", "), file: b.file})
      : t("Nothing changed.");
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", t("Not saved"), String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || t("code"))}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="${esc(t("Copy code"))}">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
const DEFAULTS = {thinking: "high", temperature: 0.9, top_p: 0.95, top_k: 20, max: "", seed: "7", show: true, esp: true, mcp: true};
// DACAN 19.10.2026: temperature 0.9 and seed 7 by default. A browser that kept an earlier default untouched
// (temperature 0.6 / 0.55 / 0.2 with top_p 0.95, top_k 20 and no seed) takes the new one; a value the user set stays.
const OLD_DEFAULT_TEMPERATURES = [0.6, 0.55, 0.2];
let settings = (() => {
  const s = store.get("sampling", {});
  if (OLD_DEFAULT_TEMPERATURES.includes(+s.temperature) && +s.top_p === 0.95 && +s.top_k === 20 && !s.seed) {
    s.temperature = DEFAULTS.temperature;
    s.seed = DEFAULTS.seed;
  }
  return {...DEFAULTS, ...s};
})();
let messages = store.get("chat", []);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name}))})));
}
function timeStr(ms) { return new Date(ms).toLocaleTimeString(locale(), {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || t("image"); wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || t("image")); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = t("You · {time}", {time: timeStr(m.time)});
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="${esc(t("Copy the answer"))}" title="${esc(t("Copy"))}">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(tc, k) {
  const [cls, label] = TOOL_STATE[tc.state] ? [TOOL_STATE[tc.state][0], t(TOOL_STATE[tc.state][1])] : ["", tc.state];
  const args = tc.arguments == null ? "" : JSON.stringify(tc.arguments, null, 2);
  const preview = tc.result != null ? tc.result : args.replace(/\s+/g, " ");
  let body = "";
  if (tc.open) {
    body = `<div class="tool-call__label">${esc(t("Arguments"))}</div><pre class="tool-call__pre">${esc(args || t("(being written)"))}</pre>`;
    if (tc.result != null) {
      body += `<div class="tool-call__label">${esc(t(tc.ok ? "Result" : "Error"))}${tc.chars ? esc(t(" · {n} characters", {n: tc.chars})) : ""}` +
              `${tc.truncated ? esc(t(", cut for the model")) : ""}</div><pre class="tool-call__pre">${esc(tc.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(tc.state)}"${tc.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(tc.name || "")}">${esc(tc.tool || tc.name || t("tool"))}</span>` +
    (tc.server ? `<span class="muted small">${esc(tc.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${tc.ms != null && tc.state !== "skipped" ? `<span class="muted small">${fmt(tc.ms / 1000, 1)} ${esc(t("s"))}</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `dacan_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? t("Thinking…") :
      m.thinkSecs != null ? t("Thought for {x} s", {x: fmt(m.thinkSecs, 1)}) : t("Thoughts");
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">${esc(t("Writing"))}</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = metaText(m) || (streaming ? "" : m.stopped ? t("Stopped") : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : t("Shift+Enter: new line");
}
// the line under an answer, written in the page's language from its facts (metaInfo); a chat kept by an older page
// has the English line itself (meta)
function metaText(m) {
  const f = m.metaInfo;
  if (!f) return m.meta || "";
  let s = "";
  if (f.n) {
    s = t("{n} tokens", {n: f.n}) + (f.rate ? ` · ${fmt(f.rate, 1)} ${t("tok/s")}` : "") + (f.stopped ? t(" · stopped") : "") +
        (f.projection == null ? "" : t(f.projection ? " · projection on" : " · projection off"));
  } else if (f.stopped) {
    s = t("Stopped");
  }
  if (f.tools) s = (s ? `${s} · ` : "") + t("{n} tool calls", {n: f.tools});
  if (f.limit) s += t(" · stopped at the limit of {n} tool rounds (mcp.max_rounds)", {n: f.limit});
  return s;
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (settings.temperature > 0) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  } else {
    body.temperature = 0;
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.dacan_mcp = true;   // this server may run MCP tools for it

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = t("This server needs an API key: add it under About > Settings.");
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || t("the engine reported an error"));
        if (j.usage) usage = j.usage;
        if (j.dacan_mcp) onTool(m, j.dacan_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", t("The request failed"), m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  // the line under the answer: its facts, written in the page's language when shown (metaText)
  const n = usage ? usage.completion_tokens : null;
  const info = {};
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    Object.assign(info, {n, rate: n >= 32 && secs > 0.25 ? n / secs : null, stopped: !!m.stopped,
                         projection: projectionLoaded() ? !!settings.esp : null});
  } else if (m.stopped) {
    info.stopped = true;
  }
  for (const tc of m.tools || []) if (tc.state === "writing" || tc.state === "running") { tc.state = "skipped"; tc.ms = null; }
  const ran = (m.tools || []).filter((tc) => tc.state === "done" || tc.state === "error").length;
  if (ran) info.tools = ran;
  if (m.limit) info.limit = m.limit;
  if (Object.keys(info).length) m.metaInfo = info;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
  if (chatStale) { chatStale = false; renderChat(); }   // the language changed while it was writing
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => {
  if (busy) { toast("warn", t("Still writing"), t("Stop the answer first.")); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat();
  renderChat();
  toast("info", t("New chat"), t("The last one was cleared."), 6000, {label: t("Undo"), run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", t("Nothing to save yet")); return; }
  const tools = (m) => (m.tools || []).filter((tc) => tc.result != null).map((tc) =>
    `<details><summary>${t("Tool")} ${tc.server ? `${tc.server} / ` : ""}${tc.tool || tc.name}${tc.ok ? "" : t(" (error)")}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(tc.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${tc.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## ${t("You")}\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>${t("Thinking")}</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `dacan-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", t("Pictures are off"), t("This model was set up for text only.")); continue; }
      if (f.size > 20e6) { toast("warn", t("Picture too large"), t("{name} is over 20 MB.", {name: f.name})); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || t("pasted image"), url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) {
      toast("warn", t("Not a text file"), t(health.images ? "{name}: attach text files (code, notes, logs, data) or pictures."
                                                          : "{name}: attach text files (code, notes, logs, data).", {name: f.name}));
      continue;
    }
    if (f.size > MAX_TEXT_FILE) { toast("warn", t("File too large"), t("{name} is over 512 KB.", {name: f.name})); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", t("Not a text file"), t("{name} looks like a binary file.", {name: f.name})); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", t("Remove"));
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); loadThreads(); }
}
// 19.10.2026: the CPU threads slider - the run config's cpu_threads (GET / POST /config); the server starts the
// engine again with them.  21.10.2026 (Vadim: the context size and the KV cache's quant right in the web page): the
// run config's max_context and kv_cache the same way; Apply sends what changed at once, so the model restarts once.
// Each control is hidden when the server has no such key (an older server, or no run config).
let threadsKey = null, ctxKey = null, kvKey = null;
const KV_TEXT = {fp16: "fp16 - full precision", int8: "int8 - half the memory", q4_0: "q4_0 - a quarter, 4-bit",
                 k8v4: "k8v4 - int8 K + 4-bit V"};
function fmtCtx(n) { return n % 1024 === 0 ? t("{x}K tokens", {x: String(n / 1024)}) : t("{n} tokens", {n}); }
function engineOptions(now, list, text) {
  return (now == null ? `<option value="" selected>${esc(t("engine default"))}</option>` : "") +
    list.map((x) => `<option value="${esc(String(x))}"${x === now ? " selected" : ""}>${esc(text(x))}</option>`).join("");
}
async function loadThreads() {
  threadsKey = ctxKey = kvKey = null;
  try {
    const r = await fetch("config", {headers: headers()});
    if (r.ok) {
      const keys = (await r.json()).keys || [];
      threadsKey = keys.find((k) => k.key === "cpu_threads") || null;
      ctxKey = keys.find((k) => k.key === "max_context") || null;
      kvKey = keys.find((k) => k.key === "kv_cache") || null;
      if (kvKey && !(kvKey.choices || []).length) kvKey = null;   // an engine without --kv
    }
  } catch (e) { /* the controls stay hidden */ }
  $("ctx-field").hidden = !ctxKey;
  if (ctxKey) {
    const now = ctxKey.value ?? null;
    const list = [...new Set([...(ctxKey.choices || []), ...(now == null ? [] : [now])])].sort((a, b) => a - b);
    $("s-ctx").innerHTML = engineOptions(now, list, fmtCtx);
  }
  $("kv-field").hidden = !kvKey;
  if (kvKey) $("s-kv").innerHTML = engineOptions(kvKey.value ?? null, kvKey.choices, (x) => (KV_TEXT[x] ? t(KV_TEXT[x]) : x));
  $("threads-field").hidden = !threadsKey;
  if (!threadsKey) return;
  const el = $("s-threads");
  el.min = threadsKey.min || 2; el.max = threadsKey.max || 64;
  el.value = threadsKey.value ?? threadsKey.max;
  outputThreads();
}
function outputThreads() {
  if (!threadsKey) return;
  const v = +$("s-threads").value, max = +threadsKey.max, usable = +threadsKey.usable || max;
  $("o-threads").textContent = t("{v} of {max} threads", {v, max}) + (v === max ? t(" (all)") : "");
  $("threads-note").textContent = (usable < max ? t("this system lets the engine use {u} of the CPU's {max} threads (more than {u} runs on those {u}); ",
                                                    {u: usable, max}) : "") + t("a change restarts the model (about a minute)");
}
async function saveEngine() {
  const set = {};
  if (threadsKey) {
    const v = +$("s-threads").value, now = threadsKey.value ?? +threadsKey.max;
    if (v !== +now) set.cpu_threads = v;
  }
  if (ctxKey && $("s-ctx").value !== "" && +$("s-ctx").value !== (ctxKey.value ?? null)) set.max_context = +$("s-ctx").value;
  if (kvKey && $("s-kv").value !== "" && $("s-kv").value !== (kvKey.value ?? null)) set.kv_cache = $("s-kv").value;
  const names = Object.keys(set);
  if (!names.length) return;
  const what = names.map((k) => k === "cpu_threads" ? t("CPU threads: {v}", {v: String(set[k])}) :
                                k === "max_context" ? t("context: {x}", {x: fmtCtx(set[k])}) : t("KV cache: {x}", {x: set[k]})).join(", ");
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    toast("success", what, b.engine === "restarting" ? t("The model restarts with it (about a minute); requests wait meanwhile.")
                                 : t("Saved: the model uses it from its next start."), 6000);
  } catch (e) {
    toast("error", t("Not changed - {what}", {what}), e.message, 6000);
  }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking, temperature: +s.temperature};
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const temp = +$("s-temp").value;
  $("o-temp").textContent = temp === 0 ? t("0 · greedy") : temp.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? t({none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v]) : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = temp === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-threads").oninput = outputThreads;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  saveEngine();
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", t("Sampling saved"), share ? t("Other apps (omp, API clients) use these settings from their next request.")
                                                  : t("Other apps use their own settings again."));
    } catch (e) {
      toast("error", t("Saved here, but not for other apps"), e.message, 6000);
    }
    return;
  }
  toast("success", t("Sampling saved"), settings.temperature === 0 ? t("Greedy: the same question gives the same answer.") : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ the page's language (web/i18n.js)
// 26.10.2026 (Vadim: «сделай выбор языка в веб морде, на анг и рус, для совсем глупых»): EN | RU in the header and
// under About > Settings.  A switch rewrites the fixed texts (applyI18n) and draws again what the page wrote itself;
// an answer being written is drawn again once it ends (its element is the one the stream writes into).
let chatStale = false;
function relocalize() {
  buildMetricCards();
  devicesKey = "";                                  // the GPU / socket cards are built again by the next render
  healthTexts();
  setBusy(!!busy);
  outputs();
  if (threadsKey) outputThreads();
  if ($("drawer").dataset.open === "true") loadThreads();
  renderMcp();
  if (tab === "about") loadConfig();
  if (lastMetrics) render(lastMetrics);
  if (busy) chatStale = true; else renderChat();
}
for (const b of document.querySelectorAll("[data-lang]")) {
  b.addEventListener("click", () => { if (b.dataset.lang !== LANG && setLangValue(b.dataset.lang)) relocalize(); });
}

// ------------------------------------------------------------------ start
setBusy(false);
healthTexts();
renderChat();
// /?lang=ru (or en): a link that opens the page in that language - kept, as a click on the switch is
const langParam = new URLSearchParams(location.search).get("lang");
if (langParam && langParam !== LANG && setLangValue(langParam)) relocalize();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();
