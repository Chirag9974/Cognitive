/* Cognitive Load Predictor — UI application (v2 "Aurora Glass").
   Vanilla ES module. Hash router, API client, four views (overview, session
   detail, synthetic run, and the in-browser cognitive battery).
   Visuals: SVG ring gauges, sparklines, probability spectrum ribbon,
   diverging factor bars, cursor spotlight tilt on glass cards.
   Motion: compositor-only, tokenized, fully cancelable. */

import { testView } from "./test.js";

const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const toastEl = $("#toast");

const BAND_COLOR = { Low: "#34d399", Manageable: "#22d3ee", Elevated: "#fbbf24", High: "#fb7185" };
const BAND_ORDER = ["Low", "Manageable", "Elevated", "High"];

/* ---------------- API client ---------------- */

async function api(method, path, body) {
  let res;
  try {
    res = await fetch(`/api/v1${path}`, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error("API unreachable — is the server running?");
  }
  let json = null;
  try { json = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const message = json && json.error ? json.error.message : `Request failed (${res.status})`;
    const code = json && json.error ? json.error.code : "HTTP_ERROR";
    const err = new Error(message);
    err.code = code;
    err.status = res.status;
    throw err;
  }
  return json;
}

/* ---------------- Health pill ---------------- */

async function pollHealth() {
  const dot = $("#api-dot");
  const label = $("#api-label");
  try {
    const health = await api("GET", "/health");
    dot.className = "dot live";
    label.textContent = `api ${health.api_version} · model ${health.model_version}`;
  } catch {
    dot.className = "dot down";
    label.textContent = "api offline";
  }
}
pollHealth();
setInterval(pollHealth, 30000);

/* ---------------- Toast ---------------- */

let toastTimer;
function toast(message, kind = "error") {
  toastEl.className = `toast show ${kind}`;
  toastEl.textContent = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 4200);
}

/* ---------------- Formatting helpers ---------------- */

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " · " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
function fmtPct(value, digits = 1) { return `${(value * 100).toFixed(digits)}%`; }
function esc(text) {
  const div = document.createElement("div");
  div.textContent = String(text ?? "");
  return div.innerHTML;
}

function countUp(el, target, decimals = 1, suffix = "") {
  if (reduceMotion) { el.textContent = target.toFixed(decimals) + suffix; return; }
  const start = performance.now();
  const duration = 700;
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * eased).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* Arm every animated visual inside a container: progress bars, spectrum
   segments, ring gauges, and sparkline draw-ins. Runs after insertion. */
function armVisuals(container) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll(".bar-fill, .spec-seg, .fdiv-bar").forEach(el => {
      el.style.setProperty("--w", el.dataset.w || "0");
    });
    container.querySelectorAll(".ring-val[data-off]").forEach(el => {
      el.style.strokeDashoffset = el.dataset.off;
    });
    container.querySelectorAll(".spark-line[data-draw]").forEach(el => {
      el.style.strokeDashoffset = "0";
    });
    container.querySelectorAll(".spark-area, .spark-dot").forEach(el => {
      el.style.opacity = "1";
    });
  }));
}

/* ---------------- Visual builders ---------------- */

function ringGauge(score, color, size = 148) {
  const pct = Math.max(0, Math.min(1, score / 100));
  const r = 62;
  const circ = 2 * Math.PI * r;
  const off = (circ * (1 - pct)).toFixed(1);
  return `
    <svg class="ring" style="width:${size}px;height:${size}px" viewBox="0 0 148 148" role="img" aria-label="score ${score.toFixed(1)} of 100">
      <circle class="ring-track" cx="74" cy="74" r="${r}"></circle>
      <circle class="ring-val" cx="74" cy="74" r="${r}"
        style="--circ:${circ.toFixed(1)}; stroke:${color}" data-off="${off}"></circle>
    </svg>`;
}

function sparkline(values, width = 560, height = 110) {
  const pts = values.filter(v => Number.isFinite(v));
  if (pts.length < 2) return "";
  const min = Math.min(...pts), max = Math.max(...pts);
  const span = (max - min) || 1;
  const px = 12, py = 16;
  const iw = width - px * 2, ih = height - py * 2;
  const coords = pts.map((v, i) => [
    px + (i / (pts.length - 1)) * iw,
    py + ih - ((v - min) / span) * ih,
  ]);
  const line = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" ");
  const area = `${line} L${coords[coords.length - 1][0].toFixed(1)},${height - 2} L${coords[0][0].toFixed(1)},${height - 2} Z`;
  const last = coords[coords.length - 1];
  const gid = "sg" + Math.random().toString(36).slice(2, 8);
  return `
    <svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#818cf8" stop-opacity=".28"></stop>
          <stop offset="1" stop-color="#818cf8" stop-opacity="0"></stop>
        </linearGradient>
      </defs>
      <path class="spark-area" d="${area}" fill="url(#${gid})" style="opacity:0;transition:opacity var(--t-slow) var(--ease-out) .45s"></path>
      <path class="spark-line" d="${line}" pathLength="1" fill="none" stroke="#a5b4fc" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"
        style="stroke-dasharray:1;stroke-dashoffset:1;transition:stroke-dashoffset 1.1s var(--ease-out) .15s" data-draw="1"></path>
      <circle class="spark-dot" cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="4.5" fill="#22d3ee"
        style="opacity:0;transition:opacity var(--t-base) var(--ease-out) 1s"></circle>
    </svg>`;
}

function spectrum(probs) {
  const cls = { Low: "s-low", Manageable: "s-man", Elevated: "s-elv", High: "s-high" };
  const segs = BAND_ORDER.map(b => {
    const p = Math.max(probs[b] || 0, 0);
    return `<div class="spec-seg ${cls[b]}" data-w="${p.toFixed(4)}"
      style="flex-grow:${Math.max(p, 0.0001).toFixed(4)}"></div>`;
  }).join("");
  const legend = BAND_ORDER.map(b => `
    <span class="legend-item"><span class="swatch" style="background:${BAND_COLOR[b]}"></span>
      ${b} <span class="pct">${fmtPct(probs[b] || 0)}</span></span>`).join("");
  return `<div class="spectrum">${segs}</div><div class="spectrum-legend">${legend}</div>`;
}

function bandPill(band, large = false) {
  const safe = band && band !== "insufficient_history" ? band : "insufficient_history";
  return `<span class="band${large ? " lg" : ""}" data-band="${esc(safe)}">${esc(safe === "insufficient_history" ? "not enough history" : safe)}</span>`;
}

function deltaPill(delta) {
  if (delta === null || delta === undefined || !Number.isFinite(delta)) return "";
  const up = delta >= 0;
  return `<span class="delta ${up ? "up" : "down"}">${up ? "▲" : "▼"} ${up ? "+" : ""}${delta.toFixed(1)}</span>`;
}

function barRow(label, value, max = 1, tone = "neutral", display) {
  const w = max > 0 ? Math.max(0.004, Math.min(1, value / max)) : 0;
  return `
    <div class="bar-row">
      <span class="bar-label" title="${esc(label)}">${esc(label)}</span>
      <div class="bar-track"><div class="bar-fill ${tone}" data-w="${w.toFixed(4)}"></div></div>
      <span class="bar-value">${esc(display ?? fmtPct(value))}</span>
    </div>`;
}

/* Diverging factor rows: raised-load bars grow leftward from center,
   lowered-load bars grow rightward — the sign is visible at a glance. */
function divergingFactors(contribs) {
  const maxAbs = Math.max(...contribs.map(c => Math.abs(c.points)), 0.001);
  return contribs.map(c => {
    const w = (Math.abs(c.points) / maxAbs).toFixed(4);
    const raised = c.impact === "raised_load";
    const bar = `<div class="fdiv-bar ${raised ? "raised" : "lowered"}" data-w="${w}"
      style="transform-origin:${raised ? "right" : "left"}"></div>`;
    return `
      <div class="fdiv-row">
        <span class="fdiv-what" title="${esc(c.detail || "")}">${esc(c.label)}
          <span class="mono" style="color:var(--text-3)">${esc(c.value_display || "")}</span></span>
        <div class="fdiv-left">${raised ? bar : ""}</div>
        <div class="fdiv-mid"><span class="pts ${raised ? "raised" : "lowered"}">${c.points >= 0 ? "+" : ""}${c.points.toFixed(2)}</span></div>
        <div class="fdiv-right">${raised ? "" : bar}</div>
      </div>`;
  }).join("");
}

/* ---------------- Cursor spotlight tilt ---------------- */

function initTilt(root) {
  if (reduceMotion || matchMedia("(pointer: coarse)").matches) return;
  root.querySelectorAll(".card:not([data-tilt])").forEach(card => {
    card.dataset.tilt = "1";
    card.addEventListener("pointermove", e => {
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width;
      const py = (e.clientY - r.top) / r.height;
      card.style.setProperty("--mx", `${(px * 100).toFixed(2)}%`);
      card.style.setProperty("--my", `${(py * 100).toFixed(2)}%`);
      card.style.setProperty("--rx", `${((0.5 - py) * 2.2).toFixed(2)}deg`);
      card.style.setProperty("--ry", `${((px - 0.5) * 2.2).toFixed(2)}deg`);
      card.classList.add("tilt");
    });
    card.addEventListener("pointerleave", () => {
      card.classList.remove("tilt");
      card.style.setProperty("--rx", "0deg");
      card.style.setProperty("--ry", "0deg");
    });
  });
}
new MutationObserver(() => initTilt(view)).observe(view, { childList: true, subtree: true });

/* ---------------- Synthetic session generator (client-side demo) ---------------- */

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeSession(participantId, profile, seed = Date.now() % 100000) {
  const rnd = mulberry32(seed);
  const P = {
    low:        { rt: 300, rs: 40,  fs: .03, acc: .95, cost: 60,  mem: .85, span: 6, str: .95, scost: 55 },
    manageable: { rt: 420, rs: 60,  fs: .07, acc: .88, cost: 130, mem: .65, span: 5, str: .88, scost: 105 },
    elevated:   { rt: 560, rs: 95,  fs: .13, acc: .76, cost: 205, mem: .45, span: 4, str: .78, scost: 175 },
    high:       { rt: 760, rs: 140, fs: .20, acc: .58, cost: 320, mem: .24, span: 3, str: .58, scost: 285 },
  }[profile];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const rt = (m, s) => Math.round(clamp(m + (rnd() + rnd() + rnd() - 1.5) * 2 * s, 120, m + 4 * s));
  const trials = (n, make) => Array.from({ length: n }, make);
  const started = new Date();
  return {
    session_id: `${participantId}-web-${Date.now()}`,
    participant_id: participantId,
    started_at: started.toISOString().slice(0, 19),
    finished_at: new Date(started.getTime() + 370000).toISOString().slice(0, 19),
    duration_seconds: 370,
    baseline_session: false,
    questionnaire: { sleep_hours: 7, stress_level: 2, caffeine_mg: 50, fatigue_level: 2, notes: "generated in browser demo" },
    self_report_load_band: profile,
    tasks: {
      reaction_time: { trials: trials(8, () => {
        const fs = rnd() < P.fs;
        return { false_start: fs, missed: false, response_ms: fs ? null : rt(P.rt, P.rs),
                 response_seconds: null, response_window_ms: 60000, correct: !fs };
      })},
      task_switching: { trials: trials(24, () => {
        const sw = rnd() < 0.5;
        return { is_switch: sw, correct: rnd() < (sw ? P.acc : Math.min(.99, P.acc + .05)), missed: false,
                 response_ms: rt(P.rt + (sw ? P.cost : 0), P.rs * 1.1), response_window_ms: 60000 };
      })},
      working_memory: { trials: [3,3,4,4,5,5,6,6,7,7].map(len => {
        const full = rnd() < P.mem * Math.max(.45, 1.1 - (len - 3) * .1);
        return { sequence_length: len, full_correct: full, position_accuracy: full ? 1.0 : clamp(rnd(), 0, 1),
                 response_ms: rt(1400 + len * 260, 350), response_window_ms: 60000, correct: full };
      })},
      stroop: { trials: trials(28, () => {
        const con = rnd() < 0.4;
        return { congruent: con, correct: rnd() < P.str, missed: false,
                 response_ms: rt(P.rt + (con ? 0 : P.scost), 80), response_window_ms: 60000 };
      })},
    },
  };
}

/* ---------------- Shared payload renderer (detail + run views) ---------------- */

function renderPredictionSection(p) {
  const u = p.uncertainty;
  const drivers = (u.drivers || []).map(d => `<li>${esc(typeof d === "string" ? d : d.detail)}</li>`).join("");
  const score = +p.overall_score || 0;
  const color = BAND_COLOR[p.band] || "#818cf8";
  return `
    <div class="bento">
      <div class="card hero span-7">
        <div class="head"><h3>Prediction</h3><span class="kpi-sub">${esc(p.prediction_method)} · ${esc(p.history_size)} prior session(s)</span></div>
        <div class="hero-score-row">
          ${ringGauge(score, color)}
          <div class="hero-kpis">
            <div class="kpi-value grad-text" data-count="${score.toFixed(1)}">0.0</div>
            <div class="kpi-row" style="margin-top:10px">${bandPill(p.band, true)}
              <span class="kpi-sub">overall score · higher is lower load</span></div>
          </div>
        </div>
      </div>
      <div class="card span-5">
        <div class="head"><h3>Uncertainty</h3><span class="band" data-band="">${esc(u.level)}</span></div>
        <p class="narrative">${esc(u.statement || "")}</p>
        ${drivers ? `<ul class="narrative" style="margin-top:10px;padding-left:18px">${drivers}</ul>` : ""}
      </div>
      <div class="card span-12">
        <div class="head"><h3>Band probabilities</h3></div>
        ${spectrum(p.probabilities || {})}
      </div>
    </div>`;
}

function renderExplanationSection(p) {
  const x = p.explanation;
  const rows = divergingFactors((x.all_contributions || []).slice(0, 8));
  return `
    <div class="bento" style="margin-top:16px">
      <div class="card span-7">
        <div class="head"><h3>Why this result</h3><span class="kpi-sub">${esc(x.method)} · exact decomposition</span></div>
        <ul class="narrative" style="padding-left:18px">${(x.narrative || []).map(s => `<li>${esc(s)}</li>`).join("")}</ul>
        <div class="factors">${rows}</div>
        <p class="kpi-sub" style="margin-top:12px">Points are exact contributions to the overall score — negative values pushed the score down (raised load).</p>
      </div>
      <div class="card span-5">
        <div class="head"><h3>Data quality</h3>${bandPill(p.anomaly.status)}</div>
        ${barRow("quality", p.quality.score, 100, p.quality.score >= 80 ? "positive" : p.quality.score >= 50 ? "warn" : "negative", `${p.quality.score.toFixed(0)}/100`)}
        <div style="margin-top:10px">
          ${(p.anomaly.reasons || []).map(r => `
            <div class="flag ${esc(r.severity)}"><span class="sev">${esc(r.severity)}</span><span>${esc(r.detail)}</span></div>`).join("")
            || `<div class="flag"><span class="sev">none</span><span>No reliability concerns detected.</span></div>`}
        </div>
      </div>
    </div>`;
}

function renderForwardSection(p) {
  const f = p.forecast, r = p.recommendations, d = p.difficulty;
  const trend = p.trend || {};
  const forecastHtml = f.available
    ? `<div class="kpi-row">
         <span class="kpi-value md grad-text">${esc(f.band)}</span>${bandPill(f.band)}
       </div>
       <p class="kpi-sub" style="margin-top:8px">p=${fmtPct(f.probability, 2)} · projected score ${f.projected_score} · ${esc(f.method)}</p>
       <div style="margin-top:12px">${spectrum(f.probabilities || {})}</div>`
    : `<p class="narrative">${esc(f.reason)}</p>
       <p class="kpi-sub" style="margin-top:6px">Guard: forecasting waits for ${f.min_sessions} valid prior sessions.</p>`;
  const recos = (r.items || []).map(item => `
    <div class="reco">
      <div class="title">${esc(item.title)} <span class="pri">priority ${item.priority}</span></div>
      <p class="narrative" style="margin-top:4px">${esc(item.action)}</p>
      <p class="basis">Basis: ${esc(item.basis)}</p>
    </div>`).join("") || `<p class="narrative">No specific recommendations — the session looks healthy.</p>`;
  return `
    <div class="bento" style="margin-top:16px">
      <div class="card span-5">
        <div class="head"><h3>Difficulty &amp; trend</h3></div>
        ${barRow("difficulty", d.score, 100, "neutral", `${d.score.toFixed(0)}/100`)}
        <p class="kpi-sub" style="margin-top:6px">adjusted error rate ${d.difficulty_adjusted_error_rate.toFixed(3)} (raw ${d.raw_error_rate.toFixed(3)})</p>
        <p class="narrative" style="margin-top:10px">Trend: <strong>${esc(trend.direction)}</strong>${
          trend.score_change !== null && trend.score_change !== undefined
            ? ` · recent change ${trend.score_change >= 0 ? "+" : ""}${trend.score_change} over ${trend.sessions_used} session(s)` : ""
        }</p>
      </div>
      <div class="card span-7">
        <div class="head"><h3>Next-session forecast</h3></div>
        ${forecastHtml}
      </div>
      <div class="card span-12">
        <div class="head"><h3>Recommendations</h3><span class="kpi-sub">rule-based · ${r.items.length}/4</span></div>
        ${recos}
        <p class="kpi-sub" style="margin-top:10px">${esc(r.caveat)}</p>
      </div>
    </div>`;
}

function renderPayload(root, p) {
  root.innerHTML = `
    ${renderPredictionSection(p)}
    ${renderExplanationSection(p)}
    ${renderForwardSection(p)}
  `;
  armVisuals(root);
  root.querySelectorAll(".kpi-value[data-count]").forEach(el => {
    countUp(el, parseFloat(el.dataset.count), 1);
  });
}

/* ---------------- View: Overview ---------------- */

/* Render-generation guard: only the latest overview run may touch the DOM
   or the URL — stale async runs (rapid participant switches, initial load
   racing a param change) abort instead of clobbering fresh renders. */
let overviewGen = 0;

async function overviewView(params) {
  const gen = ++overviewGen;
  const selected = params.get("p") || "";
  view.innerHTML = `
    <div class="page-head">
      <h1>Cognitive load, <span class="serif-accent">made visible.</span></h1>
      <p class="lede">Behavioral assessment across participants — scores, bands, history trends, and next-session forecasts.</p>
    </div>
    <div class="toolbar">
      <div class="field">
        <label for="participant">Participant</label>
        <select id="participant"><option value="">Loading…</option></select>
      </div>
      <div class="field">
        <label>&nbsp;</label>
        <a class="btn primary" href="#/test">Take the cognitive test</a>
      </div>
    </div>
    <div id="overview-body"><div class="bento"><div class="sk-block skeleton span-5"></div><div class="sk-block skeleton span-4"></div><div class="sk-block skeleton span-3"></div></div></div>
  `;

  const select = $("#participant");
  let participants = [];
  try {
    const data = await api("GET", "/participants");
    if (gen !== overviewGen) return;
    participants = data.participants || [];
  } catch (e) { if (gen === overviewGen) toast(e.message); }

  if (!participants.length) {
    if (gen !== overviewGen) return;
    $("#overview-body").innerHTML = `
      <div class="card"><div class="empty">
        <div class="glyph">◍</div>
        <p><strong>No sessions yet</strong></p>
        <p class="hint">Take the four-task cognitive test and your scores, bands and suggestions appear here.</p>
        <div class="actions">
          <a class="btn primary" href="#/test">Take the cognitive test</a>
          <a class="btn ghost" href="#/run">Simulate a session</a>
        </div>
      </div></div>`;
    select.innerHTML = `<option value="">No participants</option>`;
    return;
  }

  select.innerHTML = participants.map(pt =>
    `<option value="${esc(pt.participant_id)}" ${pt.participant_id === selected ? "selected" : ""}>
      ${esc(pt.participant_id)} (${pt.sessions})</option>`).join("");
  const current = selected || participants[0].participant_id;
  select.value = current;
  if (!selected && gen === overviewGen) history.replaceState(null, "", `#/?p=${encodeURIComponent(current)}`);
  select.addEventListener("change", () => { location.hash = `#/?p=${encodeURIComponent(select.value)}`; });

  await renderOverviewBody(current, gen);
}

async function renderOverviewBody(pid, gen) {
  if (gen !== undefined && gen !== overviewGen) return;
  const body = $("#overview-body");
  body.innerHTML = `<div class="bento"><div class="sk-block skeleton span-5"></div><div class="sk-block skeleton span-7"></div></div>`;

  let history = [];
  try {
    history = (await api("GET", `/participants/${encodeURIComponent(pid)}/history`)).sessions || [];
    if (gen !== undefined && gen !== overviewGen) return;
  } catch (e) { if (gen === overviewGen) toast(e.message); return; }

  if (!history.length) {
    if (gen !== undefined && gen !== overviewGen) return;
    body.innerHTML = `<div class="card"><div class="empty"><div class="glyph">◍</div>
      <p><strong>No sessions for ${esc(pid)}</strong></p>
      <p class="hint">Take the cognitive test under this id, or simulate one.</p>
      <div class="actions"><a class="btn primary" href="#/test">Take the cognitive test</a>
        <a class="btn ghost" href="#/run">Simulate</a></div></div></div>`;
    return;
  }

  const sorted = [...history].sort((a, b) => (a.started_at || "").localeCompare(b.started_at || ""));
  const latest = sorted[sorted.length - 1];
  const prev = sorted.length > 1 ? sorted[sorted.length - 2] : null;
  const latestScore = Number.parseFloat(latest.overall_score);
  const prevScore = prev ? Number.parseFloat(prev.overall_score) : Number.NaN;
  const hasLatest = Number.isFinite(latestScore);
  const delta = hasLatest && Number.isFinite(prevScore) ? latestScore - prevScore : null;
  const band = latest.final_load_band || latest.self_report_load_band || "";
  const sparkValues = sorted.map(s => Number.parseFloat(s.overall_score)).filter(Number.isFinite);

  let forecastHtml;
  if (sorted.length >= 5) {
    try {
      const fc = await api("POST", "/forecast", { participant_id: pid });
      if (gen !== overviewGen) return;
      forecastHtml = `
        <div class="kpi-row">
          <span class="kpi-value md grad-text">${esc(fc.band)}</span>${bandPill(fc.band)}
        </div>
        <p class="kpi-sub" style="margin-top:8px">p=${fmtPct(fc.probability, 2)} · projected score ${fc.projected_score} · from ${fc.inputs.sessions_used} sessions</p>
        <div style="margin-top:12px">${spectrum(fc.probabilities || {})}</div>`;
    } catch (e) {
      forecastHtml = `<p class="narrative">${esc(e.message)}</p>`;
    }
  } else {
    forecastHtml = `<p class="narrative">Not enough history yet — forecasting unlocks at 5 valid prior sessions (${sorted.length}/5).</p>`;
  }

  const rows = [...sorted].reverse().slice(0, 8).map(row => {
    const score = row.overall_score === "" || row.overall_score === undefined ? "—" : parseFloat(row.overall_score).toFixed(1);
    const rband = row.final_load_band || row.self_report_load_band || "";
    return `
      <tr data-sid="${esc(row.session_id)}">
        <td class="mono">${esc(row.session_id)}</td>
        <td>${esc(fmtDateTime(row.started_at))}</td>
        <td class="num">${esc(score)}</td>
        <td>${rband ? bandPill(rband) : `<span class="kpi-sub">—</span>`}</td>
      </tr>`;
  }).join("");

  body.innerHTML = `
    <div class="bento stagger">
      <div class="card hero span-5">
        <div class="head"><h3>Latest score</h3><span class="kpi-sub">${esc(fmtDate(latest.started_at))}</span></div>
        <div class="hero-score-row">
          ${hasLatest ? ringGauge(latestScore, BAND_COLOR[band] || "#818cf8", 132) : ""}
          <div class="hero-kpis">
            ${hasLatest ? `<div class="kpi-value grad-text" data-count="${latestScore.toFixed(1)}">0.0</div>` : `<div class="kpi-value md">—</div>`}
            <div class="kpi-row" style="margin-top:10px">${bandPill(band, true)}${deltaPill(delta)}</div>
          </div>
        </div>
      </div>
      <div class="card span-4">
        <div class="head"><h3>Score history</h3><span class="kpi-sub">${sparkValues.length} scored</span></div>
        ${sparkline(sparkValues) || `<p class="narrative">Not enough scored sessions to chart yet.</p>`}
      </div>
      <div class="card span-3">
        <div class="head"><h3>Record</h3></div>
        <div class="kpi-value md" data-count="${sorted.length}" data-dec="0">0</div>
        <p class="kpi-sub">sessions on record for ${esc(pid)}</p>
      </div>
      <div class="card span-5">
        <div class="head"><h3>Next-session forecast</h3><span class="kpi-sub">FR-7</span></div>
        ${forecastHtml}
      </div>
      <div class="card span-7">
        <div class="head"><h3>Recent sessions</h3><span class="kpi-sub">click a row for the full V2 report</span></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Session</th><th>When</th><th>Score</th><th>Band</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </div>`;

  armVisuals(body);
  body.querySelectorAll(".kpi-value[data-count]").forEach(el => {
    const target = parseFloat(el.dataset.count);
    if (Number.isFinite(target)) countUp(el, target, +(el.dataset.dec || 1));
  });
  body.querySelectorAll("tr[data-sid]").forEach(tr => {
    tr.addEventListener("click", () => { location.hash = `#/session/${encodeURIComponent(tr.dataset.sid)}`; });
  });
}

/* ---------------- View: Session detail ---------------- */

async function sessionView(sessionId) {
  view.innerHTML = `
    <div class="page-head">
      <h1 style="font-size:28px" class="mono">${esc(decodeURIComponent(sessionId))}</h1>
      <p class="lede">Full V2 intelligence report — prediction, uncertainty, exact explanation, data quality, and recommendations.</p>
    </div>
    <div id="detail-body">
      <div class="bento"><div class="sk-block skeleton span-7"></div><div class="sk-block skeleton span-5"></div></div>
    </div>`;

  let payload;
  try {
    payload = await api("GET", `/sessions/${encodeURIComponent(sessionId)}/analysis`);
  } catch (e) {
    $("#detail-body").innerHTML = `
      <div class="card"><div class="empty">
        <div class="glyph">◌</div>
        <p><strong>${esc(e.code === "SESSION_NOT_FOUND" ? "Session not found" : "Could not analyze session")}</strong></p>
        <p class="hint">${esc(e.message)}</p>
        <div class="actions"><a class="btn" href="#/">Back to overview</a></div>
      </div></div>`;
    return;
  }
  renderPayload($("#detail-body"), payload);
  document.title = `${sessionId} — Cognitive Load Predictor`;
}

/* ---------------- View: Run session ---------------- */

function runView() {
  view.innerHTML = `
    <div class="page-head">
      <h1>Run a <span class="serif-accent">session.</span></h1>
      <p class="lede">Generate a synthetic session (same trial schema as the desktop app) and see the full V2 analysis instantly. Persist it to include it in history and forecasts.</p>
    </div>
    <div class="toolbar">
      <div class="field"><label for="run-pid">Participant</label>
        <input type="text" id="run-pid" value="DEMO" size="10" spellcheck="false"></div>
      <div class="field"><label for="run-profile">Expected load</label>
        <select id="run-profile">
          <option value="low">low</option>
          <option value="manageable" selected>manageable</option>
          <option value="elevated">elevated</option>
          <option value="high">high</option>
        </select></div>
      <div class="field"><label>&nbsp;</label>
        <button class="btn primary" id="run-analyze">Analyze</button></div>
      <div class="field"><label>&nbsp;</label>
        <button class="btn" id="run-enrich">Analyze + save</button></div>
    </div>
    <div id="run-result"></div>
  `;

  async function execute(persist) {
    const pid = ($("#run-pid").value || "DEMO").trim();
    const profile = $("#run-profile").value;
    const result = $("#run-result");
    const buttons = [$("#run-analyze"), $("#run-enrich")];
    buttons.forEach(b => { b.disabled = true; });
    result.innerHTML = `<div class="bento"><div class="sk-block skeleton span-7"></div><div class="sk-block skeleton span-5"></div></div>`;
    try {
      const session = makeSession(pid, profile);
      const payload = await api("POST", persist ? "/sessions/analyze/enrich" : "/sessions/analyze", session);
      toast(persist ? "Session analyzed and saved to history." : "Session analyzed (not saved).", "ok");
      result.innerHTML = `
        <div class="section-title">Result <span class="meta">${esc(session.session_id)}</span>
          ${persist ? `<a class="mono" href="#/session/${encodeURIComponent(session.session_id)}">open saved report →</a>` : ""}
        </div>
        <div id="run-payload"></div>`;
      renderPayload($("#run-payload"), payload);
    } catch (e) {
      toast(e.message);
      result.innerHTML = `<div class="card"><div class="empty"><div class="glyph">◌</div>
        <p><strong>Analysis failed</strong></p><p class="hint">${esc(e.message)}</p></div></div>`;
    } finally {
      buttons.forEach(b => { b.disabled = false; });
    }
  }

  $("#run-analyze").addEventListener("click", () => execute(false));
  $("#run-enrich").addEventListener("click", () => execute(true));
}

/* ---------------- Router ---------------- */

const routes = [
  { pattern: /^#\/(?:\?(.*))?$/, handler: m => overviewView(new URLSearchParams(m[1] || "")), nav: "overview" },
  { pattern: /^#\/test\/?$/, handler: () => { activeTeardown = testView({ api, toast, esc, renderPayload }); }, nav: "test" },
  { pattern: /^#\/run\/?$/, handler: () => runView(), nav: "run" },
  { pattern: /^#\/session\/([^?]+)(?:\?.*)?$/, handler: m => sessionView(m[1]), nav: null },
];

/* A view may register cleanup (the test battery aborts its timers and drops
   focus mode when you navigate away). */
let activeTeardown = null;

function router() {
  if (activeTeardown) {
    try { activeTeardown(); } catch { /* teardown is best-effort */ }
    activeTeardown = null;
  }
  const hash = location.hash || "#/";
  for (const route of routes) {
    const match = hash.match(route.pattern);
    if (match) {
      if (route.nav) {
        document.querySelectorAll("[data-nav]").forEach(a => a.setAttribute("aria-current", a.dataset.nav === route.nav ? "page" : "false"));
      }
      document.title = "Cognitive Load Predictor";
      const fresh = $("#view");
      fresh.classList.remove("view");
      void fresh.offsetWidth; /* restart the entrance animation */
      fresh.classList.add("view");
      route.handler(match);
      return;
    }
  }
  location.hash = "#/";
}
window.addEventListener("hashchange", router);
router();
