/* Cognitive Load Predictor — in-browser cognitive battery.
   A faithful port of the desktop tkinter battery (src/cognitive_load/app.py):
   same four tasks, same trial counts, same timings, same trial schema — so a
   session taken here scores identically to one taken in the desktop app.

   Tasks: reaction time (8) · task switching (26) · working memory (10) · stroop (28)
   Timing is measured with performance.now(); every wait is abortable.

   A test seam is exposed on window.__clpTest so the battery can be driven
   headlessly (state + synthetic key presses). */

import { TASK_INTRO_MS, TASK_TIMING } from "./config.js";

/* Fail loudly if the shared timing config drifts: a missing key used to
   silently turn every keypress trial into a missed one. */
const TIMING_SHAPE = {
  reaction_time: ["response_window_ms", "feedback_ms", "pre_signal_delay_ms"],
  task_switching: ["response_window_ms", "feedback_ms"],
  working_memory: ["pre_load_ms", "digit_display_ms", "digit_gap_ms", "post_load_ms", "recall_window_ms", "feedback_ms"],
  stroop: ["response_window_ms", "feedback_ms"],
};
for (const [taskName, keys] of Object.entries(TIMING_SHAPE)) {
  const missing = keys.filter((key) => TASK_TIMING[taskName]?.[key] === undefined);
  if (missing.length) throw new Error(`[cognitive-battery] TASK_TIMING.${taskName} is missing: ${missing.join(", ")}`);
}

const TASK_ORDER = ["reaction_time", "task_switching", "working_memory", "stroop"];

const TASK_META = {
  reaction_time: {
    label: "Reaction time",
    trials: 8,
    keys: "SPACE",
    blurb: "Press SPACE the moment the panel turns green. Pressing early is a false start.",
  },
  task_switching: {
    label: "Task switching",
    trials: 26,
    keys: "F / J",
    blurb: "A cue tells you the rule: NUMBER → F=odd, J=even. LETTER → F=vowel, J=consonant.",
  },
  working_memory: {
    label: "Working memory",
    trials: 10,
    keys: "type + ENTER",
    blurb: "Digits flash one at a time. Type the sequence exactly, then press Enter.",
  },
  stroop: {
    label: "Selective attention",
    trials: 28,
    keys: "R / G / B / Y",
    blurb: "Respond to the ink colour only and ignore what the word says.",
  },
};

const MEMORY_LENGTHS = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
const STROOP_COLORS = [
  ["RED", "red", "r", "#f87171"],
  ["GREEN", "green", "g", "#34d399"],
  ["BLUE", "blue", "b", "#60a5fa"],
  ["YELLOW", "goldenrod", "y", "#fbbf24"],
];
const STROOP_CSS = { red: "#f87171", green: "#34d399", blue: "#60a5fa", goldenrod: "#fbbf24" };
const VOWELS = "AEIOU";
const CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ";

const TOTAL_TRIALS = TASK_ORDER.reduce((sum, name) => sum + TASK_META[name].trials, 0);

/* ---------------------------------------------------------------- utilities */

const pick = (list) => list[Math.floor(Math.random() * list.length)];
const randInt = (min, max) => min + Math.floor(Math.random() * (max - min + 1));
const round2 = (value) => Math.round(value * 100) / 100;

function isTypingTarget(target) {
  if (!target) return false;
  const tag = (target.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

/** Abortable controller: every timer/listener is torn down on abort(). */
function controller() {
  const timers = new Set();
  const cleanups = new Set();
  const ctl = {
    aborted: false,
    setTimer(fn, ms) {
      const id = setTimeout(() => {
        timers.delete(id);
        fn();
      }, ms);
      timers.add(id);
      return id;
    },
    sleep(ms) {
      return new Promise((resolve) => ctl.setTimer(resolve, ms));
    },
    onCleanup(fn) {
      cleanups.add(fn);
    },
    abort() {
      if (ctl.aborted) return;
      ctl.aborted = true;
      timers.forEach((id) => clearTimeout(id));
      timers.clear();
      [...cleanups].forEach((fn) => fn());
      cleanups.clear();
    },
  };
  return ctl;
}

/** Accepts only the keys the current trial defines; other presses are ignored
    (matching the desktop app, which filters keysyms). */
function keyOrTimeout(ctl, accept, ms) {
  // A missing/zero limit would resolve instantly and record a phantom miss
  // (that is a data-corrupting silent failure), so fall back loudly instead.
  if (!Number.isFinite(ms) || ms <= 0) {
    console.warn("[cognitive-battery] invalid response window", ms, "— falling back to 60s");
    ms = 60000;
  }
  return new Promise((resolve) => {
    let settled = false;
    let teardown = () => {};
    const finish = (result) => {
      if (settled) return;
      settled = true;
      teardown();
      resolve(result);
    };
    const onKey = (event) => {
      if (isTypingTarget(event.target)) return;
      const key = normalizeKey(event);
      if (!key) return;
      if (event.code === "Space") event.preventDefault();
      if (accept(key) !== true) return;
      finish({ type: "key", key, ts: performance.now() });
    };
    document.addEventListener("keydown", onKey);
    const timer = ctl.setTimer(() => finish({ type: "timeout" }), ms);
    teardown = () => {
      document.removeEventListener("keydown", onKey);
      clearTimeout(timer);
    };
    ctl.onCleanup(() => finish({ type: "abort" }));
  });
}

/** Enter-to-submit (or window expiry) for the memory recall phase. */
function submitOrTimeout(ctl, read, ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    console.warn("[cognitive-battery] invalid recall window", ms, "— falling back to 60s");
    ms = 60000;
  }
  return new Promise((resolve) => {
    let settled = false;
    let teardown = () => {};
    const finish = (result) => {
      if (settled) return;
      settled = true;
      teardown();
      resolve(result);
    };
    const onKey = (event) => {
      if (event.key !== "Enter" || isTypingTarget(event.target) === false) return;
      event.preventDefault();
      finish({ type: "submit", text: read(), ts: performance.now() });
    };
    document.addEventListener("keydown", onKey);
    const timer = ctl.setTimer(() => finish({ type: "timeout", text: read(), ts: performance.now() }), ms);
    teardown = () => {
      document.removeEventListener("keydown", onKey);
      clearTimeout(timer);
    };
    ctl.onCleanup(() => finish({ type: "abort", text: "" }));
  });
}

function normalizeKey(event) {
  const raw = event.key || "";
  if (raw === " " || event.code === "Space" || raw === "Spacebar") return " ";
  if (raw.length === 1) return raw.toLowerCase();
  return raw.length > 1 ? raw.toLowerCase() : "";
}

/* ------------------------------------------------------------ trial builders */

function buildSwitchingTrials() {
  const trials = [];
  let previousRule = pick(["number", "letter"]);
  for (let index = 0; index < TASK_META.task_switching.trials; index += 1) {
    let isSwitch;
    let rule;
    if (index === 0) {
      rule = previousRule;
      isSwitch = false;
    } else {
      isSwitch = Math.random() < 0.5;
      rule = isSwitch ? (previousRule === "number" ? "letter" : "number") : previousRule;
    }
    previousRule = rule;
    const letter = pick(Math.random() < 0.4 ? VOWELS.split("") : CONSONANTS.split(""));
    const digit = pick("123456789".split(""));
    const correctKey = rule === "number" ? (Number(digit) % 2 === 1 ? "f" : "j") : VOWELS.includes(letter) ? "f" : "j";
    trials.push({ rule, letter, digit, stimulus: `${letter} ${digit}`, correct_key: correctKey, is_switch: isSwitch });
  }
  return trials;
}

function buildStroopTrials() {
  const trials = [];
  for (let index = 0; index < TASK_META.stroop.trials; index += 1) {
    const congruent = Math.random() < 0.4;
    const [word, displayColor, correctKey] = pick(STROOP_COLORS);
    const shownWord = congruent ? word : pick(STROOP_COLORS.filter((entry) => entry[0] !== word))[0];
    trials.push({ word: shownWord, color_name: displayColor, correct_key: correctKey, congruent });
  }
  return trials;
}

/* ------------------------------------------------------------------ battery */

export function testView(ctx) {
  const { api, toast, esc, renderPayload } = ctx;
  const view = document.querySelector("#view");
  const S = { status: "intro", task: null, phase: null, trialIndex: 0, correctKey: null,
              expectedRecall: null, doneTrials: 0, sessionId: null, score: null, band: null };

  let ctl = controller();
  let tasks = {};
  let startedAt = null;
  let recallInput = null;
  let automation = null;

  /* ---- automation / debugging seam ---- */
  window.__clpTest = {
    state: S,
    start: (participantId) => startBattery(participantId),
    press: (key) => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key, code: key === " " ? "Space" : undefined, bubbles: true }));
    },
    /* Dispatch on the field itself so the target looks like real typing. */
    recall: (text) => {
      if (!recallInput) return;
      recallInput.value = text;
      recallInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    },
    setRecallInput: (input) => { recallInput = input; },
    abort: () => ctl.abort(),
  };

  const meta = () => document.querySelector("#t-trial");
  const stim = () => document.querySelector("#t-stim");
  const stimSub = () => document.querySelector("#t-feedback");
  const keys = () => document.querySelector("#t-keys");

  function setProgress() {
    const fill = document.querySelector(".test-progress-fill");
    const next = document.querySelector(".test-next");
    if (fill) fill.style.transform = `scaleX(${(S.doneTrials / TOTAL_TRIALS).toFixed(4)})`;
    if (next) next.textContent = `${S.doneTrials} / ${TOTAL_TRIALS} trials`;
  }

  function setMeta(text) {
    const el = meta();
    if (el) el.textContent = text;
  }

  function setStim(text, { tone = "", size = "lg" } = {}) {
    const el = stim();
    if (!el) return;
    el.className = `stim ${size}`;
    el.dataset.tone = tone;
    el.textContent = text;
    el.style.color = "";
  }

  function setInk(text, color) {
    const el = stim();
    if (!el) return;
    el.className = "stim xl";
    el.dataset.tone = "ink";
    el.textContent = text;
    el.style.color = color;
  }

  function setFeedback(text, tone = "") {
    const el = stimSub();
    if (!el) return;
    el.dataset.tone = tone;
    el.textContent = text;
  }

  function setKeyHints(hints) {
    const el = keys();
    if (!el) return;
    el.innerHTML = hints
      .map(([cap, label]) => `<span class="keycap"><kbd>${esc(cap)}</kbd>${label ? `<em>${esc(label)}</em>` : ""}</span>`)
      .join("");
  }

  /* ---------------------------------------------------------- task runners */

  async function runReaction() {
    const { response_window_ms: win, feedback_ms: feedback } = TASK_TIMING.reaction_time;
    tasks.reaction_time = { trials: [] };
    setKeyHints([["SPACE", "press when green"]]);
    for (let index = 0; index < TASK_META.reaction_time.trials; index += 1) {
      if (ctl.aborted) return;
      S.phase = "waiting";
      S.correctKey = " ";
      S.trialIndex = index + 1;
      setMeta(`Trial ${index + 1} of ${TASK_META.reaction_time.trials}`);
      setStim("Wait for green", { tone: "" });
      setFeedback("Stay ready — press only after the cue.");
      const [minDelay, maxDelay] = TASK_TIMING.reaction_time.pre_signal_delay_ms;
      const early = await keyOrTimeout(ctl, (key) => key === " ", randInt(minDelay, maxDelay));
      if (ctl.aborted) return;
      if (early.type === "key") {
        tasks.reaction_time.trials.push({ false_start: true, missed: false, response_ms: null,
          response_seconds: null, response_window_ms: win, correct: false });
        setStim("Too early", { tone: "bad" });
        setFeedback("False start recorded.", "bad");
        S.doneTrials += 1; setProgress();
        await ctl.sleep(feedback);
        continue;
      }
      S.phase = "go";
      setStim("PRESS SPACE NOW", { tone: "go" });
      setFeedback(`Respond within ${(win / 1000).toFixed(0)}s — faster scores better.`);
      const signalAt = performance.now();
      const result = await keyOrTimeout(ctl, (key) => key === " ", win);
      if (ctl.aborted) return;
      S.phase = "feedback";
      if (result.type === "key") {
        const rt = round2(result.ts - signalAt);
        tasks.reaction_time.trials.push({ false_start: false, missed: false, response_ms: rt,
          response_seconds: round2(rt / 1000), response_window_ms: win, correct: true });
        setStim("Captured", { tone: "good" });
        setFeedback(`Recorded in ${(rt / 1000).toFixed(2)}s.`, "good");
      } else {
        tasks.reaction_time.trials.push({ false_start: false, missed: true, response_ms: null,
          response_seconds: null, response_window_ms: win, correct: false });
        setStim("Too slow", { tone: "warn" });
        setFeedback(`No response within ${(win / 1000).toFixed(0)}s.`, "warn");
      }
      S.doneTrials += 1; setProgress();
      await ctl.sleep(feedback);
    }
  }

  async function runSwitching() {
    const { response_window_ms: win, feedback_ms: feedback } = TASK_TIMING.task_switching;
    const trials = buildSwitchingTrials();
    tasks.task_switching = { trials: [] };
    setKeyHints([["F", "odd / vowel"], ["J", "even / consonant"]]);
    for (let index = 0; index < trials.length; index += 1) {
      if (ctl.aborted) return;
      const trial = trials[index];
      S.task = "task_switching";
      S.phase = "live";
      S.trialIndex = index + 1;
      S.correctKey = trial.correct_key;
      setMeta(`Trial ${index + 1} of ${trials.length}`);
      const cue = document.querySelector("#t-cue");
      if (cue) {
        cue.textContent = `${trial.rule.toUpperCase()} RULE`;
        cue.dataset.rule = trial.rule;
      }
      setStim(trial.stimulus, { tone: "" });
      setFeedback("F or J");
      const t0 = performance.now();
      const result = await keyOrTimeout(ctl, (key) => key === "f" || key === "j", win);
      if (ctl.aborted) return;
      const record = { ...trial, pressed_key: result.type === "key" ? result.key : null,
        response_ms: result.type === "key" ? round2(result.ts - t0) : null,
        response_seconds: result.type === "key" ? round2((result.ts - t0) / 1000) : null,
        response_window_ms: win, missed: result.type !== "key" };
      record.correct = result.type === "key" && result.key === trial.correct_key;
      tasks.task_switching.trials.push(record);
      S.phase = "feedback";
      setFeedback(record.correct ? `Correct · ${(record.response_ms / 1000).toFixed(2)}s`
        : result.type === "key" ? `Incorrect · expected ${trial.correct_key.toUpperCase()}` : "No response.",
        record.correct ? "good" : result.type === "key" ? "bad" : "warn");
      S.doneTrials += 1; setProgress();
      await ctl.sleep(feedback);
    }
  }

  async function runMemory() {
    const timing = TASK_TIMING.working_memory; /* read directly: keys are validated above */
    tasks.working_memory = { trials: [] };
    const stage = document.querySelector(".test-stage");
    stage.classList.add("memory");
    setKeyHints([]);
    for (let index = 0; index < MEMORY_LENGTHS.length; index += 1) {
      if (ctl.aborted) return;
      const length = MEMORY_LENGTHS[index];
      const sequence = Array.from({ length }, () => pick("123456789".split(""))).join("");
      S.task = "working_memory";
      S.phase = "display";
      S.trialIndex = index + 1;
      S.expectedRecall = sequence;
      setMeta(`Trial ${index + 1} of ${MEMORY_LENGTHS.length}`);
      setStim(`Length ${length}`, { tone: "", size: "md" });
      setFeedback("Memorise the sequence.");
      await ctl.sleep(timing.pre_load_ms);
      for (const digit of sequence) {
        if (ctl.aborted) return;
        setStim(digit, { tone: "digit", size: "digit" });
        await ctl.sleep(timing.digit_display_ms);
        if (ctl.aborted) return;
        setStim("", { tone: "digit", size: "digit" });
        await ctl.sleep(timing.digit_gap_ms);
      }
      await ctl.sleep(timing.post_load_ms);
      if (ctl.aborted) return;
      S.phase = "recall";
      const input = document.querySelector("#t-recall");
      recallInput = input;
      input.value = "";
      input.disabled = false;
      input.classList.add("armed");
      input.focus();
      setStim("?", { tone: "", size: "md" });
      setFeedback(`Type the sequence and press Enter (up to ${(timing.recall_window_ms / 1000).toFixed(0)}s).`);
      const t0 = performance.now();
      const result = await submitOrTimeout(ctl, () => input.value.trim(), timing.recall_window_ms);
      if (ctl.aborted) return;
      input.disabled = true;
      input.classList.remove("armed");
      recallInput = null;
      const response = result.text || "";
      const record = generateMemoryRecord(sequence, response, round2(result.ts - t0), timing.recall_window_ms, result.type === "timeout");
      tasks.working_memory.trials.push(record);
      S.phase = "feedback";
      setStim(record.full_correct ? "Correct" : "Missed", { tone: record.full_correct ? "good" : "warn", size: "md" });
      setFeedback(record.full_correct ? `Recalled in ${(record.response_ms / 1000).toFixed(2)}s`
        : `Expected ${sequence} · got ${response || "—"}`, record.full_correct ? "good" : "warn");
      S.doneTrials += 1; setProgress();
      await ctl.sleep(timing.feedback_ms);
    }
    stage.classList.remove("memory");
  }

  async function runStroop() {
    const { response_window_ms: win, feedback_ms: feedback } = TASK_TIMING.stroop;
    const trials = buildStroopTrials();
    tasks.stroop = { trials: [] };
    setKeyHints([["R", "red"], ["G", "green"], ["B", "blue"], ["Y", "yellow"]]);
    for (let index = 0; index < trials.length; index += 1) {
      if (ctl.aborted) return;
      const trial = trials[index];
      S.task = "stroop";
      S.phase = "live";
      S.trialIndex = index + 1;
      S.correctKey = trial.correct_key;
      setMeta(`Trial ${index + 1} of ${trials.length}`);
      setInk(trial.word, STROOP_CSS[trial.color_name]);
      setFeedback("R / G / B / Y — the ink colour, not the word");
      const t0 = performance.now();
      const result = await keyOrTimeout(ctl, (key) => "rgby".includes(key), win);
      if (ctl.aborted) return;
      const record = { ...trial, pressed_key: result.type === "key" ? result.key : null,
        response_ms: result.type === "key" ? round2(result.ts - t0) : null,
        response_seconds: result.type === "key" ? round2((result.ts - t0) / 1000) : null,
        response_window_ms: win, missed: result.type !== "key" };
      record.correct = result.type === "key" && result.key === trial.correct_key;
      tasks.stroop.trials.push(record);
      S.phase = "feedback";
      setFeedback(record.correct ? `Correct · ${(record.response_ms / 1000).toFixed(2)}s`
        : result.type === "key" ? `Incorrect · ink was ${trial.color_name}` : "No response.",
        record.correct ? "good" : result.type === "key" ? "bad" : "warn");
      S.doneTrials += 1; setProgress();
      await ctl.sleep(feedback);
    }
  }

  const RUNNERS = { reaction_time: runReaction, task_switching: runSwitching,
                    working_memory: runMemory, stroop: runStroop };

  /* ------------------------------------------------------------- intro card */

  function introView() {
    S.status = "intro";
    document.body.classList.remove("testing");
    const remembered = localStorage.getItem("clp.participant") || "ME";
    view.innerHTML = `
      <div class="page-head">
        <h1>Take the <span class="serif-accent">cognitive test.</span></h1>
        <p class="lede">Four short behavioral tasks, about 8 minutes end to end. When you finish, your session is
          scored by the full V2 pipeline — band, confidence, exact score breakdown and suggestions — and saved to your history.</p>
      </div>
      <div class="bento stagger">
        <div class="card span-7">
          <div class="head"><h3>The battery</h3><span class="kpi-sub">${TOTAL_TRIALS} trials · keyboard required</span></div>
          <div class="test-list">
            ${TASK_ORDER.map((name, index) => `
              <div class="test-item">
                <span class="test-num">${index + 1}</span>
                <div>
                  <div class="test-name">${esc(TASK_META[name].label)}
                    <span class="mono">${TASK_META[name].trials} trials · ${esc(TASK_META[name].keys)}</span></div>
                  <p class="kpi-sub">${esc(TASK_META[name].blurb)}</p>
                </div>
              </div>`).join("")}
          </div>
          <p class="kpi-sub" style="margin-top:14px">Keep this tab focused while the test runs — background tabs
            throttle timers, which would distort your response times.</p>
        </div>
        <div class="card span-5">
          <div class="head"><h3>Before you start</h3></div>
          <div class="field" style="margin-bottom:var(--s-4)">
            <label for="test-pid">Participant id</label>
            <input type="text" id="test-pid" value="${esc(remembered)}" spellcheck="false" maxlength="40" autocomplete="off">
          </div>
          <div class="field" style="margin-bottom:var(--s-4)">
            <label for="test-self-report">How loaded do you feel right now?</label>
            <select id="test-self-report">
              <option value="">prefer not to say</option>
              <option value="low">low</option>
              <option value="manageable">manageable</option>
              <option value="elevated">elevated</option>
              <option value="high">high</option>
            </select>
          </div>
          <label class="check"><input type="checkbox" id="test-baseline"> Mark as a baseline calibration session</label>
          <details class="test-details">
            <summary>Context questionnaire <span class="mono">optional</span></summary>
            <div class="field"><label for="q-sleep">Sleep hours last night</label><input type="number" id="q-sleep" step="0.5" min="0" max="24"></div>
            <div class="field"><label for="q-stress">Stress level 1-5</label><input type="number" id="q-stress" min="1" max="5"></div>
            <div class="field"><label for="q-caffeine">Caffeine today (mg)</label><input type="number" id="q-caffeine" min="0" max="1000"></div>
            <div class="field"><label for="q-fatigue">Fatigue level 1-5</label><input type="number" id="q-fatigue" min="1" max="5"></div>
            <div class="field"><label for="q-notes">Notes</label><input type="text" id="q-notes" maxlength="120" placeholder="e.g. after a long meeting"></div>
          </details>
          <div class="actions" style="margin-top:var(--s-6); display:flex; gap:var(--s-3)">
            <button class="btn primary" id="test-start">Start the battery</button>
            <a class="btn ghost" href="#/">Not now</a>
          </div>
        </div>
      </div>`;

    document.querySelector("#test-start").addEventListener("click", () => {
      startBattery(document.querySelector("#test-pid").value.trim() || "ME");
    });
  }

  /* ---------------------------------------------------------- task shell */

  function stageView(taskName) {
    S.status = "running";
    S.task = taskName;
    document.body.classList.add("testing");
    view.innerHTML = `
      <div class="test-top">
        <div class="test-progress"><div class="test-progress-fill" style="transform:scaleX(${(S.doneTrials / TOTAL_TRIALS).toFixed(4)})"></div></div>
        <div class="test-meta">
          <span class="test-task">${esc(TASK_META[taskName].label)}</span>
          <span id="t-trial"></span>
          <span class="test-next"></span>
          <button class="btn ghost sm" id="test-quit">Abort</button>
        </div>
      </div>
      <div class="card test-stage">
        <div class="cue" id="t-cue"></div>
        <div class="stim" id="t-stim">Ready</div>
        <p class="stim-sub" id="t-feedback"></p>
        ${taskName === "working_memory"
          ? `<input type="text" id="t-recall" class="recall-input" autocomplete="off" spellcheck="false" inputmode="numeric" placeholder="type digits, press Enter">`
          : ""}
      </div>
      <div class="keyrow" id="t-keys"></div>`;
    document.querySelector("#test-quit").addEventListener("click", () => quit("You aborted the test — nothing was saved."));
    if (taskName === "working_memory") window.__clpTest.setRecallInput(document.querySelector("#t-recall"));
  }

  function abortMessage(message) {
    document.body.classList.remove("testing");
    ctl.abort();
    S.status = "intro";
    view.innerHTML = `<div class="card"><div class="empty"><div class="glyph">◌</div>
      <p><strong>Test cancelled</strong></p><p class="hint">${esc(message)}</p>
      <div class="actions"><a class="btn" href="#/test">Back to the test</a><a class="btn ghost" href="#/">Overview</a></div>
      </div></div>`;
  }

  async function quit(message) {
    abortMessage(message);
  }

  /* --------------------------------------------------------- orchestration */

  async function startBattery(participantId) {
    if (S.status === "running") return;
    ctl.abort();
    ctl = controller();
    tasks = {};
    S.doneTrials = 0;
    S.score = null;
    S.band = null;
    const started = new Date();
    startedAt = started;
    const questionnaire = collectQuestionnaire();
    const selfReport = document.querySelector("#test-self-report")?.value || "";
    const baseline = !!document.querySelector("#test-baseline")?.checked;
    localStorage.setItem("clp.participant", participantId);

    for (const taskName of TASK_ORDER) {
      if (ctl.aborted) return;
      stageView(taskName);
      setStim("Get ready", { tone: "" });
      setFeedback(TASK_META[taskName].blurb);
      await ctl.sleep(TASK_INTRO_MS);
      await RUNNERS[taskName]();
      if (ctl.aborted) return;
      await ctl.sleep(500);
    }
    if (ctl.aborted) return;
    await save(participantId, questionnaire, selfReport, baseline, started);
  }

  function collectQuestionnaire() {
    const value = (id) => {
      const raw = document.querySelector(id)?.value;
      return raw === "" || raw === undefined ? null : Number(raw);
    };
    const notes = document.querySelector("#q-notes")?.value?.trim() || "";
    return {
      sleep_hours: value("#q-sleep"),
      stress_level: value("#q-stress"),
      caffeine_mg: value("#q-caffeine"),
      fatigue_level: value("#q-fatigue"),
      notes,
    };
  }

  async function save(participantId, questionnaire, selfReport, baseline, started) {
    S.status = "saving";
    document.body.classList.add("testing");
    const duration = Math.max(1, Math.round((Date.now() - started.getTime()) / 1000));
    const payload = {
      participant_id: participantId,
      duration_seconds: duration,
      baseline_session: baseline,
      questionnaire,
      tasks,
    };
    if (selfReport) payload.self_report_load_band = selfReport;

    view.innerHTML = `
      <div class="page-head">
        <h1>Scoring your <span class="serif-accent">session…</span></h1>
        <p class="lede">${TOTAL_TRIALS} trials collected across four tasks in ${Math.floor(duration / 60)}m ${duration % 60}s.
          Running the V2 pipeline: prediction, uncertainty, explanation, quality, forecast and suggestions.</p>
      </div>
      <div class="bento"><div class="sk-block skeleton span-7"></div><div class="sk-block skeleton span-5"></div></div>`;

    let result;
    try {
      result = await api("POST", "/assessments", payload);
    } catch (error) {
      S.status = "error";
      document.body.classList.remove("testing");
      toast(error.message);
      view.innerHTML = `<div class="card"><div class="empty"><div class="glyph">◌</div>
        <p><strong>Could not score the session</strong></p><p class="hint">${esc(error.message)}</p>
        <div class="actions"><a class="btn primary" href="#/test">Take the test again</a><a class="btn ghost" href="#/">Overview</a></div>
        </div></div>`;
      return;
    }

    S.status = "done";
    S.sessionId = result.session_id;
    S.score = result.overall_score;
    S.band = result.band;
    document.body.classList.remove("testing");
    localStorage.setItem("clp.participant", participantId);

    view.innerHTML = `
      <div class="page-head">
        <h1>Test complete — <span class="serif-accent">here is your read.</span></h1>
        <p class="lede">Saved as <span class="mono">${esc(result.session_id)}</span>
          ${result.history_size
            ? `· compared against ${result.history_size} of your earlier session(s).`
            : `· this is your first session, so trends and forecasts unlock as you take more.`}</p>
      </div>
      <div id="test-result"></div>
      <div class="actions" style="margin-top:var(--s-6); display:flex; gap:var(--s-3)">
        <a class="btn primary" href="#/?p=${encodeURIComponent(participantId)}">See it on the dashboard</a>
        <a class="btn" href="#/session/${encodeURIComponent(result.session_id)}">Open the full report</a>
        <a class="btn ghost" href="#/test">Take it again</a>
      </div>`;

    renderPayload(document.querySelector("#test-result"), result.payload);
    toast(`Session ${result.session_id} scored ${Number(result.overall_score).toFixed(1)} (${result.band}).`, "ok");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function teardown() {
    ctl.abort();
    document.body.classList.remove("testing");
    if (window.__clpTest) delete window.__clpTest;
  }

  introView();
  return teardown;
}

/** Trial record for the memory task (kept separate: pure, so it is easy to test). */
export function generateMemoryRecord(sequence, response, responseMs, windowMs, timedOut) {
  const matches = [...sequence].filter((digit, position) => response[position] === digit).length;
  return {
    sequence_length: sequence.length,
    target_sequence: sequence,
    response,
    position_accuracy: Math.round((matches / sequence.length) * 10000) / 10000,
    full_correct: response === sequence,
    response_ms: responseMs,        response_seconds: Math.round(responseMs / 10) / 100,
    response_window_ms: windowMs,
    correct: response === sequence,
    missed: timedOut && !response,
  };
}

export { TASK_ORDER, TASK_META, TOTAL_TRIALS, buildStroopTrials, buildSwitchingTrials };
