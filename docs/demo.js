/* Static GitHub Pages demo — artifact replay only (no live Ollama). */
const $ = (id) => document.getElementById(id);

let manifest = null;
let rows = [];
let rowsById = {};
let selected = null;
let panelMeta = null;

const COND_KEY = {
  A: "A_direct_verdict",
  B: "B_analysis_then_verdict",
  C: "C_structured_pipeline",
  D: "D_analysis_then_structure",
};

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return res.json();
}

function basename(path) {
  if (!path) return "";
  const s = String(path).replace(/\\/g, "/");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

function verdictClass(ok) {
  return ok ? "v-ok" : "v-bad";
}

function stageClass(stage) {
  return stage && stage !== "none" ? "stage stage-loss" : "stage";
}

function renderSummary(summary, model) {
  if (!summary) {
    $("summary-bar").innerHTML = "";
    return;
  }
  const m = summary.match_gold || {};
  const n = summary.n || 0;
  const rep =
    summary.error_stage_counts_when_verdict_misses_gold?.D?.representation_loss ?? "—";
  $("summary-bar").innerHTML = `
    <span>Model: <b>${esc(model || "—")}</b></span>
    <span>A <b>${m.A ?? "?"}/${n}</b></span>
    <span>B <b>${m.B ?? "?"}/${n}</b></span>
    <span>C <b>${m.C ?? "?"}/${n}</b></span>
    <span>D <b>${m.D ?? "?"}/${n}</b></span>
    <span>D rep-loss misses: <b>${rep}</b></span>
  `;
}

function renderSidebar() {
  $("sidebar").innerHTML = rows
    .map((r) => {
      const abcd = ["A", "B", "C", "D"]
        .map((c) => {
          const ok = r.match_gold?.[c];
          const v = r.conditions?.[COND_KEY[c]]?.verdict || "?";
          return `${c}:${String(v).slice(0, 4)}${ok ? "✓" : "✗"}`;
        })
        .join(" ");
      return `<button class="case ${r.id === selected ? "active" : ""}" data-id="${esc(r.id)}" type="button">
      <b>${esc(r.id)}</b>
      <span><span class="mini">${esc(abcd)}</span><br><span class="mini">${esc(r.gold)}</span></span>
    </button>`;
    })
    .join("");
  $("sidebar").querySelectorAll("button.case").forEach((btn) => {
    btn.onclick = () => {
      selected = btn.dataset.id;
      renderDetail();
      renderSidebar();
    };
  });
}

function condBlock(title, data, stage, match) {
  if (!data) return `<div class="card"><h2>${title}</h2><p class="status">—</p></div>`;
  let inner = `<p class="${verdictClass(match)} v">verdict: ${esc(data.verdict)} ${
    match ? "✓ gold" : "✗ gold"
  }</p>`;
  inner += `<p class="${stageClass(stage)}">loss: ${esc(stage)}</p>`;
  if (data.analysis) inner += `<pre>${esc(data.analysis)}</pre>`;
  if (data.rationale) inner += `<p>${esc(data.rationale)}</p>`;
  if (data.analysis_flags)
    inner += `<pre>flags: ${esc(JSON.stringify(data.analysis_flags))}</pre>`;
  if (data.extraction)
    inner += `<pre>extract: ${esc(JSON.stringify(data.extraction, null, 2))}</pre>`;
  if (data.grounding)
    inner += `<pre>ground: ${esc(JSON.stringify(data.grounding, null, 2))}</pre>`;
  return `<div class="card"><h2>${title}</h2>${inner}</div>`;
}

function renderDetail() {
  const r = rowsById[selected];
  if (!r) {
    $("main").innerHTML = `<div class="card"><p>Select a case.</p></div>`;
    return;
  }
  const c = r.conditions || {};
  $("main").innerHTML = `
    <div class="card">
      <h2>${esc(r.id)} · gold ${esc(r.gold)} · ${esc(r.expected_distinction)}</h2>
      <p>${esc(r.evidence)}</p>
      <p class="status">${esc(r.why || "")}</p>
    </div>
    <table class="grid">
      <tr><th></th><th>Verdict</th><th>Match</th><th>Loss stage</th></tr>
      ${["A", "B", "C", "D"]
        .map((k) => {
          const v = c[COND_KEY[k]]?.verdict || "—";
          return `<tr><td>${k}</td><td class="v">${esc(v)}</td><td class="${verdictClass(
            r.match_gold?.[k]
          )}">${r.match_gold?.[k] ? "yes" : "no"}</td><td class="${stageClass(
            r.loss_stage?.[k]
          )}">${esc(r.loss_stage?.[k])}</td></tr>`;
        })
        .join("")}
    </table>
    <div class="cond-grid" style="margin-top:12px">
      ${condBlock("A — direct verdict", c.A_direct_verdict, r.loss_stage?.A, r.match_gold?.A)}
      ${condBlock("B — analysis → verdict", c.B_analysis_then_verdict, r.loss_stage?.B, r.match_gold?.B)}
      ${condBlock("C — structured pipeline", c.C_structured_pipeline, r.loss_stage?.C, r.match_gold?.C)}
      ${condBlock("D — analysis → structure", c.D_analysis_then_structure, r.loss_stage?.D, r.match_gold?.D)}
    </div>
  `;
}

function ingestRows(reportRows, summary, model) {
  rows = reportRows || [];
  rowsById = {};
  rows.forEach((r) => {
    rowsById[r.id] = r;
  });
  if (!selected || !rowsById[selected]) selected = rows[0]?.id || null;
  renderSummary(summary, model);
  renderSidebar();
  renderDetail();
}

async function loadReportFile(name) {
  const data = await getJson(`artifacts/${name}`);
  return {
    model: data.model,
    rows: data.rows || [],
    summary: data.summary || {},
    phase4_regex_screen: data.phase4_regex_screen,
  };
}

async function loadArtifactRows(panelOrReportName, model) {
  const data = await getJson(`artifacts/${panelOrReportName}`);

  if (data.rows) {
    ingestRows(data.rows, data.summary || {}, data.model || model);
    return { model: data.model || model };
  }

  if (!data.models) throw new Error("artifact has no rows or models");

  panelMeta = data;
  const box = $("panel-models");
  box.hidden = false;
  const n = (data.case_ids || []).length || "?";
  box.innerHTML = data.models
    .map((m) => {
      const matchA = m.match_gold?.A ?? m.summary?.match_gold?.A ?? "?";
      const active = m.model === model ? "active" : "";
      return `<button type="button" data-model="${esc(m.model)}" class="${active}">${esc(
        m.model
      )} A${matchA}/${n}</button>`;
    })
    .join("");

  box.querySelectorAll("button").forEach((btn) => {
    btn.onclick = async () => {
      box.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("model").value = btn.dataset.model;
      await selectPanelModel(panelOrReportName, btn.dataset.model);
      $("status").textContent = `loaded ${panelOrReportName} · ${btn.dataset.model}`;
    };
  });

  const pick =
    model && data.models.some((m) => m.model === model)
      ? model
      : data.models[0]?.model;
  if (pick) {
    $("model").value = pick;
    await selectPanelModel(panelOrReportName, pick);
  }
  return { model: pick };
}

async function selectPanelModel(panelName, model) {
  const data = panelMeta || (await getJson(`artifacts/${panelName}`));
  const entry = (data.models || []).find((m) => m.model === model);
  if (!entry) throw new Error(`model not in panel: ${model}`);
  const perName = basename(entry.artifact);
  if (!perName) throw new Error(`no per-model artifact for ${model}`);
  const per = await loadReportFile(perName);
  ingestRows(per.rows, per.summary || entry.summary || {}, per.model || model);
  if (per.phase4_regex_screen || entry.phase4_regex_screen) {
    const screen = per.phase4_regex_screen || entry.phase4_regex_screen;
    const rep = screen?.D?.representation_loss;
    const n = screen?.D?.n_screen;
    if (rep != null) $("status").textContent += ` · regex screen D-rep ${rep}/${n}`;
  }
}

function fillModelSelectFromPanel(data) {
  const models = (data.models || []).map((m) => m.model).filter(Boolean);
  if (!models.length && data.model) models.push(data.model);
  $("model").innerHTML = models
    .map((m) => `<option value="${esc(m)}">${esc(m)}</option>`)
    .join("");
}

async function onLoad() {
  const name = $("artifact").value;
  $("status").textContent = "loading…";
  $("panel-models").hidden = true;
  panelMeta = null;
  try {
    const peek = await getJson(`artifacts/${name}`);
    fillModelSelectFromPanel(peek);
    if (peek.models) {
      await loadArtifactRows(name, $("model").value);
    } else {
      await loadArtifactRows(name, peek.model);
    }
    $("status").textContent = `loaded ${name} · replay only (no inference)`;
  } catch (err) {
    $("status").innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
}

async function init() {
  try {
    manifest = await getJson("manifest.json");
  } catch (err) {
    $("status").innerHTML = `<span class="err">Missing manifest.json — run scripts/prepare-github-pages.py</span>`;
    return;
  }

  $("artifact").innerHTML = (manifest.artifacts || [])
    .map(
      (a) =>
        `<option value="${esc(a.name)}">${esc(a.label || a.name)}</option>`
    )
    .join("");

  if (manifest.default_artifact) {
    $("artifact").value = manifest.default_artifact;
  }

  $("load-btn").onclick = onLoad;
  $("artifact").onchange = async () => {
    // Prefill models from panel header without waiting for Load
    try {
      const peek = await getJson(`artifacts/${$("artifact").value}`);
      fillModelSelectFromPanel(peek);
    } catch (_) {
      /* ignore until Load */
    }
  };

  $("status").textContent = manifest.note || "Replay only";
  await onLoad();
}

init();
