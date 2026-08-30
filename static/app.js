// ============================================================
// NumPy Transformer — Live Internals Inspector
// All data rendered here comes straight from the backend's real
// forward/backward passes. Nothing on this page is simulated.
// ============================================================

const API_BASE = "https://transformer-from-scratch-numpy.onrender.com";

function $(id) { return document.getElementById(id); }

function el(tag, className, children) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (children) {
    (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
  }
  return e;
}

async function postJSON(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

async function getJSON(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

// ---------------- color scale: gunmetal -> steel -> champagne ----------------

function lerp(a, b, t) { return a + (b - a) * t; }

function colorStops(t) {
  // t in [0,1]. Three stop gradient: dark gunmetal -> mid steel -> champagne gold.
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [26, 29, 34],   // gunmetal
    [124, 135, 148], // steel
    [212, 175, 106], // champagne
  ];
  const seg = t < 0.5 ? 0 : 1;
  const localT = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const c0 = stops[seg], c1 = stops[seg + 1];
  return [
    Math.round(lerp(c0[0], c1[0], localT)),
    Math.round(lerp(c0[1], c1[1], localT)),
    Math.round(lerp(c0[2], c1[2], localT)),
  ];
}

function valueToColor(v, vmin, vmax) {
  const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
  const [r, g, b] = colorStops(t);
  return `rgb(${r},${g},${b})`;
}

function textColorFor(t) {
  return t > 0.55 ? "rgba(20,22,26,0.75)" : "rgba(240,238,230,0.85)";
}

// ============================================================
// Shared renderers
// ============================================================

function sectionShell(eyebrow, title, note) {
  const wrap = el("div", "panel");
  const head = el("div", "section-title", [
    el("span", "eyebrow", eyebrow),
    el("h3", null, title),
  ]);
  wrap.appendChild(head);
  if (note) wrap.appendChild(el("div", "section-note", note));
  return wrap;
}

function renderTokenChips(tokens, ids, opts) {
  opts = opts || {};
  const row = el("div", "token-row");
  tokens.forEach((tok, i) => {
    const classes = ["token-chip"];
    if (opts.highlightIdx && opts.highlightIdx.includes(i)) classes.push("chosen");
    if (opts.targetIdx && opts.targetIdx.includes(i)) classes.push("target");
    if (opts.mismatchIdx && opts.mismatchIdx.includes(i)) classes.push("mismatch");
    const idVal = ids ? ids[i] : null;
    row.appendChild(el("span", classes.join(" "), [
      document.createTextNode(tok),
      idVal != null ? el("span", "id", "#" + idVal) : null,
    ]));
  });
  return row;
}

function renderEmbeddingHeat(tokens, vectors, maxDims) {
  maxDims = maxDims || 32;
  const wrap = el("div", null);
  let vmin = Infinity, vmax = -Infinity;
  vectors.forEach(v => v.slice(0, maxDims).forEach(x => { vmin = Math.min(vmin, x); vmax = Math.max(vmax, x); }));
  vectors.forEach((vec, i) => {
    const row = el("div", "heat-row");
    row.appendChild(el("span", "rlabel", tokens[i]));
    const cells = el("div", "heat-cells");
    vec.slice(0, maxDims).forEach(v => {
      const cell = el("div", "heat-cell");
      cell.style.background = valueToColor(v, vmin, vmax);
      cell.title = v.toFixed(4);
      cells.appendChild(cell);
    });
    row.appendChild(cells);
    wrap.appendChild(row);
  });
  return wrap;
}

function renderMatrixHeatmap(matrix, rowLabels, colLabels) {
  const rows = matrix.length, cols = matrix[0] ? matrix[0].length : 0;
  let vmin = Infinity, vmax = -Infinity;
  matrix.forEach(r => r.forEach(v => { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); }));

  const wrap = el("div", "matrix-wrap");
  const grid = el("div", "matrix-grid");
  grid.style.gridTemplateColumns = `70px repeat(${cols}, 34px)`;

  grid.appendChild(el("div", "matrix-axis-label", ""));
  colLabels.forEach(c => grid.appendChild(el("div", "matrix-axis-label", c)));

  for (let i = 0; i < rows; i++) {
    grid.appendChild(el("div", "matrix-axis-label", rowLabels[i]));
    for (let j = 0; j < cols; j++) {
      const v = matrix[i][j];
      const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
      const cell = el("div", "matrix-cell", v.toFixed(2));
      cell.style.background = valueToColor(v, vmin, vmax);
      cell.style.color = textColorFor(t);
      cell.title = `${rowLabels[i]} → ${colLabels[j]}: ${v.toFixed(4)}`;
      grid.appendChild(cell);
    }
  }
  wrap.appendChild(grid);
  return wrap;
}

function renderAttentionBlock(attnData, titlePrefix) {
  const container = el("div", null);
  const headCount = attnData.num_heads;
  const chipRow = el("div", "chip-toggle");
  const matrixHost = el("div", null);

  const options = ["mean"];
  for (let h = 0; h < headCount; h++) options.push("head " + h);

  function draw(sel) {
    matrixHost.innerHTML = "";
    const matrix = sel === "mean" ? attnData.mean_over_heads : attnData.per_head[parseInt(sel.split(" ")[1], 10)];
    matrixHost.appendChild(renderMatrixHeatmap(matrix, attnData.query_tokens, attnData.key_tokens));
  }

  options.forEach((opt, i) => {
    const btn = el("button", i === 0 ? "active" : "", opt);
    btn.addEventListener("click", () => {
      chipRow.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      draw(opt);
    });
    chipRow.appendChild(btn);
  });

  container.appendChild(el("div", "section-note", `${titlePrefix} · ${headCount} heads · rows = query tokens, columns = key tokens, cell = softmax(QKᵀ/√d)`));
  container.appendChild(chipRow);
  container.appendChild(matrixHost);
  draw("mean");
  return container;
}

function renderFFNBlock(ffnData) {
  const wrap = el("div", null);
  wrap.appendChild(el("div", "section-note",
    `hidden_dim=${ffnData.hidden_dim} · ${(ffnData.fraction_neurons_active * 100).toFixed(1)}% of hidden units fired (ReLU > 0) · first 16 dims shown`));
  const twoCol = el("div", "two-col");

  const preWrap = el("div", null, [el("div", "section-note", "pre-activation (Linear₁ output)")]);
  preWrap.appendChild(renderEmbeddingHeat(
    ffnData.pre_activation_sample.map((_, i) => "pos " + i),
    ffnData.pre_activation_sample,
    16
  ));
  const postWrap = el("div", null, [el("div", "section-note", "post-ReLU (before Linear₂)")]);
  postWrap.appendChild(renderEmbeddingHeat(
    ffnData.post_relu_sample.map((_, i) => "pos " + i),
    ffnData.post_relu_sample,
    16
  ));

  twoCol.appendChild(preWrap);
  twoCol.appendChild(postWrap);
  wrap.appendChild(twoCol);
  return wrap;
}

function renderBarList(items, opts) {
  opts = opts || {};
  const wrap = el("div", "bar-list");
  const maxVal = Math.max(...items.map(i => i.value), 1e-9);
  items.forEach(item => {
    const row = el("div", "bar-item");
    row.appendChild(el("span", "lbl", item.label));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill" + (item.chosen ? " chosen-fill" : ""));
    fill.style.width = (100 * item.value / maxVal).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "valnum", opts.formatValue ? opts.formatValue(item.value) : item.value.toFixed(4)));
    wrap.appendChild(row);
  });
  return wrap;
}

function renderLayerToggle(count, onSelect) {
  const chipRow = el("div", "chip-toggle");
  for (let i = 0; i < count; i++) {
    const btn = el("button", i === 0 ? "active" : "", "layer " + i);
    btn.addEventListener("click", () => {
      chipRow.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      onSelect(i);
    });
    chipRow.appendChild(btn);
  }
  return chipRow;
}

function renderSparkline(values, width, height) {
  width = width || 260; height = height || 46;
  if (!values.length) return el("div", "section-note", "no steps yet this session");
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const vmin = Math.min(...values), vmax = Math.max(...values);
  const pad = 4;
  const pts = values.map((v, i) => {
    const x = values.length > 1 ? (i / (values.length - 1)) * (width - 2 * pad) + pad : width / 2;
    const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
    const y = height - pad - t * (height - 2 * pad);
    return [x, y];
  });

  const path = document.createElementNS(svgNS, "polyline");
  path.setAttribute("points", pts.map(p => p.join(",")).join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#cda86a");
  path.setAttribute("stroke-width", "2");
  svg.appendChild(path);

  const last = pts[pts.length - 1];
  const dot = document.createElementNS(svgNS, "circle");
  dot.setAttribute("cx", last[0]); dot.setAttribute("cy", last[1]); dot.setAttribute("r", "3");
  dot.setAttribute("fill", "#ddc08b");
  svg.appendChild(dot);

  const box = el("div", "sparkline-wrap");
  box.appendChild(svg);
  return box;
}

// ---------------- shared "full trace" renderer (embedding/pos/encoder/decoder) ----------------

function renderFullTrace(container, trace) {
  // Embedding
  const embSection = sectionShell("01 · tokenization → embedding", "Token embeddings",
    `${trace.input_tokens.length} tokens · embedding_dim=${trace.embedding_dims_total} · row = one token's real learned embedding vector`);
  embSection.appendChild(renderTokenChips(trace.input_tokens, trace.input_ids));
  embSection.appendChild(document.createElement("hr"));
  embSection.appendChild(renderEmbeddingHeat(trace.input_tokens, trace.embedding_sample));
  container.appendChild(embSection);

  // Positional encoding
  const posSection = sectionShell("02 · positional encoding", "Sinusoidal position vectors",
    "added element-wise to the token embedding before entering the encoder (first 16 dims shown)");
  posSection.appendChild(renderEmbeddingHeat(trace.input_tokens.map((_, i) => "pos " + i), trace.positional_sample));
  container.appendChild(posSection);

  // Encoder
  const encSection = sectionShell("03 · encoder", "Encoder self-attention & feed-forward",
    `${trace.num_layers} layer(s) · ${trace.num_heads} heads/layer — pick a layer below`);
  const encBody = el("div", null);
  function drawEncLayer(idx) {
    encBody.innerHTML = "";
    const layer = trace.encoder_layers[idx];
    encBody.appendChild(el("h4", null, "Self-attention (encoder input attends to itself)"));
    encBody.appendChild(renderAttentionBlock(layer.self_attention, "Encoder self-attention"));
    encBody.appendChild(el("h4", null, "Feed-forward network"));
    encBody.appendChild(renderFFNBlock(layer.feed_forward));
  }
  encSection.appendChild(renderLayerToggle(trace.encoder_layers.length, drawEncLayer));
  encSection.appendChild(encBody);
  drawEncLayer(0);
  container.appendChild(encSection);

  // Decoder
  const decSection = sectionShell("04 · decoder", "Decoder self-, cross-attention & feed-forward",
    `${trace.num_layers} layer(s) — masked self-attention (causal) then cross-attention over the encoder output`);
  const decBody = el("div", null);
  function drawDecLayer(idx) {
    decBody.innerHTML = "";
    const layer = trace.decoder_layers[idx];
    decBody.appendChild(el("h4", null, "Masked self-attention (causal — can't see future tokens)"));
    decBody.appendChild(renderAttentionBlock(layer.self_attention, "Decoder self-attention"));
    decBody.appendChild(el("h4", null, "Cross-attention (decoder queries, encoder keys/values)"));
    decBody.appendChild(renderAttentionBlock(layer.cross_attention, "Decoder cross-attention"));
    decBody.appendChild(el("h4", null, "Feed-forward network"));
    decBody.appendChild(renderFFNBlock(layer.feed_forward));
  }
  decSection.appendChild(renderLayerToggle(trace.decoder_layers.length, drawDecLayer));
  decSection.appendChild(decBody);
  drawDecLayer(0);
  container.appendChild(decSection);
}

// ============================================================
// INFERENCE TAB
// ============================================================

function initInferenceTab() {
  const textEl = $("inf-text"), maxlenEl = $("inf-maxlen"), tempEl = $("inf-temp"), topkEl = $("inf-topk");
  const maxlenVal = $("inf-maxlen-val"), tempVal = $("inf-temp-val"), topkVal = $("inf-topk-val");
  const runBtn = $("inf-run"), statusEl = $("inf-status"), errorEl = $("inf-error"), results = $("inf-results");

  maxlenEl.addEventListener("input", () => maxlenVal.textContent = maxlenEl.value);
  tempEl.addEventListener("input", () => tempVal.textContent = parseFloat(tempEl.value).toFixed(2));
  topkEl.addEventListener("input", () => topkVal.textContent = topkEl.value);

  runBtn.addEventListener("click", async () => {
    const text = textEl.value.trim();
    errorEl.classList.remove("show");
    if (!text) { errorEl.textContent = "Enter a sentence first."; errorEl.classList.add("show"); return; }

    runBtn.disabled = true;
    statusEl.textContent = "running real forward pass + autoregressive decoding…";
    try {
      const data = await postJSON("/generate/trace", {
        text,
        max_length: parseInt(maxlenEl.value, 10),
        temperature: parseFloat(tempEl.value),
        top_k: parseInt(topkEl.value, 10),
      });
      renderInferenceResults(results, data);
      statusEl.textContent = `done · ${data.trace.input_tokens.length} input tokens · ${data.generation_steps.length} decoding steps`;
    } catch (err) {
      errorEl.textContent = "Error: " + err.message;
      errorEl.classList.add("show");
      statusEl.textContent = "";
    } finally {
      runBtn.disabled = false;
    }
  });
}

function renderInferenceResults(container, data) {
  container.innerHTML = "";

  renderFullTrace(container, data.trace);

  // Generation steps — the real top-k / temperature softmax at every position
  const genSection = sectionShell("05 · autoregressive generation", "Top-k / temperature sampling, step by step",
    "at each step the raw logits are divided by temperature, restricted to the top-k candidates, then softmax'd — this is exactly that real distribution");
  data.generation_steps.forEach(step => {
    const card = el("div", "step-card");
    card.appendChild(el("div", "step-head", [
      el("span", null, `step ${step.step} · so far: “${step.decoder_tokens_so_far.join(" ")}”`),
      el("b", null, "→ " + step.chosen_token),
    ]));
    const items = step.candidates.map(c => ({
      label: c.token,
      value: c.probability,
      chosen: c.token_id === step.chosen_token_id,
    }));
    card.appendChild(renderBarList(items, { formatValue: v => (v * 100).toFixed(1) + "%" }));
    genSection.appendChild(card);
  });
  container.appendChild(genSection);

  // Final output
  const outSection = sectionShell("06 · result", "Final generated text", null);
  const outBox = el("div", "stat-row");
  outBox.appendChild(el("div", "stat-card accent", [
    el("div", "stat-label", "model output"),
    el("div", "stat-value", data.output || "(empty)"),
  ]));
  outSection.appendChild(outBox);
  if (data.unknown_words && data.unknown_words.length) {
    outSection.appendChild(el("div", "section-note", "unknown words (mapped to <UNK>): " + data.unknown_words.join(", ")));
  }
  container.appendChild(outSection);
}

// ============================================================
// TRAINING TAB
// ============================================================

function initTrainingTab() {
  const encEl = $("tr-enc"), decEl = $("tr-dec"), lrEl = $("tr-lr"), lrVal = $("tr-lr-val");
  const applyEl = $("tr-apply"), runBtn = $("tr-run"), resetBtn = $("tr-reset"), sampleBtn = $("tr-sample");
  const statusEl = $("tr-status"), errorEl = $("tr-error"), results = $("tr-results");

  lrEl.addEventListener("input", () => lrVal.textContent = parseFloat(lrEl.value).toFixed(3));

  sampleBtn.addEventListener("click", async () => {
    try {
      const pair = await getJSON("/train/sample");
      encEl.value = pair.encoder_text;
      decEl.value = pair.decoder_text;
    } catch (err) {
      errorEl.textContent = "Error: " + err.message;
      errorEl.classList.add("show");
    }
  });

  resetBtn.addEventListener("click", async () => {
    resetBtn.disabled = true;
    try {
      await postJSON("/model/reset", {});
      statusEl.textContent = "model weights reset to original checkpoint.";
      results.innerHTML = '<div class="placeholder-note">Weights restored. Run a training step to see fresh real internals.</div>';
    } catch (err) {
      errorEl.textContent = "Error: " + err.message;
      errorEl.classList.add("show");
    } finally {
      resetBtn.disabled = false;
    }
  });

  runBtn.addEventListener("click", async () => {
    const encoder_text = encEl.value.trim();
    const decoder_text = decEl.value.trim();
    errorEl.classList.remove("show");
    if (!encoder_text || !decoder_text) {
      errorEl.textContent = "Fill in both the encoder input and decoder target."; errorEl.classList.add("show"); return;
    }
    runBtn.disabled = true;
    statusEl.textContent = "running real forward pass, loss, backward pass" + (applyEl.checked ? ", and SGD update…" : "…");
    try {
      const data = await postJSON("/train/step", {
        encoder_text, decoder_text,
        learning_rate: parseFloat(lrEl.value),
        apply_update: applyEl.checked,
      });
      renderTrainingResults(results, data);
      statusEl.textContent = `done · session step ${data.session_step_count}`;
    } catch (err) {
      errorEl.textContent = "Error: " + err.message;
      errorEl.classList.add("show");
      statusEl.textContent = "";
    } finally {
      runBtn.disabled = false;
    }
  });
}

function renderTrainingResults(container, data) {
  container.innerHTML = "";

  // Tokenization + prediction vs target
  const tokSection = sectionShell("01 · tokenization", "Encoder input & decoder target/prediction", null);
  tokSection.appendChild(el("div", "section-note", "encoder input tokens"));
  tokSection.appendChild(renderTokenChips(data.input_tokens));
  tokSection.appendChild(el("div", "section-note", "decoder target vs. what the model currently predicts at each position"));
  const mismatchIdx = [];
  data.target_tokens.forEach((t, i) => { if (t !== data.predicted_tokens[i]) mismatchIdx.push(i); });
  tokSection.appendChild(el("div", null, [
    el("div", "section-note", "target →"),
    renderTokenChips(data.target_tokens, null, { targetIdx: data.target_tokens.map((_, i) => i) }),
    el("div", "section-note", "predicted →"),
    renderTokenChips(data.predicted_tokens, null, { mismatchIdx }),
  ]));
  container.appendChild(tokSection);

  // Full forward trace (embedding / positional / encoder / decoder)
  renderFullTrace(container, data.forward_trace);

  // Per-token loss
  const lossTokSection = sectionShell("05 · per-token loss", "Cross-entropy loss at each target position",
    "higher bar = the model was more surprised by / more wrong about that token");
  lossTokSection.appendChild(renderBarList(
    data.target_tokens.map((t, i) => ({ label: t, value: data.per_token_loss[i] })),
    { formatValue: v => v.toFixed(3) }
  ));
  container.appendChild(lossTokSection);

  // Loss before/after + session sparkline
  const lossSection = sectionShell("06 · loss", "Real cross-entropy loss, before vs. after this step", null);
  const statRow = el("div", "stat-row");
  statRow.appendChild(el("div", "stat-card", [
    el("div", "stat-label", "loss before step"),
    el("div", "stat-value", data.loss_before.toFixed(4)),
  ]));
  if (data.loss_after != null) {
    const improved = data.loss_after < data.loss_before;
    statRow.appendChild(el("div", "stat-card " + (improved ? "good" : "bad"), [
      el("div", "stat-label", "loss after SGD update"),
      el("div", "stat-value", data.loss_after.toFixed(4)),
    ]));
  } else {
    statRow.appendChild(el("div", "stat-card", [
      el("div", "stat-label", "loss after"),
      el("div", "stat-value", "— (update not applied)"),
    ]));
  }
  statRow.appendChild(el("div", "stat-card accent", [
    el("div", "stat-label", "learning rate"),
    el("div", "stat-value", data.learning_rate.toFixed(3)),
  ]));
  lossSection.appendChild(statRow);
  if (data.session_loss_history && data.session_loss_history.length > 1) {
    lossSection.appendChild(el("div", "section-note", `loss after each real step this session (${data.session_loss_history.length} steps)`));
    lossSection.appendChild(renderSparkline(data.session_loss_history));
  }
  container.appendChild(lossSection);

  // Backward pass: gradient report
  const gradSection = sectionShell("07 · backward pass", "Real gradients produced by model.backward()",
    "∂loss/∂param for every parameter tensor — sorted by gradient norm (top 15 shown)");
  const sorted = [...data.gradient_report.per_parameter_summary].sort((a, b) => b.norm - a.norm).slice(0, 15);
  gradSection.appendChild(renderBarList(sorted.map(p => ({ label: p.name, value: p.norm })), { formatValue: v => v.toFixed(4) }));
  container.appendChild(gradSection);

  // Spotlight gradient matrices
  const spotlightSection = sectionShell("08 · gradient close-up", "Full gradient matrices for selected weight tensors",
    "same heatmap convention as attention — every cell is a real ∂loss/∂weight value");
  Object.entries(data.gradient_report.spotlight_matrices).forEach(([name, matrix]) => {
    spotlightSection.appendChild(el("h4", null, name));
    const rowLabels = matrix.map((_, i) => "r" + i);
    const colLabels = matrix[0].map((_, i) => "c" + i);
    spotlightSection.appendChild(renderMatrixHeatmap(matrix, rowLabels, colLabels));
  });
  container.appendChild(spotlightSection);

  // Embedding row gradients (sparse — only tokens actually used)
  const embGradSection = sectionShell("09 · embedding gradients", "∂loss/∂embedding for each token actually used",
    "only the rows for tokens that appeared in this example receive nonzero gradient");
  embGradSection.appendChild(renderBarList(
    data.gradient_report.embedding_row_gradients.map(r => ({ label: r.token, value: r.grad_row_norm })),
    { formatValue: v => v.toFixed(4) }
  ));
  container.appendChild(embGradSection);

  // Output projection gradient — which vocab tokens moved the most
  const outGradSection = sectionShell("10 · output projection gradients", "Vocabulary tokens whose output weights shifted the most",
    "the 10 vocabulary columns of the final projection matrix with the largest gradient norm");
  outGradSection.appendChild(renderBarList(
    data.gradient_report.top_output_projection_gradients.map(t => ({ label: t.token, value: t.grad_col_norm })),
    { formatValue: v => v.toFixed(4) }
  ));
  container.appendChild(outGradSection);

  // Weight update summary
  if (data.applied_update) {
    const updSection = sectionShell("11 · weight update", "Real SGD step: param -= learning_rate × gradient",
      "update norm = ‖learning_rate × gradient‖ actually subtracted from each parameter tensor — top 15 shown");
    const sortedUpd = [...data.weight_update_norms].sort((a, b) => b.update_norm - a.update_norm).slice(0, 15);
    updSection.appendChild(renderBarList(sortedUpd.map(u => ({ label: u.name, value: u.update_norm })), { formatValue: v => v.toFixed(5) }));
    container.appendChild(updSection);
  }
}

// ============================================================
// Tabs + boot
// ============================================================

function initTabs() {
  const infBtn = $("tab-btn-inference"), trBtn = $("tab-btn-training");
  const infPanel = $("tab-inference"), trPanel = $("tab-training");
  infBtn.addEventListener("click", () => {
    infBtn.classList.add("active"); trBtn.classList.remove("active");
    infPanel.style.display = ""; trPanel.style.display = "none";
  });
  trBtn.addEventListener("click", () => {
    trBtn.classList.add("active"); infBtn.classList.remove("active");
    trPanel.style.display = ""; infPanel.style.display = "none";
  });
}

async function loadModelChip() {
  const chip = $("model-chip");
  try {
    const info = await getJSON("/api/status");
    chip.innerHTML = `<b>${info.checkpoint}</b> &nbsp;·&nbsp; vocab ${info.vocab_size} &nbsp;·&nbsp; ${info.config.num_layers} layers &nbsp;·&nbsp; ${info.config.num_heads} heads &nbsp;·&nbsp; d=${info.config.embedding_dim}`;
  } catch (err) {
    chip.textContent = "could not reach API";
  }
}

initTabs();
initInferenceTab();
initTrainingTab();
loadModelChip();
