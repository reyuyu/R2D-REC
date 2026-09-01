const state = { snapshot: null, selectedStage: null, selectedMetric: null, autoRefresh: true, timer: null };
const $ = id => document.getElementById(id);

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 0.001) return number.toExponential(2);
  return number.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function labelStatus(value) {
  const labels = {
    pass: "合同通过", running: "运行中", stopped: "已停止", incomplete: "不完整", not_started: "未开始",
    pending: "待产生", fail: "合同失败", complete: "完整", invalid: "异常", missing: "缺失",
    within_reference_band: "参考带内", review: "需复核", bitwise_match: "位级一致",
    different_expected_stochastic: "SHA 不同", recorded: "已录入", healthy: "健康", ready: "就绪",
    contract_failure: "合同失败", numerically_compared: "已数值对比", structure_mismatch: "结构不一致",
    shape_mismatch: "形状不一致"
  };
  return labels[value] || value || "未知";
}

function pill(status) {
  return `<span class="status-pill status-${status}">${labelStatus(status)}</span>`;
}

function overallCopy(snapshot) {
  if (snapshot.overall_status === "contract_failure") return ["发现复现合同失败", "先处理数据、配置、父检查点或保存结构问题，训练曲线不应掩盖合同错误。"];
  if (snapshot.overall_status === "review") return ["训练轨迹需要人工复核", "合同未失败，但至少一个历史窗口超出宽松参考带。"];
  if (snapshot.summary.completed_stage_count === 4) return ["四阶段复现链已完成", "继续录入固定外部评测，区分工程复现与效果复现。"];
  if (snapshot.stages.some(stage => stage.runtime_status === "running")) return ["四阶段复现正在进行", "实时读取训练步数、同阶段指标窗口和检查点完整性；不会修改训练进程。"];
  return ["复现链已就绪", "监控只读训练产物；合同门禁失败与随机轨迹偏离分别处理。"];
}

function renderSummary(snapshot) {
  const [title, note] = overallCopy(snapshot);
  $("rootPath").textContent = snapshot.root;
  $("overallTitle").textContent = title;
  $("overallNote").textContent = note;
  const metrics = [
    [snapshot.summary.completed_stage_count + "/4", "完成阶段"],
    [snapshot.summary.contract_fail_count, "合同失败"],
    [snapshot.summary.trajectory_review_count, "轨迹复核"],
    [snapshot.summary.external_evaluation_count + "/4", "外评录入"]
  ];
  $("summaryMetrics").innerHTML = metrics.map(([value, label]) => `<div class="summary-stat"><strong>${value}</strong><span>${label}</span></div>`).join("");
}

function renderStageRail(snapshot) {
  $("stageRail").innerHTML = snapshot.stages.map((stage, index) => `
    <button class="stage-card ${stage.id === state.selectedStage ? "active" : ""}" data-stage="${stage.id}" type="button">
      <span class="stage-index">STAGE ${index + 1}</span>
      <span class="stage-title">${stage.short_label}</span>
      ${pill(stage.runtime_status)} ${pill(stage.trajectory_status)}
      <div class="stage-progress"><span style="width:${Math.round(stage.progress * 100)}%"></span></div>
      <div class="stage-meta"><span>${stage.current_step} / ${stage.target_step}</span><span>${Math.round(stage.progress * 100)}%</span></div>
    </button>`).join("");
  document.querySelectorAll(".stage-card").forEach(button => button.addEventListener("click", () => selectStage(button.dataset.stage)));
}

function drawAxes(ctx, width, height, bounds, labels) {
  const pad = { left: 58, right: 50, top: 18, bottom: 42 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfc"; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9e0e2"; ctx.lineWidth = 1;
  ctx.fillStyle = "#66747b"; ctx.font = "12px Segoe UI";
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + innerH * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    const value = bounds.maxY - (bounds.maxY - bounds.minY) * i / 4;
    ctx.fillText(fmt(value, 3), 5, y + 4);
  }
  labels.forEach((label, index) => {
    const x = labels.length === 1 ? pad.left + innerW / 2 : pad.left + innerW * index / (labels.length - 1);
    ctx.textAlign = "center"; ctx.fillText(label, x, height - 14);
  });
  ctx.textAlign = "left";
  return { pad, innerW, innerH, x: t => pad.left + t * innerW, y: value => pad.top + (bounds.maxY - value) / (bounds.maxY - bounds.minY || 1) * innerH };
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.round(rect.width * ratio));
  const height = Math.max(220, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function renderScore(snapshot) {
  const canvas = $("scoreChart");
  const { ctx, width, height } = setupCanvas(canvas);
  const reference = snapshot.stages.map(stage => stage.external_evaluation.reference_score);
  const reproduced = snapshot.stages.map(stage => stage.external_evaluation.reproduced_score);
  const finite = [...reference, ...reproduced].filter(Number.isFinite);
  const min = Math.min(...finite), max = Math.max(...finite);
  const margin = Math.max((max - min) * .25, .006);
  const axes = drawAxes(ctx, width, height, { minY: min - margin, maxY: max + margin }, snapshot.stages.map(stage => stage.short_label));
  const draw = (values, color, dashed = false) => {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5; ctx.setLineDash(dashed ? [7, 5] : []);
    ctx.beginPath();
    let started = false;
    values.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const x = axes.x(index / Math.max(values.length - 1, 1)); const y = axes.y(value);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.setLineDash([]);
    values.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const x = axes.x(index / Math.max(values.length - 1, 1)); const y = axes.y(value);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    });
  };
  draw(reference, "#16835f"); draw(reproduced, "#286da8", true);
  $("scoreLegend").innerHTML = `<span class="legend-key" style="--legend-color:#16835f">历史参考</span><span class="legend-key" style="--legend-color:#286da8">复现评测</span>`;
  $("scoreRows").innerHTML = snapshot.stages.map(stage => {
    const ext = stage.external_evaluation;
    return `<tr><td>${stage.short_label}</td><td>${fmt(ext.reference_score)}</td><td>${fmt(ext.reproduced_score)}</td><td>${ext.delta == null ? "—" : (ext.delta >= 0 ? "+" : "") + fmt(ext.delta)}</td><td>${pill(ext.status)}</td></tr>`;
  }).join("");
}

function selectStage(stageId) {
  state.selectedStage = stageId;
  const stage = state.snapshot.stages.find(item => item.id === stageId);
  const metrics = Object.keys(stage.metric_definitions);
  if (!metrics.includes(state.selectedMetric)) state.selectedMetric = metrics[0];
  renderStageRail(state.snapshot);
  renderSelectors(stage);
  renderDetail(stage);
}

function renderSelectors(stage) {
  $("stageSelect").innerHTML = state.snapshot.stages.map(item => `<option value="${item.id}" ${item.id === stage.id ? "selected" : ""}>${item.short_label}</option>`).join("");
  $("metricSelect").innerHTML = Object.keys(stage.metric_definitions).map(metric => `<option value="${metric}" ${metric === state.selectedMetric ? "selected" : ""}>${metric}</option>`).join("");
}

function metricTrend(metric) {
  if (["clip_fraction", "approx_kl", "zero_std_ratio"].includes(metric)) return "持续抬升通常需要关注；单点波动不作结论。";
  if (metric.includes("loss")) return "只与同阶段同口径参考窗口比较，不要求单调下降。";
  if (metric === "grad_norm") return "应保持有限且无持续爆炸；绝对大小跨算法不可比。";
  if (metric.includes("completion_mean_length")) return "长期骤降可能表示生成结构收缩，需结合 route 与 Probe。";
  if (metric.includes("reward")) return "用于训练健康和方差判断，不能替代固定外部评测。";
  return "以历史窗口和合同语义联合判断，不用单点下结论。";
}

function nearestPoint(points, step) {
  if (!points.length) return null;
  return points.reduce((best, point) => Math.abs(point.step - step) < Math.abs(best.step - step) ? point : best, points[0]);
}

function drawMetricSeries(ctx, axes, points, metric, maxStep, color, dashed = false) {
  if (!points.length) return;
  ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.setLineDash(dashed ? [7, 5] : []); ctx.beginPath();
  points.forEach((point, index) => {
    const x = axes.x(point.step / maxStep), y = axes.y(point[metric]);
    index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke(); ctx.setLineDash([]);
}

function bindMetricTooltip(canvas, axes, maxStep, metric, historical, reproduced) {
  const tooltip = $("metricTooltip");
  canvas.onmousemove = event => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < axes.pad.left || x > rect.width - axes.pad.right) { tooltip.hidden = true; return; }
    const step = Math.max(0, Math.min(maxStep, (x - axes.pad.left) / axes.innerW * maxStep));
    const historyPoint = nearestPoint(historical, step);
    const reproducedPoint = nearestPoint(reproduced, step);
    tooltip.innerHTML = `<strong>${metric}</strong><span class="history">历史 step ${historyPoint?.step ?? "—"} · ${fmt(historyPoint?.[metric])}</span><span class="reproduced">复现 step ${reproducedPoint?.step ?? "—"} · ${fmt(reproducedPoint?.[metric])}</span>`;
    tooltip.style.left = `${Math.min(Math.max(x + 14, 10), rect.width - 210)}px`;
    tooltip.style.top = `${Math.max(event.clientY - rect.top - 76, 8)}px`;
    tooltip.hidden = false;
  };
  canvas.onmouseleave = () => { tooltip.hidden = true; };
}

function renderMetricChart(stage) {
  const metric = state.selectedMetric;
  const points = stage.curve.filter(point => Number.isFinite(point[metric]));
  const historical = stage.historical_curve.filter(point => Number.isFinite(point[metric]));
  const refs = stage.milestones.map(item => {
    const gap = item.gaps.find(entry => entry.metric === metric);
    return gap ? { step: item.step, ...gap } : null;
  }).filter(Boolean);
  const values = [...points.map(point => point[metric]), ...historical.map(point => point[metric]), ...refs.flatMap(item => [item.reference_mean, ...item.reference_band])].filter(Number.isFinite);
  const canvas = $("metricChart"); const { ctx, width, height } = setupCanvas(canvas);
  if (!values.length) {
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#66747b"; ctx.font = "14px Segoe UI"; ctx.fillText("当前阶段尚无该指标数据", 24, 42);
    $("metricHint").textContent = `${stage.metric_definitions[metric]} 健康趋势：${metricTrend(metric)}`;
    $("metricLegend").innerHTML = ""; $("metricTooltip").hidden = true; return;
  }
  const min = Math.min(...values), max = Math.max(...values), margin = Math.max((max - min) * .12, Math.abs(max || 1) * .03);
  const maxStep = Math.max(stage.target_step, ...points.map(point => point.step), ...historical.map(point => point.step));
  const axes = drawAxes(ctx, width, height, { minY: min - margin, maxY: max + margin }, ["0", String(Math.round(maxStep / 2)), String(maxStep)]);
  refs.forEach(ref => {
    const x = axes.x(ref.step / maxStep); const y1 = axes.y(ref.reference_band[1]); const y2 = axes.y(ref.reference_band[0]);
    ctx.fillStyle = "rgba(167,104,19,.14)"; ctx.fillRect(x - 5, y1, 10, y2 - y1);
    ctx.fillStyle = "#a76813"; ctx.beginPath(); ctx.arc(x, axes.y(ref.reference_mean), 4, 0, Math.PI * 2); ctx.fill();
  });
  drawMetricSeries(ctx, axes, historical, metric, maxStep, "#16835f");
  drawMetricSeries(ctx, axes, points, metric, maxStep, "#286da8", true);
  bindMetricTooltip(canvas, axes, maxStep, metric, historical, points);
  $("metricLegend").innerHTML = `<span class="legend-key" style="--legend-color:#16835f">历史完整轨迹 · ${historical.length} 点</span><span class="legend-key dashed" style="--legend-color:#286da8">当前复现 · ${points.length} 点</span><span class="legend-key milestone" style="--legend-color:#a76813">历史里程碑均值 / 参考带</span>`;
  $("metricHint").textContent = `${stage.metric_definitions[metric]} 健康趋势：${metricTrend(metric)}`;
}

function renderDetail(stage) {
  renderMetricChart(stage);
  $("qualityFacts").innerHTML = [
    ["运行状态", labelStatus(stage.runtime_status)],
    ["合同门禁", labelStatus(stage.contract_status)],
    ["轨迹判断", labelStatus(stage.trajectory_status)],
    ["训练进度", `${stage.current_step} / ${stage.target_step}`],
    ["Adapter", labelStatus(stage.adapter.status)],
    ["外部评测", labelStatus(stage.external_evaluation.status)]
  ].map(([label, value]) => `<div class="fact"><span>${label}</span><strong>${value}</strong></div>`).join("");
  const rows = [];
  stage.milestones.forEach(milestone => milestone.gaps.forEach(gap => rows.push(`<tr><td>${milestone.step}</td><td>${gap.metric}</td><td>${fmt(gap.reference_mean)}</td><td>${fmt(gap.reproduced_mean)}</td><td>${gap.absolute_delta >= 0 ? "+" : ""}${fmt(gap.absolute_delta)}</td><td>${fmt(gap.reference_band[0])} ~ ${fmt(gap.reference_band[1])}</td><td>${pill(gap.status)}</td></tr>`)));
  $("gapRows").innerHTML = rows.length ? rows.join("") : `<tr><td colspan="7">尚未到达可比较的历史里程碑。</td></tr>`;
  $("definitions").innerHTML = Object.entries(stage.metric_definitions).map(([name, description]) => `<div class="definition"><strong>${name}</strong><p>${description}</p><div class="trend">${metricTrend(name)}</div></div>`).join("");
}

function renderIntegrity(snapshot) {
  $("integrityGrid").innerHTML = snapshot.stages.map(stage => `
    <article class="integrity-item">
      <h3>${stage.short_label}</h3>
      <div>${pill(stage.contract_status)} ${pill(stage.adapter.status)}</div>
      <div class="checkpoint-list">${stage.checkpoints.map(cp => `<span class="checkpoint ${cp.status}" title="${cp.path}" aria-label="检查点 ${cp.step} ${labelStatus(cp.status)}">${cp.step} · ${labelStatus(cp.status)}</span>`).join("")}</div>
      <span class="hash">历史 ${stage.adapter.reference_sha256.slice(0, 16)}…<br>复现 ${stage.adapter.reproduced_sha256 ? stage.adapter.reproduced_sha256.slice(0, 16) + "…" : "待产生"}</span>
      <div class="adapter-caption">同 step 历史 vs 复现 LoRA adapter</div>
      <div class="table-wrap adapter-table"><table><thead><tr><th>Step</th><th>Cosine</th><th>Rel L2</th><th>Max abs</th><th>状态</th></tr></thead><tbody>
        ${stage.adapter_comparisons.map(item => `<tr><td>${item.step}</td><td>${fmt(item.cosine_similarity, 7)}</td><td>${fmt(item.relative_l2, 7)}</td><td>${fmt(item.max_abs_delta, 7)}</td><td>${pill(item.status)}</td></tr>`).join("")}
      </tbody></table></div>
    </article>`).join("");
}

function render(snapshot) {
  state.snapshot = snapshot;
  if (!state.selectedStage || !snapshot.stages.some(stage => stage.id === state.selectedStage)) {
    state.selectedStage = snapshot.summary.current_stage || snapshot.stages[0].id;
  }
  renderSummary(snapshot); renderStageRail(snapshot); renderScore(snapshot); renderIntegrity(snapshot);
  selectStage(state.selectedStage);
  $("liveState").innerHTML = `<span class="live-dot"></span>已更新 ${new Date(snapshot.generated_at).toLocaleTimeString()}`;
}

async function refresh() {
  try {
    const response = await fetch(`/api/snapshot?_=${Date.now()}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    render(payload); $("errorBanner").hidden = true;
  } catch (error) {
    $("liveState").innerHTML = `<span class="live-dot" style="background:#d86a6a"></span>连接失败`;
    $("errorBanner").textContent = `监控读取失败：${error.message}`; $("errorBanner").hidden = false;
  }
}

$("refreshNow").addEventListener("click", refresh);
$("autoRefresh").addEventListener("click", event => {
  state.autoRefresh = !state.autoRefresh; event.currentTarget.setAttribute("aria-pressed", String(state.autoRefresh));
  event.currentTarget.textContent = `自动刷新：${state.autoRefresh ? "开" : "关"}`;
});
$("stageSelect").addEventListener("change", event => selectStage(event.target.value));
$("metricSelect").addEventListener("change", event => { state.selectedMetric = event.target.value; renderDetail(state.snapshot.stages.find(stage => stage.id === state.selectedStage)); });
window.addEventListener("resize", () => { if (state.snapshot) { renderScore(state.snapshot); renderDetail(state.snapshot.stages.find(stage => stage.id === state.selectedStage)); } });
refresh(); state.timer = setInterval(() => { if (state.autoRefresh) refresh(); }, 5000);
