(() => {
  const RECOMMENDATION = 'recommendation_grpo';
  const USER = 'user_grpo';
  const USER_KINDS = [
    ['hallucinated_sid', 'Action / Chain'],
    ['duplicate_sid', 'Action'],
    ['date_mismatch', 'Chain'],
    ['action_mismatch', 'Chain'],
    ['duplicate_event', 'Chain'],
    ['chronology_violation', 'Chain'],
    ['excess_event', 'Chain'],
  ];
  const recommendationRenderOverview = renderOverview;
  const recommendationRenderPerformance = renderPerformance;
  const recommendationRenderExplorer = renderExplorer;
  const recommendationRenderExplorerOptions = renderExplorerOptions;
  const recommendationRenderTraceIndex = renderTraceIndex;
  const recommendationRenderProbes = renderProbes;
  const recommendationRenderDsr = renderDsr;
  const recommendationRefresh = refresh;
  const recommendationActivateView = activateView;

  state.allRuns = [];
  state.activeKind = RECOMMENDATION;
  state.mcSummary = {available: false};
  const userSampleContextCache = new Map();
  let userRefreshInFlight = false;
  const monitorScrollableSelectors = [
    '#trace .probe-context-block pre',
    '#userProbeDetail .probe-context-block pre',
    '#trace .candidate .text',
    '#userProbeDetail .candidate .text',
    '#mcCreditTrace .mc-credit-completion',
    '#trace .beam-details[open] .beam-grid',
    '#traceIndex .trace-links',
  ];

  function captureMonitorScrollState() {
    const root = document.scrollingElement;
    return {
      pageTop: root?.scrollTop || 0,
      pageLeft: root?.scrollLeft || 0,
      containers: monitorScrollableSelectors.map(selector => ({
        selector,
        positions: [...document.querySelectorAll(selector)].map(element => ({
          top: element.scrollTop,
          left: element.scrollLeft,
        })),
      })),
    };
  }

  function restoreMonitorScrollState(snapshot) {
    if (!snapshot) return;
    for (const group of snapshot.containers) {
      [...document.querySelectorAll(group.selector)].forEach((element, index) => {
        const position = group.positions[index];
        if (!position) return;
        element.scrollTop = position.top;
        element.scrollLeft = position.left;
      });
    }
    const root = document.scrollingElement;
    if (root) {
      root.scrollTop = snapshot.pageTop;
      root.scrollLeft = snapshot.pageLeft;
    }
  }

  function isUserRun() {
    return state.activeKind === USER || state.manifest.run_kind === USER || state.capabilities.user_grpo === true;
  }

  function isMcRun() {
    return state.manifest.algorithm === 'mc_user_v1' || state.capabilities.mc_user === true;
  }

  function isHybridMcRun() {
    return state.manifest.algorithm === 'mc_user_hybrid_grpo_v1' || state.capabilities.mc_user_hybrid === true;
  }

  function buildUserShell() {
    document.querySelector('.experiment-bar').insertAdjacentHTML('beforebegin', `
      <section class="run-kind-switcher" aria-label="GRPO 任务类型">
        <span class="switch-label">任务入口</span>
        <div class="kind-segments">
          <button class="kind-segment active" type="button" data-run-kind="${RECOMMENDATION}">懂推荐 GRPO</button>
          <button class="kind-segment" type="button" data-run-kind="${USER}">懂用户 GRPO</button>
          <button class="kind-segment" id="trueRecEntry" type="button">TrueRec GRPO</button>
        </div>
        <span class="kind-context" id="kindContext">Recommendation / DSR 训练实验</span>
      </section>`);
    const nav = document.querySelector('.tabs');
    document.querySelector('#trueRecEntry').onclick = () => { window.location.href = '/truerec'; };
    nav.querySelectorAll('.tab').forEach(tab => tab.dataset.kind = 'recommendation');
    nav.querySelector('[data-view="explorer"]').dataset.kind = 'shared';
    nav.querySelector('[data-view="explorer"]').insertAdjacentHTML('beforebegin', `
      <button class="tab user-tab" data-kind="user" data-view="userOverview" hidden>总览</button>
      <button class="tab user-tab" data-kind="user" data-view="userAction" hidden>Action</button>
      <button class="tab user-tab" data-kind="user" data-view="userChain" hidden>Chain</button>
      <button class="tab user-tab" data-kind="user" data-view="userToken" hidden>Token Advantage</button>
      <button class="tab user-tab" data-kind="user" data-view="userProbe" hidden>Prob / 固定探针</button>`);
    document.querySelector('#dsr').insertAdjacentHTML('beforebegin', `
      <main id="userOverview" class="view"><div class="wrap">
        <div class="user-callout demo" id="userDemoNotice" hidden>当前为合成 UI 演示数据，不代表真实训练结果。</div>
        <section class="status user-status">
          <div class="stat"><div class="label" id="uStepLabel">训练步数</div><div class="value" id="uStep">-</div></div>
          <div class="stat" id="uOptimizerStat" hidden><div class="label">Optimizer Step</div><div class="value" id="uOptimizerStep">-</div></div>
          <div class="stat"><div class="label">当前路由</div><div class="value" id="uRoute">-</div></div>
          <div class="stat"><div class="label">总体进度</div><div class="value" id="uProgress">-</div></div>
          <div class="stat"><div class="label">Loss</div><div class="value" id="uLoss">-</div></div>
          <div class="stat"><div class="label">Grad Norm</div><div class="value" id="uGrad">-</div></div>
          <div class="stat"><div class="label">Learning Rate</div><div class="value small" id="uLr">-</div></div>
        </section>
        <div class="progress-track" aria-label="User GRPO 总体训练进度"><div class="progress-fill" id="uProgressFill"></div></div>
        <section class="user-metric-grid" id="userOverviewCards"></section>
        <section class="charts">
          <div class="panel"><h2>Loss 与梯度范数</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Loss</span><span class="key" style="--c:#2d6cdf">Grad Norm</span></div><canvas class="chart" id="userLossChart"></canvas></div>
          <div class="panel"><h2>Task Reward</h2><div class="legend"><span class="key" style="--c:#2d6cdf">Action F1</span><span class="key" style="--c:#bd3f4c">Chain Reward</span></div><canvas class="chart" id="userRewardChart"></canvas></div>
          <div class="panel"><h2>Sequence / Token Advantage</h2><div class="legend"><span class="key" style="--c:#16835f">Sequence mean</span><span class="key" style="--c:#2d6cdf">Sequence std</span><span class="key" style="--c:#b36b08">Token mean</span><span class="key" style="--c:#7654b5">Token std</span></div><canvas class="chart" id="userAdvantageChart"></canvas></div>
          <div class="panel"><h2>局部 Penalty 覆盖</h2><div class="legend"><span class="key" style="--c:#b36b08">Masked candidates</span><span class="key" style="--c:#bd3f4c">Masked tokens</span></div><canvas class="chart" id="userMaskChart"></canvas></div>
          <div class="panel"><h2>Policy Ratio 与 Clip</h2><div class="legend"><span class="key" style="--c:#16835f">Ratio mean</span><span class="key" style="--c:#b36b08">Clip fraction</span></div><canvas class="chart" id="userPolicyChart"></canvas></div>
          <div class="panel"><h2>Reward Std 与 Zero-std</h2><div class="legend"><span class="key" style="--c:#2d6cdf">Task reward std</span><span class="key" style="--c:#bd3f4c">Zero-std ratio</span></div><canvas class="chart" id="userVarianceChart"></canvas></div>
        </section>
      </div></main>
      <main id="userAction" class="view"><div class="wrap">
        <div class="user-section-head"><h2>Action Set-F1</h2><p>wrong selection 由主 F1 处理；hallucination / duplicate 同时具有局部 token penalty</p></div>
        <section class="user-metric-grid" id="userActionCards"></section>
        <section class="charts">
          <div class="panel"><h2>F1 / Precision / Recall</h2><div class="legend"><span class="key" style="--c:#16835f">F1</span><span class="key" style="--c:#2d6cdf">Precision</span><span class="key" style="--c:#b36b08">Recall</span></div><canvas class="chart" id="userActionScoreChart"></canvas></div>
          <div class="panel"><h2>Exact Match</h2><div class="legend"><span class="key" style="--c:#16835f">Exact</span></div><canvas class="chart" id="userActionExactChart"></canvas></div>
          <div class="panel"><h2>Wrong Selection</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Candidate rate</span></div><canvas class="chart" id="userWrongSelectionChart"></canvas></div>
          <div class="panel"><h2>Hallucination / Duplicate</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Hallucination</span><span class="key" style="--c:#b36b08">Duplicate</span></div><canvas class="chart" id="userActionConstraintChart"></canvas></div>
        </section>
      </div></main>
      <main id="userChain" class="view"><div class="wrap">
        <div class="user-section-head"><h2>Chain Alignment</h2><p>同时观察 Action Alignment、Logic Alignment 与局部 grounding/constraint 信号</p></div>
        <section class="user-metric-grid" id="userChainCards"></section>
        <section class="charts">
          <div class="panel"><h2>Total / Action / Logic</h2><div class="legend"><span class="key" style="--c:#16835f">Total Reward</span><span class="key" style="--c:#2d6cdf">Action Alignment</span><span class="key" style="--c:#7654b5">Logic Alignment</span></div><canvas class="chart" id="userChainScoreChart"></canvas></div>
          <div class="panel"><h2>Date / Action Mismatch</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Date</span><span class="key" style="--c:#b36b08">Action</span></div><canvas class="chart" id="userChainMismatchChart"></canvas></div>
          <div class="panel"><h2>Chain Constraint Count</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Hallucinated SID</span><span class="key" style="--c:#2d6cdf">Duplicate event</span><span class="key" style="--c:#b36b08">Chronology</span><span class="key" style="--c:#7654b5">Excess event</span></div><canvas class="chart" id="userChainConstraintChart"></canvas></div>
          <div class="panel"><h2>Grounding</h2><div class="legend"><span class="key" style="--c:#16835f">Grounded</span><span class="key" style="--c:#b36b08">Partial</span><span class="key" style="--c:#bd3f4c">Ungrounded</span></div><canvas class="chart" id="userGroundingChart"></canvas></div>
        </section>
      </div></main>
      <main id="userToken" class="view"><div class="wrap" id="userTokenLegacy">
        <div class="user-section-head"><h2>Token Advantage</h2><p>Sqrt-normalized local penalty；overlap 取最强值，不累加</p></div>
        <section class="user-metric-grid" id="userTokenCards"></section>
        <section class="charts">
          <div class="panel"><h2>Per-kind Masked Token Count</h2><div class="legend" id="userKindCountLegend"></div><canvas class="chart" id="userKindCountChart"></canvas></div>
          <div class="panel"><h2>Per-kind Incremental Negative Mass</h2><div class="legend" id="userKindMassLegend"></div><canvas class="chart" id="userKindMassChart"></canvas></div>
        </section>
        <section class="panel section-gap"><h2>最新各类局部 Penalty</h2><div class="table-scroll"><table class="penalty-table"><thead><tr><th>Violation kind</th><th>适用路由</th><th>Masked tokens</th><th>Negative mass</th><th>Mass share</th></tr></thead><tbody id="userPenaltyRows"></tbody></table></div></section>
      </div><div class="wrap" id="userMcCredit" hidden>
        <div class="user-section-head"><h2>Marginal Credit</h2><p>当前 on-policy 训练样本；这些 credit 直接参与本次参数更新</p></div>
        <div class="user-callout" id="mcCreditNotice"></div>
        <section class="user-metric-grid" id="mcCreditCards"></section>
        <section class="charts">
          <div class="panel"><h2>Action Credit Mass</h2><div class="legend"><span class="key" style="--c:#16835f">Positive</span><span class="key" style="--c:#bd3f4c">Negative</span></div><canvas class="chart" id="mcActionCreditChart"></canvas></div>
          <div class="panel"><h2>Chain Credit Mass</h2><div class="legend"><span class="key" style="--c:#16835f">Positive</span><span class="key" style="--c:#bd3f4c">Negative</span></div><canvas class="chart" id="mcChainCreditChart"></canvas></div>
        </section>
        <section class="panel section-gap" id="mcCreditTracePanel">
          <div class="mc-credit-controls"><label>Route<select id="mcCreditRoute"><option value="">全部</option><option value="action">Action</option><option value="chain">Chain</option></select></label><label>Prompt<select id="mcCreditPrompt"></select></label></div>
          <div id="mcCreditTrace"></div>
        </section>
      </div></main>
      <main id="userProbe" class="view"><div class="wrap" id="userProbeLegacy">
        <div class="user-section-head"><h2>固定 Probe 趋势</h2><p>固定样本、固定随机种子、无梯度；用于比较不同训练时间步的真实泛化方向</p></div>
        <div class="user-callout" id="userProbeNotice">等待固定 Probe 数据。该评估不进入 reward、advantage 或 optimizer。</div>
        <section class="user-metric-grid" id="userProbeCards"></section>
        <section class="charts">
          <div class="panel"><h2>Action Probe</h2><div class="legend"><span class="key" style="--c:#16835f">F1</span><span class="key" style="--c:#2d6cdf">Precision</span><span class="key" style="--c:#b36b08">Recall</span></div><canvas class="chart" id="userProbeActionChart"></canvas></div>
          <div class="panel"><h2>Chain Probe</h2><div class="legend"><span class="key" style="--c:#16835f">Total</span><span class="key" style="--c:#2d6cdf">Action Alignment</span><span class="key" style="--c:#7654b5">Logic Alignment</span></div><canvas class="chart" id="userProbeChainChart"></canvas></div>
        </section>
        <section class="panel section-gap">
          <div class="user-probe-controls"><label>路由<select id="userProbeRoute"><option value="action">Action</option><option value="chain">Chain</option></select></label><label>固定样本<select id="userProbeGroup"></select></label><label>时间步<select id="userProbeStep"></select></label></div>
          <div id="userProbeDetail" class="user-probe-detail"></div>
        </section>
      </div><div class="wrap" id="userMcProbe" hidden>
        <div class="user-section-head"><h2>Probe / 固定探针</h2><p>固定 3+3、独立 inference-only sidecar；不参与训练，不等于官方分数</p></div>
        <div class="user-callout" id="mcProbeNotice"></div>
        <div class="mc-probe-schedule" id="mcProbeSchedule"></div>
        <section class="user-metric-grid" id="mcProbeCards"></section>
        <section class="charts" id="mcProbeCharts">
          <div class="panel"><h2>Action F1 / User Proxy</h2><div class="legend"><span class="key" style="--c:#16835f">Action F1</span><span class="key" style="--c:#2d6cdf">User Proxy</span></div><canvas class="chart" id="mcProbeActionChart"></canvas></div>
          <div class="panel"><h2>Chain Total</h2><div class="legend"><span class="key" style="--c:#7654b5">Chain Total</span></div><canvas class="chart" id="mcProbeChainChart"></canvas></div>
        </section>
        <section class="panel section-gap"><h2>Paired Checkpoints</h2><div class="table-scroll" id="mcProbeTable"></div><div class="mc-probe-deltas" id="mcProbeDeltas"></div></section>
        <section class="panel section-gap"><h2>固定样本详情</h2><div class="user-probe-controls"><label>路由<select id="mcProbeSampleRoute"><option value="action">Action</option><option value="chain">Chain</option></select></label><label>固定样本<select id="mcProbeSampleId"></select></label><label>Checkpoint<select id="mcProbeSampleStep"></select></label></div><div id="mcProbeSampleDetail" class="user-probe-detail"></div></section>
        <section class="panel section-gap"><h2>Recommendation Guard</h2><div id="mcRecommendationGuard"></div></section>
      </div></main>`);
  }

  const HYBRID_METRIC_GUIDE = {
    'Group Reward Mean': ['当前 K4 候选 evaluator reward 的均值。', 'stable', '健康：结合固定 Probe 看，单步波动正常'],
    'Reward Spread': ['同组最高与最低 reward 之差，决定序列优势是否有区分度。', 'stable', '健康：持续非零且不过度尖峰'],
    'Sequence Loss': ['完整 completion 上由组内标准化优势产生的损失项。', 'stable', '健康：有限波动，无 NaN/Inf'],
    'Local Loss': ['SID/Event marginal delta 产生的局部辅助损失，进入总损失前乘 0.3。', 'stable', '健康：有限且不长期压倒序列项'],
    'Total Loss': ['Sequence Loss + 0.3 × Local Loss。', 'stable', '健康：有限波动，无持续爆炸'],
    'Grad Norm': ['本次 LoRA 更新前的全局梯度范数。', 'stable', '健康：有限稳定，无尖峰或非有限值'],
    'Candidate Mean F1': ['Action 当前所有 on-policy candidate 的集合 F1 均值。', 'up', '健康：固定 cohort 上升；在线值仅看长期趋势'],
    'Mean SID Count': ['Action candidate 平均预测完整 SID 数。', 'stable', '健康：与 Gold 规模匹配，避免坍缩或膨胀'],
    'Negative Candidate Rate': ['至少含一个负 marginal unit 的 candidate 比例。', 'stable', '健康：非零且稳定；过高或归零都需检查'],
    'Positive credit mass': ['正 marginal delta 的累计绝对质量。', 'stable', '健康：持续存在，并与负质量保持合理平衡'],
    'Negative credit mass': ['负 marginal delta 的累计绝对质量。', 'stable', '健康：持续存在但不应长期压倒正质量'],
    'Mean Reward': ['Chain evaluator Total Reward 的 on-policy candidate 均值。', 'up', '健康：固定 cohort 上升；在线值仅作训练信号'],
    'Action Alignment': ['Chain 事件 action 与 Gold ordered matching 的对齐分数。', 'up', '健康：固定 cohort 趋势上升'],
    'Logic Alignment': ['Chain 事件 logic 与 Gold 逻辑关系的对齐分数。', 'up', '健康：固定 cohort 趋势上升'],
    'Mean Event Count': ['Chain candidate 平均输出事件数量。', 'stable', '健康：贴近 Gold 事件数，避免过短或冗长'],
    'Positive Credit Mass': ['正 marginal delta 的累计绝对质量。', 'stable', '健康：持续存在，并与负质量平衡'],
    'Negative Credit Mass': ['负 marginal delta 的累计绝对质量。', 'stable', '健康：持续存在但不过度主导'],
    'Action Mean F1': ['全部 Action on-policy candidates 的累计平均 F1。', 'up', '健康：固定 Probe 上升；在线累计值仅作参考'],
    'Action Positive Mass': ['Action 正 SID marginal delta 的累计质量。', 'stable', '健康：持续增长且信号不过度集中'],
    'Action Negative Mass': ['Action 负 SID marginal delta 的累计绝对质量。', 'stable', '健康：非零但不长期压倒正质量'],
    'Action Negative Rate': ['Action 中含负 SID credit 的 candidate 比例。', 'stable', '健康：保持辨别力，不应长期为 0 或 100%'],
    'Chain Action Align': ['全部 Chain candidates 的累计平均 action alignment。', 'up', '健康：固定 Probe 上升'],
    'Chain Logic Align': ['全部 Chain candidates 的累计平均 logic alignment。', 'up', '健康：固定 Probe 上升'],
    'Chain Positive Mass': ['Chain 正 event marginal delta 的累计质量。', 'stable', '健康：持续存在且不过度集中'],
    'Chain Negative Mass': ['Chain 负 event marginal delta 的累计绝对质量。', 'stable', '健康：非零但不长期压倒正质量'],
    'Action F1': ['固定 3+3 Probe 中 Action 的集合 F1。', 'up', '健康：相对 BETA 持续为正或改善'],
    'ΔAction': ['当前 checkpoint Action F1 减去 BETA Action F1。', 'up', '健康：大于 0，且多个 milestone 方向一致'],
    'Chain Total': ['固定 Probe 中 0.5 × Action Alignment + 0.5 × Logic Alignment。', 'up', '健康：相对 BETA 改善'],
    'ΔChain': ['当前 checkpoint Chain Total 减去 BETA。', 'up', '健康：大于 0，且不是单点偶然波动'],
    'ΔChainAction': ['当前 checkpoint Chain Action Alignment 减去 BETA。', 'up', '健康：大于 0'],
    'ΔChainLogic': ['当前 checkpoint Chain Logic Alignment 减去 BETA。', 'up', '健康：大于 0'],
    'User Proxy': ['固定 Action F1 与 Chain Total 的综合本地代理分。', 'up', '健康：仅用于本地 checkpoint 排序，越高越好'],
    'ΔProxy': ['当前 User Proxy 减去 BETA User Proxy。', 'up', '健康：大于 0，且多时间点保持'],
  };

  const HYBRID_CHART_GUIDE = {
    'Loss / Grad Norm': ['总损失与梯度尺度。', '有限稳定、无 NaN/Inf；不以越低越好判断'],
    'On-policy Reward': ['当前生成候选的 evaluator 均值。', '只看长期训练信号，模型效果以固定 Probe 为准'],
    'Marginal Credit Mass': ['正负 SID/Event marginal delta 的每步质量。', '两侧持续有信号，避免一侧长期归零或压倒另一侧'],
    'Negative Candidate Rate': ['含负 marginal unit 的候选累计比例。', '稳定在非退化区间，不追求单调下降'],
    'Training Signal': ['活跃 unit/token 及 optimizer 是否实际更新。', 'active signal 持续存在，非预期 skipped update 接近 0'],
    'Pipeline Health': ['格式有效、投影需求和 token overlap 的累计比例。', 'Valid 越高越好；Projection/Overlap 低且稳定'],
    'F1 / Precision / Recall': ['Action K4 候选的集合匹配质量。', '固定 cohort 上升且 Precision/Recall 不严重失衡'],
    'Predicted SID Count': ['每个 Action candidate 的平均完整 SID 数。', '贴近 Gold 规模，避免输出坍缩或膨胀'],
    'Positive / Negative Credit Mass': ['marginal delta 的正负训练质量。', '两侧均有且比例稳定，避免单侧主导'],
    'Positive / Negative Unit Count': ['产生正负 credit 的 SID/Event 数量。', '持续有区分信号，避免长期全零'],
    'Total / Action / Logic': ['Chain 总分及 action、logic 两个组成分。', '固定 cohort 上升，且两项不明显背离'],
    'Predicted Event Count': ['每个 Chain candidate 的平均事件数。', '贴近 Gold 事件数，避免过短或冗长'],
    'Action Credit Mass': ['Action SID marginal credit 正负质量。', '正负信号持续存在且比例稳定'],
    'Chain Credit Mass': ['Chain Event marginal credit 正负质量。', '正负信号持续存在且比例稳定'],
  };

  function metricCards(id, items) {
    $(id).innerHTML = items.map(([label, value, tone]) => {
      const guide = isHybridMcRun() ? HYBRID_METRIC_GUIDE[label] : null;
      return `<div class="stat"><div class="label">${escapeHtml(label)}</div><div class="value small${tone ? ` ${tone}` : ''}">${escapeHtml(value)}</div>${guide ? `<div class="user-metric-definition">${escapeHtml(guide[0])}</div><span class="user-health-tag ${guide[1]}">${escapeHtml(guide[2])}</span>` : ''}</div>`;
    }).join('');
  }

  function latestRoute(route) {
    return state.metrics.filter(row => row.route === route).at(-1) || {};
  }

  function objectPoint(rows, field, key) {
    return rows.filter(row => row[field]?.[key] != null).map(row => [row.step, Number(row[field][key])]);
  }

  function mcCandidateMean(row, key) {
    const values = (row.candidates || []).map(candidate => key === 'f1' ? (candidate.f1 ?? candidate.reward) : candidate[key]).filter(value => value != null).map(Number);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  function mcCandidatePoints(rows, key) {
    return rows.map(row => [Number(row.prompt_step ?? row.step), mcCandidateMean(row, key)]).filter(([, value]) => value != null);
  }

  function mcCandidateTotal(row, key) {
    return (row.candidates || []).reduce((sum, candidate) => sum + Number(candidate[key] || 0), 0);
  }

  function mcTotalPoints(rows, key) {
    return rows.map(row => [Number(row.prompt_step ?? row.step), mcCandidateTotal(row, key)]);
  }

  function mcCumulativeRouteRate(route, predicate) {
    let numerator = 0, denominator = 0;
    const output = [];
    for (const row of state.metrics) {
      if (row.route !== route) continue;
      for (const candidate of row.candidates || []) {
        denominator += 1;
        numerator += predicate(candidate) ? 1 : 0;
      }
      output.push([Number(row.prompt_step ?? row.step), denominator ? numerator / denominator : 0]);
    }
    return output;
  }

  function mcCumulativeHealth(predicate) {
    let numerator = 0, denominator = 0;
    return state.metrics.map(row => {
      for (const candidate of row.candidates || []) {
        denominator += 1;
        numerator += predicate(candidate) ? 1 : 0;
      }
      return [Number(row.prompt_step ?? row.step), denominator ? numerator / denominator : 0];
    });
  }

  function setUserChartPanel(id, title, labels, note = '', palette = colors) {
    const panel = $(id)?.closest('.panel');
    if (!panel) return;
    panel.hidden = false;
    panel.querySelector('h2').textContent = title;
    panel.querySelector('.legend').innerHTML = labels.map((label, index) => `<span class="key" style="--c:${palette[index % palette.length]}">${escapeHtml(label)}</span>`).join('');
    let noteNode = panel.querySelector('.mc-chart-note');
    if (note && !noteNode) {
      noteNode = document.createElement('p');
      noteNode.className = 'mc-chart-note';
      panel.querySelector('h2').after(noteNode);
    }
    if (noteNode) {
      noteNode.textContent = note;
      noteNode.hidden = !note;
    }
    let guideNode = panel.querySelector('.mc-metric-guide');
    const guide = isHybridMcRun() ? HYBRID_CHART_GUIDE[title] : null;
    if (guide && !guideNode) {
      guideNode = document.createElement('p');
      guideNode.className = 'mc-metric-guide';
      (noteNode || panel.querySelector('h2')).after(guideNode);
    }
    if (guideNode) {
      guideNode.innerHTML = guide ? `<strong>定义：</strong>${escapeHtml(guide[0])} <strong>健康趋势：</strong>${escapeHtml(guide[1])}` : '';
      guideNode.hidden = !guide;
    }
    chartLabels[id] = labels;
  }

  function configureOverviewCharts(mc) {
    if (mc) {
      setUserChartPanel('userLossChart', 'Loss / Grad Norm', ['Loss', 'Grad Norm'], '', ['#bd3f4c', '#2d6cdf']);
      setUserChartPanel('userRewardChart', 'On-policy Reward', ['Action mean reward', 'Chain mean reward'], '在线训练信号，不等同于固定 Probe 泛化分数', ['#2d6cdf', '#bd3f4c']);
      setUserChartPanel('userAdvantageChart', 'Marginal Credit Mass', ['Action Positive', 'Action Negative', 'Chain Positive', 'Chain Negative'], '', ['#16835f', '#bd3f4c', '#2d6cdf', '#b36b08']);
      setUserChartPanel('userMaskChart', 'Negative Candidate Rate', ['Action cumulative', 'Chain cumulative'], '', ['#2d6cdf', '#bd3f4c']);
      setUserChartPanel('userPolicyChart', 'Training Signal', ['Active units', 'Active tokens', 'Optimizer update', 'Skipped update'], '', ['#16835f', '#2d6cdf', '#b36b08', '#bd3f4c']);
      setUserChartPanel('userVarianceChart', 'Pipeline Health', ['Valid candidates', 'Projection required', 'Overlap candidates'], '', ['#16835f', '#2d6cdf', '#bd3f4c']);
      return;
    }
    setUserChartPanel('userLossChart', 'Loss 与梯度范数', ['Loss', 'Grad Norm'], '', ['#bd3f4c', '#2d6cdf']);
    setUserChartPanel('userRewardChart', 'Task Reward', ['Action F1', 'Chain Reward'], '', ['#2d6cdf', '#bd3f4c']);
    setUserChartPanel('userAdvantageChart', 'Sequence / Token Advantage', ['Sequence mean', 'Sequence std', 'Token mean', 'Token std'], '', ['#16835f', '#2d6cdf', '#b36b08', '#7654b5']);
    setUserChartPanel('userMaskChart', '局部 Penalty 覆盖', ['Masked candidates', 'Masked tokens'], '', ['#b36b08', '#bd3f4c']);
    setUserChartPanel('userPolicyChart', 'Policy Ratio 与 Clip', ['Ratio mean', 'Clip fraction'], '', ['#16835f', '#b36b08']);
    setUserChartPanel('userVarianceChart', 'Reward Std 与 Zero-std', ['Task reward std', 'Zero-std ratio'], '', ['#2d6cdf', '#bd3f4c']);
  }

  function renderUserOverview() {
    const current = state.metrics.at(-1) || {};
    const action = latestRoute('action');
    const chain = latestRoute('chain');
    const mc = isMcRun();
    const max = Number(state.manifest.max_steps || state.manifest.config?.prompt_count || 0);
    const promptStep = Number(current.prompt_step ?? current.step ?? 0);
    const progress = max ? Math.min(100, 100 * promptStep / max) : 0;
    setText('uStepLabel', mc ? 'Prompt Step' : '训练步数');
    $('uOptimizerStat').hidden = !mc;
    setText('uOptimizerStep', current.optimizer_step ?? 0);
    setText('uStep', `${promptStep} / ${max || '-'}`);
    setText('uRoute', routeName(current.route));
    setText('uProgress', max ? `${progress.toFixed(1)}%` : '-');
    setText('uLoss', fmt(current.loss));
    setText('uGrad', fmt(current.grad_norm));
    const learningRate = current.learning_rate ?? state.manifest.config?.learning_rate;
    setText('uLr', learningRate == null ? '-' : Number(learningRate).toExponential(2));
    $('uProgressFill').style.width = `${progress}%`;
    $('userDemoNotice').hidden = !state.manifest.demo;
    configureOverviewCharts(mc);
    if (mc) {
      const summary = state.mcSummary || {};
      metricCards('userOverviewCards', isHybridMcRun() ? [
        ['Group Reward Mean', fmt(current.group_reward_mean)], ['Reward Spread', fmt(current.group_reward_spread)],
        ['Sequence Loss', fmt(current.sequence_loss)], ['Local Loss', fmt(current.local_loss)],
        ['Total Loss', fmt(current.total_loss)], ['Grad Norm', fmt(current.grad_norm)],
      ] : [
        ['Valid Candidate', pct(summary.valid_candidate_rate)], ['No-credit Prompt', pct(summary.no_credit_prompt_rate)],
        ['Projection Required', pct(summary.projection_required_rate)], ['Negative Rate · Action', pct(summary.action?.negative_candidate_rate)],
        ['Negative Rate · Chain', pct(summary.chain?.negative_candidate_rate)], ['Peak VRAM', summary.peak_vram_mib == null ? '-' : `${fmt(summary.peak_vram_mib, 0)} MiB`],
      ]);
      draw('userLossChart', [{data: points(state.metrics, 'loss'), color: colors[3]}, {data: points(state.metrics, 'grad_norm'), color: colors[0]}]);
      draw('userRewardChart', [
        {data: mcCandidatePoints(state.metrics.filter(row => row.route === 'action'), 'reward'), color: colors[0]},
        {data: mcCandidatePoints(state.metrics.filter(row => row.route === 'chain'), 'reward'), color: colors[3]},
      ]);
      const actionRows = state.metrics.filter(row => row.route === 'action');
      const chainRows = state.metrics.filter(row => row.route === 'chain');
      draw('userAdvantageChart', [
        ...(isHybridMcRun() ? [
          {data: points(state.metrics, 'group_reward_mean'), color: '#16835f'},
          {data: points(state.metrics, 'group_reward_std'), color: '#2d6cdf'},
          {data: points(state.metrics, 'sequence_loss'), color: '#b36b08'},
          {data: points(state.metrics, 'local_loss'), color: '#7654b5'},
        ] : [
          {data: mcTotalPoints(actionRows, 'positive_credit_mass'), color: '#16835f'},
          {data: mcTotalPoints(actionRows, 'negative_credit_mass'), color: '#bd3f4c'},
          {data: mcTotalPoints(chainRows, 'positive_credit_mass'), color: '#2d6cdf'},
          {data: mcTotalPoints(chainRows, 'negative_credit_mass'), color: '#b36b08'},
        ]),
      ]);
      draw('userMaskChart', [
        {data: mcCumulativeRouteRate('action', candidate => Number(candidate.negative_unit_count || 0) > 0), color: '#2d6cdf'},
        {data: mcCumulativeRouteRate('chain', candidate => Number(candidate.negative_unit_count || 0) > 0), color: '#bd3f4c'},
      ]);
      draw('userPolicyChart', [
        {data: points(state.metrics, 'active_unit_count'), color: '#16835f'},
        {data: points(state.metrics, 'active_token_count'), color: '#2d6cdf'},
        {data: state.metrics.map(row => [Number(row.step), row.optimizer_step_performed ? 1 : 0]), color: '#b36b08', dash: [3, 3]},
        {data: state.metrics.map(row => [Number(row.step), row.skipped_update ? 1 : 0]), color: '#bd3f4c', dash: [3, 3]},
      ]);
      draw('userVarianceChart', [
        {data: mcCumulativeHealth(candidate => Boolean(candidate.format_valid)), color: '#16835f'},
        {data: mcCumulativeHealth(candidate => Boolean(candidate.projection_required)), color: '#2d6cdf'},
        {data: mcCumulativeHealth(candidate => Number(candidate.overlap_token_count || 0) > 0), color: '#bd3f4c'},
      ]);
      return;
    }
    metricCards('userOverviewCards', [
      ['Ratio mean', fmt(current.ratio_mean)], ['Clip fraction', pct(current.clip_fraction)],
      ['Task reward std', fmt(current.task_reward_std)], ['Zero-std ratio', pct(current.zero_std_ratio)],
      ['Masked candidates', pct(current.masked_candidate_rate)], ['Masked tokens', pct(current.masked_token_rate)],
      ['Action F1', fmt(action.f1_mean)], ['Chain Reward', fmt(chain.total_reward_mean)],
    ]);
    draw('userLossChart', [{data: points(state.metrics, 'loss'), color: colors[3]}, {data: points(state.metrics, 'grad_norm'), color: colors[0]}]);
    draw('userRewardChart', [{data: points(state.metrics, 'task_reward_mean', 'action'), color: colors[0]}, {data: points(state.metrics, 'task_reward_mean', 'chain'), color: colors[3]}]);
    draw('userAdvantageChart', [
      {data: points(state.metrics, 'sequence_advantage_mean'), color: colors[1]},
      {data: points(state.metrics, 'sequence_advantage_std'), color: colors[0]},
      {data: points(state.metrics, 'token_advantage_mean'), color: colors[2]},
      {data: points(state.metrics, 'token_advantage_std'), color: colors[4]},
    ]);
    draw('userMaskChart', [{data: points(state.metrics, 'masked_candidate_rate'), color: colors[2]}, {data: points(state.metrics, 'masked_token_rate'), color: colors[3]}]);
    draw('userPolicyChart', [{data: points(state.metrics, 'ratio_mean'), color: colors[1]}, {data: points(state.metrics, 'clip_fraction'), color: colors[2]}]);
    draw('userVarianceChart', [{data: points(state.metrics, 'task_reward_std'), color: colors[0]}, {data: points(state.metrics, 'zero_std_ratio'), color: colors[3]}]);
  }

  function renderUserAction() {
    const rows = state.metrics.filter(row => row.route === 'action');
    const row = rows.at(-1) || {};
    if (isMcRun()) {
      const summary = state.mcSummary?.action || {};
      document.querySelector('#userAction .user-section-head h2').textContent = 'Action · SID Marginal Credit';
      metricCards('userActionCards', [
        ['Candidate Mean F1', fmt(summary.mean_f1 ?? summary.mean_reward)], ['Mean SID Count', fmt(summary.mean_predicted_sid_count, 1)], ['Negative Candidate Rate', pct(summary.negative_candidate_rate)],
        ['Positive credit mass', fmt(summary.positive_credit_mass)], ['Negative credit mass', fmt(summary.negative_credit_mass)],
      ]);
      setUserChartPanel('userActionScoreChart', isHybridMcRun() ? 'F1 / Precision / Recall' : 'Candidate Mean F1', isHybridMcRun() ? ['F1', 'Precision', 'Recall'] : ['Mean F1'], '当前 on-policy candidates 的 evaluator 均值；不等同于固定 Probe 泛化分数', ['#16835f', '#2d6cdf', '#b36b08']);
      setUserChartPanel('userActionExactChart', 'Predicted SID Count', ['Mean SID count'], '', ['#2d6cdf']);
      setUserChartPanel('userWrongSelectionChart', 'Positive / Negative Credit Mass', ['Positive mass', 'Negative mass'], '', ['#16835f', '#bd3f4c']);
      setUserChartPanel('userActionConstraintChart', 'Positive / Negative Unit Count', ['Positive units', 'Negative units'], '', ['#16835f', '#bd3f4c']);
      draw('userActionScoreChart', isHybridMcRun() ? [
        {data: mcCandidatePoints(rows, 'f1'), color: colors[1]},
        {data: mcCandidatePoints(rows, 'precision'), color: colors[0]},
        {data: mcCandidatePoints(rows, 'recall'), color: colors[2]},
      ] : [{data: mcCandidatePoints(rows, 'f1'), color: colors[1]}]);
      draw('userActionExactChart', [{data: mcCandidatePoints(rows, isHybridMcRun() ? 'predicted_sid_count' : 'predicted_sid_unit_count'), color: colors[0]}]);
      draw('userWrongSelectionChart', [
        {data: mcTotalPoints(rows, 'positive_credit_mass'), color: colors[1]},
        {data: mcTotalPoints(rows, 'negative_credit_mass'), color: colors[3]},
      ]);
      draw('userActionConstraintChart', [
        {data: mcTotalPoints(rows, 'positive_unit_count'), color: colors[1]},
        {data: mcTotalPoints(rows, 'negative_unit_count'), color: colors[3]},
      ]);
      return;
    }
    document.querySelector('#userAction .user-section-head h2').textContent = 'Action Set-F1';
    setUserChartPanel('userActionScoreChart', 'F1 / Precision / Recall', ['F1', 'Precision', 'Recall'], '', ['#16835f', '#2d6cdf', '#b36b08']);
    setUserChartPanel('userActionExactChart', 'Exact Match', ['Exact'], '', ['#16835f']);
    setUserChartPanel('userWrongSelectionChart', 'Wrong Selection', ['Candidate rate'], '', ['#bd3f4c']);
    setUserChartPanel('userActionConstraintChart', 'Hallucination / Duplicate', ['Hallucination', 'Duplicate'], '', ['#bd3f4c', '#b36b08']);
    metricCards('userActionCards', [
      ['F1', fmt(row.f1_mean)], ['Precision', fmt(row.precision_mean)], ['Recall', fmt(row.recall_mean)],
      ['Exact Match', fmt(row.exact_match_rate)], ['Wrong selection', pct(row.wrong_selection_candidate_rate)],
      ['Hallucination', pct(row.hallucination_candidate_rate)], ['Duplicate', pct(row.duplicate_candidate_rate)],
    ]);
    draw('userActionScoreChart', [{data: points(rows, 'f1_mean'), color: colors[1]}, {data: points(rows, 'precision_mean'), color: colors[0]}, {data: points(rows, 'recall_mean'), color: colors[2]}]);
    draw('userActionExactChart', [{data: points(rows, 'exact_match_rate'), color: colors[1]}]);
    draw('userWrongSelectionChart', [{data: points(rows, 'wrong_selection_candidate_rate'), color: colors[3]}]);
    draw('userActionConstraintChart', [{data: points(rows, 'hallucination_candidate_rate'), color: colors[3]}, {data: points(rows, 'duplicate_candidate_rate'), color: colors[2]}]);
  }

  function renderUserChain() {
    const rows = state.metrics.filter(row => row.route === 'chain');
    const row = rows.at(-1) || {};
    const counts = row.violation_counts || {};
    if (isMcRun()) {
      const summary = state.mcSummary?.chain || {};
      document.querySelector('#userChain .user-section-head h2').textContent = 'Chain · Event Marginal Credit';
      metricCards('userChainCards', [
        ['Mean Reward', fmt(summary.mean_reward)], ['Action Alignment', fmt(summary.mean_action_alignment)],
        ['Logic Alignment', fmt(summary.mean_logic_alignment)], ['Mean Event Count', fmt(summary.mean_predicted_event_count, 1)],
        ['Negative Candidate Rate', pct(summary.negative_candidate_rate)], ['Positive Credit Mass', fmt(summary.positive_credit_mass)],
        ['Negative Credit Mass', fmt(summary.negative_credit_mass)],
      ]);
      setUserChartPanel('userChainScoreChart', 'Total / Action / Logic', ['Total Reward', 'Action Alignment', 'Logic Alignment'], '', ['#16835f', '#2d6cdf', '#7654b5']);
      setUserChartPanel('userChainMismatchChart', 'Predicted Event Count', ['Mean event count'], '', ['#2d6cdf']);
      setUserChartPanel('userChainConstraintChart', 'Positive / Negative Credit Mass', ['Positive mass', 'Negative mass'], '', ['#16835f', '#bd3f4c']);
      setUserChartPanel('userGroundingChart', 'Positive / Negative Unit Count', ['Positive units', 'Negative units'], '', ['#16835f', '#bd3f4c']);
      draw('userChainScoreChart', [
        {data: mcCandidatePoints(rows, 'reward'), color: colors[1]},
        {data: mcCandidatePoints(rows, 'full_action_alignment'), color: colors[0]},
        {data: mcCandidatePoints(rows, 'full_logic_alignment'), color: colors[4]},
      ]);
      draw('userChainMismatchChart', [{data: mcCandidatePoints(rows, 'predicted_event_count'), color: colors[0]}]);
      draw('userChainConstraintChart', [
        {data: mcTotalPoints(rows, 'positive_credit_mass'), color: colors[1]},
        {data: mcTotalPoints(rows, 'negative_credit_mass'), color: colors[3]},
      ]);
      draw('userGroundingChart', [
        {data: mcTotalPoints(rows, 'positive_unit_count'), color: colors[1]},
        {data: mcTotalPoints(rows, 'negative_unit_count'), color: colors[3]},
      ]);
      return;
    }
    document.querySelector('#userChain .user-section-head h2').textContent = 'Chain Alignment';
    setUserChartPanel('userChainScoreChart', 'Total / Action / Logic', ['Total Reward', 'Action Alignment', 'Logic Alignment'], '', ['#16835f', '#2d6cdf', '#7654b5']);
    setUserChartPanel('userChainMismatchChart', 'Date / Action Mismatch', ['Date', 'Action'], '', ['#bd3f4c', '#b36b08']);
    setUserChartPanel('userChainConstraintChart', 'Chain Constraint Count', ['Hallucinated SID', 'Duplicate event', 'Chronology', 'Excess event'], '', ['#bd3f4c', '#2d6cdf', '#b36b08', '#7654b5']);
    setUserChartPanel('userGroundingChart', 'Grounding', ['Grounded', 'Partial', 'Ungrounded'], '', ['#16835f', '#b36b08', '#bd3f4c']);
    metricCards('userChainCards', [
      ['Total Reward', fmt(row.total_reward_mean)], ['Action Alignment', fmt(row.action_alignment_mean)],
      ['Logic Alignment', fmt(row.logic_alignment_mean)], ['Date mismatch', counts.date_mismatch == null ? '-' : String(counts.date_mismatch)],
      ['Action mismatch', counts.action_mismatch == null ? '-' : String(counts.action_mismatch)],
      ['Hallucinated SID', counts.hallucinated_sid == null ? '-' : String(counts.hallucinated_sid)],
      ['Duplicate event', counts.duplicate_event == null ? '-' : String(counts.duplicate_event)],
      ['Chronology', counts.chronology_violation == null ? '-' : String(counts.chronology_violation)],
      ['Excess event', counts.excess_event == null ? '-' : String(counts.excess_event)],
    ]);
    draw('userChainScoreChart', [{data: points(rows, 'total_reward_mean'), color: colors[1]}, {data: points(rows, 'action_alignment_mean'), color: colors[0]}, {data: points(rows, 'logic_alignment_mean'), color: colors[4]}]);
    draw('userChainMismatchChart', [{data: points(rows, 'date_mismatch_candidate_rate'), color: colors[3]}, {data: points(rows, 'action_mismatch_candidate_rate'), color: colors[2]}]);
    draw('userChainConstraintChart', [
      {data: objectPoint(rows, 'violation_counts', 'hallucinated_sid'), color: colors[3]},
      {data: objectPoint(rows, 'violation_counts', 'duplicate_event'), color: colors[0]},
      {data: objectPoint(rows, 'violation_counts', 'chronology_violation'), color: colors[2]},
      {data: objectPoint(rows, 'violation_counts', 'excess_event'), color: colors[4]},
    ]);
    draw('userGroundingChart', [{data: points(rows, 'grounded_rate'), color: colors[1]}, {data: points(rows, 'partially_grounded_rate'), color: colors[2]}, {data: points(rows, 'ungrounded_rate'), color: colors[3]}]);
  }

  function mcCreditSpanHtml(candidate) {
    const text = String(candidate.completion || '');
    const units = (candidate.credit_units || []).map((unit, index) => ({...unit, unit_index: unit.unit_index ?? index}));
    const spans = units.filter(unit => Number.isInteger(unit.char_start) && Number.isInteger(unit.char_end) && unit.char_start >= 0 && unit.char_end > unit.char_start && unit.char_end <= text.length);
    if (!spans.length) return `<pre class="mc-credit-completion">${escapeHtml(text || '该 candidate 没有 completion 文本')}</pre>`;
    const boundaries = [...new Set([0, text.length, ...spans.flatMap(unit => [unit.char_start, unit.char_end])])].sort((a, b) => a - b);
    let html = '';
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const start = boundaries[index], end = boundaries[index + 1];
      const active = spans.filter(unit => unit.char_start <= start && unit.char_end >= end);
      const chunk = escapeHtml(text.slice(start, end));
      if (!active.length) { html += chunk; continue; }
      const types = new Set(active.map(unit => unit.credit_type || (Number(unit.delta) > 0 ? 'positive' : Number(unit.delta) < 0 ? 'negative' : 'zero')));
      const css = types.size > 1 ? 'credit-overlap' : `credit-${[...types][0]}`;
      const indices = active.map(unit => unit.unit_index).join(',');
      html += `<button type="button" class="mc-unit-mark ${css}" data-unit-indices="${indices}" title="Unit ${indices}">${chunk}</button>`;
    }
    return `<pre class="mc-credit-completion">${html}</pre>`;
  }

  function mcUnitLabel(unit) {
    if (unit.sid) return `SID ${unit.sid} · occurrence ${unit.occurrence_index ?? unit.occurrence ?? '-'}`;
    return `Event ${unit.event_index ?? '-'} `;
  }

  function mcUnitEffectiveAdvantage(candidate, unit) {
    const indices = Array.isArray(unit.generated_token_indices) ? unit.generated_token_indices : [];
    const completionTokens = Number(candidate.generated_token_count ?? candidate.completion_length ?? candidate.completion_token_count ?? 0);
    const sequenceWeight = Number(state.manifest.sequence_weight ?? state.manifest.config?.sequence_weight ?? 1);
    const localWeight = Number(state.manifest.local_weight ?? state.manifest.config?.local_weight ?? 0.3);
    const candidateCount = Number(state.manifest.K ?? state.manifest.config?.K ?? 4);
    const sequencePerToken = completionTokens > 0 ? sequenceWeight * Number(candidate.sequence_advantage || 0) / completionTokens : 0;
    const localByToken = new Map();
    for (const candidateUnit of candidate.credit_units || []) {
      const unitIndices = Array.isArray(candidateUnit.generated_token_indices) ? candidateUnit.generated_token_indices : [];
      if (!unitIndices.length) continue;
      const contribution = localWeight * Number(candidateUnit.delta || 0) / unitIndices.length;
      unitIndices.forEach(tokenIndex => localByToken.set(tokenIndex, (localByToken.get(tokenIndex) || 0) + contribution));
    }
    const localValues = indices.map(tokenIndex => localByToken.get(tokenIndex) || 0);
    const combinedValues = localValues.map(value => sequencePerToken + value);
    const lossCoefficientValues = combinedValues.map(value => -value / candidateCount);
    return {sequencePerToken, localValues, combinedValues, lossCoefficientValues, localWeight, candidateCount};
  }

  function mcAdvantageRange(values) {
    if (!values.length) return '-';
    const minimum = Math.min(...values), maximum = Math.max(...values);
    return Math.abs(maximum - minimum) < 1e-12 ? fmt(minimum, 6) : `${fmt(minimum, 6)} .. ${fmt(maximum, 6)}`;
  }

  function mcUnitDetail(unit, candidate) {
    const indices = Array.isArray(unit.generated_token_indices) ? unit.generated_token_indices : [];
    const effective = mcUnitEffectiveAdvantage(candidate, unit);
    const extras = unit.sid
      ? `<dt>SID</dt><dd>${escapeHtml(unit.sid)}</dd><dt>Occurrence</dt><dd>${unit.occurrence_index ?? unit.occurrence ?? '-'}</dd>`
      : `<dt>Event index</dt><dd>${unit.event_index ?? '-'}</dd><dt>Δ Action</dt><dd>${fmt(unit.delta_action_alignment ?? unit.delta_action)}</dd><dt>Δ Logic</dt><dd>${fmt(unit.delta_logic_alignment ?? unit.delta_logic)}</dd>`;
    return `<dl class="mc-unit-detail-grid"><dt>Delta</dt><dd>${fmt(unit.delta)}</dd><dt>Credit type</dt><dd>${escapeHtml(unit.credit_type || 'zero')}</dd><dt>Token count</dt><dd>${indices.length}</dd><dt>Sequence A / token</dt><dd>${fmt(effective.sequencePerToken, 6)}</dd><dt>Aux A / token</dt><dd>${mcAdvantageRange(effective.localValues)}</dd><dt>Combined A / token</dt><dd>${mcAdvantageRange(effective.combinedValues)}</dd><dt>Loss coefficient / token</dt><dd>${mcAdvantageRange(effective.lossCoefficientValues)}</dd><dt>Objective weights</dt><dd>sequence 1.0 · local ${fmt(effective.localWeight)} · K${effective.candidateCount} mean</dd>${extras}</dl>`;
  }

  function renderMcTraceCandidate(candidate, candidateKey) {
    const units = candidate.credit_units || [];
    const overlap = candidate.overlap_metadata || candidate.overlaps || [];
    const overlapHtml = Array.isArray(overlap) && overlap.length
      ? `<div class="mc-overlap-list">${overlap.map(item => `<span>shared token ${escapeHtml(String(item.token_index ?? item.shared_token ?? '-'))} · units ${escapeHtml((item.unit_indices || []).join(',') || '-')} · mixed-sign ${item.mixed_sign ? 'yes' : 'no'} · net ${fmt(item.net_coefficient)}</span>`).join('')}</div>`
      : (Number(candidate.overlap_token_count || 0) > 0 ? `<div class="mc-overlap-list"><span>${candidate.overlap_token_count} shared token(s) · mixed-sign ${candidate.mixed_sign_overlap_token_count || 0}</span></div>` : '');
    const evaluatorScore = candidate.route === 'action'
      ? `F1 ${fmt(candidate.f1 ?? candidate.full_reward ?? candidate.reward)} · P/R ${fmt(candidate.precision)} / ${fmt(candidate.recall)}`
      : `Action ${fmt(candidate.full_action_alignment)} · Logic ${fmt(candidate.full_logic_alignment)}`;
    const hybridScore = isHybridMcRun()
      ? ` · A ${fmt(candidate.sequence_advantage)} · Lseq ${fmt(candidate.sequence_loss)} · Llocal ${fmt(candidate.local_loss)} · Ltotal ${fmt(candidate.total_loss)}`
      : '';
    return `<article class="mc-credit-candidate" data-candidate-key="${candidateKey}">
      <header><strong>${escapeHtml(routeName(candidate.route))} · Prompt ${candidate.prompt_step ?? candidate.step ?? '-'} · Candidate ${candidate.candidate_index ?? '-'}</strong><span>Reward ${fmt(candidate.full_reward ?? candidate.reward)} · ${evaluatorScore}${hybridScore} · ${units.length} units</span></header>
      ${mcCreditSpanHtml(candidate)}
      <div class="mc-unit-list">${units.map((unit, index) => `<button type="button" class="mc-unit-chip credit-${escapeHtml(unit.credit_type || 'zero')}" data-unit-index="${index}">${escapeHtml(mcUnitLabel(unit))} · Δ ${fmt(unit.delta)} · Aeff ${mcAdvantageRange(mcUnitEffectiveAdvantage(candidate, unit).combinedValues)}</button>`).join('') || '<span class="empty-inline">没有 marginal credit unit</span>'}</div>
      ${overlapHtml}<div class="mc-unit-detail" id="mcUnitDetail-${candidateKey}">${units.length ? mcUnitDetail(units[0], candidate) : '该 candidate 没有 unit-level credit。'}</div>
    </article>`;
  }

  function renderMcCredit() {
    const summary = state.mcSummary || {}, action = summary.action || {}, chain = summary.chain || {};
    metricCards('mcCreditCards', [
      ['Action Mean F1', fmt(action.mean_f1 ?? action.mean_reward)], ['Action Positive Mass', fmt(action.positive_credit_mass)], ['Action Negative Mass', fmt(action.negative_credit_mass)], ['Action Negative Rate', pct(action.negative_candidate_rate)],
      ['Chain Action Align', fmt(chain.mean_action_alignment)], ['Chain Logic Align', fmt(chain.mean_logic_alignment)], ['Chain Positive Mass', fmt(chain.positive_credit_mass)], ['Chain Negative Mass', fmt(chain.negative_credit_mass)],
    ]);
    const actionRows = state.metrics.filter(row => row.route === 'action'), chainRows = state.metrics.filter(row => row.route === 'chain');
    draw('mcActionCreditChart', [{data: mcTotalPoints(actionRows, 'positive_credit_mass'), color: colors[1]}, {data: mcTotalPoints(actionRows, 'negative_credit_mass'), color: colors[3]}]);
    draw('mcChainCreditChart', [{data: mcTotalPoints(chainRows, 'positive_credit_mass'), color: colors[1]}, {data: mcTotalPoints(chainRows, 'negative_credit_mass'), color: colors[3]}]);
    const rows = Array.isArray(state.rollouts) ? state.rollouts.filter(row => Array.isArray(row.credit_units)) : [];
    $('mcCreditNotice').textContent = rows.length
      ? `Rollout：当前训练样本 / on-policy / 参与训练。已落盘 ${rows.length} 个 candidate 的 unit-level marginal trace；前端只读，不重新计算 scorer 或 delta。`
      : '该历史 run 未落盘 unit-level marginal trace。可展示 aggregate credit；未来 MC run 已启用完整 trace。';
    $('mcCreditTracePanel').hidden = !rows.length;
    if (!rows.length) { $('mcCreditTrace').innerHTML = ''; return; }
    const routeSelect = $('mcCreditRoute'), oldRoute = routeSelect.value;
    if (oldRoute && !rows.some(row => row.route === oldRoute)) routeSelect.value = '';
    const routeRows = rows.filter(row => !routeSelect.value || row.route === routeSelect.value);
    const prompts = [...new Set(routeRows.map(row => Number(row.prompt_step ?? row.step)))].sort((a, b) => a - b);
    const oldPrompt = $('mcCreditPrompt').value;
    $('mcCreditPrompt').innerHTML = prompts.map(step => `<option value="${step}">Prompt ${step}</option>`).join('');
    $('mcCreditPrompt').value = prompts.includes(Number(oldPrompt)) ? oldPrompt : String(prompts.at(-1));
    const selected = routeRows.filter(row => Number(row.prompt_step ?? row.step) === Number($('mcCreditPrompt').value));
    $('mcCreditTrace').innerHTML = selected.map((candidate, index) => renderMcTraceCandidate(candidate, `${candidate.prompt_step ?? candidate.step}-${candidate.candidate_index ?? index}`)).join('');
    selected.forEach((candidate, candidateIndex) => {
      const key = `${candidate.prompt_step ?? candidate.step}-${candidate.candidate_index ?? candidateIndex}`;
      const root = document.querySelector(`[data-candidate-key="${key}"]`);
      root?.querySelectorAll('[data-unit-index]').forEach(button => button.onclick = () => {
        const unit = candidate.credit_units[Number(button.dataset.unitIndex)];
        $(`mcUnitDetail-${key}`).innerHTML = mcUnitDetail(unit, candidate);
      });
      root?.querySelectorAll('[data-unit-indices]').forEach(button => button.onclick = () => {
        const unit = candidate.credit_units[Number(button.dataset.unitIndices.split(',')[0])];
        $(`mcUnitDetail-${key}`).innerHTML = mcUnitDetail(unit, candidate);
      });
    });
  }

  function renderUserToken() {
    const mc = isMcRun();
    $('userTokenLegacy').hidden = mc;
    $('userMcCredit').hidden = !mc;
    if (mc) { renderMcCredit(); return; }
    const row = state.metrics.at(-1) || {};
    metricCards('userTokenCards', [
      ['Sequence advantage mean', fmt(row.sequence_advantage_mean)], ['Sequence advantage std', fmt(row.sequence_advantage_std)],
      ['Token advantage mean', fmt(row.token_advantage_mean)], ['Token advantage std', fmt(row.token_advantage_std)],
      ['Masked candidates', pct(row.masked_candidate_rate)], ['Masked tokens', pct(row.masked_token_rate)],
      ['Positive sequence flips', row.positive_sequence_masked_token_flip_count == null ? '-' : String(row.positive_sequence_masked_token_flip_count)],
    ]);
    const countSeries = USER_KINDS.map(([kind], index) => ({data: objectPoint(state.metrics, 'per_kind_masked_token_count', kind), color: colors[index % colors.length]}));
    const massSeries = USER_KINDS.map(([kind], index) => ({data: objectPoint(state.metrics, 'per_kind_incremental_negative_mass', kind), color: colors[index % colors.length]}));
    $('userKindCountLegend').innerHTML = USER_KINDS.map(([kind], index) => `<span class="key" style="--c:${colors[index % colors.length]}">${kind}</span>`).join('');
    $('userKindMassLegend').innerHTML = $('userKindCountLegend').innerHTML;
    draw('userKindCountChart', countSeries);
    draw('userKindMassChart', massSeries);
    const latestByKind = new Map();
    for (const metric of state.metrics) {
      for (const [kind] of USER_KINDS) {
        if (metric.per_kind_masked_token_count?.[kind] != null || metric.per_kind_incremental_negative_mass?.[kind] != null) latestByKind.set(kind, metric);
      }
    }
    const masses = USER_KINDS.map(([kind]) => Number(latestByKind.get(kind)?.per_kind_incremental_negative_mass?.[kind] || 0));
    const totalMass = masses.reduce((sum, value) => sum + value, 0);
    $('userPenaltyRows').innerHTML = USER_KINDS.map(([kind, route], index) => {
      const metric = latestByKind.get(kind) || {};
      const count = metric.per_kind_masked_token_count?.[kind];
      const mass = metric.per_kind_incremental_negative_mass?.[kind];
      return `<tr><td><span class="penalty-kind">${kind}</span><span class="penalty-route">${route}</span></td><td>${route}</td><td>${count ?? '-'}</td><td>${mass == null ? '-' : fmt(mass)}</td><td>${mass == null || !totalMass ? '-' : pct(Number(mass) / totalMass)}</td></tr>`;
    }).join('');
  }

  function highlightedCompletion(candidate) {
    const text = String(candidate.completion || '');
    const valid = span => Number.isInteger(span.start) && Number.isInteger(span.end) && span.start >= 0 && span.end > span.start && span.end <= text.length;
    const spans = [
      ...(Array.isArray(candidate.match_spans) ? candidate.match_spans.filter(valid).map(span => ({...span, priority: 1, css: 'match-mark'})) : []),
      ...(Array.isArray(candidate.masked_spans) ? candidate.masked_spans.filter(valid).map(span => ({...span, priority: 2, css: 'penalty-mark'})) : []),
    ];
    if (!spans.length) return escapeHtml(text);
    const boundaries = [...new Set([0, text.length, ...spans.flatMap(span => [span.start, span.end])])].sort((a, b) => a - b);
    let html = '';
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const start = boundaries[index], end = boundaries[index + 1];
      const active = spans.filter(span => span.start <= start && span.end >= end).sort((a, b) => b.priority - a.priority)[0];
      const chunk = escapeHtml(text.slice(start, end));
      html += active ? `<mark class="${active.css}" title="${escapeHtml(active.kind || (active.priority === 2 ? 'local penalty' : 'correct match'))}">${chunk}</mark>` : chunk;
    }
    return html;
  }

  function renderUserCandidate(candidate, route) {
    const violations = Array.isArray(candidate.violations) ? candidate.violations : [];
    const counts = new Map();
    for (const item of violations) {
      const kind = typeof item === 'string' ? item : item.kind || JSON.stringify(item);
      counts.set(kind, (counts.get(kind) || 0) + 1);
    }
    const penaltyKinds = new Set(candidate.penalty_kinds || (candidate.masked_spans || []).map(span => span.kind));
    const violationHtml = counts.size ? [...counts].map(([kind, count]) => `<span class="violation-chip ${penaltyKinds.has(kind) ? 'penalized' : 'diagnostic'}" title="${penaltyKinds.has(kind) ? '进入局部 token penalty' : '仅参与主 Reward 或诊断'}">${escapeHtml(kind)}${count > 1 ? ` ×${count}` : ''}</span>`).join('') : '<span class="violation-chip none">无 violation</span>';
    const scoreHtml = route === 'action'
      ? `<div><div class="label">F1 / P / R</div>${fmt(candidate.f1)} / ${fmt(candidate.precision)} / ${fmt(candidate.recall)}</div>`
      : `<div><div class="label">Action / Logic</div>${fmt(candidate.action_alignment)} / ${fmt(candidate.logic_alignment)}</div>`;
    const goldSet = new Set(candidate.gold_sids || []);
    const sidHtml = route === 'action' && (candidate.gold_sids || candidate.pred_sids)
      ? `<div class="sid-list">Gold: ${escapeHtml((candidate.gold_sids || []).join('；') || '-')}<br>Pred: ${(candidate.pred_sids || []).map(sid => `<span class="${goldSet.has(sid) ? 'correct-inline' : ''}">${escapeHtml(sid)}</span>`).join('；') || '-'}</div>` : '';
    return `<div class="candidate user-candidate"><div><span class="badge">#${candidate.candidate_id ?? '-'}</span></div><div><div class="text user-completion">${highlightedCompletion(candidate)}</div>${sidHtml}<div class="violation-list">${violationHtml}</div></div><div class="user-candidate-meta"><div><div class="label">Reward / Sequence A</div>${fmt(candidate.reward)} / ${fmt(candidate.sequence_advantage)}</div>${scoreHtml}</div><div class="user-candidate-meta"><div><div class="label">Masked tokens</div>${candidate.masked_token_count ?? 0}</div><div><div class="label">Completion tokens</div>${candidate.completion_length ?? '-'}</div></div></div>`;
  }

  function userProbePoints(route, path) {
    const grouped = new Map();
    for (const row of state.probes.filter(item => item.route === route)) {
      const value = path.split('.').reduce((object, key) => object?.[key], row);
      if (value == null) continue;
      if (!grouped.has(Number(row.step))) grouped.set(Number(row.step), []);
      grouped.get(Number(row.step)).push(Number(value));
    }
    return [...grouped].sort((a, b) => a[0] - b[0]).map(([step, values]) => [step, values.reduce((sum, value) => sum + value, 0) / values.length]);
  }

  function userProbeDelta(points) {
    return points.length > 1 ? points.at(-1)[1] - points[0][1] : null;
  }

  function renderUserSampleContext(selected, route) {
    const prompt = String(selected.prompt || '当前记录没有可用输入样本');
    const gold = route === 'chain'
      ? JSON.stringify(selected.gold_events || [], null, 2)
      : (selected.gold_sids || []).join('\n');
    return `<section class="user-probe-context" aria-label="${route === 'chain' ? 'Chain' : 'Action'} 样本上下文">
      <div class="probe-context-block"><div class="probe-context-title">输入样本</div><pre>${escapeHtml(prompt)}</pre></div>
      <div class="probe-context-block"><div class="probe-context-title">Ground Truth · ${route === 'chain' ? 'Chain Events' : 'Action SIDs'}</div><pre>${escapeHtml(gold || '-')}</pre></div>
    </section>`;
  }

  function mcProbeRows() {
    if (!state.userLightProbe?.available) return [];
    return (state.userLightProbe.checkpoints || []).map(checkpoint => {
      const summary = checkpoint.summary || checkpoint;
      return {
        step: Number(checkpoint.step ?? summary.step),
        action: summary.action || {},
        chain: summary.chain || {},
        proxy: Number(summary.overall_user_proxy ?? summary.user_proxy),
      };
    }).sort((a, b) => a.step - b.step);
  }

  function signed(value) {
    return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${fmt(Number(value))}` : '-';
  }

  function guardMetric(row, key) {
    const source = row?.summary || row?.metrics || row || {};
    return source[key] ?? source.overall?.[key];
  }

  function renderMcRecommendationGuard() {
    const guard = state.recommendationGuard || {};
    if (!guard.available) {
      $('mcRecommendationGuard').innerHTML = '<div class="empty mc-empty">尚未执行 Recommendation guard evaluation</div>';
      return;
    }
    const rows = Array.isArray(guard.checkpoints) ? guard.checkpoints : [guard];
    const baseline = rows[0] || {}, latest = rows.at(-1) || {};
    const metrics = [
      ['Hit Rate', 'hit_rate'], ['History Copy Rate', 'history_copy_rate'], ['Unique History Copy Rate', 'unique_history_copy_rate'],
    ];
    $('mcRecommendationGuard').innerHTML = `<div class="guard-grid">${metrics.map(([label, key]) => {
      const current = guardMetric(latest, key), base = guardMetric(baseline, key);
      const delta = current != null && base != null ? Number(current) - Number(base) : guardMetric(guard.deltas, `delta_${key}`);
      return `<div class="guard-metric"><span>${label}</span><strong>${current == null ? '-' : pct(current)}</strong><small>vs BETA ${delta == null ? '-' : signed(delta)}</small></div>`;
    }).join('')}</div>`;
  }

  function mcProbeSchedule() {
    const schedule = state.userLightProbe?.checkpoint_schedule;
    return Array.isArray(schedule) && schedule.length ? schedule.map(Number) : [0, 128, 256, 384, 512];
  }

  function mcProbeStepLabel(step) {
    if (Number(step) !== 0) return `Step ${step}`;
    const checkpoint = (state.userLightProbe?.checkpoints || []).find(item => Number(item.step) === 0);
    const queueItem = (state.userLightProbe?.items || []).find(item => Number(item.step) === 0);
    return checkpoint?.name || queueItem?.label || state.userLightProbe?.baseline_label || 'BETA';
  }

  function mcProbeQueueState(step, evaluated) {
    if (evaluated.has(Number(step))) return {className: 'done', label: 'Done'};
    const item = (state.userLightProbe?.items || []).find(value => Number(value.step) === Number(step));
    const status = String(item?.status || 'waiting').toLowerCase();
    if (status === 'ready') return {className: 'ready', label: 'Ready for Probe'};
    if (status === 'evaluating') return {className: 'evaluating', label: 'Evaluating'};
    if (status === 'pending') return {className: 'pending', label: 'Pending'};
    return {className: 'waiting', label: 'Waiting'};
  }

  function renderMcProbeSampleDetail() {
    const checkpoints = state.userLightProbe?.checkpoints || [];
    const routeSelect = $('mcProbeSampleRoute');
    const sampleSelect = $('mcProbeSampleId');
    const stepSelect = $('mcProbeSampleStep');
    if (!checkpoints.length) {
      const baselineLabel = mcProbeStepLabel(0);
      sampleSelect.innerHTML = `<option value="">等待 ${escapeHtml(baselineLabel)}</option>`;
      stepSelect.innerHTML = `<option value="">等待 ${escapeHtml(baselineLabel)}</option>`;
      $('mcProbeSampleDetail').innerHTML = `<div class="empty mc-empty">${escapeHtml(baselineLabel)} 固定探针完成后可查看 6 个固定样本及其真实 completion。</div>`;
      return;
    }
    const route = routeSelect.value || 'action';
    const baselineSamples = (checkpoints.find(item => Number(item.step) === 0) || checkpoints[0]).samples || [];
    const sampleIds = baselineSamples.filter(sample => sample.route === route).map(sample => String(sample.sample_id));
    const oldSample = sampleSelect.value;
    sampleSelect.innerHTML = sampleIds.map((sampleId, index) => `<option value="${escapeHtml(sampleId)}">${route === 'action' ? 'Action' : 'Chain'} ${index + 1} · ${escapeHtml(sampleId.slice(0, 12))}</option>`).join('') || '<option value="">暂无样本</option>';
    if (sampleIds.includes(oldSample)) sampleSelect.value = oldSample;
    const sampleId = sampleSelect.value;
    const available = checkpoints.filter(checkpoint => (checkpoint.samples || []).some(sample => String(sample.sample_id) === sampleId));
    const oldStep = stepSelect.value;
    stepSelect.innerHTML = [...available].reverse().map(checkpoint => `<option value="${checkpoint.step}">${mcProbeStepLabel(checkpoint.step)}</option>`).join('') || '<option value="">等待评估</option>';
    if (available.some(checkpoint => String(checkpoint.step) === oldStep)) stepSelect.value = oldStep;
    const checkpoint = available.find(item => String(item.step) === stepSelect.value) || available.at(-1);
    const sample = checkpoint?.samples?.find(item => String(item.sample_id) === sampleId);
    if (!sample) {
      $('mcProbeSampleDetail').innerHTML = '<div class="empty mc-empty">当前固定样本尚无评估结果。</div>';
      return;
    }
    const candidates = sample.candidates || [];
    const candidateHtml = candidates.map(candidate => {
      const gold = new Set(candidate.gold_sids || sample.gold_sids || []);
      const sidHtml = route === 'action'
        ? `<div class="sid-list">Gold: ${escapeHtml([...(candidate.gold_sids || sample.gold_sids || [])].join('；') || '-')}<br>Pred: ${(candidate.pred_sids || []).map(sid => `<span class="${gold.has(sid) ? 'correct-inline' : ''}">${escapeHtml(sid)}</span>`).join('；') || '-'}</div>`
        : '';
      const score = route === 'action'
        ? `F1 ${fmt(candidate.f1)} · P ${fmt(candidate.precision)} · R ${fmt(candidate.recall)}`
        : `Total ${fmt(candidate.total_reward ?? candidate.reward)} · Action ${fmt(candidate.action_alignment)} · Logic ${fmt(candidate.logic_alignment)} · Events ${candidate.predicted_event_count ?? '-'}`;
      return `<article class="mc-probe-candidate"><header><strong>Candidate ${candidate.candidate_id ?? '-'}</strong><span>${score} · ${candidate.completion_length ?? '-'} tokens</span></header><pre>${escapeHtml(candidate.completion || '')}</pre>${sidHtml}</article>`;
    }).join('');
    $('mcProbeSampleDetail').innerHTML = `${renderUserSampleContext(sample, route)}<div class="mc-probe-sample-meta"><strong>${mcProbeStepLabel(checkpoint.step)}</strong><span>Mean reward ${fmt(sample.mean_reward)}</span><span>Δ vs ${escapeHtml(mcProbeStepLabel(0))} ${signed(sample.relative_to_parent_delta ?? sample.relative_to_beta_delta)}</span><span>Seed ${sample.seed ?? '-'}</span></div>${candidateHtml}`;
  }

  function renderMcProbe() {
    const rows = mcProbeRows();
    const schedule = mcProbeSchedule();
    const evaluated = new Set(rows.map(row => row.step));
    const baselineLabel = mcProbeStepLabel(0);
    const status = state.userLightProbe?.status || 'WAITING_FOR_PARENT';
    $('mcProbeNotice').textContent = rows.length
      ? `固定 3+3、独立 inference-only sidecar、不参与训练。当前状态：${status}；结果为本地趋势代理，不是官方分数。`
      : `固定 3+3 inference-only sidecar 尚未写入 ${escapeHtml(baselineLabel)} 结果；该探针不参与 reward、loss 或 optimizer。`;
    $('mcProbeSchedule').innerHTML = schedule.map(step => {
      const queueState = mcProbeQueueState(step, evaluated);
      return `<span class="mc-probe-stage ${queueState.className}"><strong>${mcProbeStepLabel(step)}</strong> · ${queueState.label}</span>`;
    }).join('');
    $('mcProbeCharts').hidden = !rows.length;
    if (!rows.length) {
      $('mcProbeCards').innerHTML = '';
      $('mcProbeTable').innerHTML = `<div class="empty mc-empty">等待 ${escapeHtml(baselineLabel)} 固定探针先完成。</div>`;
      $('mcProbeDeltas').innerHTML = '';
      renderMcProbeSampleDetail();
      renderMcRecommendationGuard();
      return;
    }
    const baseline = rows[0], latest = rows.at(-1);
    const delta = (getter) => getter(latest) - getter(baseline);
    metricCards('mcProbeCards', [
      ['Action F1', fmt(latest.action.f1)], ['ΔAction', signed(delta(row => Number(row.action.f1)))],
      ['Chain Total', fmt(latest.chain.total_reward)], ['ΔChain', signed(delta(row => Number(row.chain.total_reward)))],
      ['ΔChainAction', signed(delta(row => Number(row.chain.action_alignment)))], ['ΔChainLogic', signed(delta(row => Number(row.chain.logic_alignment)))],
      ['User Proxy', fmt(latest.proxy)], ['ΔProxy', signed(delta(row => row.proxy))],
    ]);
    draw('mcProbeActionChart', [
      {data: rows.map(row => [row.step, Number(row.action.f1)]), color: colors[1]},
      {data: rows.map(row => [row.step, row.proxy]), color: colors[0]},
    ]);
    draw('mcProbeChainChart', [{data: rows.map(row => [row.step, Number(row.chain.total_reward)]), color: colors[4]}]);
    const headers = ['Checkpoint', 'Action F1', 'Precision', 'Recall', 'Chain Total', 'Chain Action', 'Chain Logic', 'User Proxy'];
    const byStep = new Map(rows.map(row => [row.step, row]));
    $('mcProbeTable').innerHTML = `<table class="penalty-table mc-probe-table"><thead><tr>${headers.map(value => `<th>${value}</th>`).join('')}</tr></thead><tbody>${schedule.map(step => {
      const row = byStep.get(step);
      return row ? `<tr><td>${mcProbeStepLabel(step)}</td><td>${fmt(row.action.f1)}</td><td>${fmt(row.action.precision)}</td><td>${fmt(row.action.recall)}</td><td>${fmt(row.chain.total_reward)}</td><td>${fmt(row.chain.action_alignment)}</td><td>${fmt(row.chain.logic_alignment)}</td><td>${fmt(row.proxy)}</td></tr>` : `<tr class="mc-probe-waiting-row"><td>${mcProbeStepLabel(step)}</td><td colspan="7">Waiting for adapter-only checkpoint</td></tr>`;
    }).join('')}</tbody></table>`;
    $('mcProbeDeltas').innerHTML = rows.slice(1).map(row => `<div><strong>${mcProbeStepLabel(row.step)} vs ${escapeHtml(baselineLabel)}</strong><span>ΔAction ${signed(Number(row.action.f1) - Number(baseline.action.f1))}</span><span>ΔChain ${signed(Number(row.chain.total_reward) - Number(baseline.chain.total_reward))}</span><span>ΔChainAction ${signed(Number(row.chain.action_alignment) - Number(baseline.chain.action_alignment))}</span><span>ΔChainLogic ${signed(Number(row.chain.logic_alignment) - Number(baseline.chain.logic_alignment))}</span><span>ΔProxy ${signed(row.proxy - baseline.proxy)}</span></div>`).join('') || `<div class="empty-inline">${escapeHtml(baselineLabel)} 已完成；等待首个 checkpoint 后计算 paired delta。</div>`;
    renderMcProbeSampleDetail();
    renderMcRecommendationGuard();
  }

  function renderUserProbe() {
    const mc = isMcRun();
    $('userProbeLegacy').hidden = mc;
    $('userMcProbe').hidden = !mc;
    if (mc) { renderMcProbe(); return; }
    const actionF1 = userProbePoints('action', 'action.f1_mean');
    const actionP = userProbePoints('action', 'action.precision_mean');
    const actionR = userProbePoints('action', 'action.recall_mean');
    const chainTotal = userProbePoints('chain', 'chain.total_reward_mean');
    const chainAction = userProbePoints('chain', 'chain.action_alignment_mean');
    const chainLogic = userProbePoints('chain', 'chain.logic_alignment_mean');
    draw('userProbeActionChart', [{data: actionF1, color: colors[1]}, {data: actionP, color: colors[0]}, {data: actionR, color: colors[2]}]);
    draw('userProbeChainChart', [{data: chainTotal, color: colors[1]}, {data: chainAction, color: colors[0]}, {data: chainLogic, color: colors[4]}]);
    const latest = points => points.at(-1)?.[1];
    const deltaText = points => points.length > 1 ? `${userProbeDelta(points) >= 0 ? '+' : ''}${fmt(userProbeDelta(points))}` : '-';
    metricCards('userProbeCards', [
      ['Action F1', fmt(latest(actionF1))], ['Action Δ vs Step 0', deltaText(actionF1)],
      ['Chain Total', fmt(latest(chainTotal))], ['Chain Total Δ', deltaText(chainTotal)],
      ['Chain Action', fmt(latest(chainAction))], ['Chain Action Δ', deltaText(chainAction)],
      ['Chain Logic', fmt(latest(chainLogic))], ['Chain Logic Δ', deltaText(chainLogic)],
    ]);
    const probeRows = state.probes.filter(row => ['action', 'chain'].includes(row.route));
    $('userProbeNotice').textContent = probeRows.length
      ? `固定 ${new Set(probeRows.map(row => row.group_id)).size} 个样本 · ${new Set(probeRows.map(row => row.step)).size} 个时间点 · 绿色为正确命中，红色为局部 penalty`
      : '当前实验尚无 User 固定 Probe 数据；历史训练曲线不等价于固定样本评估。';
    const route = $('userProbeRoute').value || 'action';
    const routeRows = probeRows.filter(row => row.route === route);
    const groups = [...new Set(routeRows.map(row => row.group_id))];
    const oldGroup = $('userProbeGroup').value;
    $('userProbeGroup').innerHTML = groups.length ? groups.map((group, index) => `<option value="${escapeHtml(group)}">${route === 'action' ? 'Action' : 'Chain'} ${index + 1} · ${escapeHtml(group.slice(0, 12))}</option>`).join('') : '<option value="">暂无样本</option>';
    if (groups.includes(oldGroup)) $('userProbeGroup').value = oldGroup;
    const group = $('userProbeGroup').value;
    const available = routeRows.filter(row => row.group_id === group).sort((a, b) => a.step - b.step);
    const oldStep = $('userProbeStep').value;
    $('userProbeStep').innerHTML = available.length ? [...available].reverse().map(row => `<option value="${row.step}">Step ${row.step} · ${row.reason === 'baseline' ? '基线' : row.reason === 'final' ? '最终' : '定期'}</option>`).join('') : '<option value="">暂无记录</option>';
    if (available.some(row => String(row.step) === oldStep)) $('userProbeStep').value = oldStep;
    const selected = available.find(row => String(row.step) === $('userProbeStep').value) || available.at(-1);
    if (!selected) { $('userProbeDetail').innerHTML = '<div class="empty">没有可展示的固定 Probe 样例</div>'; return; }
    const source = available.find(row => row.prompt) || selected;
    const score = route === 'action'
      ? `F1 ${fmt(selected.action?.f1_mean)} · P ${fmt(selected.action?.precision_mean)} · R ${fmt(selected.action?.recall_mean)}`
      : `Total ${fmt(selected.chain?.total_reward_mean)} · Action ${fmt(selected.chain?.action_alignment_mean)} · Logic ${fmt(selected.chain?.logic_alignment_mean)}`;
    $('userProbeDetail').innerHTML = `<div class="trace-head">Step ${selected.step} · ${score}<span class="candidate-count">${selected.candidates?.length || 0}/4 candidates</span></div>${renderUserSampleContext({...selected, prompt: source.prompt}, route)}<div class="trace user-trace">${(selected.candidates || []).map(candidate => renderUserCandidate(candidate, route)).join('')}</div>`;
  }

  function renderUserTraceIndex() {
    const current = $('rolloutSelect').value;
    const lane = route => {
      const traces = state.traces.filter(trace => trace.route === route).sort((a, b) => a.rollout_id - b.rollout_id);
      const traceByRollout = new Map(traces.map(trace => [Number(trace.rollout_id), trace]));
      const rollouts = state.rollouts.filter(rollout => rollout.route === route);
      if (!rollouts.length) return `<div class="trace-lane"><div class="trace-lane-title ${route === 'action' ? 'user-route-action' : 'user-route-chain'}">${routeName(route)}</div><span class="trace-none">尚无记录</span></div>`;
      const selected = rollouts.find(rollout => userRolloutKey(rollout) === current);
      const rollout = selected || rollouts.at(-1);
      const key = userRolloutKey(rollout);
      const trace = traceByRollout.get(Number(rollout.rollout_id));
      const complete = Boolean(trace) || Array.isArray(rollout.credit_units);
      const completeCount = rollouts.filter(row => traceByRollout.has(Number(row.rollout_id)) || Array.isArray(row.credit_units)).length;
      const detail = trace ? `${trace.candidates?.length ?? 0} candidates` : complete ? '完整样本' : '仅汇总';
      const button = `<button class="trace-link ${complete ? '' : 'summary-only '}${key === current ? 'active' : ''}" type="button" data-rollout-id="${escapeHtml(key)}" data-route="${route}">第 ${rollout.step ?? '-'} 步 · #${rollout.rollout_id ?? '-'} · ${detail}</button>`;
      return `<div class="trace-lane"><div class="trace-lane-title ${route === 'action' ? 'user-route-action' : 'user-route-chain'}">${routeName(route)}</div><div class="trace-lane-compact"><div class="trace-links">${button}</div><span class="trace-count">共 ${rollouts.length} 条 · ${completeCount} 条完整</span></div></div>`;
    };
    $('traceIndex').innerHTML = lane('action') + lane('chain');
  }

  function requestUserSampleContext(sampleId) {
    const key = `${state.activeRun}:${sampleId}`;
    if (userSampleContextCache.has(key)) return;
    userSampleContextCache.set(key, null);
    fetch(`${apiUrl('/api/sample-context')}&sample_id=${encodeURIComponent(sampleId)}`, {cache: 'no-store'})
      .then(response => response.ok ? response.json() : null)
      .then(context => {
        userSampleContextCache.set(key, context || false);
        if (isUserRun() && String($('rolloutSelect').value)) {
          const scrollState = captureMonitorScrollState();
          renderUserExplorer();
          restoreMonitorScrollState(scrollState);
        }
      })
      .catch(() => userSampleContextCache.set(key, false));
  }

  function renderUserExplorer() {
    const selectedKey = $('rolloutSelect').value;
    const route = $('routeSelect').value;
    const rollout = state.rollouts.find(row => userRolloutKey(row) === selectedKey && (!route || row.route === route));
    if (!rollout) {
      $('rolloutSummary').innerHTML = '<div class="empty">请选择一个 User rollout</div>';
      $('trace').innerHTML = '';
      return;
    }
    if (isMcRun() && Array.isArray(rollout.credit_units)) {
      const sampleId = String(rollout.sample_id || '');
      const contextKey = sampleId ? `${state.activeRun}:${sampleId}` : '';
      const context = contextKey ? userSampleContextCache.get(contextKey) : false;
      if (sampleId && !userSampleContextCache.has(contextKey)) requestUserSampleContext(sampleId);
      $('rolloutSummary').innerHTML = [
        ['路由', routeName(rollout.route)], ['Prompt', String(rollout.prompt_step ?? rollout.step ?? '-')],
        ['Candidate', String(rollout.candidate_index ?? '-')], ['Reward', fmt(rollout.full_reward ?? rollout.reward)],
        [rollout.route === 'action' ? 'F1' : 'Action / Logic', rollout.route === 'action' ? fmt(rollout.f1) : `${fmt(rollout.full_action_alignment)} / ${fmt(rollout.full_logic_alignment)}`],
        ['Completion tokens', String(rollout.generated_token_count ?? '-')],
        ...(isHybridMcRun() ? [
          ['Sequence A', fmt(rollout.sequence_advantage)], ['Sequence Loss', fmt(rollout.sequence_loss)],
          ['Local Loss', fmt(rollout.local_loss)], ['Total Loss', fmt(rollout.total_loss)],
        ] : []),
      ].map(([label, value]) => `<div class="stat"><div class="label">${label}</div><div class="value small">${escapeHtml(value)}</div></div>`).join('');
      const contextHtml = context
        ? renderUserSampleContext(context, rollout.route)
        : context === false
          ? '<div class="empty">该记录没有可恢复的输入样本与 Ground Truth。</div>'
          : '<div class="empty">正在读取输入样本与 Ground Truth...</div>';
      $('trace').classList.add('user-trace');
      $('trace').innerHTML = `${contextHtml}${renderMcTraceCandidate(rollout, `${rollout.prompt_step ?? rollout.step}-${rollout.candidate_index ?? 0}`)}`;
      return;
    }
    const id = Number(rollout.rollout_id);
    const trace = state.traces.find(row => row.rollout_id === id && (!route || row.route === route));
    const candidates = Array.isArray(trace?.candidates) ? trace.candidates : [];
    const meanCandidateValue = key => {
      const values = candidates.map(candidate => Number(candidate[key])).filter(Number.isFinite);
      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    };
    const actionF1 = meanCandidateValue('f1');
    const chainAction = meanCandidateValue('action_alignment');
    const chainLogic = meanCandidateValue('logic_alignment');
    const stats = [
      ['路由', routeName(rollout.route)], ['平均 Reward', fmt(rollout.reward_mean)], ['Reward Std', fmt(rollout.reward_std)],
      ['Zero-std', pct(rollout.zero_std_ratio)], ['Masked candidates', pct(rollout.masked_candidate_rate)],
      ['Masked tokens', pct(rollout.masked_token_rate)], ['平均长度', fmt(rollout.completion_length_mean, 1)], ['G', String(rollout.g ?? state.manifest.G ?? 4)],
    ];
    if (rollout.route === 'action') stats.splice(2, 0, ['平均 F1', actionF1 == null ? '未落盘' : fmt(actionF1)]);
    if (rollout.route === 'chain') stats.splice(2, 0, ['平均 Action / Logic', chainAction == null || chainLogic == null ? '未落盘' : `${fmt(chainAction)} / ${fmt(chainLogic)}`]);
    $('rolloutSummary').innerHTML = stats.map(([label, value]) => `<div class="stat"><div class="label">${label}</div><div class="value small">${escapeHtml(value)}</div></div>`).join('');
    if (!trace) {
      const sampleIds = Array.isArray(rollout.group_ids) ? rollout.group_ids : [];
      const sampleId = sampleIds[0];
      const contextKey = sampleId ? `${state.activeRun}:${sampleId}` : '';
      const context = contextKey ? userSampleContextCache.get(contextKey) : false;
      if (sampleId && !userSampleContextCache.has(contextKey)) requestUserSampleContext(sampleId);
      const contextHtml = context
        ? `<div class="trace-head">代表输入 1/${sampleIds.length} · ${escapeHtml(sampleId.slice(0, 12))}</div>${renderUserSampleContext(context, rollout.route)}`
        : context === false
          ? '<div class="empty">该汇总没有可恢复的输入上下文。</div>'
          : '<div class="empty">正在读取该 rollout 的代表输入...</div>';
      $('trace').innerHTML = `<div class="empty">该时间步保存了 rollout 汇总，但未保存 candidate trace，因此没有历史候选输出可展示。</div>${contextHtml}`;
      return;
    }
    const expected = Number(state.manifest.G || 4);
    $('trace').classList.add('user-trace');
    $('trace').innerHTML = `<div class="trace-head">${escapeHtml(trace.group_id || '-')} · ${escapeHtml(routeName(trace.route))}<span class="candidate-count ${candidates.length === expected ? '' : 'incomplete'}">${candidates.length}/${expected} candidates</span></div>${renderUserSampleContext(trace, trace.route)}${candidates.map(candidate => renderUserCandidate(candidate, trace.route)).join('') || '<div class="empty">trace 中没有 candidate</div>'}`;
  }

  function userRolloutKey(row) {
    if (row.rollout_id != null) return String(row.rollout_id);
    return `mc:${row.prompt_step ?? row.step ?? 0}:${row.candidate_index ?? 0}`;
  }

  function renderUserExplorerOptions() {
    const select = $('rolloutSelect');
    const current = select.value;
    const traceIds = new Set(state.traces.map(trace => Number(trace.rollout_id)));
    select.innerHTML = [...state.rollouts].reverse().map(rollout => {
      const recorded = traceIds.has(Number(rollout.rollout_id)) || Array.isArray(rollout.credit_units);
      const key = userRolloutKey(rollout);
      const label = rollout.rollout_id != null ? `#${rollout.rollout_id}` : `Prompt ${rollout.prompt_step ?? rollout.step} · Candidate ${rollout.candidate_index ?? '-'}`;
      return `<option value="${escapeHtml(key)}">${escapeHtml(label)} · ${escapeHtml(routeName(rollout.route))} · 第 ${rollout.step} 步 · ${recorded ? '完整样本' : '仅汇总'}</option>`;
    }).join('');
    if (current && [...select.options].some(option => option.value === current)) {
      select.value = current;
    } else {
      const latestTrace = [...state.traces].sort((a, b) => Number(b.rollout_id) - Number(a.rollout_id))[0];
      if (latestTrace) select.value = String(latestTrace.rollout_id);
      else if (state.rollouts.length) select.value = userRolloutKey(state.rollouts.at(-1));
    }
    renderTraceIndex();
    renderExplorer();
  }

  function configureNavigation() {
    const user = isUserRun();
    const mc = isMcRun();
    document.body.classList.toggle('user-mode', user);
    document.querySelectorAll('.tab[data-kind="recommendation"]').forEach(tab => tab.hidden = user || (tab.id === 'dsrTab' && !state.capabilities.dsr));
    document.querySelectorAll('.tab[data-kind="user"]').forEach(tab => tab.hidden = !user);
    const explorerTab = document.querySelector('.tab[data-view="explorer"]');
    explorerTab.textContent = user ? 'Rollout 样本' : '采样检视';
    $('runKind').textContent = isHybridMcRun() ? 'MC_USER Hybrid' : (mc ? 'MC_USER_v1' : (user ? '懂用户 GRPO' : (state.capabilities.dsr ? 'DSR 实验' : '懂推荐 GRPO')));
    $('runKind').classList.toggle('user', user);
    $('runKind').classList.toggle('dsr', !user && Boolean(state.capabilities.dsr));
    $('kindContext').textContent = isHybridMcRun() ? `Action / Chain · K${state.manifest.K ?? 4} · GRPO + Marginal Credit` : (mc ? `Action / Chain · K${state.manifest.K ?? 2} · Marginal Credit` : (user ? 'Action / Chain · G4 · Token-local penalty' : 'Recommendation / DSR 训练实验'));
    document.querySelector('#userAction .user-section-head p').textContent = mc
      ? '候选 SID 的正负 credit 仅由删除该 SID 后的 Action F1 边际变化决定'
      : 'wrong selection 由主 F1 处理；hallucination / duplicate 同时具有局部 token penalty';
    document.querySelector('#userChain .user-section-head p').textContent = mc
      ? '候选 event 的 credit 由重新计算后的 Action / Logic Alignment 边际变化决定'
      : '同时观察 Action Alignment、Logic Alignment 与局部 grounding/constraint 信号';
    const tokenTab = document.querySelector('.tab[data-view="userToken"]');
    if (tokenTab) {
      tokenTab.hidden = !user;
      tokenTab.textContent = mc ? 'Marginal Credit' : 'Token Advantage';
    }
    const probeTab = document.querySelector('.tab[data-view="userProbe"]');
    if (probeTab) probeTab.textContent = mc ? 'Probe / 固定探针' : 'Prob / 固定探针';
    const demo = Boolean(state.manifest.demo);
    if (demo && !$('demoChip')) $('experimentMeta').insertAdjacentHTML('afterend', '<span class="demo-chip" id="demoChip">DEMO</span>');
    $('demoChip')?.toggleAttribute('hidden', !demo);
    const active = document.querySelector('.view.active')?.id;
    if (user && !['userOverview', 'userAction', 'userChain', 'userToken', 'userProbe', 'explorer'].includes(active)) activateView('userOverview');
    if (!user && ['userOverview', 'userAction', 'userChain', 'userToken', 'userProbe'].includes(active)) activateView('overview');
    if (user) {
      $('routeSelect').innerHTML = '<option value="">全部</option><option value="action">Action</option><option value="chain">Chain</option>';
    } else if (![...$('routeSelect').options].some(option => option.value === 'think')) {
      $('routeSelect').innerHTML = '<option value="">全部</option><option value="think">思考路线</option><option value="no_think">直答路线</option>';
      $('trace').classList.remove('user-trace');
    }
  }

  renderExperimentList = function() {
    const select = $('experimentSelect');
    const current = state.activeRun;
    select.innerHTML = state.runs.map(run => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${run.latest_step ?? 0}/${run.max_steps ?? '-'}${run.demo ? ' · DEMO' : ''}</option>`).join('');
    if (current) select.value = current;
    const run = state.runs.find(item => item.run_id === state.activeRun);
    setText('experimentMeta', run ? `最新进度 ${run.latest_step ?? 0} / ${run.max_steps ?? '-'} · ${routeName(run.latest_route)}` : '当前类型没有可用实验');
    document.querySelectorAll('.kind-segment').forEach(button => button.classList.toggle('active', button.dataset.runKind === state.activeKind));
    if (!isUserRun()) populateRewardCompareOptions();
  };

  loadRuns = async function() {
    const response = await fetch('/api/runs', {cache: 'no-store'});
    if (!response.ok) throw new Error('实验列表接口返回异常状态');
    state.allRuns = (await response.json()).map(run => ({...run, run_kind: run.run_kind === USER ? USER : RECOMMENDATION}));
    const query = new URLSearchParams(location.search);
    const requestedRun = query.get('run');
    const requested = state.allRuns.find(run => run.run_id === requestedRun);
    const requestedKind = query.get('kind');
    if (requested) state.activeKind = requested.run_kind;
    else if ([RECOMMENDATION, USER].includes(requestedKind)) state.activeKind = requestedKind;
    else if (!state.allRuns.some(run => run.run_kind === state.activeKind)) state.activeKind = state.allRuns.some(run => run.run_kind === RECOMMENDATION) ? RECOMMENDATION : USER;
    state.runs = state.allRuns.filter(run => run.run_kind === state.activeKind);
    if (!state.runs.some(run => run.run_id === state.activeRun)) state.activeRun = requested && requested.run_kind === state.activeKind ? requested.run_id : (state.runs[0]?.run_id || '');
    renderExperimentList();
    return Boolean(state.activeRun);
  };

  function stageRunKind(kind) {
    state.manifest = {run_kind: kind};
    state.capabilities = {run_kind: kind, user_grpo: kind === USER, dsr: false};
    setText('headerRun', '正在加载...');
    setLiveState('stale', '切换实验中');
    configureNavigation();
    activateView(kind === USER ? 'userOverview' : 'overview');
    renderCheckpoints();
  }

  async function selectRunKind(kind) {
    if (![RECOMMENDATION, USER].includes(kind) || kind === state.activeKind) return;
    state.activeKind = kind;
    state.runs = state.allRuns.filter(run => run.run_kind === kind);
    state.activeRun = state.runs[0]?.run_id || '';
    resetRunState();
    stageRunKind(kind);
    const url = new URL(location.href);
    url.searchParams.set('kind', kind);
    if (state.activeRun) url.searchParams.set('run', state.activeRun); else url.searchParams.delete('run');
    history.replaceState(null, '', url);
    renderExperimentList();
    if (state.activeRun) await refresh(true); else setLiveState('stale', '当前类型没有实验');
  }

  const baseSelectExperiment = selectExperiment;
  selectExperiment = async function(runId) {
    const run = state.allRuns.find(item => item.run_id === runId);
    if (run) state.activeKind = run.run_kind;
    const kind = state.activeKind;
    const pending = baseSelectExperiment(runId);
    stageRunKind(kind);
    await pending;
    const url = new URL(location.href);
    url.searchParams.set('kind', state.activeKind);
    history.replaceState(null, '', url);
  };

  renderTraceIndex = function() { isUserRun() ? renderUserTraceIndex() : recommendationRenderTraceIndex(); };
  renderExplorer = function() { isUserRun() ? renderUserExplorer() : recommendationRenderExplorer(); };
  renderExplorerOptions = function() { isUserRun() ? renderUserExplorerOptions() : recommendationRenderExplorerOptions(); };

  refresh = async function(force = false) {
    if (!isUserRun()) return recommendationRefresh(force);
    if (!autoRefresh && !force) return;
    if (userRefreshInFlight) return;
    userRefreshInFlight = true;
    let scrollState = null;
    try {
      if (!state.activeRun && !(await loadRuns())) { setLiveState('stale', '等待实验数据'); return; }
      const requestedRun = state.activeRun;
      const endpoints = ['/api/manifest', '/api/capabilities', '/api/metrics', '/api/rollouts', '/api/ranks', '/api/traces', '/api/checkpoints', '/api/probes', '/api/dsr/metrics', '/api/dsr/steps', '/api/dsr/traces', '/api/dsr/gate', '/api/mc-user/summary', '/api/user-light-probe', '/api/recommendation-guard', '/api/model-publish/capabilities', '/api/model-publish/jobs'];
      const responses = await Promise.all(endpoints.map(url => fetch(apiUrl(url), {cache: 'no-store'})));
      if (responses.some(response => !response.ok)) throw new Error('接口返回异常状态');
      const [manifest, capabilities, metrics, rollouts, ranks, traces, checkpoints, probes, dsrMetrics, dsrSteps, dsrTraces, gate, mcSummary, userLightProbe, recommendationGuard, modelPublish, publishJobs] = await Promise.all(responses.map(response => response.json()));
      if (requestedRun !== state.activeRun) return;
      scrollState = captureMonitorScrollState();
      Object.assign(state, {manifest, capabilities, metrics, rollouts, ranks, traces, checkpoints, probes, dsrMetrics, dsrSteps, dsrTraces, gate, mcSummary, userLightProbe, recommendationGuard, modelPublish, publishJobs});
      if (typeof mergeDsrCandidateTraces === 'function') mergeDsrCandidateTraces();
      state.activeKind = manifest.run_kind === USER ? USER : RECOMMENDATION;
      state.runs = state.allRuns.filter(run => run.run_kind === state.activeKind);
      lastSuccessAt = Date.now();
      setText('headerRun', manifest.run_id || state.activeRun);
      const listed = state.allRuns.find(run => run.run_id === state.activeRun);
      if (listed && metrics.length) listed.latest_step = metrics.at(-1).step;
      renderExperimentList();
      renderCheckpoints();
      configureNavigation();
      if (autoRefresh) setLiveState('', `实时 · ${isMcRun() ? `Prompt ${metrics.at(-1)?.prompt_step ?? 0} · Optimizer ${metrics.at(-1)?.optimizer_step ?? 0}` : `step ${metrics.at(-1)?.step ?? '-'}`} · ${new Date(lastSuccessAt).toLocaleTimeString()}`);
      $('error').style.display = 'none';
      if (isUserRun()) {
        renderUserOverview(); renderUserAction(); renderUserChain(); renderUserToken(); renderUserProbe(); renderExplorerOptions();
      } else {
        recommendationRenderOverview(); recommendationRenderPerformance(); renderExplorerOptions(); recommendationRenderProbes();
        if ($('dsr').classList.contains('active')) recommendationRenderDsr();
      }
      if (activeChart) drawZoom();
    } catch (error) {
      $('error').textContent = `监控数据读取失败：${error.message}`;
      $('error').style.display = 'block';
      if (autoRefresh) setLiveState('stale', '连接异常 · 展示上次数据');
    } finally {
      restoreMonitorScrollState(scrollState);
      userRefreshInFlight = false;
    }
  };

  activateView = function(view) {
    if (!isUserRun()) return recommendationActivateView(view);
    document.querySelectorAll('.tab,.view').forEach(element => element.classList.remove('active'));
    const tab = document.querySelector(`.tab[data-view="${view}"]`);
    if (tab && !tab.hidden) tab.classList.add('active');
    $(view).classList.add('active');
    if (isUserRun()) {
      renderUserOverview(); renderUserAction(); renderUserChain(); renderUserToken(); renderUserProbe();
      if (view === 'explorer') renderExplorer();
    } else {
      recommendationRenderOverview(); recommendationRenderPerformance(); recommendationRenderProbes();
      if (view === 'dsr') recommendationRenderDsr();
    }
  };

  buildUserShell();
  const mcPromptCharts = new Set([
    'userLossChart', 'userRewardChart', 'userAdvantageChart', 'userMaskChart', 'userPolicyChart', 'userVarianceChart',
    'userActionScoreChart', 'userActionExactChart', 'userWrongSelectionChart', 'userActionConstraintChart',
    'userChainScoreChart', 'userChainMismatchChart', 'userChainConstraintChart', 'userGroundingChart',
    'mcActionCreditChart', 'mcChainCreditChart',
  ]);
  const drawWithoutMcMilestones = draw;
  draw = function(id, series) {
    drawWithoutMcMilestones(id, series);
    const resolved = id === 'zoomChart' ? activeChart : id;
    if (!isMcRun() || !mcPromptCharts.has(resolved)) return;
    const steps = series.flatMap(item => item.data.map(point => Number(point[0]))).filter(Number.isFinite);
    if (!steps.length) return;
    const min = Math.min(...steps), max = Math.max(...steps), span = Math.max(1, max - min);
    const milestones = (state.checkpoints || []).map(item => Number(item.step)).filter(step => Number.isFinite(step) && step >= min && step <= max);
    if (!milestones.length) return;
    const canvas = $(id), width = canvas.clientWidth, height = canvas.clientHeight, padding = {l: 54, r: 18, t: 16, b: 31};
    const context = canvas.getContext('2d'), toX = step => padding.l + (width - padding.l - padding.r) * (step - min) / span;
    context.save();
    context.font = '10px "Segoe UI", sans-serif';
    milestones.forEach((step, index) => {
      const x = toX(step);
      context.strokeStyle = '#81929c'; context.lineWidth = 1; context.setLineDash([3, 4]);
      context.beginPath(); context.moveTo(x, padding.t); context.lineTo(x, height - padding.b); context.stroke();
      context.setLineDash([]); context.fillStyle = '#5c6b74';
      context.textAlign = x > width - 60 ? 'right' : x < 75 ? 'left' : 'center';
      context.fillText(`ckpt ${step}`, x, padding.t + 10 + (index % 2) * 11);
    });
    context.restore();
  };
  attachChartControls();
  chartLabels.userLossChart = ['Loss', 'Grad Norm'];
  chartLabels.userRewardChart = ['Action F1', 'Chain Reward'];
  chartLabels.userAdvantageChart = ['Sequence mean', 'Sequence std', 'Token mean', 'Token std'];
  chartLabels.userMaskChart = ['Masked candidates', 'Masked tokens'];
  chartLabels.userPolicyChart = ['Ratio mean', 'Clip fraction'];
  chartLabels.userVarianceChart = ['Task reward std', 'Zero-std ratio'];
  chartLabels.userActionScoreChart = ['F1', 'Precision', 'Recall'];
  chartLabels.userActionExactChart = ['Exact Match'];
  chartLabels.userWrongSelectionChart = ['Wrong selection'];
  chartLabels.userActionConstraintChart = ['Hallucination', 'Duplicate'];
  chartLabels.userChainScoreChart = ['Total Reward', 'Action Alignment', 'Logic Alignment'];
  chartLabels.userChainMismatchChart = ['Date mismatch', 'Action mismatch'];
  chartLabels.userChainConstraintChart = ['Hallucinated SID', 'Duplicate event', 'Chronology', 'Excess event'];
  chartLabels.userGroundingChart = ['Grounded', 'Partially grounded', 'Ungrounded'];
  chartLabels.userKindCountChart = USER_KINDS.map(([kind]) => kind);
  chartLabels.userKindMassChart = USER_KINDS.map(([kind]) => kind);
  chartLabels.userProbeActionChart = ['F1', 'Precision', 'Recall'];
  chartLabels.userProbeChainChart = ['Total', 'Action Alignment', 'Logic Alignment'];
  chartLabels.mcActionCreditChart = ['Positive mass', 'Negative mass'];
  chartLabels.mcChainCreditChart = ['Positive mass', 'Negative mass'];
  chartLabels.mcProbeActionChart = ['Action F1', 'User Proxy'];
  chartLabels.mcProbeChainChart = ['Chain Total'];
  document.querySelectorAll('.user-tab').forEach(button => button.onclick = () => activateView(button.dataset.view));
  document.querySelectorAll('.kind-segment[data-run-kind]').forEach(button => button.onclick = () => selectRunKind(button.dataset.runKind));
  $('userProbeRoute').onchange = renderUserProbe;
  $('userProbeGroup').onchange = renderUserProbe;
  $('userProbeStep').onchange = renderUserProbe;
  $('mcCreditRoute').onchange = renderMcCredit;
  $('mcCreditPrompt').onchange = renderMcCredit;
  $('mcProbeSampleRoute').onchange = renderMcProbeSampleDetail;
  $('mcProbeSampleId').onchange = renderMcProbeSampleDetail;
  $('mcProbeSampleStep').onchange = renderMcProbeSampleDetail;
  document.querySelectorAll('canvas.chart:not(.zoomed)').forEach(canvas => {
    if (!canvas.onclick) canvas.onclick = () => openChart(canvas.id);
    if (!canvas.onmousemove) canvas.onmousemove = showChartTooltip;
    if (!canvas.onmouseleave) canvas.onmouseleave = hideChartTooltip;
  });
  const baseResize = window.onresize;
  window.onresize = () => { baseResize?.(); if (isUserRun()) { renderUserOverview(); renderUserAction(); renderUserChain(); renderUserToken(); renderUserProbe(); } };
  loadRuns().then(() => refresh(true)).catch(error => {
    $('error').textContent = `监控数据读取失败：${error.message}`;
    $('error').style.display = 'block';
  });
})();
