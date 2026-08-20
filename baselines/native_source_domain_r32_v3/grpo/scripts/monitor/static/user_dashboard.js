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
  const userSampleContextCache = new Map();
  let userRefreshInFlight = false;
  const monitorScrollableSelectors = [
    '#trace .probe-context-block pre',
    '#userProbeDetail .probe-context-block pre',
    '#trace .candidate .text',
    '#userProbeDetail .candidate .text',
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

  function buildUserShell() {
    document.querySelector('.experiment-bar').insertAdjacentHTML('beforebegin', `
      <section class="run-kind-switcher" aria-label="GRPO 任务类型">
        <span class="switch-label">任务入口</span>
        <div class="kind-segments">
          <button class="kind-segment active" type="button" data-run-kind="${RECOMMENDATION}">懂推荐 GRPO</button>
          <button class="kind-segment" type="button" data-run-kind="${USER}">懂用户 GRPO</button>
        </div>
        <span class="kind-context" id="kindContext">Recommendation / DSR 训练实验</span>
      </section>`);
    const nav = document.querySelector('.tabs');
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
          <div class="stat"><div class="label">训练步数</div><div class="value" id="uStep">-</div></div>
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
      <main id="userToken" class="view"><div class="wrap">
        <div class="user-section-head"><h2>Token Advantage</h2><p>Sqrt-normalized local penalty；overlap 取最强值，不累加</p></div>
        <section class="user-metric-grid" id="userTokenCards"></section>
        <section class="charts">
          <div class="panel"><h2>Per-kind Masked Token Count</h2><div class="legend" id="userKindCountLegend"></div><canvas class="chart" id="userKindCountChart"></canvas></div>
          <div class="panel"><h2>Per-kind Incremental Negative Mass</h2><div class="legend" id="userKindMassLegend"></div><canvas class="chart" id="userKindMassChart"></canvas></div>
        </section>
        <section class="panel section-gap"><h2>最新各类局部 Penalty</h2><div class="table-scroll"><table class="penalty-table"><thead><tr><th>Violation kind</th><th>适用路由</th><th>Masked tokens</th><th>Negative mass</th><th>Mass share</th></tr></thead><tbody id="userPenaltyRows"></tbody></table></div></section>
      </div></main>
      <main id="userProbe" class="view"><div class="wrap">
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
      </div></main>`);
  }

  function metricCards(id, items) {
    $(id).innerHTML = items.map(([label, value, tone]) => `<div class="stat"><div class="label">${escapeHtml(label)}</div><div class="value small${tone ? ` ${tone}` : ''}">${escapeHtml(value)}</div></div>`).join('');
  }

  function latestRoute(route) {
    return state.metrics.filter(row => row.route === route).at(-1) || {};
  }

  function objectPoint(rows, field, key) {
    return rows.filter(row => row[field]?.[key] != null).map(row => [row.step, Number(row[field][key])]);
  }

  function renderUserOverview() {
    const current = state.metrics.at(-1) || {};
    const action = latestRoute('action');
    const chain = latestRoute('chain');
    const max = Number(state.manifest.max_steps || 0);
    const progress = max ? Math.min(100, 100 * Number(current.step || 0) / max) : 0;
    setText('uStep', `${current.step ?? 0} / ${max || '-'}`);
    setText('uRoute', routeName(current.route));
    setText('uProgress', max ? `${progress.toFixed(1)}%` : '-');
    setText('uLoss', fmt(current.loss));
    setText('uGrad', fmt(current.grad_norm));
    setText('uLr', current.learning_rate == null ? '-' : Number(current.learning_rate).toExponential(2));
    $('uProgressFill').style.width = `${progress}%`;
    $('userDemoNotice').hidden = !state.manifest.demo;
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

  function renderUserToken() {
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

  function renderUserProbe() {
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
    const current = Number($('rolloutSelect').value);
    const lane = route => {
      const traces = state.traces.filter(trace => trace.route === route).sort((a, b) => a.rollout_id - b.rollout_id);
      const rolloutCount = state.rollouts.filter(rollout => rollout.route === route).length;
      const links = traces.length ? traces.map(trace => `<button class="trace-link ${trace.rollout_id === current ? 'active' : ''}" type="button" data-rollout-id="${trace.rollout_id}" data-route="${route}">第 ${trace.step ?? '-'} 步 · #${trace.rollout_id} · ${trace.candidates?.length ?? 0} candidates</button>`).join('') : `<span class="trace-none">${rolloutCount ? `${rolloutCount} 条汇总 · candidate trace 未落盘` : '尚无记录'}</span>`;
      return `<div class="trace-lane"><div class="trace-lane-title ${route === 'action' ? 'user-route-action' : 'user-route-chain'}">${routeName(route)}</div><div class="trace-links">${links}</div></div>`;
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
    const id = Number($('rolloutSelect').value);
    const route = $('routeSelect').value;
    const rollout = state.rollouts.find(row => row.rollout_id === id && (!route || row.route === route));
    if (!rollout) {
      $('rolloutSummary').innerHTML = '<div class="empty">请选择一个 User rollout</div>';
      $('trace').innerHTML = '';
      return;
    }
    const stats = [
      ['路由', routeName(rollout.route)], ['平均 Reward', fmt(rollout.reward_mean)], ['Reward Std', fmt(rollout.reward_std)],
      ['Zero-std', pct(rollout.zero_std_ratio)], ['Masked candidates', pct(rollout.masked_candidate_rate)],
      ['Masked tokens', pct(rollout.masked_token_rate)], ['平均长度', fmt(rollout.completion_length_mean, 1)], ['G', String(rollout.g ?? state.manifest.G ?? 4)],
    ];
    $('rolloutSummary').innerHTML = stats.map(([label, value]) => `<div class="stat"><div class="label">${label}</div><div class="value small">${escapeHtml(value)}</div></div>`).join('');
    const trace = state.traces.find(row => row.rollout_id === id && (!route || row.route === route));
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
    const candidates = Array.isArray(trace.candidates) ? trace.candidates : [];
    const expected = Number(state.manifest.G || 4);
    $('trace').classList.add('user-trace');
    $('trace').innerHTML = `<div class="trace-head">${escapeHtml(trace.group_id || '-')} · ${escapeHtml(routeName(trace.route))}<span class="candidate-count ${candidates.length === expected ? '' : 'incomplete'}">${candidates.length}/${expected} candidates</span></div>${renderUserSampleContext(trace, trace.route)}${candidates.map(candidate => renderUserCandidate(candidate, trace.route)).join('') || '<div class="empty">trace 中没有 candidate</div>'}`;
  }

  function renderUserExplorerOptions() {
    const select = $('rolloutSelect');
    const current = select.value;
    const traceIds = new Set(state.traces.map(trace => Number(trace.rollout_id)));
    select.innerHTML = [...state.rollouts].reverse().map(rollout => {
      const recorded = traceIds.has(Number(rollout.rollout_id));
      return `<option value="${rollout.rollout_id}">#${rollout.rollout_id} · ${escapeHtml(routeName(rollout.route))} · 第 ${rollout.step} 步${recorded ? ' · 有样本' : ''}</option>`;
    }).join('');
    if (current && [...select.options].some(option => option.value === current)) {
      select.value = current;
    } else {
      const latestTrace = [...state.traces].sort((a, b) => Number(b.rollout_id) - Number(a.rollout_id))[0];
      if (latestTrace) select.value = String(latestTrace.rollout_id);
    }
    renderTraceIndex();
    renderExplorer();
  }

  function configureNavigation() {
    const user = isUserRun();
    document.body.classList.toggle('user-mode', user);
    document.querySelectorAll('.tab[data-kind="recommendation"]').forEach(tab => tab.hidden = user || (tab.id === 'dsrTab' && !state.capabilities.dsr));
    document.querySelectorAll('.tab[data-kind="user"]').forEach(tab => tab.hidden = !user);
    const explorerTab = document.querySelector('.tab[data-view="explorer"]');
    explorerTab.textContent = user ? 'Rollout 样本' : '采样检视';
    $('runKind').textContent = user ? '懂用户 GRPO' : (state.capabilities.dsr ? 'DSR 实验' : '懂推荐 GRPO');
    $('runKind').classList.toggle('user', user);
    $('runKind').classList.toggle('dsr', !user && Boolean(state.capabilities.dsr));
    $('kindContext').textContent = user ? 'Action / Chain · G4 · Token-local penalty' : 'Recommendation / DSR 训练实验';
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
      const endpoints = ['/api/manifest', '/api/capabilities', '/api/metrics', '/api/rollouts', '/api/ranks', '/api/traces', '/api/checkpoints', '/api/probes', '/api/dsr/metrics', '/api/dsr/steps', '/api/dsr/traces', '/api/dsr/gate'];
      const responses = await Promise.all(endpoints.map(url => fetch(apiUrl(url), {cache: 'no-store'})));
      if (responses.some(response => !response.ok)) throw new Error('接口返回异常状态');
      const [manifest, capabilities, metrics, rollouts, ranks, traces, checkpoints, probes, dsrMetrics, dsrSteps, dsrTraces, gate] = await Promise.all(responses.map(response => response.json()));
      if (requestedRun !== state.activeRun) return;
      scrollState = captureMonitorScrollState();
      Object.assign(state, {manifest, capabilities, metrics, rollouts, ranks, traces, checkpoints, probes, dsrMetrics, dsrSteps, dsrTraces, gate});
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
      if (autoRefresh) setLiveState('', `实时 · ${new Date(lastSuccessAt).toLocaleTimeString()}`);
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
  document.querySelectorAll('.user-tab').forEach(button => button.onclick = () => activateView(button.dataset.view));
  document.querySelectorAll('.kind-segment').forEach(button => button.onclick = () => selectRunKind(button.dataset.runKind));
  $('userProbeRoute').onchange = renderUserProbe;
  $('userProbeGroup').onchange = renderUserProbe;
  $('userProbeStep').onchange = renderUserProbe;
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
