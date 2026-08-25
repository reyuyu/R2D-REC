/* Composite Interest monitor extension. Reward math always comes from Python. */
(() => {
  const EXPERIMENT = 'GR_REC_Think_CompositeInterest_v1';
  const old = {advantageMatches, provenanceBadge, renderAdvantages, renderCandidate, renderOverview, renderProbes};
  const enabled = () => state.manifest?.experiment === EXPERIMENT;
  const beamFirstEnabled = () => enabled() && String(
    state.manifest?.frozen_contract?.reward || ''
  ).includes('min(0.25,0.5*min_positive_beam_gap)');
  const rewardPanel=$('rewardChart').closest('.panel'),rewardHelp=rewardPanel.querySelector('.metric-help .help-content'),rewardTrend=rewardPanel.querySelector('.trend-guide');
  const rewardCompare=rewardPanel.querySelector('.reward-compare-controls'),rewardCompareNote=$('rewardCompareNote');
  const signalPanel=$('signal').closest('.panel'),rewardStdPanel=$('rewardStd').closest('.panel');
  const legacyUi={
    rewardLabel:$('reward').previousElementSibling.textContent,
    rewardHelp:rewardHelp?.innerHTML||'',rewardTrend:rewardTrend?.innerHTML||'',
    signalLabel:signalPanel.querySelector('.label').textContent,signalNote:signalPanel.querySelector('p').textContent,
    rewardStdLabel:rewardStdPanel.querySelector('.label').textContent,rewardStdNote:rewardStdPanel.querySelector('p').textContent,
  };
  const num = value => Number.isFinite(Number(value)) ? Number(value) : null;
  const mean = values => {const xs=values.map(num).filter(x=>x!=null);return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null};
  const vector = values => `[${(values||[]).map(value=>fmt(value,3)).join(', ')}]`;
  const capturedBadge = () => '<span class="provenance-badge captured" title="训练时直接落盘的 Composite reward 与 Sequence Advantage">实采</span>';
  const style=document.createElement('style');style.textContent=`
    .provenance-badge.captured{border-color:#75ad91;background:#e9f6f0;color:#0d684a}
    .composite-panels[hidden]{display:none}.composite-wide{grid-column:1/-1}.composite-callout{margin:10px 14px;padding:9px 11px;border-left:3px solid #16835f;background:#edf8f3;color:#205c48}
    .composite-card-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:14px}.composite-card{min-width:0;border:1px solid #dbe1e5;border-top:3px solid #89949c;background:#fff}.composite-card.positive{border-top-color:#16835f}.composite-card.negative{border-top-color:#bd3f4c}
    .composite-card-head{display:flex;justify-content:space-between;gap:8px;padding:10px;border-bottom:1px solid #e6eaed;font-weight:700}.composite-primary{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;padding:10px}.composite-secondary{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;padding:0 10px 10px}.composite-primary div,.composite-secondary div{padding:7px;background:#f6f8f9}.composite-primary b,.composite-secondary b{display:block;margin-top:2px}
    .match-list{padding:0 10px 10px}.match-row{margin-top:7px;padding:8px;border:1px solid #dce3e7;background:#fbfcfc}.match-row strong{display:block}.match-score{color:#60717d;font-size:11px;margin-top:4px}.unmatched{margin-top:7px;color:#9a2f3b;font-size:11px}.composite-strip{margin:0 12px 10px;padding:8px 10px;border-left:3px solid #16835f;background:#f1f8f5;font-size:11px}.composite-strip button{margin-left:8px;border:0;background:none;color:#1c61a8;text-decoration:underline;cursor:pointer}@media(max-width:1050px){.composite-card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .composite-input{margin:12px 14px 0;border:1px solid #dbe3e7;background:#fbfcfc}.composite-input summary{cursor:pointer;padding:10px 12px;font-weight:700}.composite-input-meta{display:flex;gap:8px;flex-wrap:wrap;padding:0 12px 9px}.composite-input pre,.sampled-cot pre{white-space:pre-wrap;overflow:auto;max-height:280px;margin:0;padding:12px;background:#f4f7f8;border-top:1px solid #e0e6e9;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}.reward-reference{margin:10px 14px;padding:10px 12px;border-left:3px solid #b36b08;background:#fff8e8}.reward-reference strong{display:block}.sampled-cot{margin:0 10px 10px;border:1px solid #e0e6e9}.sampled-cot h4{margin:0;padding:8px 10px;font-size:12px}.beam32-box{margin:0 10px 10px;border:1px solid #d7e1e6;background:#f9fbfb}.beam32-summary{padding:10px}.beam32-flow{margin:5px 0;color:#52656f;font-size:11px}.beam32-toggle{margin-top:8px;padding:7px 10px;border:1px solid #9cb7c4;background:#fff;color:#155e75;cursor:pointer}.beam32-detail{padding:10px;border-top:1px solid #d7e1e6}.beam32-detail[hidden]{display:none}.beam-row{margin-top:8px;padding:9px;border:1px solid #dde5e8;background:#fff}.beam-row-head{display:flex;justify-content:space-between;gap:8px}.beam-token{display:inline-block;margin:6px 5px 0 0;padding:4px 6px;background:#eaf4f0;color:#0d684a;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.beam-meta{margin-top:6px;color:#60717d;font-size:11px;overflow-wrap:anywhere}.relation-EXACT{color:#087447}.relation-AB,.relation-A{color:#9a6400}.relation-INVALID{color:#b52f3e}.relation-VALID_NO_HIT{color:#52656f}@media(max-width:1050px){.composite-card-grid{grid-template-columns:1fr}}
  `;document.head.appendChild(style);

  function ensureOverview(){
    if($('compositeOverviewPanels'))return;
    const panel=document.createElement('section');panel.id='compositeOverviewPanels';panel.className='composite-panels';panel.hidden=true;
    panel.innerHTML=`<div class="diagnostic-head"><h2>Composite Interest 辅助诊断</h2><p id="compositeOverviewDescription">最终 Composite 主奖励与原始 Beam Reward 纵向分图对照，避免不同量纲叠线。</p></div><section class="charts">
      <div class="panel composite-wide"><h2 id="compositeBeamRawTitle">原始 Beam Reward</h2><div class="legend"><span class="key" style="--c:#2d6cdf">Beam Raw</span></div><canvas class="chart" id="compositeBeamRawChart"></canvas></div>
      <div class="panel"><h2 id="compositeRewardComponentsTitle">奖励分量对照</h2><div class="legend" id="compositeRewardComponentsLegend"></div><canvas class="chart" id="compositeRewardChart"></canvas></div>
      <div class="panel"><h2>零方差与救活率</h2><div class="legend"><span class="key" style="--c:#bd3f4c">Beam Zero</span><span class="key" style="--c:#b36b08">Composite Zero</span><span class="key" style="--c:#16835f">Rescued</span><span class="key" style="--c:#2d6cdf">CoT Active</span></div><canvas class="chart" id="compositeSignalChart"></canvas></div>
      <div class="panel"><h2>兴趣结构</h2><div class="legend"><span class="key" style="--c:#16835f">Matched</span><span class="key" style="--c:#7654b5">Raw N</span><span class="key" style="--c:#2d6cdf">Grounded N</span><span class="key" style="--c:#b36b08">Q Active</span></div><canvas class="chart" id="compositeInterestChart"></canvas></div></section>`;
    $('overview').querySelector('.wrap').appendChild(panel);
    chartLabels.compositeBeamRawChart=['原始 Beam Reward'];
    chartLabels.compositeRewardChart=['U_cot','兴趣加成'];
    chartLabels.compositeSignalChart=['Beam Zero','Composite Zero','Rescued','CoT Active'];
    chartLabels.compositeInterestChart=['Matched','Raw N','Grounded N','Q Active'];
    rawDiagnosticCharts.add('compositeSignalChart');
    attachChartControls(panel);
  }
  let loading=false;
  async function loadSummary(){if(loading||!enabled())return;loading=true;try{const r=await fetch(apiUrl('/api/composite-interest/summary'),{cache:'no-store'});if(r.ok)state.compositeSummary=await r.json()}finally{loading=false;drawOverview()}}
  function configurePrimaryReward(rows){
    if(!enabled()){
      rewardCompare.hidden=false;rewardCompareNote.hidden=false;
      $('reward').previousElementSibling.textContent=legacyUi.rewardLabel;
      signalPanel.querySelector('.label').textContent=legacyUi.signalLabel;signalPanel.querySelector('p').textContent=legacyUi.signalNote;
      rewardStdPanel.querySelector('.label').textContent=legacyUi.rewardStdLabel;rewardStdPanel.querySelector('p').textContent=legacyUi.rewardStdNote;
      if(rewardHelp)rewardHelp.innerHTML=legacyUi.rewardHelp;if(rewardTrend)rewardTrend.innerHTML=legacyUi.rewardTrend;
      return;
    }
    rewardCompare.hidden=true;rewardCompareNote.hidden=true;
    setText('rewardChartTitle','主奖励 · 最终 Composite Reward');
    $('rewardLegend').innerHTML='<span class="key" style="--c:#16835f">最终 Composite 奖励</span>';
    chartLabels.rewardChart=['最终 Composite 奖励'];
    draw('rewardChart',[{data:rows.map(row=>[row.step,row.composite_reward_mean]),color:colors[1]}]);
    const latest=rows.at(-1);$('reward').previousElementSibling.textContent='最新主奖励（Composite）';
    signalPanel.querySelector('.label').textContent='Composite 有效信号密度';signalPanel.querySelector('p').textContent='1 - Composite zero-std rate';
    rewardStdPanel.querySelector('.label').textContent='Composite 奖励离散度';rewardStdPanel.querySelector('p').textContent='当前最终奖励 population std';
    if(latest){setText('reward',fmt(latest.composite_reward_mean));setText('signal',pct(1-num(latest.composite_zero_std_rate)));setText('rewardStd',fmt(latest.composite_reward_std));}
    if(rewardHelp)rewardHelp.innerHTML=beamFirstEnabled()
      ? '<p><strong>名词解释：</strong>最终奖励 = Beam raw + 实际 tie-break scale × U_cot。Beam 层级绝对优先，兴趣分只负责层内排序。</p><p class="trend-good"><strong>健康趋势：</strong>主奖励随 Beam 改善，STRICT_REVERSAL_COUNT 始终为 0。</p>'
      : '<p><strong>名词解释：</strong>最终 Composite Reward 是 Beam 贡献与 CoT 兴趣命中贡献按冻结公式合成的实际训练主奖励。</p><p class="trend-good"><strong>健康趋势：</strong>平滑均值稳定改善，同时有效信号密度不持续下降。</p>';
    if(rewardTrend)rewardTrend.innerHTML=beamFirstEnabled()
      ? '<strong>怎么看：</strong>先看 Beam raw，再看同 Beam 内的兴趣加成；低 Beam 不允许越过高 Beam。'
      : '<strong>怎么看：</strong>这里只看最终 Composite 主奖励；Beam 与 CoT 分量在下方辅助诊断中分开查看。';
  }
  function drawOverview(){ensureOverview();const panel=$('compositeOverviewPanels'),rows=state.compositeSummary?.rows||[];panel.hidden=!enabled();configurePrimaryReward(rows);if(!enabled())return;
    const beamFirst=beamFirstEnabled();
    setText('compositeOverviewDescription',beamFirst?'Beam raw 是主层级；兴趣分仅以不跨 Beam 层级的实际 scale 做 tie-break。所有曲线均来自训练时 Python 实采。':'最终 Composite 主奖励与原始 Beam Reward 纵向分图对照，避免不同量纲叠线。');
    setText('compositeBeamRawTitle',beamFirst?'Beam 主奖励（第一优先级）':'原始 Beam Reward（旧口径，仅对照）');
    setText('compositeRewardComponentsTitle',beamFirst?'兴趣 Tie-break（辅助信号）':'奖励分量对照');
    $('compositeRewardComponentsLegend').innerHTML=beamFirst
      ? '<span class="key" style="--c:#7654b5">U_cot</span><span class="key" style="--c:#16835f">实际兴趣加成</span>'
      : '<span class="key" style="--c:#2d6cdf">U_beam（Beam 命中得分）</span><span class="key" style="--c:#7654b5">U_cot（CoT 兴趣命中得分）</span>';
    draw('compositeBeamRawChart',[{data:rows.map(r=>[r.step,r.beam_raw_mean]),color:colors[0]}]);
    chartLabels.compositeRewardChart=beamFirst?['U_cot','实际兴趣加成']:['U_beam','U_cot'];
    draw('compositeRewardChart',beamFirst
      ? [{data:rows.map(r=>[r.step,r.cot_utility_mean]),color:colors[4]},{data:rows.map(r=>[r.step,r.cot_contribution_mean]),color:colors[1]}]
      : [{data:rows.map(r=>[r.step,r.beam_utility_mean]),color:colors[0]},{data:rows.map(r=>[r.step,r.cot_utility_mean]),color:colors[4]}]);
    draw('compositeSignalChart',[{data:rows.map(r=>[r.step,r.beam_zero_std_rate]),color:colors[3]},{data:rows.map(r=>[r.step,r.composite_zero_std_rate]),color:colors[2]},{data:rows.map(r=>[r.step,r.rescued_rate]),color:colors[1]},{data:rows.map(r=>[r.step,r.cot_active_rate]),color:colors[0]}]);
    draw('compositeInterestChart',[{data:rows.map(r=>[r.step,r.matched_interest_mean]),color:colors[1]},{data:rows.map(r=>[r.step,r.raw_n_mean]),color:colors[4]},{data:rows.map(r=>[r.step,r.grounded_n_mean]),color:colors[0]},{data:rows.map(r=>[r.step,r.q_active_rate]),color:colors[2]}]);if(!state.compositeSummary)loadSummary();
  }
  function controls(){const intro=document.querySelector(".adv-intro .provenance-badge");if(intro&&enabled()){intro.textContent="实采";intro.title="训练 Composite reward 与 Sequence Advantage";intro.classList.add("captured")}else if(intro){intro.textContent="复算";intro.title="由已落盘 trace 只读重构，非训练时直接采集";intro.classList.remove("captured")}const route=$('advRouteSelect'),focus=$('advFocusSelect');if(enabled()){route.innerHTML='<option value="think">思考路线</option>';focus.innerHTML=[['','全部信号'],['positive','Composite 正优势'],['negative','Composite 负优势'],['rescued','Beam zero-std 被救活'],['composite_zero','Composite zero-std'],['full','Full interest coverage'],['partial','Partial interest coverage'],['zero_match','Zero interest match'],['tie','Beam top tie-break'],['reversal','Strict Beam reversal'],['parser','Parser failure'],['q','Q active']].map(([v,l])=>`<option value="${v}">${l}</option>`).join('')}else if(route.options.length===1){route.innerHTML='<option value="">全部</option><option value="think">思考路线</option><option value="no_think">直答路线</option>';focus.innerHTML='<option value="">全部信号</option><option value="positive">正信用</option><option value="negative">负信用</option><option value="clamp">ExactClamp</option><option value="zero">零信号组</option><option value="b_singleton">B singleton</option><option value="c_singleton">C singleton</option><option value="bridge">Bridge active</option><option value="fallback">Domain fallback</option>'}}
  function matches(g,f){const c=g.candidates||[];if(!f)return true;if(f==='positive')return c.some(x=>num(x.final_sequence_advantage)>0);if(f==='negative')return c.some(x=>num(x.final_sequence_advantage)<0);if(f==='rescued')return g.beam_all_equal&&!g.composite_all_equal;if(f==='composite_zero')return g.composite_all_equal;if(f==='full')return c.some(x=>num(x.interest_coverage)===1);if(f==='partial')return c.some(x=>num(x.interest_coverage)>0&&num(x.interest_coverage)<1);if(f==='zero_match')return c.some(x=>num(x.matched_interest_count)===0);if(f==='tie')return g.top_set_tie_break;if(f==='reversal')return g.strict_beam_reversal;if(f==='parser')return c.some(x=>x.parser_success===false);if(f==='q')return c.some(x=>num(x.Q)>0);return true}
  function matching(c,g){const pred=c.pred_interest_units||[],gold=g.gold_interest_units||[];const rows=(c.match_details||[]).map(m=>{const p=pred.find(x=>x.index===m.pred_index)||{},q=gold.find(x=>x.index===m.gold_index)||{};return `<div class="match-row"><strong>Pred #${m.pred_index}：${escapeHtml(p.normalized_text||'—')}</strong><span>匹配 Gold #${m.gold_index}：${escapeHtml(q.normalized_text||'—')}</span><div class="match-score">Text ${fmt(m.text_similarity,3)} · Evidence ${fmt(m.evidence_similarity,3)} · Combined ${fmt(m.combined_similarity,3)} · MATCH ✓</div></div>`}).join('');const up=(c.unmatched_pred_indices||[]).map(i=>`Pred #${i}`).join(', '),ug=(c.unmatched_gold_indices||[]).map(i=>`Gold #${i}`).join(', ');return rows+(up?`<div class="unmatched">Candidate 额外兴趣：${up}</div>`:'')+(ug?`<div class="unmatched">Gold 兴趣未恢复：${ug}</div>`:'')}
  function card(c,g){const a=num(c.final_sequence_advantage),tone=a>0?'positive':a<0?'negative':'';return `<article class="composite-card ${tone}"><div class="composite-card-head"><span>Candidate #${c.candidate_id}</span><span class="summary-chip">${c.parser_success===false?'Parser failure':`Matched ${c.matched_interest_count}/${c.gold_interest_count}`}</span></div><div class="composite-primary"><div><span class="label">Composite Reward</span><b>${fmt(c.composite_reward,4)}</b></div><div><span class="label">Sequence Advantage</span><b class="adv-${creditTone(a)}">${signed(a,4)}</b></div><div><span class="label">Matched K/N</span><b>${c.matched_interest_count??'—'}/${c.gold_interest_count??'—'}</b></div></div><div class="composite-secondary"><div><span class="label">Beam 得分 / 归一化 U_beam</span><b>${fmt(c.beam_raw)} / ${fmt(c.beam_utility)}</b></div><div><span class="label">Beam 对最终奖励贡献</span><b>${fmt(c.beam_contribution)}</b></div><div><span class="label">CoT 命中得分 / 对最终奖励贡献</span><b>${fmt(c.cot_utility)} / ${fmt(c.cot_contribution)}</b></div><div><span class="label">Coverage tier / Q</span><b>${fmt(c.coverage_tier)} / ${fmt(c.Q)}</b></div><div><span class="label">Raw N / Grounded N</span><b>${c.raw_n??'—'} / ${c.grounded_n??'—'}</b></div><div><span class="label">Grounding Coverage</span><b>${c.grounding_coverage==null?'未定义':pct(c.grounding_coverage)}</b></div><div><span class="label">Parser</span><b>${c.parser_success===false?escapeHtml(c.parser_failure_reason||'失败'):'成功'}</b></div><div><span class="label">Completion Length</span><b>${c.completion_length??'—'}</b></div></div><div class="match-list">${matching(c,g)}</div></article>`}
  window.renderCompositeThinkAdvantage=g=>{const flags=[g.beam_all_equal&&!g.composite_all_equal?'<span class="summary-chip good">CoT Reward 救活零方差组 · RESCUED</span>':'',g.composite_all_equal?'<span class="summary-chip alert">Composite 仍无组内信号</span>':'',g.top_set_tie_break?'<span class="summary-chip good">Beam 并列第一 → CoT 负责打破平局</span>':'',g.strict_beam_reversal?'<span class="summary-chip alert">Composite 改变 Beam 排名</span>':''].join('');return `<section class="adv-group"><header class="adv-group-head"><div><div class="adv-title">Step ${g.step??'—'} · Think G4 · Balanced Composite Reward ${capturedBadge()}</div><div class="adv-sub">Sequence-level Advantage：整个 CoT 共享一个 final advantage。兴趣匹配用于解释 Composite Reward，不是 token advantage。</div></div><div class="summary-chips">${flags}</div></header><div class="composite-callout">Beam ${vector(g.beam_raw_vector)} · U_cot ${vector(g.cot_utility_vector)} · Composite ${vector(g.composite_reward_vector)} · population std ${fmt(g.composite_reward_population_std,5)} · Beam ${g.beam_active?'active':'zero'} / CoT ${g.cot_active?'active':'zero'} / Composite ${g.composite_active?'active':'zero'}</div><div class="composite-card-grid">${(g.candidates||[]).map(c=>card(c,g)).join('')}</div></section>`};
  provenanceBadge=()=>enabled()?capturedBadge():old.provenanceBadge();advantageMatches=(g,r,f)=>g.formula==='composite_interest_v1'?(!r||r==='think')&&matches(g,f):old.advantageMatches(g,r,f);
  renderAdvantages=function(){controls();if(!enabled())return old.renderAdvantages();const groups=(state.advantages?.groups||[]).filter(g=>g.valid&&matches(g,$('advFocusSelect').value)),select=$('advGroupSelect'),current=select.value;select.innerHTML=groups.length?[...groups].reverse().map(g=>`<option value="${escapeHtml(groupKey(g))}">Step ${g.step??'—'} · Think · ${escapeHtml(String(g.group_id))}</option>`).join(''):'<option value="">暂无实采 Composite 数据</option>';if(groups.some(g=>groupKey(g)===current))select.value=current;const selected=groups.find(g=>groupKey(g)===select.value)||groups.at(-1);if(!selected){$('advantageContent').innerHTML='<div class="adv-empty">暂无实采 Composite 数据</div>';return}select.value=groupKey(selected);$('advantageContent').innerHTML=renderCompositeThinkAdvantage(selected)};
  renderCandidate=function(c,t,g,a){const html=old.renderCandidate(c,t,g,a);if(!enabled()||a?.formula!=='composite_interest_v1')return html;const x=a.candidates?.find(row=>Number(row.candidate_id)===Number(c.candidate_id));if(!x)return html;return html+`<div class="composite-strip">Composite Reward <b>${fmt(x.composite_reward)}</b> · Sequence Advantage <b>${signed(x.final_sequence_advantage)}</b> · Beam ${fmt(x.beam_contribution)} + CoT ${fmt(x.cot_contribution)} · Matched ${x.matched_interest_count}/${x.gold_interest_count}<button type="button" data-open-advantage="${escapeHtml(groupKey(a))}">查看完整 Composite 详情</button></div>`};
  function probeCandidate(c,selected){const html=old.renderCandidate(c,{route:'think'},selected.gold_sids||[]);return html+`<div class="composite-strip">最终 Composite <b>${fmt(c.composite_reward)}</b> · Sequence Advantage <b>${signed(c.final_sequence_advantage)}</b> · Beam 贡献 ${fmt(c.beam_contribution)} + CoT 贡献 ${fmt(c.cot_contribution)} · Matched ${c.matched_interest_count}/${c.gold_interest_count}</div>`}
  function compositeProbes(){const rows=state.probes||[],configured=state.manifest.fixed_probe_ids||state.manifest.fixed_probe_group_ids||state.manifest.fixed_probe?.group_ids||[],groups=[...new Set([...configured,...rows.map(r=>r.group_id)])],gs=$('probeGroupSelect'),previous=gs.value;gs.innerHTML=groups.length?groups.map(g=>{const r=rows.find(x=>x.group_id===g)||{};return `<option value="${escapeHtml(g)}">${escapeHtml(r.target_domain||'domain')} · R${Number(r.probe_round??0)+1} · ${escapeHtml(String(g).slice(0,12))}</option>`}).join(''):'<option value="">未启用</option>';if(groups.includes(previous))gs.value=previous;const available=rows.filter(r=>r.group_id===gs.value).sort((a,b)=>a.step-b.step),ss=$('probeStepSelect'),before=ss.value,labels={0:'Baseline',200:'Step 200',400:'Step 400',600:'Step 600',716:'Final 716'};ss.innerHTML=available.length?[...available].reverse().map(r=>`<option value="${r.step}">${labels[r.step]||`Step ${r.step}`}</option>`).join(''):'<option value="">暂无记录</option>';if(available.some(r=>String(r.step)===before))ss.value=before;setText('probeMeta','固定 milestone：0 / 200 / 400 / 600 / 716 · Think-only 12 probes');const steps=[...new Set(rows.map(r=>r.step))].sort((a,b)=>a-b),avg=(step,field)=>mean(rows.filter(r=>r.step===step).flatMap(field));draw('probeRewardChart',[{data:steps.map(s=>[s,avg(s,r=>[r.think?.beam_reward_mean])]),color:colors[2]},{data:steps.map(s=>[s,avg(s,r=>[r.think?.composite_reward_mean])]),color:colors[1]},{data:steps.map(s=>[s,avg(s,r=>(r.think?.candidates||[]).map(c=>c.cot_utility))]),color:colors[4]}]);draw('probeClosureChart',[{data:steps.map(s=>[s,avg(s,r=>(r.think?.candidates||[]).map(c=>c.matched_interest_count))]),color:colors[1]},{data:steps.map(s=>[s,avg(s,r=>(r.think?.candidates||[]).map(c=>c.raw_n))]),color:colors[4]},{data:steps.map(s=>[s,avg(s,r=>(r.think?.candidates||[]).map(c=>c.grounded_n))]),color:colors[0]}]);const selected=available.find(r=>String(r.step)===ss.value)||available.at(-1);if(!selected){$('probeOverview').innerHTML='';$('probeDetail').innerHTML='<div class="probe-empty">暂无实采 Composite Probe 数据</div>';return}ss.value=String(selected.step);const c=selected.think?.candidates||[];diagnosticCards('probeOverview',[['Domain / Group',`${selected.target_domain??'—'} / ${selected.group_id}`],['Probe Round',`R${Number(selected.probe_round??0)+1}`],['Beam 命中均值',fmt(selected.think?.beam_reward_mean)],['最终 Composite 均值 / 标准差',`${fmt(selected.think?.composite_reward_mean)} / ${fmt(selected.think?.composite_reward_std)}`],['CoT 兴趣命中均值（U_cot）',fmt(mean(c.map(x=>x.cot_utility)))],['Matched / Raw / Grounded',`${fmt(mean(c.map(x=>x.matched_interest_count)),2)} / ${fmt(mean(c.map(x=>x.raw_n)),2)} / ${fmt(mean(c.map(x=>x.grounded_n)),2)}`],['Mean Completion Length',fmt(mean(c.map(x=>x.completion_length)),1)],['Parser Failure / Q Active',`${c.filter(x=>x.parser_success===false).length}/${c.length} / ${c.filter(x=>num(x.Q)>0).length}`]]);$('probeDsrTimeline').hidden=true;$('probeDetail').innerHTML=`<div class="trace-head">Step ${selected.step} · ${escapeHtml(selected.group_id)} · Think-only</div><div class="trace">${c.map(x=>probeCandidate(x,selected)).join('')}</div>`}
  renderProbes=()=>enabled()?compositeProbes():old.renderProbes();renderOverview=()=>{old.renderOverview();drawOverview()};ensureOverview();
  function modelInput(g){
    const first=(g.candidates||[])[0]||{};
    return `<details class="composite-input" open><summary>Model Input</summary><div class="composite-input-meta"><span class="summary-chip">recommendation_group_id: ${escapeHtml(String(g.group_id??'—'))}</span><span class="summary-chip">target_domain: ${escapeHtml(String(first.target_domain??'—'))}</span></div><pre>${escapeHtml(first.prompt||'该历史记录未捕获完整模型输入。')}</pre></details>`;
  }
  function rewardReference(g){
    const units=(g.gold_interest_units||[]).map(unit=>`<div><b>Gold #${unit.index}</b> ${escapeHtml(unit.normalized_text||'—')} <span class="beam-meta">${escapeHtml(JSON.stringify(unit.grounded_evidence_sids||[]))}</span></div>`).join('');
    return `<div class="reward-reference"><strong>Reward-only Reference · NOT MODEL INPUT</strong><span>以下 Gold 仅供奖励解释，模型生成 sampled CoT 时不可见。</span>${units?`<div class="match-list">${units}</div>`:''}</div>`;
  }
  function capturedBeamSummary(c,g){
    return `<div class="beam32-box"><div class="beam32-summary"><b>Beam32 Summary</b><div class="beam32-flow">Sampled CoT + Fixed Domain Prefix → Beam32 → exactly 3 generated tokens: A / B / C</div><div>target_domain <b>${escapeHtml(String(c.target_domain??'—'))}</b> · fixed domain prefix <b>${escapeHtml(String(c.domain_prefix??'—'))}</b> · fixed <b>${c.beam_fixed_domain_prefix===true?'YES':'NO'}</b></div><div>Beam raw <b>${fmt(c.beam_raw)}</b> · Exact / AB / A / Invalid <b>${c.beam_exact_count??'—'} / ${c.beam_ab_count??'—'} / ${c.beam_a_count??'—'} / ${c.beam_invalid_count??'—'}</b></div><button type="button" class="beam32-toggle" data-step="${g.step}" data-rollout-id="${g.rollout_id}" data-group-id="${escapeHtml(String(g.group_id))}" data-origin-rank="${c.rank}" data-local-index="${c.local_index}">Show 32 Beams</button></div><div class="beam32-detail" hidden></div></div>`;
  }
  function richCard(c,g){
    const a=num(c.final_sequence_advantage),tone=a>0?'positive':a<0?'negative':'';
    const beamFirst=c.interest_tiebreak_scale!=null;
    const rewardDetail=beamFirst
      ? `<div><span class="label">实际 Tie-break Scale / 上限</span><b>${fmt(c.interest_tiebreak_scale,4)} / 0.2500</b></div><div><span class="label">兴趣加成</span><b>${fmt(c.cot_contribution,4)}</b></div><div><span class="label">Reward 公式</span><b>${fmt(c.beam_raw)} + ${fmt(c.cot_contribution,4)}</b></div><div><span class="label">U_beam（仅监控）</span><b>${fmt(c.beam_utility)}</b></div>`
      : `<div><span class="label">U_beam</span><b>${fmt(c.beam_utility)}</b></div><div><span class="label">Reward Contributions</span><b>Beam ${fmt(c.beam_contribution)} + CoT ${fmt(c.cot_contribution)}</b></div>`;
    return `<article class="composite-card ${tone}"><div class="composite-card-head"><span>Candidate #${c.candidate_id}</span><span class="summary-chip">${c.parser_success===false?'Parser failure':`Matched ${c.matched_interest_count}/${c.gold_interest_count}`}</span></div><div class="composite-primary"><div><span class="label">${beamFirst?'Beam-first Reward':'Composite Reward'}</span><b>${fmt(c.composite_reward,4)}</b></div><div><span class="label">Final Advantage</span><b class="adv-${creditTone(a)}">${signed(a,4)}</b></div><div><span class="label">Beam 主层级</span><b>${fmt(c.beam_raw)}</b></div></div><div class="composite-secondary"><div><span class="label">U_cot</span><b>${fmt(c.cot_utility)}</b></div>${rewardDetail}<div><span class="label">Completion Length</span><b>${c.completion_length??'—'}</b></div><div><span class="label">Raw N / Grounded N</span><b>${c.raw_n??'—'} / ${c.grounded_n??'—'}</b></div><div><span class="label">Grounding Coverage</span><b>${c.grounding_coverage==null?'未定义':pct(c.grounding_coverage)}</b></div><div><span class="label">Parser Status</span><b>${c.parser_success===false?escapeHtml(c.parser_failure_reason||'失败'):'成功'}</b></div></div><div class="sampled-cot"><h4>Sampled CoT</h4><pre>${escapeHtml(c.completion||'—')}</pre></div>${capturedBeamSummary(c,g)}<div class="match-list">${matching(c,g)}</div></article>`;
  }
  function relationLabel(value){return value==='VALID_NO_HIT'?'VALID NO HIT':value||'—'}
  function renderBeamDetails(payload){
    if(!payload.supported)return `<div class="adv-empty">${escapeHtml(payload.unavailable_reason||'Beam details unavailable for this legacy run.')}</div>`;
    const summary=payload.summary||{},counts=summary.relation_counts||{},beams=payload.beams||[];
    const rows=beams.map(beam=>{
      const relation=beam.relation_to_gold||'INVALID',tokens=(beam.generated_tokens||[]).map(token=>`<span class="beam-token">${escapeHtml(String(token))}</span>`).join('');
      return `<div class="beam-row"><div class="beam-row-head"><b>Beam #${String(beam.beam_index??0).padStart(2,'0')}</b><b class="relation-${escapeHtml(relation)}">${escapeHtml(relationLabel(relation))}</b></div><div>${tokens}</div><div class="beam-meta">generated_token_ids: ${escapeHtml(JSON.stringify(beam.generated_token_ids||[]))}</div><div class="beam-meta">parsed_sid: ${escapeHtml(JSON.stringify(beam.parsed_sid??null))}</div><div class="beam-meta">text: ${escapeHtml(beam.generated_continuation_text||'—')}</div></div>`;
    }).join('');
    return `<div><b>Captured Beam32 · ${summary.abc_parse_success_count??'—'}/${summary.beam_count??'—'} ABC parse success</b><div class="beam-meta">EXACT ${counts.EXACT??0} · AB ${counts.AB??0} · A ${counts.A??0} · VALID NO HIT ${counts.VALID_NO_HIT??0} · INVALID ${counts.INVALID??0}</div></div>${rows}`;
  }
  async function toggleBeamDetails(button){
    const target=button.closest('.beam32-box').querySelector('.beam32-detail');
    if(button.dataset.loaded==='true'){
      target.hidden=!target.hidden;
      button.textContent=target.hidden?'Show 32 Beams':'Hide 32 Beams';
      return;
    }
    button.disabled=true;button.textContent='Loading captured Beams…';target.hidden=false;target.innerHTML='<div class="adv-empty">正在按联合键读取真实 Beam32…</div>';
    try{
      const url=new URL(apiUrl('/api/composite-beams'),window.location.origin);
      for(const [key,value] of Object.entries({step:button.dataset.step,rollout_id:button.dataset.rolloutId,recommendation_group_id:button.dataset.groupId,origin_rank:button.dataset.originRank,local_index:button.dataset.localIndex}))url.searchParams.set(key,value);
      const response=await fetch(url,{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      target.innerHTML=renderBeamDetails(await response.json());button.dataset.loaded='true';button.textContent='Hide 32 Beams';
    }catch(error){target.innerHTML=`<div class="adv-empty">Beam32 读取失败：${escapeHtml(error.message||String(error))}</div>`;button.textContent='Retry 32 Beams'}finally{button.disabled=false}
  }
  function bindBeamToggles(){
    document.querySelectorAll('#advantageContent .beam32-toggle').forEach(button=>{
      if(button.dataset.bound)return;
      button.dataset.bound='true';button.addEventListener('click',()=>toggleBeamDetails(button));
    });
  }
  window.renderCompositeThinkAdvantage=g=>{
    const beamFirst=g.interest_tiebreak_scale!=null||(g.candidates||[]).some(c=>c.interest_tiebreak_scale!=null);
    const flags=[beamFirst&&g.strict_reversal_count===0?'<span class="summary-chip good">Beam-first 排序已校验 · STRICT REVERSAL 0</span>':'',g.beam_all_equal&&!g.composite_all_equal?'<span class="summary-chip good">同 Beam 内由 CoT 打破平局 · RESCUED</span>':'',g.composite_all_equal?'<span class="summary-chip alert">Composite 仍无组内信号</span>':'',g.top_set_tie_break?'<span class="summary-chip good">Beam 并列第一 → CoT 负责打破平局</span>':'',g.strict_beam_reversal?'<span class="summary-chip alert">错误：兴趣分跨越 Beam 层级</span>':''].join('');
    const formula=beamFirst?`实际 scale ${fmt(g.interest_tiebreak_scale,4)}（上限 0.25） · R = Beam raw + scale × U_cot · strict reversal ${g.strict_reversal_count??0}`:'旧版 Balanced Composite 公式';
    return `<section class="adv-group"><header class="adv-group-head"><div><div class="adv-title">Step ${g.step??'—'} · Think G4 · ${beamFirst?'Beam-first Composite Reward':'Balanced Composite Reward'} ${capturedBadge()}</div><div class="adv-sub">${beamFirst?'Beam 决定主层级，CoT 兴趣分只在安全边界内辅助排序。':'Sequence-level Advantage：整个 CoT 共享一个 final advantage。'} 最终 Advantage 仍按组内 population std 归一化。</div></div><div class="summary-chips">${flags}</div></header>${modelInput(g)}${rewardReference(g)}<div class="composite-callout">${formula}<br>Beam ${vector(g.beam_raw_vector)} · U_cot ${vector(g.cot_utility_vector)} · Final Reward ${vector(g.composite_reward_vector)} · population std ${fmt(g.composite_reward_population_std,5)}</div><div class="composite-card-grid">${(g.candidates||[]).map(c=>richCard(c,g)).join('')}</div></section>`;
  };
  const refreshWithoutCompositeSummary=refresh;
  refresh=async function(force=false){
    await refreshWithoutCompositeSummary(force);
    if(enabled()&&(autoRefresh||force))await loadSummary();
  };
  const renderCompositeAdvantages=renderAdvantages;
  renderAdvantages=function(){renderCompositeAdvantages();if(enabled())bindBeamToggles()};
})();
