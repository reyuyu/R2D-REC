(()=>{
  const SAMPLE8_EXPERIMENT='GR_REC_ThinkSample8_FullSID_v3';
  const POSITIVE_A0_EXPERIMENT='GR_REC_ThinkSample8_FullSID_PositiveA0_v3';
  const positiveA0Enabled=()=>state?.manifest?.experiment===POSITIVE_A0_EXPERIMENT;
  const sample8Enabled=()=>[SAMPLE8_EXPERIMENT,POSITIVE_A0_EXPERIMENT].includes(state?.manifest?.experiment);
  const enabled=()=>['GR_REC_ThinkDualBeam8_v2',SAMPLE8_EXPERIMENT,POSITIVE_A0_EXPERIMENT].includes(state?.manifest?.experiment);
  const expandedDetails=new Map();
  const number=(value,digits=4)=>{
    if(value==null||!Number.isFinite(Number(value)))return '—';
    return Number(value).toFixed(digits);
  };
  const signedNumber=(value,digits=4)=>{
    if(value==null||!Number.isFinite(Number(value)))return '—';
    const numeric=Number(value);
    return `${numeric>0?'+':''}${numeric.toFixed(digits)}`;
  };
  const tone=value=>Number(value)>0?'positive':Number(value)<0?'negative':'zero';
  const relationLabel=level=>({
    exact:'Exact',ab:'AB',a:'A',domain:'Domain',wrong_domain:'Wrong Domain',invalid:'Invalid',
  }[String(level||'').toLowerCase()]||level||'—');
  const captureBadge=()=>'<span class="provenance-badge" title="训练时由 Python 直接捕获，前端未重算">实采</span>';
  const stableGroupKey=group=>`${group.group_id||'group'}:${group.step??'step'}:${group.rollout_id??'rollout'}`;
  const detailsAttributes=(key,defaultOpen=false)=>{
    const open=expandedDetails.has(key)?expandedDetails.get(key):defaultOpen;
    return `data-dual-detail="${escapeHtml(key)}"${open?' open':''}`;
  };
  const rememberExpandedDetails=()=>{
    document.querySelectorAll('.dual-beam8-group details[data-dual-detail]').forEach(details=>{
      expandedDetails.set(details.dataset.dualDetail,details.open);
    });
  };
  const domainLabel=domain=>({video:'Video',ad:'Ad',prod:'Prod',living:'Living'}[String(domain||'').toLowerCase()]||domain||'—');

  function renderSidRows(candidate,cotKey){
    return (candidate.sid_candidates||[]).map(sid=>{
      const adv=tone(sid.advantage);
      const level=String(sid.reward_level||'').toLowerCase();
      const aOnlyRemoved=positiveA0Enabled()&&level==='a';
      const sidKey=`${cotKey}:sid:${sid.candidate_id}`;
      const multi=sid.multi_sid_output?'<span class="summary-chip alert">多 SID · 训练取第一个</span>':'';
      const status=sid.parser_status?`<span class="summary-chip ${sid.parser_status==='ok'?'good':'alert'}">${escapeHtml(sid.parser_status)}</span>`:'';
      const suppression=aOnlyRemoved?'<span class="dual-a0-badge">A-only 已清零</span>':'';
      const continuation=sid.continuation_text==null?'':`<details class="dual-continuation" ${detailsAttributes(`${sidKey}:continuation`)}><summary>查看完整 sampled continuation</summary><pre>${escapeHtml(sid.continuation_text)}</pre></details>`;
      return `<tr class="dual-sid-row ${adv}${aOnlyRemoved?' a0-suppressed':''}">
        <td>#${String(sid.candidate_id).padStart(2,'0')}</td>
        <td><code>${escapeHtml(sid.parsed_sid_text||'解析失败')}</code><div class="dual-parser-badges">${status}${multi}</div>${continuation}</td>
        <td><span class="dual-level level-${escapeHtml(level)}">${escapeHtml(relationLabel(sid.reward_level))}</span>${suppression}</td>
        <td><b>${number(sid.reward)}</b>${aOnlyRemoved?'<small class="dual-a0-note">V3 0.5 → 本阶段 0</small>':''}</td>
        <td class="adv-${adv}"><b>${signedNumber(sid.advantage)}</b></td>
        <td><code>${escapeHtml((sid.sid_action_span||[]).join(' → ')||'—')}</code><details ${detailsAttributes(`${sidKey}:tokens`)}><summary>token IDs (${(sid.generated_token_ids||[]).length})</summary><code class="dual-token-ids">${escapeHtml((sid.generated_token_ids||[]).join(', '))}</code></details></td>
      </tr>`;
    }).join('');
  }

  function renderCot(candidate,groupKey){
    const adv=tone(candidate.final_advantage);
    const label=adv==='positive'?'正优势':adv==='negative'?'负优势':'零优势';
    const cotKey=`${groupKey}:cot:${candidate.candidate_id}`;
    const aOnly=positiveA0Enabled()?(candidate.sid_candidates||[]).filter(sid=>String(sid.reward_level||'').toLowerCase()==='a').length:0;
    return `<article class="dual-cot-card ${adv}">
      <header class="dual-cot-head">
        <div><strong>CoT #${candidate.candidate_id}</strong><span class="summary-chip ${adv==='positive'?'good':adv==='negative'?'alert':''}">${label}</span></div>
        <div class="dual-cot-score">
          <span>CoT 分数 ${captureBadge()} <b>${number(candidate.reward)}</b></span>
          <span>CoT 优势 ${captureBadge()} <b class="adv-${adv}">${signedNumber(candidate.final_advantage)}</b></span>
        </div>
      </header>
      <div class="dual-meta">
        <span>长度 ${candidate.cot_length??'—'}</span>
        <span>${candidate.closed?'已闭合 </think>':'未闭合'}</span>
        <span>${escapeHtml(candidate.generation_mode||'Beam8')} ${number(candidate.beam8_wall_sec,2)}s</span>
        <span>SID std ${number(candidate.sid_population_std,5)}</span>
        <span>${candidate.sid_zero_std?'SID 零方差':'SID 有组内信号'}</span>
      </div>
      <details class="dual-cot-text" ${detailsAttributes(`${cotKey}:text`)}><summary>查看采样 CoT 原文</summary><pre>${escapeHtml(candidate.cot_text||'')}</pre></details>
      <div class="dual-beam-summary">
        <span>Exact ${candidate.exact??0}</span><span>AB ${candidate.ab??0}</span>
        <span>A ${candidate.a??0}</span><span>Domain ${candidate.domain??0}</span>
        <span>Wrong Domain ${candidate.wrong_domain??0}</span><span>Invalid ${candidate.invalid??0}</span>
        ${positiveA0Enabled()?`<span class="dual-a0-summary">A-only 清零 ${aOnly}/8</span>`:''}
        ${sample8Enabled()?'<span>无固定 Domain 前缀 · 全文扫描首个完整 SID</span>':`<span>固定前缀 <code>${escapeHtml(candidate.domain_prefix||'—')}</code></span>`}
      </div>
      <div class="table-scroll"><table class="dual-sid-table">
        <thead><tr><th>答案</th><th>${sample8Enabled()?'首个完整 SID / continuation':'Beam8 SID'}</th><th>层级</th><th>答案分数 ${captureBadge()}</th><th>答案优势 ${captureBadge()}</th><th>${sample8Enabled()?'SID span / continuation IDs':'3 token IDs'}</th></tr></thead>
        <tbody>${renderSidRows(candidate,cotKey)}</tbody>
      </table></div>
    </article>`;
  }

  const oldThinkAdvantage=renderThinkAdvantage;
  renderThinkAdvantage=function(group){
    if(!enabled()||group?.kind!=='dual_beam8_advantage')return oldThinkAdvantage(group);
    const groupKey=stableGroupKey(group);
    const sidRows=(group.candidates||[]).flatMap(candidate=>candidate.sid_candidates||[]);
    const positiveSid=sidRows.filter(sid=>Number(sid.advantage)>0).length;
    const negativeSid=sidRows.filter(sid=>Number(sid.advantage)<0).length;
    const aOnlyRemoved=positiveA0Enabled()?sidRows.filter(sid=>String(sid.reward_level||'').toLowerCase()==='a').length:0;
    const signalG8=(group.candidates||[]).filter(candidate=>!candidate.sid_zero_std).length;
    const positiveCot=(group.candidates||[]).filter(candidate=>Number(candidate.final_advantage)>0).length;
    const negativeCot=(group.candidates||[]).filter(candidate=>Number(candidate.final_advantage)<0).length;
    const gold=(group.gold_sids||[]).map(value=>`<code>${escapeHtml(value)}</code>`).join('');
    return `<section class="adv-group dual-beam8-group">
      <header class="adv-group-head">
        <div>
          <div class="adv-title">Step ${group.step??'—'} · Rollout ${group.rollout_id??'—'} · ${group.sample8_fullsid?'Sample8 FullSID':'Dual Beam8'} 两级优势 ${captureBadge()}</div>
          <div class="adv-sub">G4 只归一化四条 CoT；每条 CoT 下的 8 个 SID 独立做 G8 归一化，绝不按 G32 混合。</div>
        </div>
        <div class="summary-chips">
          <span class="dual-domain-badge domain-${escapeHtml(String(group.target_domain||'').toLowerCase())}">${escapeHtml(domainLabel(group.target_domain))}</span>
          <span class="summary-chip ${group.cot_zero_std?'alert':'good'}">${group.cot_zero_std?'CoT 零方差':'CoT 有组内信号'}</span>
          <span class="summary-chip">CoT std ${number(group.cot_population_std,5)}</span>
        </div>
      </header>
      ${positiveA0Enabled()?'<div class="dual-contract"><strong>Positive-A0 训练合同</strong><span>invalid -1 · wrong-domain -0.25 · no-hit 0 · <b>A-only 0</b> · AB 2 · Exact 8</span><span>Probe 仍使用 production 原始 reward（A=0.5），不会与本栏训练分数混算。</span></div>':''}
      <div class="dual-kpis">
        <div><span>CoT 正 / 负优势</span><strong class="kpi-split"><b>${positiveCot}</b> / <em>${negativeCot}</em></strong></div>
        <div><span>SID 正 / 负优势</span><strong class="kpi-split"><b>${positiveSid}</b> / <em>${negativeSid}</em></strong></div>
        <div><span>有信号 G8</span><strong>${signalG8} / 4</strong></div>
        ${positiveA0Enabled()?`<div class="a0"><span>A-only 奖励被清零</span><strong>${aOnlyRemoved} / 32</strong></div>`:''}
      </div>
      <details class="dual-input" ${detailsAttributes(`${groupKey}:input`,true)}><summary>模型输入原文</summary><pre>${escapeHtml(group.prompt||'当前记录无法定位原始 prompt')}</pre></details>
      <details class="dual-gold" ${detailsAttributes(`${groupKey}:gold`)}><summary>Reward-only Reference · NOT MODEL INPUT</summary><div class="dual-gold-list">${gold||'—'}</div></details>
      <div class="dual-cot-list">${(group.candidates||[]).map(candidate=>renderCot(candidate,groupKey)).join('')}</div>
    </section>`;
  };

  const oldRenderAdvantages=renderAdvantages;
  renderAdvantages=function(){
    rememberExpandedDetails();
    const tab=document.querySelector('.tab[data-view="advantages"]');
    if(tab)tab.textContent=enabled()?'CoT / SID 优势':'优势可解释性';
    oldRenderAdvantages();
  };

  const oldRenderProbes=renderProbes;
  renderProbes=function(){
    oldRenderProbes();
    if(!positiveA0Enabled())return;
    const detail=document.getElementById('probeDetail');
    if(detail)detail.insertAdjacentHTML('afterbegin','<div class="dual-probe-contract"><strong>Probe 是 production 评价</strong><span><b class="probe-hit exact">Exact</b> 8 · <b class="probe-hit ab">AB</b> 2 · <b class="probe-hit a">A</b> 0.5。这里不应用 Positive-A0 的 A-only 清零。</span></div>');
  };

  document.addEventListener('toggle',event=>{
    const details=event.target.closest?.('details[data-dual-detail]');
    if(details)expandedDetails.set(details.dataset.dualDetail,details.open);
  },true);

  const style=document.createElement('style');
  style.textContent=`
    .dual-beam8-group{max-width:1500px;color:#18212b}
    .dual-contract,.dual-probe-contract{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0;padding:10px 12px;border:1px solid #b8c5d1;border-left:4px solid #245d7d;background:#f2f7fa;color:#23313e;font-size:12px}
    .dual-contract strong,.dual-probe-contract strong{font-size:13px;color:#123d56}.dual-contract span:last-child{margin-left:auto;color:#52606d}
    .dual-kpis{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;margin:12px 0}
    .dual-kpis>div{display:flex;justify-content:space-between;gap:10px;padding:9px 11px;border:1px solid #d5dce3;background:#f8fafb;font-size:12px}
    .dual-kpis span{color:#5b6672}.dual-kpis strong{color:#17212b}.dual-kpis .a0{border-color:#d2ae57;background:#fff9e8}.dual-kpis .a0 strong{color:#755200}
    .kpi-split b{color:#087752}.kpi-split em{color:#b72f44;font-style:normal}
    .dual-domain-badge{display:inline-flex;align-items:center;padding:4px 8px;border:1px solid #99a6b2;background:#f4f6f8;color:#24313c;font-size:11px;font-weight:800;text-transform:uppercase}
    .domain-video{border-color:#4583a5;background:#e9f4fa;color:#174e6c}.domain-ad{border-color:#a17c2c;background:#fff6d8;color:#654a0a}.domain-prod{border-color:#5c8a62;background:#edf7ee;color:#2d5c34}.domain-living{border-color:#9a6076;background:#faedf2;color:#6b2942}
    .dual-input,.dual-gold,.dual-cot-text{margin-top:12px;border:1px solid #d7dde5;background:#f8fafc;color:#1c2732}
    .dual-input summary,.dual-gold summary,.dual-cot-text summary{cursor:pointer;padding:9px 12px;font-weight:750}
    .dual-input pre,.dual-cot-text pre{max-height:320px;overflow:auto;margin:0;padding:12px;border-top:1px solid #d7dde5;white-space:pre-wrap;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}
    .dual-gold-list{display:flex;flex-wrap:wrap;gap:6px;padding:10px 12px;border-top:1px solid #d7dde5}
    .dual-cot-list{display:grid;gap:14px;margin-top:14px}
    .dual-cot-card{border:1px solid #d6dce4;border-left:5px solid #8b96a5;background:#fff;color:#18212b}
    .dual-cot-card.positive{border-left-color:#13815b;background:#fbfffd}.dual-cot-card.negative{border-left-color:#c84555;background:#fffafb}.dual-cot-card.zero{background:#fbfcfd}
    .dual-cot-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 14px;border-bottom:1px solid #e3e7ec}
    .dual-cot-head>div:first-child{display:flex;align-items:center;gap:8px}.dual-cot-score{display:flex;gap:18px;flex-wrap:wrap}
    .dual-meta,.dual-beam-summary{display:flex;gap:12px;flex-wrap:wrap;padding:8px 14px;color:#55606f;font-size:12px}
    .dual-beam-summary{background:#f5f7f9;border-top:1px solid #e5e9ee;border-bottom:1px solid #e5e9ee}
    .dual-sid-table{min-width:980px;width:100%;border-collapse:collapse;font-size:12px}
    .dual-sid-table th,.dual-sid-table td{padding:8px 10px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:top}
    .dual-sid-table th{background:#f7f8fa;color:#4a5563;white-space:nowrap}
    .dual-sid-row.positive td{background:#f0faf5}.dual-sid-row.negative td{background:#fff3f4}.dual-sid-row.zero td{background:#fff}
    .dual-sid-row.positive td:first-child{box-shadow:inset 4px 0 #159467}.dual-sid-row.negative td:first-child{box-shadow:inset 4px 0 #d1495b}
    .dual-sid-row.a0-suppressed td{background:#fff9e8}.dual-sid-row.a0-suppressed td:first-child{box-shadow:inset 4px 0 #cf9f22}
    .dual-sid-table code{font-size:11px;white-space:nowrap}
    .dual-parser-badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
    .dual-continuation{margin-top:7px}.dual-continuation summary{cursor:pointer;color:#315a7d}
    .dual-continuation pre{max-width:640px;max-height:190px;overflow:auto;white-space:pre-wrap;background:#f7f9fb;border:1px solid #dde3e9;padding:8px;font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}
    .dual-token-ids{display:block;max-width:420px;max-height:100px;overflow:auto;white-space:normal!important;margin-top:5px}
    .dual-level{display:inline-block;padding:2px 6px;border:1px solid #ccd3dc;background:#f7f8fa}
    .level-exact{color:#086947;border-color:#81bba4;background:#eaf7f1}.level-ab{color:#174e6c;border-color:#78a9c3;background:#eaf4fa}.level-a{color:#765300;border-color:#d2ae57;background:#fff7dc}.level-invalid,.level-wrong_domain{color:#a32738;border-color:#d89da6;background:#fff1f2}
    .dual-a0-badge{display:block;width:max-content;margin-top:5px;padding:2px 5px;background:#7a5907;color:#fff;font-size:10px;font-weight:800}.dual-a0-note{display:block;margin-top:4px;color:#765300;font-weight:750}.dual-a0-summary{color:#765300;font-weight:800}
    .probe-hit{display:inline-block;padding:2px 5px;border:1px solid}.probe-hit.exact{color:#086947;border-color:#81bba4;background:#eaf7f1}.probe-hit.ab{color:#174e6c;border-color:#78a9c3;background:#eaf4fa}.probe-hit.a{color:#765300;border-color:#d2ae57;background:#fff7dc}
    @media(max-width:900px){.dual-cot-head{align-items:flex-start;flex-direction:column}.dual-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.dual-contract span:last-child{margin-left:0}}
    @media(max-width:560px){.dual-kpis{grid-template-columns:1fr}.dual-cot-score{gap:8px}.dual-meta,.dual-beam-summary{gap:7px}.dual-input pre,.dual-cot-text pre{font-size:11px}.dual-contract,.dual-probe-contract{align-items:flex-start;flex-direction:column}}
  `;
  document.head.appendChild(style);
})();
