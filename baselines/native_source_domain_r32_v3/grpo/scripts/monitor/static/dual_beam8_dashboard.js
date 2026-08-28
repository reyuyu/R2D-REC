(()=>{
  const enabled=()=>state?.manifest?.experiment==='GR_REC_ThinkDualBeam8_v2';
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
    exact:'Exact',ab:'AB',a:'A',domain:'Domain',invalid:'Invalid',
  }[String(level||'').toLowerCase()]||level||'—');
  const captureBadge=()=>'<span class="provenance-badge" title="训练时由 Python 直接捕获，前端未重算">实采</span>';

  function renderSidRows(candidate){
    return (candidate.sid_candidates||[]).map(sid=>{
      const adv=tone(sid.advantage);
      return `<tr>
        <td>#${String(sid.candidate_id).padStart(2,'0')}</td>
        <td><code>${escapeHtml(sid.parsed_sid_text||'解析失败')}</code></td>
        <td><span class="dual-level level-${escapeHtml(String(sid.reward_level||'').toLowerCase())}">${escapeHtml(relationLabel(sid.reward_level))}</span></td>
        <td><b>${number(sid.reward)}</b></td>
        <td class="adv-${adv}"><b>${signedNumber(sid.advantage)}</b></td>
        <td><code>${escapeHtml((sid.generated_token_ids||[]).join(', '))}</code></td>
      </tr>`;
    }).join('');
  }

  function renderCot(candidate){
    const adv=tone(candidate.final_advantage);
    const label=adv==='positive'?'正优势':adv==='negative'?'负优势':'零优势';
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
        <span>Beam8 ${number(candidate.beam8_wall_sec,2)}s</span>
        <span>SID std ${number(candidate.sid_population_std,5)}</span>
        <span>${candidate.sid_zero_std?'SID 零方差':'SID 有组内信号'}</span>
      </div>
      <details class="dual-cot-text"><summary>查看采样 CoT</summary><pre>${escapeHtml(candidate.cot_text||'')}</pre></details>
      <div class="dual-beam-summary">
        <span>Exact ${candidate.exact??0}</span><span>AB ${candidate.ab??0}</span>
        <span>A ${candidate.a??0}</span><span>Invalid ${candidate.invalid??0}</span>
        <span>固定前缀 <code>${escapeHtml(candidate.domain_prefix||'—')}</code></span>
      </div>
      <div class="table-scroll"><table class="dual-sid-table">
        <thead><tr><th>答案</th><th>Beam8 SID</th><th>层级</th><th>答案分数 ${captureBadge()}</th><th>答案优势 ${captureBadge()}</th><th>3 token IDs</th></tr></thead>
        <tbody>${renderSidRows(candidate)}</tbody>
      </table></div>
    </article>`;
  }

  const oldThinkAdvantage=renderThinkAdvantage;
  renderThinkAdvantage=function(group){
    if(!enabled()||group?.kind!=='dual_beam8_advantage')return oldThinkAdvantage(group);
    const gold=(group.gold_sids||[]).map(value=>`<code>${escapeHtml(value)}</code>`).join('');
    return `<section class="adv-group dual-beam8-group">
      <header class="adv-group-head">
        <div>
          <div class="adv-title">Step ${group.step??'—'} · Rollout ${group.rollout_id??'—'} · Dual Beam8 两级优势 ${captureBadge()}</div>
          <div class="adv-sub">G4 只归一化四条 CoT；每条 CoT 下的 8 个 SID 独立做 G8 归一化，绝不按 G32 混合。</div>
        </div>
        <div class="summary-chips">
          <span class="summary-chip">Domain ${escapeHtml(group.target_domain||'—')}</span>
          <span class="summary-chip ${group.cot_zero_std?'alert':'good'}">${group.cot_zero_std?'CoT 零方差':'CoT 有组内信号'}</span>
          <span class="summary-chip">CoT std ${number(group.cot_population_std,5)}</span>
        </div>
      </header>
      <details class="dual-input" open><summary>模型输入</summary><pre>${escapeHtml(group.prompt||'当前记录无法定位原始 prompt')}</pre></details>
      <details class="dual-gold"><summary>Reward-only Reference · NOT MODEL INPUT</summary><div class="dual-gold-list">${gold||'—'}</div></details>
      <div class="dual-cot-list">${(group.candidates||[]).map(renderCot).join('')}</div>
    </section>`;
  };

  const oldRenderAdvantages=renderAdvantages;
  renderAdvantages=function(){
    const tab=document.querySelector('.tab[data-view="advantages"]');
    if(tab)tab.textContent=enabled()?'CoT / 答案优势':'优势可解释性';
    oldRenderAdvantages();
  };

  const style=document.createElement('style');
  style.textContent=`
    .dual-beam8-group{max-width:1500px}
    .dual-input,.dual-gold,.dual-cot-text{margin-top:12px;border:1px solid #d7dde5;background:#f8fafc}
    .dual-input summary,.dual-gold summary,.dual-cot-text summary{cursor:pointer;padding:9px 12px;font-weight:750}
    .dual-input pre,.dual-cot-text pre{max-height:320px;overflow:auto;margin:0;padding:12px;border-top:1px solid #d7dde5;white-space:pre-wrap;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}
    .dual-gold-list{display:flex;flex-wrap:wrap;gap:6px;padding:10px 12px;border-top:1px solid #d7dde5}
    .dual-cot-list{display:grid;gap:14px;margin-top:14px}
    .dual-cot-card{border:1px solid #d6dce4;border-left:4px solid #8b96a5;background:#fff}
    .dual-cot-card.positive{border-left-color:#13815b}.dual-cot-card.negative{border-left-color:#c84555}
    .dual-cot-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 14px;border-bottom:1px solid #e3e7ec}
    .dual-cot-head>div:first-child{display:flex;align-items:center;gap:8px}.dual-cot-score{display:flex;gap:18px;flex-wrap:wrap}
    .dual-meta,.dual-beam-summary{display:flex;gap:12px;flex-wrap:wrap;padding:8px 14px;color:#55606f;font-size:12px}
    .dual-beam-summary{background:#f5f7f9;border-top:1px solid #e5e9ee;border-bottom:1px solid #e5e9ee}
    .dual-sid-table{min-width:980px;width:100%;border-collapse:collapse;font-size:12px}
    .dual-sid-table th,.dual-sid-table td{padding:8px 10px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:top}
    .dual-sid-table th{background:#f7f8fa;color:#4a5563;white-space:nowrap}
    .dual-sid-table code{font-size:11px;white-space:nowrap}
    .dual-level{display:inline-block;padding:2px 6px;border:1px solid #ccd3dc;background:#f7f8fa}
    .level-exact{color:#086947;border-color:#81bba4;background:#eaf7f1}.level-invalid{color:#a32738;border-color:#d89da6;background:#fff1f2}
    @media(max-width:900px){.dual-cot-head{align-items:flex-start;flex-direction:column}}
  `;
  document.head.appendChild(style);
})();
