(()=>{
  const enabled=()=>state?.manifest?.experiment==='GR_REC_ThinkSuffixSID_Resample_v1';
  const sidTextLocal=sid=>Array.isArray(sid)?`<|${sid[0]}_begin|><s_a_${sid[1]}><s_b_${sid[2]}><s_c_${sid[3]}>`:'-';
  const oldCandidate=renderCandidate;
  renderCandidate=function(candidate,trace,golds,advantageGroup){
    if(!enabled()||trace?.route!=='think')return oldCandidate(candidate,trace,golds,advantageGroup);
    const advantageCandidate=(advantageGroup?.candidates||[]).find(row=>Number(row.candidate_id)===Number(candidate.candidate_id));
    const completion=renderRolloutCreditCompletion(candidate,advantageCandidate);
    const credit=renderRolloutCredit(advantageGroup,advantageCandidate);
    const kind=sidKind(candidate.parsed_sid,golds);
    const multi=candidate.multi_sid_output
      ? `<span class="multi-sid-warning" title="仅监控标记；reward仍按最后一个完整SID计算">多个 SID · ${candidate.sid_count}</span>`
      : `<span class="suffix-ok">单 SID</span>`;
    const all=(candidate.all_parsed_sids||[]).map((sid,index)=>`<span>${index+1}. ${escapeHtml(sidTextLocal(sid))}</span>`).join('');
    return `<div class="candidate suffix-sid-candidate"><div><span class="badge">#${candidate.candidate_id}</span></div><div>${multi}<div class="text">${completion}</div><div class="sid parsed-sid ${kind}">${escapeHtml(sidTextLocal(candidate.parsed_sid))}${kind==='exact'?' · 正确答案':''}</div>${all?`<div class="suffix-sid-list">${all}</div>`:''}</div><div><div class="label">SID Reward</div>${fmt(candidate.reward)}</div><div><div class="label">Parser</div>${escapeHtml(candidate.parser_status||'-')}</div><div><div class="label">Suffix / CoT tokens</div>${candidate.suffix_token_count??'-'} / ${candidate.cot_token_count??'-'}</div><div><div class="label">Loss范围</div>仅 &lt;/think&gt; 后</div>${credit}</div>`;
  };

  const oldExplorer=renderExplorer;
  renderExplorer=function(){
    oldExplorer();
    if(!enabled())return;
    const rollout=selectedExplorerRollout();
    if(!rollout)return;
    const trace=selectedExplorerTrace();
    const count=document.querySelector('#trace .candidate-count');
    if(count&&trace){count.textContent=`${trace.candidates?.length??0}/${state.manifest.group_size??8} 个候选`;count.classList.toggle('incomplete',(trace.candidates?.length??0)!==Number(state.manifest.group_size??8));}
    const summary=$('rolloutSummary');
    const fields=[
      ['采样轮次',`${Number(rollout.resample_rounds_used??0)+1}/4`],
      ['累计候选',rollout.generated_candidate_total??8],
      ['零方差救活',rollout.zero_std_rescued?'是':'否'],
      ['救援耗尽',rollout.zero_std_rescue_exhausted?'是':'否'],
    ];
    summary.insertAdjacentHTML('beforeend',fields.map(([label,value])=>`<div class="stat"><div class="label">${label}</div><div class="value small">${escapeHtml(value)}</div></div>`).join(''));
  };

  const oldThinkAdvantage=renderThinkAdvantage;
  renderThinkAdvantage=function(group){
    if(!enabled()||group?.kind!=='suffix_sequence_advantage')return oldThinkAdvantage(group);
    const cards=(group.candidates||[]).map(candidate=>{
      const tone=candidate.final_advantage>0?'positive':candidate.final_advantage<0?'negative':'neutral';
      const warning=candidate.multi_sid_output?`<span class="summary-chip alert">多个 SID · ${candidate.sid_count}</span>`:'';
      return `<article class="think-card ${tone}"><div class="think-card-head"><span>Candidate ${candidate.candidate_id}</span>${warning}</div><div class="adv-metrics"><div class="adv-metric"><span class="label">SID Reward（实采）</span><b>${fmt(candidate.reward)}</b></div><div class="adv-metric"><span class="label">Group Mean（复算）</span><b>${fmt(candidate.group_mean,5)}</b></div><div class="adv-metric"><span class="label">Final Advantage（复算）</span><b class="adv-${creditTone(candidate.final_advantage)}">${signed(candidate.final_advantage)}</b></div><div class="adv-metric"><span class="label">Suffix / CoT tokens</span><b>${candidate.suffix_token_count??'-'} / ${candidate.cot_token_count??'-'}</b></div><div class="adv-metric"><span class="label">Parser</span><b>${escapeHtml(candidate.parser_status||'-')}</b></div><div class="adv-metric"><span class="label">Loss范围</span><b>仅 &lt;/think&gt; 后</b></div></div><div class="sequence-credit adv-${creditTone(candidate.final_advantage)}">Suffix token advantage：${signed(candidate.final_advantage)}</div><pre class="adv-completion">${escapeHtml(candidate.completion||'')}</pre></article>`;
    }).join('');
    return `<section class="adv-group"><header class="adv-group-head"><div><div class="adv-title">Step ${group.step??'-'} · Think G8 · SID suffix-only GRPO</div><div class="adv-sub">完整CoT仅作上下文；Final Advantage只作用于第一个 &lt;/think&gt; 后的answer tokens。</div></div><div class="summary-chips"><span class="summary-chip ${group.zero_std?'alert':'good'}">${group.zero_std?'零方差':'有效组内信号'}</span></div></header><div class="think-grid">${cards}</div></section>`;
  };

  const style=document.createElement('style');
  style.textContent='.multi-sid-warning{display:inline-flex;margin:0 0 7px;padding:3px 7px;border:1px solid #d28d98;border-radius:3px;background:#fff0f1;color:#9c2939;font-size:11px;font-weight:750}.suffix-ok{display:inline-flex;margin:0 0 7px;padding:3px 7px;border:1px solid #9cc8b2;border-radius:3px;background:#eaf6f0;color:#126648;font-size:11px}.suffix-sid-list{display:grid;gap:3px;margin-top:7px;padding:7px 9px;border-left:3px solid #9c2939;background:#fff7f7;color:#6b3a40;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.suffix-sid-candidate{grid-template-columns:56px minmax(320px,1fr) 110px 145px 150px 135px}';
  document.head.appendChild(style);
})();
