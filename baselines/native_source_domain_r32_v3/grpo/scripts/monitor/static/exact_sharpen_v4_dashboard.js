(() => {
  const advantageRouteOptions = $('advRouteSelect').innerHTML;
  const advantageFocusOptions = $('advFocusSelect').innerHTML;
  const baseRenderAdvantages = renderAdvantages;
  const baseRenderProbes = renderProbes;
  const style = document.createElement('style');
  style.textContent = `
    .v4-branch-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
    .v4-branch{border:1px solid var(--line);background:#fff;min-width:0}
    .v4-branch header{padding:10px 12px;border-bottom:1px solid var(--line);background:#f6f8f7;display:flex;justify-content:space-between;gap:8px;align-items:center}
    .v4-table{width:100%;border-collapse:collapse;font-size:11px}.v4-table th,.v4-table td{padding:7px 8px;border-bottom:1px solid #e6eaed;text-align:right;white-space:nowrap}.v4-table th:nth-child(2),.v4-table td:nth-child(2){text-align:left;white-space:normal;overflow-wrap:anywhere}.v4-table th{color:var(--muted);background:#fbfcfc}.v4-table tr.a-removed{background:#fff4e3}.v4-table tr.positive{background:#f2faf6}
    .v4-cot{margin-top:14px;border:1px solid var(--line);background:#fff}.v4-cot-head{padding:12px 14px;background:#eef3f1;border-bottom:1px solid var(--line)}.v4-cot-text{margin:9px 0 0;max-height:130px;overflow:auto;white-space:pre-wrap;font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;color:#3c4d55}
    .v4-flag{display:inline-block;padding:2px 6px;border:1px solid #d79c43;background:#fff4df;color:#81530d;font-weight:700;font-size:10px}.v4-flag.ok{border-color:#82b39e;background:#eff8f3;color:#28634c}
    .v4-probe-table td,.v4-probe-table th{text-align:left}.v4-probe-beams{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px;margin-top:7px;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#455}
    @media(max-width:1000px){.v4-branch-grid{grid-template-columns:1fr}.v4-probe-beams{grid-template-columns:1fr}}
  `;
  document.head.appendChild(style);

  function isV4(){return Boolean(state.capabilities?.exact_sharpen_v4||state.manifest?.experiment==='GR_REC_ThinkExactSharpen_v4')}
  function setV4Controls(){
    if($('advRouteSelect').dataset.v4)return;
    $('advRouteSelect').dataset.v4='1';
    $('advRouteSelect').innerHTML='<option value="">Free + Official</option><option value="free">Free G8</option><option value="official">Official G8</option>';
    $('advFocusSelect').innerHTML='<option value="">全部候选</option><option value="a_removed">A 奖励已删除</option><option value="saturated">已饱和分支</option><option value="nonzero">非零 Advantage</option><option value="duplicate">重复惩罚</option><option value="exact">Exact</option>';
  }
  function restoreControls(){
    if(!$('advRouteSelect').dataset.v4)return;
    delete $('advRouteSelect').dataset.v4;
    $('advRouteSelect').innerHTML=advantageRouteOptions;
    $('advFocusSelect').innerHTML=advantageFocusOptions;
  }
  function sid(value){return Array.isArray(value)?`${value[0]}:${value.slice(1).join('/')}`:'未解析'}
  function branchRows(cot,branch,focus){
    return (cot[branch]||[]).filter(row=>{
      if(focus==='a_removed')return row.a_reward_removed;
      if(focus==='saturated')return true;
      if(focus==='nonzero')return Number(row.advantage)!==0;
      if(focus==='duplicate')return Number(row.duplicate_penalty)!==0;
      if(focus==='exact')return Number(row.raw_reward)===8;
      return true
    })
  }
  function renderBranch(group,cot,branch,focus){
    const saturated=group[`${branch}_saturated`],aPlus=group[`${branch}_a_plus_count`],rows=branchRows(cot,branch,focus);
    const body=rows.map(row=>`<tr class="${row.a_reward_removed?'a-removed':Number(row.advantage)>0?'positive':''}"><td>#${row.candidate_id+1}</td><td>${escapeHtml(sid(row.sid))}<br><span class="label">ids ${escapeHtml((row.token_ids||[]).join(','))}</span></td><td>${fmt(row.raw_reward)}</td><td>${fmt(row.saturation_reward)}</td><td>${signed(row.duplicate_penalty,2)}</td><td><b>${fmt(row.shaped_reward)}</b></td><td class="adv-${creditTone(row.advantage)}"><b>${signed(row.advantage,4)}</b></td><td>${row.a_reward_removed?'<span class="v4-flag">A 已删除</span>':row.exact_duplicate_exempt?'<span class="v4-flag ok">Exact 免重复罚</span>':'-'}</td></tr>`).join('');
    return `<section class="v4-branch"><header><div><b>${branch==='free'?'Free Sample8':'Official Sample8'}</b> · 独立 G8 normalization</div><div><span class="v4-flag ${saturated?'':'ok'}">${saturated?'SATURATED':'UNSATURATED'}</span> <span class="summary-chip">A+ ${aPlus}/32</span></div></header><div class="table-scroll"><table class="v4-table"><thead><tr><th>#</th><th>输出 SID / token ids</th><th>Raw</th><th>饱和后</th><th>重复罚</th><th>Final shaped</th><th>Advantage</th><th>规则命中</th></tr></thead><tbody>${body||'<tr><td colspan="8">当前筛选无候选</td></tr>'}</tbody></table></div></section>`
  }
  function renderV4Advantages(){
    setV4Controls();
    const groups=(state.advantages?.groups||[]).filter(group=>group.valid&&group.kind==='exact_sharpen_v4');
    const select=$('advGroupSelect'),current=select.value;
    select.innerHTML=groups.length?[...groups].reverse().map(group=>`<option value="${escapeHtml(groupKey(group))}">Step ${group.step} · ${escapeHtml(String(group.group_id).slice(0,42))}</option>`).join(''):'<option value="">等待首个 V4 rollout</option>';
    if(groups.some(group=>groupKey(group)===current))select.value=current;
    const selected=groups.find(group=>groupKey(group)===select.value)||groups.at(-1);
    if(!selected){$('advantageContent').innerHTML='<div class="adv-empty">等待 V4 rollout 明细写入。</div>';return}
    select.value=groupKey(selected);
    const route=$('advRouteSelect').value,focus=$('advFocusSelect').value;
    const cots=selected.cots.map(cot=>{
      const coverage=cot.cot_coverage||{};
      const branches=[route||'free',...(route?[]:['official'])].filter(branch=>focus!=='saturated'||selected[`${branch}_saturated`]);
      return `<article class="v4-cot"><div class="v4-cot-head"><div class="adv-title">CoT ${cot.cot_id+1} · reward ${fmt(cot.cot_reward)} · G4 advantage <span class="adv-${creditTone(cot.cot_advantage)}">${signed(cot.cot_advantage,4)}</span> ${provenanceBadge()}</div><div class="summary-chips"><span class="summary-chip">unique Exact ${coverage.unique_exact??0}</span><span class="summary-chip">uncovered AB ${coverage.unique_uncovered_ab??0}</span><span class="summary-chip">uncovered A ${coverage.unique_uncovered_a??0}</span></div><details><summary>查看 CoT 文本</summary><pre class="v4-cot-text">${escapeHtml(cot.cot_text||'')}</pre></details></div><div class="v4-branch-grid">${branches.map(branch=>renderBranch(selected,cot,branch,focus)).join('')}</div></article>`
    }).join('');
    $('advantageContent').innerHTML=`<section class="adv-group"><header class="adv-group-head"><div><div class="adv-title">Step ${selected.step} · Business Group ${escapeHtml(selected.group_id)}</div><div class="adv-sub">训练时捕获的两路 G8 明细；仅 CoT G4 advantage 从捕获 reward 只读复算。Fingerprint ${escapeHtml(selected.rollout_fingerprint||'-')}</div></div><div class="summary-chips"><span class="summary-chip ${selected.free_saturated?'alert':'good'}">Free A+ ${selected.free_a_plus_count}/32 · ${selected.free_saturated?'A=0':'A=0.5'}</span><span class="summary-chip ${selected.official_saturated?'alert':'good'}">Official A+ ${selected.official_a_plus_count}/32 · ${selected.official_saturated?'A=0':'A=0.5'}</span><span class="summary-chip">1×G4 + 4×Free G8 + 4×Official G8</span></div></header>${cots}</section>`
  }
  renderAdvantages=function(){if(isV4())renderV4Advantages();else{restoreControls();baseRenderAdvantages()}};

  function compactProbeCandidate(mode,candidate,index){
    if(mode==='free')return `<tr><td>#${index+1}</td><td>${escapeHtml(sid(candidate.parsed_sid))}</td><td>${fmt(candidate.reward)}</td><td>${escapeHtml(candidate.parser_status||'-')}</td><td>${candidate.closed?'是':'否'}</td></tr>`;
    const beams=(candidate.beam_sids||[]).map(value=>`<span>${escapeHtml(sid(value))}</span>`).join('');
    return `<tr><td>#${index+1}</td><td>${fmt(candidate.reward)}</td><td>${candidate.exact??0} / ${candidate.ab??0} / ${candidate.a??0}</td><td>${candidate.invalid??0}</td><td><details><summary>Beam32 SID 明细</summary><div class="v4-probe-beams">${beams}</div></details></td></tr>`
  }
  function probeBlock(mode,data){
    const free=mode==='free',rows=(data?.candidates||[]).map((candidate,index)=>compactProbeCandidate(mode,candidate,index)).join('');
    return `<section class="probe-route"><h3>${free?'Free Probe · 自由 continuation / 首个完整 SID':'Official Probe · fixed domain begin / production Beam32 ABC3'} · reward ${fmt(data?.reward_mean)} ± ${fmt(data?.reward_std)} · denominator ${escapeHtml(data?.reward_denominator||'-')}</h3><div class="table-scroll"><table class="v4-table v4-probe-table"><thead><tr>${free?'<th>#</th><th>Parsed SID</th><th>Reward</th><th>Parser</th><th>Closed</th>':'<th>CoT</th><th>Reward</th><th>Exact / AB / A</th><th>Invalid</th><th>Beam32</th>'}</tr></thead><tbody>${rows}</tbody></table></div></section>`
  }
  function renderV4Probes(){
    $('probeRewardChart').closest('.panel').hidden=true;$('v4ProbeFreePanel').hidden=false;$('v4ProbeOfficialPanel').hidden=false;
    const configured=state.manifest.fixed_probe?.group_ids||[],groups=[...new Set([...configured,...state.probes.map(row=>row.group_id)])],groupSelect=$('probeGroupSelect'),stepSelect=$('probeStepSelect'),currentGroup=groupSelect.value;
    groupSelect.innerHTML=groups.map((gid,index)=>`<option value="${escapeHtml(gid)}">Probe ${index+1} · ${escapeHtml(gid.slice(0,12))}</option>`).join('')||'<option value="">未配置</option>';if(groups.includes(currentGroup))groupSelect.value=currentGroup;
    const group=groupSelect.value,available=state.probes.filter(row=>row.group_id===group).sort((a,b)=>a.step-b.step),current=Number(stepSelect.value);
    stepSelect.innerHTML=[...available].reverse().map(row=>`<option value="${row.step}">Step ${row.step} · ${escapeHtml(row.reason||'periodic')}</option>`).join('')||'<option value="">暂无记录</option>';if(available.some(row=>row.step===current))stepSelect.value=String(current);
    chartLabels.v4ProbeFreeRewardChart=['Free reward mean'];chartLabels.v4ProbeOfficialRewardChart=['Official reward mean'];
    draw('v4ProbeFreeRewardChart',[{data:available.map(row=>[row.step,row.probe_free?.reward_mean]),color:colors[1]}]);
    draw('v4ProbeOfficialRewardChart',[{data:available.map(row=>[row.step,row.probe_official?.reward_mean]),color:colors[2]}]);
    const selected=available.find(row=>row.step===Number(stepSelect.value));
    if(!selected){$('probeOverview').innerHTML='';$('probeDetail').innerHTML='<div class="probe-empty">等待双 Probe 数据</div>';return}
    setText('probeMeta',`同一批 sampled CoT · seed ${selected.seed} · ${sec(selected.probe_wall_sec)}`);
    const free=selected.probe_free||{},official=selected.probe_official||{};
    const aliasOk=selected.think?.reward_mean===official.reward_mean&&selected.think?.candidate_count===official.candidate_count;
    diagnosticCards('probeOverview',[["Domain / Group",`${selected.target_domain} / ${selected.group_id}`],["Free reward / std",`${fmt(free.reward_mean)} / ${fmt(free.reward_std)}`],["Free A / AB / Exact",`${free.a_count??0} / ${free.ab_count??0} / ${free.exact_count??0}`],["Free candidates / denominator",`${free.candidate_count??0} / ${free.reward_denominator||'-'}`],["Official reward / std",`${fmt(official.reward_mean)} / ${fmt(official.reward_std)}`],["Official A / AB / Exact",`${official.a_count??0} / ${official.ab_count??0} / ${official.exact_count??0}`],["Official CoTs / denominator",`${official.candidate_count??0} / ${official.reward_denominator||'-'}`],["Compatibility",aliasOk?'think → Official ✓':'think alias mismatch']]);
    $('probeDsrTimeline').hidden=true;
    $('probeDetail').innerHTML=`<div class="trace-head">Step ${selected.step} · ${escapeHtml(selected.group_id)}</div>${probeBlock('free',free)}${probeBlock('official',official)}`
  }
  renderProbes=function(){
    if(isV4())renderV4Probes();
    else{$('probeRewardChart').closest('.panel').hidden=false;$('v4ProbeFreePanel').hidden=true;$('v4ProbeOfficialPanel').hidden=true;baseRenderProbes()}
  };
})();
