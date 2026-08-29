(() => {
  const advantageRouteOptions = $('advRouteSelect').innerHTML;
  const advantageFocusOptions = $('advFocusSelect').innerHTML;
  const baseRenderAdvantages = renderAdvantages;
  const baseRenderProbes = renderProbes;
  const style = document.createElement('style');
  style.textContent = `
    .v4-consistency{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:12px}.v4-stat{border-left:3px solid #719184;background:#f4f7f5;padding:8px 10px;min-width:0}.v4-stat b{display:block;font-size:15px}.v4-stat span{font-size:10px;color:var(--muted)}
    .v4-branch-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
    .v4-branch{border:1px solid var(--line);background:#fff;min-width:0}.v4-branch>summary,.v4-cot>summary{cursor:pointer;list-style:none}.v4-branch>summary::-webkit-details-marker,.v4-cot>summary::-webkit-details-marker{display:none}
    .v4-branch>summary{padding:9px 11px;border-bottom:1px solid transparent;background:#f6f8f7;color:var(--ink);display:flex;justify-content:space-between;gap:8px;align-items:center}.v4-branch[open]>summary{border-bottom-color:var(--line)}
    .v4-table{width:100%;border-collapse:collapse;font-size:11px}.v4-table th,.v4-table td{padding:7px 8px;border-bottom:1px solid #e6eaed;text-align:right;white-space:nowrap}.v4-table th:nth-child(2),.v4-table td:nth-child(2){text-align:left;white-space:normal;overflow-wrap:anywhere}.v4-table th{color:var(--muted);background:#fbfcfc}.v4-table tr.a-removed{background:#fff4e3}.v4-table tr.positive{background:#f2faf6}
    .v4-cot{margin-top:10px;border:1px solid var(--line);background:#fff;color:var(--ink)}.v4-cot>summary{padding:10px 12px;background:#eef3f1;color:var(--ink);display:flex;justify-content:space-between;align-items:center;gap:10px}.v4-cot[open]>summary{border-bottom:1px solid var(--line)}.v4-cot-body{padding:0 11px 11px}.v4-cot-text,.v4-sample-text{margin:8px 0 0;max-height:320px;overflow:auto;white-space:pre-wrap;font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;color:#283740;background:#f7f9f8;border:1px solid #dce3df;padding:9px}.v4-sample-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:7px 0;font-size:10px;color:#53616a}.v4-token-list{max-height:90px;overflow:auto;white-space:normal;overflow-wrap:anywhere;font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#34434c;background:#f7f9f8;padding:7px}.v4-source{margin:10px 0;border:1px solid var(--line);background:#fff;color:var(--ink)}.v4-source>summary{cursor:pointer;padding:9px 11px;background:#f0f4f2;color:var(--ink);font-weight:700}
    .v4-flag{display:inline-block;padding:2px 6px;border:1px solid #d79c43;background:#fff4df;color:#81530d;font-weight:700;font-size:10px}.v4-flag.ok{border-color:#82b39e;background:#eff8f3;color:#28634c}.v4-flag.copy{border-color:#b74355;background:#fff0f2;color:#8a2033}.v4-beam.copy{box-shadow:inset 0 0 0 1px #d88b97;background:#fff1f3}
    .v4-probe-table td,.v4-probe-table th{text-align:left}.v4-probe-table details{white-space:normal}.v4-probe-beams{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;margin-top:7px;min-width:0;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#455}.v4-beam{min-width:0;padding:4px 6px;border-left:3px solid #aab7b1;background:#f5f7f6;white-space:normal;overflow-wrap:anywhere}.v4-beam.exact{border-color:#16835f}.v4-beam.ab{border-color:#4f86c4}.v4-beam.a{border-color:#d29a3a}.v4-beam.invalid{border-color:#bd3f4c}.v4-contract-note{min-width:0;margin:8px 0;padding:8px 10px;background:#f5f7f6;border-left:3px solid #719184;font-size:11px;color:#455;white-space:normal;overflow-wrap:anywhere}.v4-sample-meta span{min-width:0;white-space:normal;overflow-wrap:anywhere}
    @media(max-width:1100px){.v4-consistency{grid-template-columns:repeat(2,minmax(0,1fr))}.v4-branch-grid{grid-template-columns:1fr}.v4-probe-beams{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.v4-consistency,.v4-sample-meta,.v4-probe-beams{grid-template-columns:1fr}.v4-cot>summary{align-items:flex-start;flex-direction:column}}
  `;
  document.head.appendChild(style);

  function isV4(){return Boolean(state.capabilities?.exact_sharpen_v4||state.manifest?.experiment==='GR_REC_ThinkExactSharpen_v4')}
  function setV4Controls(){
    if($('advRouteSelect').dataset.v4)return;
    $('advRouteSelect').dataset.v4='1';
    $('advRouteSelect').innerHTML='<option value="">Free + Official</option><option value="free">Free G8</option><option value="official">Official G8</option>';
    $('advFocusSelect').innerHTML='<option value="">全部候选</option><option value="history_copy">抄历史序列</option><option value="a_removed">A 奖励已删除</option><option value="saturated">已饱和分支</option><option value="nonzero">非零 Advantage</option><option value="duplicate">重复惩罚</option><option value="exact">Exact</option>';
  }
  function restoreControls(){
    if(!$('advRouteSelect').dataset.v4)return;
    delete $('advRouteSelect').dataset.v4;
    $('advRouteSelect').innerHTML=advantageRouteOptions;
    $('advFocusSelect').innerHTML=advantageFocusOptions;
  }
  function sid(value){return Array.isArray(value)?`${value[0]}:${value.slice(1).join('/')}`:'未解析'}
  function sidPart(value){const text=String(value);const domain=text.match(/^<\|([a-z_]+)_begin\|>$/);if(domain)return domain[1];const token=text.match(/^<s_[abc]_(\d+)>$/);return token?token[1]:text}
  function sidKey(value){return Array.isArray(value)&&value.length>=4?value.slice(0,4).map(sidPart).join('|'):null}
  function historySidKeys(prompt){const keys=new Set(),pattern=/<\|([a-z_]+)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>/g;for(const match of String(prompt||'').matchAll(pattern))keys.add(match.slice(1,5).join('|'));return keys}
  function copiedFromHistory(value,historyKeys){const key=sidKey(value);return Boolean(key&&historyKeys.has(key))}
  function branchRows(cot,branch,focus){
    return (cot[branch]||[]).filter(row=>{
      if(focus==='a_removed')return row.a_reward_removed;
      if(focus==='history_copy')return row.copied_from_history;
      if(focus==='saturated')return true;
      if(focus==='nonzero')return Number(row.advantage)!==0;
      if(focus==='duplicate')return Number(row.duplicate_penalty)!==0;
      if(focus==='exact')return Number(row.raw_reward)===8;
      return true
    })
  }
  function levelCounts(summary){const c=summary?.reward_counts||{};return `Exact ${c.exact??0} · AB ${c.ab??0} · A ${c.a??0} · Domain ${c.domain??0} · Wrong ${c.wrong_domain??0} · Invalid ${c.invalid??0}`}
  function ruleFlags(row){return `${row.copied_from_history?`<span class="v4-flag copy" title="完整 domain+A+B+C SID 在模型输入历史中出现 ${row.history_occurrence_count??1} 次">抄历史 ×${row.history_occurrence_count??1}</span> `:''}${row.a_reward_removed?'<span class="v4-flag">A 已删除</span>':row.exact_duplicate_exempt?'<span class="v4-flag ok">Exact 免重复罚</span>':Number(row.duplicate_penalty)!==0?'<span class="v4-flag">重复惩罚</span>':'-'}`}
  function candidateDetails(row,branch){
    const span=Array.isArray(row.first_sid_span)?row.first_sid_span.join(' → '):'-';
    return `<details><summary>采样与解析明细</summary><div class="v4-sample-meta"><span>合同：${escapeHtml(row.sampling_contract||'-')}</span><span>parser：${escapeHtml(row.parser_status||'-')}</span><span>生成 token：${row.generated_token_count??'-'}</span><span>loss mask：${row.masked_action_token_count??0} token</span><span>首 SID span：${escapeHtml(span)}</span><span>分支：${branch==='free'?'Free continuation':'Official ABC3'}</span></div><div class="label">模型输出原文（由固定 tokenizer 从捕获 token ids 只读解码）</div><pre class="v4-sample-text">${escapeHtml(row.completion_text||'解码器不可用；下方保留原始 token ids')}</pre><div class="label">完整生成 token ids</div><div class="v4-token-list">${escapeHtml((row.token_ids||[]).join(', '))}</div></details>`
  }
  function renderBranch(group,cot,branch,focus){
    const saturated=group[`${branch}_saturated`],aPlus=group[`${branch}_a_plus_count`],rows=branchRows(cot,branch,focus),summary=cot[`${branch}_summary`]||{};
    const body=rows.map(row=>`<tr class="${row.a_reward_removed?'a-removed':Number(row.advantage)>0?'positive':''}"><td>#${row.candidate_id+1}</td><td><b>${escapeHtml(sid(row.sid))}</b>${candidateDetails(row,branch)}</td><td>${fmt(row.raw_reward)}</td><td>${fmt(row.saturation_reward)}</td><td>${signed(row.duplicate_penalty,2)}</td><td><b>${fmt(row.shaped_reward)}</b></td><td class="adv-${creditTone(row.advantage)}"><b>${signed(row.advantage,4)}</b></td><td>${ruleFlags(row)}</td></tr>`).join('');
    return `<details class="v4-branch" open><summary><div><b>${branch==='free'?'Free Sample8':'Official Sample8'}</b> · shaped μ ${fmt(summary.shaped_reward_mean)} · adv σ ${fmt(summary.advantage_std)}</div><div><span class="v4-flag ${saturated?'':'ok'}">${saturated?'SATURATED':'UNSATURATED'}</span> <span class="summary-chip">A+ ${aPlus}/32</span></div></summary><div class="v4-contract-note">${levelCounts(summary)} · parsed ${summary.parsed_count??0}/8 · unique SID ${summary.unique_sid_count??0} · duplicate penalized ${summary.duplicate_penalized_count??0} · A removed ${summary.a_reward_removed_count??0} · <b>抄历史 ${summary.history_copy_count??0}/8</b></div><div class="table-scroll"><table class="v4-table"><thead><tr><th>#</th><th>输出 SID / 采样明细</th><th>Raw</th><th>饱和后</th><th>重复罚</th><th>Final shaped</th><th>Advantage</th><th>规则命中</th></tr></thead><tbody>${body||'<tr><td colspan="8">当前筛选无候选</td></tr>'}</tbody></table></div></details>`
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
      const consistency=cot.branch_consistency||{};
      return `<details class="v4-cot" ${cot.cot_id===0?'open':''}><summary><div class="adv-title">CoT ${cot.cot_id+1} · reward ${fmt(cot.cot_reward)} · G4 advantage <span class="adv-${creditTone(cot.cot_advantage)}">${signed(cot.cot_advantage,4)}</span> ${provenanceBadge()}</div><div class="summary-chips"><span class="summary-chip">Exact SID overlap ${consistency.exact_sid_overlap??0}</span><span class="summary-chip">A-prefix overlap ${consistency.a_prefix_overlap??0}</span><span class="summary-chip">展开明细</span></div></summary><div class="v4-cot-body"><div class="v4-consistency"><div class="v4-stat"><b>${coverage.unique_exact??0}</b><span>unique Exact</span></div><div class="v4-stat"><b>${coverage.unique_uncovered_ab??0}</b><span>uncovered AB</span></div><div class="v4-stat"><b>${coverage.unique_uncovered_a??0}</b><span>uncovered A</span></div><div class="v4-stat"><b>${pct(consistency.exact_sid_jaccard)}</b><span>两路 SID Jaccard</span></div><div class="v4-stat"><b>${consistency.same_best_reward_level?'一致':'不同'}</b><span>两路最佳 reward level</span></div></div><div class="v4-contract-note">${escapeHtml(consistency.interpretation||'')} · target domain ${escapeHtml(cot.target_domain||'-')} · rollout ${sec(cot.rollout_wall_sec)}</div><details><summary>展开完整 CoT 信息</summary><pre class="v4-cot-text">${escapeHtml(cot.cot_text||'')}</pre><div class="label">Official fixed prefix</div><div class="v4-token-list">${escapeHtml(cot.official_domain_prefix||'-')}</div></details><div class="v4-branch-grid">${branches.map(branch=>renderBranch(selected,cot,branch,focus)).join('')}</div></div></details>`
    }).join('');
    const consistency=selected.branch_consistency||{};
    const golds=(selected.gold_sids||[]).map(value=>`<span class="gold-chip">${escapeHtml(typeof value==='string'?value:sid(value))}</span>`).join('');
    const copies=selected.history_copy_summary||{};
    $('advantageContent').innerHTML=`<section class="adv-group"><header class="adv-group-head"><div><div class="adv-title">Step ${selected.step} · Business Group ${escapeHtml(selected.group_id)}</div><div class="adv-sub">训练捕获双路 G8；CoT G4 advantage 由捕获 reward 只读复算。Fingerprint ${escapeHtml(selected.rollout_fingerprint||'-')}</div></div><div class="summary-chips"><span class="summary-chip ${selected.free_saturated?'alert':'good'}">Free A+ ${selected.free_a_plus_count}/32 · ${selected.free_saturated?'A 删除':'A 保留'}</span><span class="summary-chip ${selected.official_saturated?'alert':'good'}">Official A+ ${selected.official_a_plus_count}/32 · ${selected.official_saturated?'A 删除':'A 保留'}</span><span class="summary-chip alert">抄历史 Free ${copies.free_count??0}/32 · Official ${copies.official_count??0}/32</span><span class="summary-chip">8 个独立 G8</span></div></header><details class="v4-source" open><summary>模型输入原文 / Gold SID</summary><pre class="v4-cot-text">${escapeHtml(selected.input_prompt||'输入原文不可用')}</pre><div class="gold-strip"><span class="label">Gold SID</span>${golds||'<span class="label">未记录</span>'}</div></details><div class="v4-consistency"><div class="v4-stat"><b>${selected.history_sid_count??0} / ${selected.history_unique_sid_count??0}</b><span>历史 SID 总数 / 去重</span></div><div class="v4-stat"><b>${copies.total_count??0}/${copies.total_denominator??64}</b><span>双路完整 SID 抄历史</span></div><div class="v4-stat"><b>${consistency.exact_sid_overlap??0}</b><span>两路 Exact SID overlap</span></div><div class="v4-stat"><b>${consistency.ab_prefix_overlap??0}</b><span>AB-prefix overlap</span></div><div class="v4-stat"><b>${consistency.a_prefix_overlap??0}</b><span>A-prefix overlap</span></div></div><div class="v4-contract-note"><b>抄历史定义：</b>候选完整 domain+A+B+C SID 精确出现在模型输入历史中；仅 A/AB 前缀相同不标记。${escapeHtml(consistency.interpretation||'')}</div>${cots}</section>`
  }
  renderAdvantages=function(){if(isV4())renderV4Advantages();else{restoreControls();baseRenderAdvantages()}};

  function probeCompletion(candidate){
    return `<div class="v4-sample-meta"><span>length ${candidate.completion_length??'-'}</span><span>closed ${candidate.closed?'YES':'NO'}</span><span>SHA ${escapeHtml(String(candidate.completion_sha256||'-').slice(0,16))}…</span></div><pre class="v4-sample-text">${escapeHtml(candidate.completion||'未保存 completion')}</pre>`
  }
  function compactProbeCandidate(mode,candidate,index,golds,historyKeys){
    if(mode==='free'){const copied=copiedFromHistory(candidate.parsed_sid,historyKeys);return `<tr><td>#${index+1}</td><td><b>${escapeHtml(sid(candidate.parsed_sid))}</b> ${copied?'<span class="v4-flag copy">抄历史</span>':''}<details><summary>完整 Free 采样</summary>${probeCompletion(candidate)}<div class="v4-contract-note">parser ${escapeHtml(candidate.parser_status||'-')} · fixed domain prefix ${candidate.fixed_domain_prefix?'YES':'NO'} · 抄历史 ${copied?'YES':'NO'}</div></details></td><td>${fmt(candidate.reward)}</td><td>${escapeHtml(candidate.parser_status||'-')}</td><td>${candidate.closed?'是':'否'}</td></tr>`}
    const beams=(candidate.beam_sids||[]).map((value,beamIndex)=>{const kind=sidKind(value,golds),copied=copiedFromHistory(value,historyKeys);return `<div class="v4-beam ${kind} ${copied?'copy':''}"><b>#${beamIndex+1}</b> ${escapeHtml(sid(value))} · ${kind==='exact'?'Exact':kind==='ab'?'AB':kind==='a'?'A':kind==='invalid'?'Invalid':'Domain'} ${copied?'<span class="v4-flag copy">抄历史</span>':''}</div>`}).join('');
    return `<tr><td>#${index+1}<details><summary>完整 CoT / Beam 合同</summary>${probeCompletion(candidate)}<div class="v4-contract-note">fixed domain prefix ${candidate.fixed_domain_prefix?'YES':'NO'} · search ${escapeHtml(candidate.beam_search_space||'-')}</div></details></td><td>${fmt(candidate.reward)}</td><td>${candidate.exact??0} / ${candidate.ab??0} / ${candidate.a??0}</td><td>${candidate.invalid??0}</td><td><details><summary>展开 32 条 Beam SID</summary><div class="v4-probe-beams">${beams}</div></details></td></tr>`
  }
  function probeBlock(mode,data,golds,historyKeys){
    const free=mode==='free',candidates=data?.candidates||[],rows=candidates.map((candidate,index)=>compactProbeCandidate(mode,candidate,index,golds,historyKeys)).join('');
    const copyCount=free?candidates.filter(candidate=>copiedFromHistory(candidate.parsed_sid,historyKeys)).length:candidates.reduce((total,candidate)=>total+(candidate.beam_sids||[]).filter(value=>copiedFromHistory(value,historyKeys)).length,0),copyDenominator=free?candidates.length:candidates.reduce((total,candidate)=>total+(candidate.beam_sids||[]).length,0);
    return `<details class="probe-route" open><summary><h3 style="display:inline">${free?'Free Probe · 自由 continuation / 首个完整 SID':'Official Probe · fixed domain begin / production Beam32 ABC3'} · reward ${fmt(data?.reward_mean)} ± ${fmt(data?.reward_std)}</h3></summary><div class="v4-contract-note"><b>评分语义：</b>${escapeHtml(data?.reward_semantics||'-')}<br><b>分母：</b>${escapeHtml(data?.reward_denominator||'-')} · candidates ${data?.candidate_count??0} · closure ${pct(data?.closure_rate)} · <b>抄历史 ${copyCount}/${copyDenominator}</b></div><div class="table-scroll"><table class="v4-table v4-probe-table"><thead><tr>${free?'<th>#</th><th>Parsed SID / 完整采样</th><th>Reward</th><th>Parser</th><th>Closed</th>':'<th>CoT / 完整文本</th><th>Reward</th><th>Exact / AB / A</th><th>Invalid</th><th>Beam32 明细</th>'}</tr></thead><tbody>${rows}</tbody></table></div></details>`
  }
  function renderV4Probes(){
    $('probeRewardChart').closest('.panel').hidden=true;$('v4ProbeFreePanel').hidden=false;$('v4ProbeOfficialPanel').hidden=false;
    const configured=state.manifest.fixed_probe?.group_ids||[],groups=[...new Set([...configured,...state.probes.map(row=>row.group_id)])],groupSelect=$('probeGroupSelect'),stepSelect=$('probeStepSelect'),currentGroup=groupSelect.value;
    groupSelect.innerHTML=groups.map((gid,index)=>`<option value="${escapeHtml(gid)}">Probe ${index+1} · ${escapeHtml(gid.slice(0,12))}</option>`).join('')||'<option value="">未配置</option>';if(groups.includes(currentGroup))groupSelect.value=currentGroup;
    const group=groupSelect.value,available=state.probes.filter(row=>row.group_id===group).sort((a,b)=>a.step-b.step),current=Number(stepSelect.value);
    stepSelect.innerHTML=[...available].reverse().map(row=>`<option value="${row.step}">Step ${row.step} · ${escapeHtml(row.reason||'periodic')}</option>`).join('')||'<option value="">暂无记录</option>';if(available.some(row=>row.step===current))stepSelect.value=String(current);
    chartLabels.v4ProbeFreeRewardChart=['Reward mean','A rate','AB rate','Exact rate','Closure'];chartLabels.v4ProbeOfficialRewardChart=['Reward mean','A Beam rate','AB Beam rate','Exact Beam rate','Closure'];
    $('v4ProbeFreePanel').querySelector('.legend').innerHTML='<span class="key" style="--c:#16835f">Reward</span><span class="key" style="--c:#d29a3a">A/32</span><span class="key" style="--c:#4f86c4">AB/32</span><span class="key" style="--c:#7f5aa2">Exact/32</span><span class="key" style="--c:#445d69">Closure</span>';
    $('v4ProbeOfficialPanel').querySelector('.legend').innerHTML='<span class="key" style="--c:#b36b08">Reward</span><span class="key" style="--c:#d29a3a">A/128 beams</span><span class="key" style="--c:#4f86c4">AB/128</span><span class="key" style="--c:#7f5aa2">Exact/128</span><span class="key" style="--c:#445d69">Closure</span>';
    const points=(path,denominator=1)=>available.map(row=>[row.step,Number(path(row)||0)/denominator]);
    draw('v4ProbeFreeRewardChart',[{data:points(row=>row.probe_free?.reward_mean),color:colors[1]},{data:points(row=>row.probe_free?.a_count,32),color:'#d29a3a'},{data:points(row=>row.probe_free?.ab_count,32),color:'#4f86c4'},{data:points(row=>row.probe_free?.exact_count,32),color:'#7f5aa2'},{data:points(row=>row.probe_free?.closure_rate),color:'#445d69'}]);
    draw('v4ProbeOfficialRewardChart',[{data:points(row=>row.probe_official?.reward_mean),color:colors[2]},{data:points(row=>row.probe_official?.a_count,128),color:'#d29a3a'},{data:points(row=>row.probe_official?.ab_count,128),color:'#4f86c4'},{data:points(row=>row.probe_official?.exact_count,128),color:'#7f5aa2'},{data:points(row=>row.probe_official?.closure_rate),color:'#445d69'}]);
    const selected=available.find(row=>row.step===Number(stepSelect.value));
    if(!selected){$('probeOverview').innerHTML='';$('probeDetail').innerHTML='<div class="probe-empty">等待双 Probe 数据</div>';return}
    setText('probeMeta',`同一批 sampled CoT · seed ${selected.seed} · ${sec(selected.probe_wall_sec)}`);
    const free=selected.probe_free||{},official=selected.probe_official||{};
    const aliasOk=selected.think?.reward_mean===official.reward_mean&&selected.think?.candidate_count===official.candidate_count;
    diagnosticCards('probeOverview',[["Domain / Group",`${selected.target_domain} / ${selected.group_id}`],["Free reward / std",`${fmt(free.reward_mean)} / ${fmt(free.reward_std)}`],["Free A / AB / Exact",`${free.a_count??0} / ${free.ab_count??0} / ${free.exact_count??0}`],["Free candidates / denominator",`${free.candidate_count??0} / ${free.reward_denominator||'-'}`],["Official reward / std",`${fmt(official.reward_mean)} / ${fmt(official.reward_std)}`],["Official A / AB / Exact",`${official.a_count??0} / ${official.ab_count??0} / ${official.exact_count??0}`],["Official CoTs / denominator",`${official.candidate_count??0} / ${official.reward_denominator||'-'}`],["Compatibility",aliasOk?'think → Official ✓':'think alias mismatch']]);
    $('probeDsrTimeline').hidden=true;
    const freePositive=(Number(free.a_count||0)+Number(free.ab_count||0)+Number(free.exact_count||0))/32,officialPositive=(Number(official.a_count||0)+Number(official.ab_count||0)+Number(official.exact_count||0))/128;
    const historyKeys=historySidKeys(selected.think_prompt);
    $('probeDetail').innerHTML=`<div class="trace-head">Step ${selected.step} · ${escapeHtml(selected.group_id)}</div><details class="v4-source" open><summary>Probe 模型输入原文 / Gold SID</summary><pre class="v4-cot-text">${escapeHtml(selected.think_prompt||'Probe 输入原文未记录')}</pre><div class="gold-strip"><span class="label">Gold SID</span>${(selected.gold_sids||[]).map(value=>`<span class="gold-chip">${escapeHtml(typeof value==='string'?value:sid(value))}</span>`).join('')}</div></details><div class="v4-consistency"><div class="v4-stat"><b>同一批 4 CoT</b><span>两路输入对齐</span></div><div class="v4-stat"><b>${historyKeys.size}</b><span>输入历史 unique SID</span></div><div class="v4-stat"><b>${pct(freePositive)}</b><span>Free A+ candidate rate</span></div><div class="v4-stat"><b>${pct(officialPositive)}</b><span>Official A+ Beam rate</span></div><div class="v4-stat"><b>${pct(Math.abs(freePositive-officialPositive))}</b><span>A+ rate 绝对差</span></div><div class="v4-stat"><b>${aliasOk?'PASS':'FAIL'}</b><span>think → Official alias</span></div></div><div class="v4-contract-note"><b>抄历史定义：</b>候选完整 domain+A+B+C SID 精确命中 Probe 输入历史；只撞 A/AB 不算。两路共用 sampled CoT，但 continuation、候选分母与 reward 量纲不同。</div>${probeBlock('free',free,selected.gold_sids||[],historyKeys)}${probeBlock('official',official,selected.gold_sids||[],historyKeys)}`
  }
  renderProbes=function(){
    if(isV4())renderV4Probes();
    else{$('probeRewardChart').closest('.panel').hidden=false;$('v4ProbeFreePanel').hidden=true;$('v4ProbeOfficialPanel').hidden=true;baseRenderProbes()}
  };
})();
