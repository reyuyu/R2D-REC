(() => {
  const phaseLabels = {
    training: "正式训练中",
    saving_training: "保存训练结果",
    offline_probes: "离线探针评测",
    finalizing: "生成最终报告",
    ready: "等待人工选择 Checkpoint",
    failed: "实验失败",
  };
  let payload = null;
  let requestedRun = "";

  const html = value => String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
  const number = (value, digits = 4) => value !== null && value !== "" && Number.isFinite(Number(value))
    ? Number(value).toFixed(digits) : "-";
  const shortHash = value => value ? `${String(value).slice(0, 12)}...${String(value).slice(-8)}` : "-";
  const download = (checkpoint, file) => `/api/checkpoints/${encodeURIComponent(checkpoint)}/download?run_id=${encodeURIComponent(requestedRun)}&file=${encodeURIComponent(file)}`;

  function install() {
    const tabs = document.querySelector(".tabs");
    if (!tabs || document.getElementById("grpo2FormalTab")) return;
    const tab = document.createElement("button");
    tab.className = "tab";
    tab.id = "grpo2FormalTab";
    tab.dataset.view = "grpo2Formal";
    tab.textContent = "GRPO-2 阶段";
    tab.hidden = true;
    tab.onclick = () => activateView("grpo2Formal");
    tabs.insertBefore(tab, document.getElementById("checkpointEvalTab"));

    const main = document.createElement("main");
    main.id = "grpo2Formal";
    main.className = "view";
    main.innerHTML = '<div class="wrap"><div class="grpo2-empty">正在读取 GRPO-2 正式实验状态</div></div>';
    document.getElementById("checkpointEval").before(main);
  }

  function contractRow(label, value, title = "") {
    return `<div><span class="label">${html(label)}</span><code title="${html(title || value)}">${html(value)}</code></div>`;
  }

  function render() {
    if (!payload) return;
    const current = payload.current_step || 0;
    const max = payload.max_steps || 300;
    const percent = Math.max(0, Math.min(100, current / Math.max(1, max) * 100));
    const latest = payload.latest || {};
    const contract = payload.contract || {};
    const phaseClass = payload.phase === "ready" ? "ready" : payload.phase === "failed" ? "failed" : "running";
    const completedCheckpoints = payload.checkpoints.filter(row => row.complete).length;
    const completedProbes = payload.offline_probes.filter(row => row.complete).length;

    const checkpoints = payload.checkpoints.map(row => {
      const fileCount = Object.values(row.files || {}).filter(Boolean).length;
      return `<article class="grpo2-checkpoint ${row.complete ? "complete" : ""}">
        <h3>Step ${row.step}</h3>
        <div class="grpo2-state">${row.complete ? "续训状态完整" : row.exists ? `正在写入 · ${fileCount}/11` : "等待训练到达"}</div>
        <div class="grpo2-state" title="${html(row.adapter_sha256 || "")}">${row.adapter_sha256 ? shortHash(row.adapter_sha256) : "SHA 待生成"}</div>
        ${row.complete ? `<div class="grpo2-downloads"><a href="${download(row.checkpoint, "adapter_model.safetensors")}" download>下载权重</a><a href="${download(row.checkpoint, "adapter_config.json")}" download>下载配置</a></div>` : ""}
      </article>`;
    }).join("");

    const probes = payload.offline_probes.map(row => `<article class="grpo2-probe ${row.complete ? "complete" : ""}">
      <h3>${html(row.model)}</h3><div class="grpo2-state">${row.complete ? `完成 · ${row.probe_rows} 条记录` : "等待顺序评测"}</div>
    </article>`).join("");

    const comparison = payload.comparison.length ? `<div class="grpo2-table-wrap"><table class="grpo2-table"><thead><tr>
      <th>模型</th><th>Think 奖励</th><th>Think Success@K</th><th>Think Success@32</th><th>NoThink 奖励</th><th>NoThink Success@K</th><th>NoThink 正样本率</th>
    </tr></thead><tbody>${payload.comparison.map(row => `<tr><td>${html(row.model)}</td><td>${number(row.think_mean_reward)}</td><td>${number(row.think_success_at_k)}</td><td>${number(row.think_success_at_32)}</td><td>${number(row.nothink_mean_reward)}</td><td>${number(row.nothink_success_at_k)}</td><td>${number(row.nothink_positive_rate)}</td></tr>`).join("")}</tbody></table></div>` : '<div class="grpo2-empty">训练完成后依次评测基线与五个 checkpoint；这里不会自动选最优权重。</div>';

    const gpus = (payload.gpu?.gpus || []).map(gpu => `<div class="grpo2-gpu"><span class="label">GPU ${gpu.index}</span><b>${number(gpu.memory_used_mb, 0)} / ${number(gpu.memory_total_mb, 0)} MiB</b></div>`).join("");
    document.querySelector("#grpo2Formal .wrap").innerHTML = `
      <section class="panel">
        <div class="grpo2-stage-head"><div><h2>Recommendation Think-only · Continued Adapter</h2><p>从 GRPO-1 step 500 的同一个 LoRA adapter 继续训练，优化器从 GRPO-2 step 0 重新初始化。</p></div><span class="grpo2-phase ${phaseClass}">${html(phaseLabels[payload.phase] || payload.phase)}</span></div>
        <div class="grpo2-progress"><i style="width:${percent}%"></i></div>
        <div class="grpo2-metrics">
          <div class="grpo2-metric"><span class="label">训练进度</span><b>${current} / ${max}</b></div>
          <div class="grpo2-metric"><span class="label">Loss</span><b>${number(latest.loss)}</b></div>
          <div class="grpo2-metric"><span class="label">Reward</span><b>${number(latest.reward_mean)}</b></div>
          <div class="grpo2-metric"><span class="label">Reward Std</span><b>${number(latest.reward_std)}</b></div>
          <div class="grpo2-metric"><span class="label">Grad Norm</span><b>${number(latest.grad_norm)}</b></div>
          <div class="grpo2-metric"><span class="label">学习率</span><b>${number(latest.learning_rate, 8)}</b></div>
        </div>
      </section>
      <section class="panel grpo2-section"><div class="grpo2-section-title"><h2>训练合同</h2><span>基座冻结 · 单一继承 adapter · 新 optimizer</span></div><div class="grpo2-contract">
        ${contractRow("继承模式", contract.mode)}${contractRow("LoRA 初始化", contract.adapter_initialization)}${contractRow("新 LoRA", contract.fresh_lora === false ? "否" : "是")}${contractRow("新 Optimizer", contract.fresh_optimizer === true ? "是" : "否")}
        ${contractRow("Full SFT SHA256", shortHash(contract.base_sha256), contract.base_sha256)}${contractRow("GRPO-1 父检查点", `step ${contract.parent_checkpoint ?? "-"}`)}${contractRow("父 Adapter SHA256", shortHash(contract.parent_adapter_sha256), contract.parent_adapter_sha256)}${contractRow("代码 Commit", shortHash(contract.code_commit), contract.code_commit)}
      </div></section>
      <section class="panel grpo2-section"><div class="grpo2-section-title"><h2>正式 Checkpoint</h2><span>${completedCheckpoints} / ${payload.checkpoints.length} 完整</span></div><div class="grpo2-checkpoints">${checkpoints}</div></section>
      <section class="panel grpo2-section"><div class="grpo2-section-title"><h2>离线固定探针</h2><span>${completedProbes} / ${payload.offline_probes.length} 完成 · 顺序执行</span></div><div class="grpo2-probes">${probes}</div></section>
      <section class="panel grpo2-section"><div class="grpo2-section-title"><h2>Checkpoint 横向结果</h2><span>同一固定样本 · 不自动选优</span></div>${comparison}</section>
      <section class="panel grpo2-section"><div class="grpo2-section-title"><h2>GPU 状态</h2><span>${payload.gpu?.released ? "四卡已释放" : `${payload.gpu?.process_count || 0} 个训练进程`}</span></div><div class="grpo2-gpus">${gpus}</div></section>`;
  }

  async function sync() {
    install();
    const run = state.activeRun;
    if (!run) return;
    try {
      const response = await fetch(`/api/grpo2-formal/status?run_id=${encodeURIComponent(run)}`, { cache: "no-store" });
      const tab = document.getElementById("grpo2FormalTab");
      if (response.status === 404) {
        tab.hidden = true;
        if (document.getElementById("grpo2Formal").classList.contains("active")) activateView("overview");
        payload = null;
        return;
      }
      if (!response.ok) throw new Error(`GRPO-2 status ${response.status}`);
      requestedRun = run;
      payload = await response.json();
      if (requestedRun !== state.activeRun) return;
      tab.hidden = false;
      render();
    } catch (error) {
      if (document.getElementById("grpo2Formal")?.classList.contains("active")) {
        document.querySelector("#grpo2Formal .wrap").innerHTML = `<div class="grpo2-empty">GRPO-2 阶段数据读取失败：${html(error.message)}</div>`;
      }
    }
  }

  install();
  sync();
  setInterval(sync, 3000);
})();
