# Prompt for independent GPT review

请直接读取 GitHub 仓库 `reyuyu/onereason-multitask-sft` 的 `main` 分支，重点审查目录：

```text
reproduction/final_chain_20260901/
```

背景：该目录用于从组织方基础模型开始，按固定父子关系复现四阶段最终链：

```text
BETA 多任务 SFT checkpoint-1106
  -> GR_REC_v1 双路懂推荐 GRPO checkpoint-1500
  -> GRPO-TK Think Sample8 FullSID checkpoint-250
  -> MC_USER Hybrid K4 prompt-step-0100
```

请把它当作一次严格的 correctness / reproducibility code review，而不是实验方案讨论。先完整读取以下文件，再下结论：

```text
README.md
run.sh
repro_control.py
ENVIRONMENT_LOCK.json
BASE_MODEL_SHA256SUMS
historical_reference.json
repro_quality.py
compare_adapters.py
repro_monitor_server.py
repro_monitor_static/index.html
repro_monitor_static/styles.css
repro_monitor_static/app.js
docs/REPRODUCTION_QUALITY.md
tests/test_repro_quality.py
tests/test_repro_monitor_server.py
```

同时检查该提交对原训练代码的必要复现参数化改动，尤其是：

```text
baselines/native_source_domain_r32_v3/scripts/run_native_source_domain_r32_v3.sh
baselines/native_source_domain_r32_v3/grpo/scripts/run_grpo_multi_positive.py
baselines/native_source_domain_r32_v3/grpo/scripts/run_grpo_think_sample8_fullsid_v3.py
baselines/native_source_domain_r32_v3/grpo/scripts/launch_gr_rec_think_sample8_fullsid_v3_formal_4gpu.sh
baselines/native_source_domain_r32_v3/grpo/user/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py
```

已知合同：

- 所有精调数据必须从 `/root/reproduce_datasets/onereason_final_chain_20260901` 注册，不能回退读取历史 `/data/...` 数据源。
- 没有教师模型；组织方预训练 base 不属于教师模型。
- 四阶段必须严格父子继承，不能从历史成品 checkpoint 偷跑。
- 数据 SHA、行数、base 16 文件 SHA、两个 Python 环境、LLaMAFactory 快照、四张 A800 拓扑和驱动均在训练前 fail-closed 校验。
- 保留中间检查点：SFT 553/1106，GR_REC 500/1000/1500，GRPO-TK 50/100/150/200/250，MC_USER 25/50/75/100。
- Monitor 必须纯只读，不导入 trainer、不占 GPU、不改变 RNG、不删除或修改训练证据。
- 由于 FlashAttention、CUDA sampling 和 NCCL reduction，不承诺跨运行 adapter SHA 位级相同。必须区分合同复现、轨迹复现、参数数值接近和外部分数复现。

请重点寻找以下问题：

1. `run.sh` 是否真的能从 stage1 一路执行到 stage4，父路径、输出路径、环境变量和 checkpoint 验证是否完全对接？
2. `--from-stage` 是否可能绕过父检查点来源验证，或误用历史 checkpoint？
3. 三份注册数据是否覆盖了所有 runner 的真实运行时数据依赖，是否还有隐式绝对路径或 cache 泄漏？
4. 随机种子是否在 rank、route、candidate、resume/iteration 边界完整冻结；是否存在 ambient RNG 依赖？
5. Trainer checkpoint 的 optimizer/scheduler/RNG 保存与 selected checkpoint 截断是否一致，后续 stage 是否拿到正确 checkpoint？
6. MC_USER 的 100 prompt 截断、严格 Action/Chain 交替、K4 candidate-parallel、LR 3e-7、Sequence 1.0 + Local 0.3 是否与记录一致？
7. `historical_reference.json` 和 `repro_quality.py` 的窗口算法是否会错误地把未到达 milestone 的少量行当作完整窗口，或把 NaN/partial append 算进去？
8. 轨迹参考带是否被错误用于硬判失败；adapter SHA 不同是否被正确标注为 review 而非失败？
9. 前端是否会把在线 reward 误当外部成绩，或把项目保守口径 1.356 误写成一次真实评测？
10. Monitor 是否存在路径穿越、任意文件读取、刷新时重复哈希大权重导致训练 I/O 干扰、前端长时间运行内存泄漏或自动刷新状态问题？
11. GitHub 发布范围是否泄露样本、rollout、权重、凭据或不必要的私有数据来源；同时是否缺少复现所必需的公开合同？
12. 当前测试是否遗漏了足以阻止正式复现的关键路径？请只建议必要的高风险回归测试，不要泛化扩测。

输出格式必须是代码审查格式：

1. Findings 优先，按 P0/P1/P2/P3 严重度排序。
2. 每条 finding 给出具体文件和行号、可复现的失败路径、为什么会影响四阶段复现、最小修复方案。
3. 然后列出“需要人工确认的外部事实”，不要把无法从仓库证明的事情猜成结论。
4. 最后分别判定：

```text
DATA_CONTRACT: PASS / FAIL / UNKNOWN
PARENT_HANDOFF: PASS / FAIL / UNKNOWN
SEED_AND_ORDER: PASS / FAIL / UNKNOWN
CHECKPOINT_RECOVERY: PASS / FAIL / UNKNOWN
MONITOR_READ_ONLY: PASS / FAIL / UNKNOWN
PUBLICATION_BOUNDARY: PASS / FAIL / UNKNOWN
OVERALL: READY / NOT_READY / NEEDS_EXTERNAL_CONFIRMATION
```

不要建议新的训练实验，不要根据在线 reward 判断模型效果，不要要求上传数据或权重到 GitHub。
