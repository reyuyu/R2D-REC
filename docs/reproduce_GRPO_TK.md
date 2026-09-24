# GRPO-TK 复现指南

## 1. 复现范围

本文复现 `GRPO-TK`，工程名为 `GR_REC_ThinkSample8_FullSID_v3`。仓库公开训练代码、测试、monitor adapter、zero-update preflight 结果和实验记录；私有训练数据、8B Base、parent Adapter 与训练 checkpoint 不上传 GitHub。

严格数值复现需要同时满足：

- 相同 Git commit。
- 相同私有数据及 group 顺序。
- 相同 8B Base 与 parent Adapter。
- 相同 tokenizer、4-GPU 拓扑、随机种子和软件版本。
- 相同 G4 CoT、每条 CoT 独立 Sample8、两次 policy iteration 合同。

缺少私有资产时仍可复刻代码结构、数据 schema、CPU contract 和算法流程，但不能声称复现原始 rollout 或 `1.3579` 外部成绩。

## 2. GitHub 项目结构

从仓库根目录看，GRPO-TK 相关文件如下：

```text
README.md
docs/
  experiment_GRPO_TK.md
  reproduce_GRPO_TK.md
baselines/native_source_domain_r32_v3/grpo/
  scripts/
    grpo_model.py
    grpo_sid.py
    grpo_probe.py
    run_grpo_trl_train.py
    grpo_trl_trainer.py
    monitor/
      server.py
      writer.py
      dual_beam8_adapter.py
  ablations/
    gr_rec_think_sample8_fullsid_v3/
      README.md
      CHANGELOG.md
      run_sample8_fullsid_train.py
      sample8_fullsid_trainer.py
      launch_sample8_fullsid_train.sh
      gpu_preflight.py
      test_sample8_fullsid_contract.py
  results/
    gr_rec_think_sample8_fullsid_v3_gpu_preflight_scan_20260828.json
```

职责边界：

- `run_sample8_fullsid_train.py`：parent SHA guard、数据/Probe 拓扑、训练配置、manifest、正式入口与可信 checkpoint 恢复。
- `sample8_fullsid_trainer.py`：G4 CoT、每 CoT Sample8、FullSID 扫描、4 个独立 G8、CoT G4、双分支 loss 和监控捕获。
- `gpu_preflight.py`：4-GPU generation/forward/backward 的 zero-update 验收。
- `launch_sample8_fullsid_train.sh`：正式 4-GPU torchrun 启动合同。
- `test_sample8_fullsid_contract.py`：CPU 公式、拓扑、parser、恢复和 cadence 合同。

## 3. 已验收代码点

- v2 参考基线：`32c22f3de8e90e226bf3671f64ca9fbfbed90e8d`
- v3 continuation-scanning 正式启动代码点：`bcff386ec81104251e93a483785d96ece042f0b2`
- checkpoint-50/probe-50 与恢复 cadence 验收代码点：`40cb740963f0dfd7f780b461f064b3dd5ee2c517`

建议从最新 `origin/main` 复现，并在 manifest 中保存实际 `git_commit`。若目标是逐 rollout 配对，则必须 checkout 计划比较的精确 commit。

## 4. 已验收环境

开发机实测环境：

| 组件 | 版本 |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| Transformers | 5.6.0 |
| TRL | 0.24.0 |
| PEFT | 0.18.1 |
| Accelerate | 1.11.0 |
| GPU | 4 × NVIDIA A800-SXM4-80GB |
| Driver | 535.129.03 |

当前 runner 对 root-owned 本地 checkpoint 提供受限的 PyTorch 2.5/Transformers 5.6 恢复兼容，但只在路径、文件、权限和 step guard 全部通过后启用。不要把该兼容逻辑扩展到不可信 checkpoint。

## 5. 私有资产布局

严格复现使用以下路径：

| 资产 | 路径/要求 |
|---|---|
| 8B Base | `/data/models/onereason-8b-pretrain-competition` |
| Parent Adapter | `/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500` |
| Parent weight SHA256 | `a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436` |
| 私有训练数据 | `/data/GRPO/data/rec_mp_grpo_v2/train.jsonl` |
| Monitor run 根目录 | `/data/GRPO/runs` |
| Checkpoint 根目录 | `/root/GRPO-checkpoints` |

代码对 parent 路径和 SHA fail closed。若更换 parent，必须新建实验身份、更新 SHA 和外部 baseline，不能继续称为同一 GRPO-TK 数值复现。

## 6. 私有数据构成与特征

数据本体不上传，只公开 schema 和聚合统计。

### 6.1 JSONL schema

每行包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `recommendation_group_id` | string | 同一业务样本跨 route 的稳定 group ID |
| `route` | string | `think` 或 `no_think` |
| `prompt` | string | 模型真实输入，包含用户历史和 route 指令 |
| `target_domain` | string | `video/prod/ad/living` 之一 |
| `all_gold_sids` | list[string] | 该业务样本可接受的完整 Gold SID 集合 |
| `gold_count` | int | Gold SID 数量 |

### 6.2 聚合规模

- 3098 行。
- 1549 个唯一 `recommendation_group_id`。
- 每个 group 恰好两行：1 条 `think` + 1 条 `no_think`。
- GRPO-TK 只选择 `think` 行。
- 固定 Probe4 排除后，实际训练 1545 个 group。
- domain 的 group 级分布：video 550、ad 427、prod 382、living 190。
- Gold SID 数量：最少 2、均值约 7.114、中位数 3、最多 20。
- prompt 字符数：最少 99、均值约 8183.6、中位数约 8019.5、P95 约 14225、最大 21803。
- 没有缺失 group ID、空 prompt 或空 Gold 集合。

Gold 数量在 2-3 和 11-20 两段较多，数据具有明显的多正例规模差异。训练 reward 因而基于 Gold SID 集合计算，而不是把某一个 Gold SID 当作唯一答案。

### 6.3 不上传内容

以下内容不得提交 GitHub：

- 原始 prompt 和用户历史。
- 原始 `all_gold_sids`。
- Base/Adapter/checkpoint 权重。
- optimizer、scheduler 和 RNG state。
- 包含可还原私有样本全文的 rollout 导出。

可以公开：字段 schema、聚合分布、不可逆 group hash、公式测试、去内容化统计和不含私有 prompt 的结果摘要。

## 7. 数据合同自检

在私有数据已挂载的机器上执行以下只读检查。脚本只输出聚合统计，不打印 prompt 或 SID：

```bash
python3 - <<'PY'
import collections
import json
from pathlib import Path

path = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
required = {
    "recommendation_group_id", "route", "prompt",
    "target_domain", "all_gold_sids", "gold_count",
}
assert len(rows) == 3098
assert all(required <= row.keys() for row in rows)

groups = collections.defaultdict(list)
for row in rows:
    groups[row["recommendation_group_id"]].append(row)

assert len(groups) == 1549
assert all(len(group) == 2 for group in groups.values())
assert all({row["route"] for row in group} == {"think", "no_think"}
           for group in groups.values())
assert all(row["prompt"] and row["all_gold_sids"] for row in rows)
print("GRPO_TK_PRIVATE_DATA_CONTRACT=PASS")
print("routes", collections.Counter(row["route"] for row in rows))
print("domains", collections.Counter(row["target_domain"] for row in rows))
PY
```

## 8. 获取代码

```bash
git clone git@github.com:reyuyu/onereason-multitask-sft.git
cd onereason-multitask-sft
git fetch origin
git checkout main
git pull --ff-only origin main

cd baselines/native_source_domain_r32_v3/grpo
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

不要把旧 `/data/GRPO/scripts` 插到当前 worktree 之前。runner 的 runtime provenance guard 要求实际 import 来自当前 checkout。

## 9. Parent 完整性检查

```bash
PARENT=/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500

test -f "$PARENT/adapter_config.json"
test -f "$PARENT/adapter_model.safetensors"
sha256sum "$PARENT/adapter_model.safetensors"
```

输出必须包含：

```text
a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436
```

## 10. CPU contract tests

不加载 8B 模型、不使用 GPU：

```bash
pytest -q \
  ablations/gr_rec_think_sample8_fullsid_v3/test_sample8_fullsid_contract.py

python3 -m py_compile \
  ablations/gr_rec_think_sample8_fullsid_v3/run_sample8_fullsid_train.py \
  ablations/gr_rec_think_sample8_fullsid_v3/sample8_fullsid_trainer.py \
  ablations/gr_rec_think_sample8_fullsid_v3/gpu_preflight.py
```

关键 gate：

- 1549 raw groups、1545 train groups、3090 optimizer steps。
- Probe4 与训练零重叠。
- 全局 G4 CoT。
- 每条 CoT 正好 Sample8。
- 四个独立 G8，不是 G32。
- CoT reward 等于对应 8 个 SID reward 之和。
- SID 只更新第一个完整 SID 的 4 tokens。
- CoT/SID loss 1:1。
- iteration 2 不重新采样。
- Base frozen、parent SHA guard 和 checkpoint cadence 合同存在。

## 11. 4-GPU zero-update preflight

必须先确认 4 张 GPU 空闲。该阶段允许 generation、forward 和 backward，但不执行 optimizer/scheduler step，不保存模型：

```bash
export NCCL_SOCKET_IFNAME=lo
export GRPO_GIT_COMMIT="$(git rev-parse HEAD)"

torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29224 \
  -m ablations.gr_rec_think_sample8_fullsid_v3.gpu_preflight \
  --run-id GRPO-TK-REPRO-PREFLIGHT \
  --output results/grpo_tk_repro_preflight.json
```

继续训练前必须确认 JSON 中：

```text
preflight_pass = true
world_size = 4
optimizer_steps = 0
scheduler_steps = 0
parameter_update = false
checkpoint_saved = false
training_started = false
```

还应通过 G4/4×G8、无 fixed domain/bridge、真实 reward variance、CoT/SID gradient、Base frozen、checksum unchanged、Probe4 和 runtime import provenance 等全部 checks。

## 12. 正式训练

```bash
export RUN_ID=GRPO-TK-REPRO-E1
export OUTPUT_ROOT=/root/GRPO-checkpoints
export MASTER_PORT=29225

bash ablations/gr_rec_think_sample8_fullsid_v3/launch_sample8_fullsid_train.sh \
  2>&1 | tee /data/GRPO/logs/${RUN_ID}.log
```

默认正式合同：

- 4 GPU。
- seed `20260816`。
- 1545 fresh rollouts。
- 3090 optimizer steps。
- G4 CoT：temperature `0.9`、top-p `0.95`。
- 每条 CoT Sample8：temperature `1.0`、top-p `1.0`、top-k `0`、max new tokens `128`。
- `num_iterations=2`。
- AdamW、`lr=1e-6`、weight decay `0`、constant scheduler。
- PPO epsilon `0.2`、beta `0`。
- 每 50 step Probe4。
- 每 50 step checkpoint，最多保留 64 个，final 为 step 3090。

启动后首先检查：

```text
manifest.expected_raw_groups = 1549
manifest.expected_training_groups = 1545
manifest.expected_optimizer_steps = 3090
manifest.sid_group_size_per_cot = 8
manifest.sid_candidates_per_business_group = 32
manifest.zero_std_reroll = false
manifest.runtime_import_provenance = PASS
```

## 13. Checkpoint 恢复

恢复 checkpoint 必须位于 `/root/GRPO-checkpoints/<run-id>/checkpoint-<even-step>`，且至少包含：

```text
adapter_config.json
adapter_model.safetensors
optimizer.pt
scheduler.pt
trainer_state.json
training_args.bin
rng_state_0.pth
rng_state_1.pth
rng_state_2.pth
rng_state_3.pth
```

恢复命令：

```bash
export RUN_ID=GRPO-TK-REPRO-E1
export OUTPUT_ROOT=/root/GRPO-checkpoints
export RESUME_FROM_CHECKPOINT=/root/GRPO-checkpoints/${RUN_ID}/checkpoint-500
export MASTER_PORT=29226

bash ablations/gr_rec_think_sample8_fullsid_v3/launch_sample8_fullsid_train.sh \
  2>&1 | tee /data/GRPO/logs/${RUN_ID}-resume500.log
```

恢复后确认 step 大于恢复点，并实际验收下一个 50-step checkpoint、对应 Probe4 以及 `trainer_state.json` 中 `save_steps=50`。

## 14. Monitor

训练会把 append-only 证据写到 `/data/GRPO/runs/<run-id>`。启动多实验 monitor：

```bash
python scripts/monitor/server.py \
  --runs-dir /data/GRPO/runs \
  --outputs-dir /data/GRPO/outputs/formal \
  --checkpoint-outputs-dir /root/GRPO-checkpoints \
  --host 127.0.0.1 \
  --port 8878
```

访问：

```text
http://127.0.0.1:8878/?kind=recommendation_grpo&run=<run-id>
```

GRPO-TK 关键流：

- `manifest.json`
- `metrics.jsonl`
- `rollouts.jsonl`
- `sample8_fullsid.jsonl`
- `probes.jsonl`
- `training_sample_exports/`

前端和离线分析应直接展示训练时捕获的 reward、advantage、SID 和 parser status，不在 JavaScript 中重算训练公式。

## 15. 外部评测对齐

外部 11 项顺序固定为：

```text
懂物料 4 项 + 懂用户 2 项 + 懂推荐 4 项 + 懂世界 1 项
```

当前记录：

- Parent 三次：`1.3436 / 1.3497 / 1.3510`，均值 `1.3481`。
- GRPO-TK checkpoint-250：`1.3579`。
- 相对 parent 均值：`+0.0098`。
- 相对 parent 最好值：`+0.0069`。

要声明稳定复现，应至少对同一 GRPO-TK checkpoint 做 3 次同配置外部评测，并同时报告四类合计和 11 个原始分项。单次 `1.3579` 只能称为当前最高单次记录和早期正向信号。

## 16. 常见非等价改动

以下任一变化都会使实验不再与 GRPO-TK 严格等价：

- 更换 parent、Base、tokenizer 或私有数据顺序。
- 在 `</think>` 后固定 target domain。
- 插入自然语言 answer bridge。
- 把四个 SID G8 flatten 成 G32。
- 只生成 4 token 并要求 SID 必须从 continuation 第一个位置开始。
- 将 CoT reward 改为 max/mean/Beam/interest score，而不是 8 个 raw SID reward 之和。
- 对 zero-std G8 自动重采样。
- iteration 2 重新 rollout。
- 用 `generate.scores` 代替 unchanged-policy full-forward old logp。
- 让 SID loss 覆盖前置自然语言或后续第二个 SID。
- 把 Probe4 混入训练数据。

复现结果出现差异时，应优先比较 manifest、parent SHA、dataset group hash、rollout fingerprint、Probe4 IDs、采样参数和 imported module paths，再讨论算法效果。
