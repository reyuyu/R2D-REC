# Native Source-Domain R32 V3 Baseline

该目录是同学 material-domain 训练路线的隔离复刻基线，不和当前多任务工程共享 Trainer 或输出目录。

## 目录约定

- `config/`：不可覆盖的训练与 smoke YAML。
- `dataset/`：活动数据快照。`manifest.json` 保存输入来源、统计和 SHA256；`versions.json` 管理活动版与归档版。
- `scripts/`：构造、审计、训练与启动脚本。
- `docs/`：实验记录和运行规范。
- 正式输出：`/data/outputs/baselines/native_source_domain_r32_v3/<RUN_ID>/`。
- 完整训练日志：`/data/logs/baselines/native_source_domain_r32_v3/<RUN_ID>/train.log`。

任何新数据实验都必须新建数据版本目录和 manifest，并在新 YAML 中显式引用；不要覆盖 `dataset/` 活动快照。

## 当前正式实验

`NSD-R32-V3-2E-GC04-4GPU-20260810`：四卡、全局 batch 64、8K neat packing、LoRA r32、SID 权重 8、2 epoch、0.4GC。详见 `docs/实验Baseline_NSD_R32_V3_GC04_4GPU.md`。

启动：

```bash
cd /data/baselines/native_source_domain_r32_v3
bash scripts/launch_4gpu_gc04_2epoch.sh
```
