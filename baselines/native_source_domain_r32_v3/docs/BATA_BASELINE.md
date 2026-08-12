# Pure bata_baseline

Goal: reproduce `onereason_material_domain_2ep_repro_20260811.tar.gz` in the active training framework with one intentional variable: understand_user uses the active BETA data.

## Dataset contract

- Material understanding, SID canonical, SID reverse, and recommendation rows come directly from the reproduction Parquet.
- The 32,848 understand_user rows come from active BETA. The first 32,452 replace the archive user positions and the remaining 396 are appended.
- World-understanding data is excluded.
- Recommendation rows retain multi-positive metadata grouped by exact user history and target domain. REC-PU remains disabled for this experiment.

The formal dataset has 222,001 rows: material 100,000, canonical 11,298, reverse 29,586, understand_user 32,848, and recommendation 48,269.

## Training contract

LoRA r32/alpha64/dropout0.05, 8K neat packing, FA2, Liger, global batch 64, LR 2e-4, cosine, warmup 0.03, weight decay 0.01, bf16/pure bf16, seed 20260806, four GPUs, and two epochs match the reproduction archive.

The archive uses full activation checkpointing. This experiment uses fractional GC 0.4 in the active framework to reduce memory without changing the forward objective. REC-PU and task-ratio sampling are disabled.

The active environment uses `report_to: none` because TensorBoard is not installed. All scalar metrics remain available in `train.log`; this runtime logging adaptation does not affect the training objective.

Monitor-only recommendation metrics reuse detached logits and the retained multi-positive metadata. Core recommendation metrics and recommendation segment/count statistics are accumulated every step, candidate hit/coverage metrics are sampled every 50 optimizer steps, and task loss/exposure metrics use the normal logging cadence. The monitor-only path never replaces baseline CE and has an exact loss/gradient parity regression.

Formal launch entry:

```bash
bash /data/baselines/native_source_domain_r32_v3/scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh
```

The launcher runs hard checks for the dataset, registry, config, and actual loss routes before creating GPU workers. Dataset/config creation does not automatically start formal training.
