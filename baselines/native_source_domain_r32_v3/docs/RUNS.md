# Run Layout

The baseline source, data, outputs, and logs are intentionally separated:

```text
/data/baselines/native_source_domain_r32_v3/
  config/       immutable run configurations
  dataset/      active dataset snapshot and manifest
  scripts/      build, audit, training, and launch scripts
  docs/         experiment records and operational documentation

/data/outputs/baselines/native_source_domain_r32_v3/<RUN_ID>/
  checkpoint-*/ model, optimizer, scheduler, and trainer state
  tensorboard/  scalar event files

/data/logs/baselines/native_source_domain_r32_v3/<RUN_ID>/
  train.log     complete stdout/stderr from torch.distributed.run
```

Every formal launch requires a unique `RUN_ID`; do not reuse an output directory after a completed run. Smoke outputs remain under the legacy `/data/outputs/onereason_native_source_domain_r32_v3_smoke_*` paths and are not formal artifacts.
