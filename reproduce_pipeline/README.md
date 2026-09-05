# Deterministic OneReason reproduction pipeline

The default one-command chain is:

```text
Full-Parameter SFT (1 epoch)
  -> GRPO1 Recommendation Bilateral (300 steps)
  -> GRPO3 User GRPO (200 steps)
```

GRPO2 Think-only is retained but skipped by default. Enable it with
`RUN_GRPO2=1 ./run.sh`; GRPO3 will then inherit the single cumulative GRPO2
adapter. When GRPO2 is skipped, GRPO3 inherits the single GRPO1 adapter.
Neither route merges, stacks, or initializes an extra adapter.

## Data

GRPO1, optional GRPO2, and GRPO3 resolve data only through
`REPRO_DATA_ROOT/registry.json`. Copy
`manifests/reproduce_dataset_registry.example.json` to that location and keep
the registered relative paths, SHA256 values, row counts, and splits exact.
The resolver never generates or edits data.

SFT data is temporarily outside the registry. Set `SFT_DATA_ROOT` to the
frozen Rec FDR V4.3 package data directory and use `SFT_DATASET_KEY` as its
audit label. TODO: migrate this SFT package into the same registry without
changing its bytes.

## Usage

From a clean checkout on a four-GPU development machine:

```bash
cd reproduce_pipeline
./run.sh
```

Useful controls:

```bash
REPRO_DATA_ROOT=/root/reproduce_datasets RUN_ROOT=/root/my-run ./run.sh
RUN_GRPO2=1 GRPO2_PROBE_DATA=/path/to/frozen/probe.jsonl ./run.sh
DRY_RUN=1 ./run.sh
RUN_GRPO2=1 DRY_RUN=1 ./run.sh
```

The pipeline fails closed on source mutation, dataset identity mismatch,
unexpected parent lineage, non-final checkpoints, incomplete resume state, or
GPU release timeout. It does not run external evaluation, merge models, upload
artifacts, retry, or resume automatically.

Default retained model links are `01_sft_final/`, `02_grpo1_final/`, and
`03_grpo3_final/`. With GRPO2 enabled they are `01_sft_final/`,
`02_grpo1_final/`, `03_grpo2_final/`, and `04_grpo3_final/`.
