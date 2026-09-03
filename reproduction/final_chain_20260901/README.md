# OneReason Final Chain Reproduction

This isolated bundle reproduces the selected final adapter chain from the frozen dataset snapshot:

```text
BETA SFT checkpoint-1106
  -> GR_REC_v1 checkpoint-1500
  -> GRPO-TK checkpoint-250
  -> MC_USER Hybrid K4 prompt-step-0100
```

The default data registry is `/root/reproduce_datasets/onereason_final_chain_20260901`. No training stage reads the historical `/data/lf_data_versions`, `/data/GRPO/data`, or `/data/GRPO_USER/data` dataset locations. No teacher model is used; `/root/teacher_expert_ckpt/README.md` records that contract.

数据来源、Recommendation SFT 多正样本元数据以及懂推荐 GRPO 域特定 prompt 的处理说明见 [docs/DATASET_PROVENANCE.md](docs/DATASET_PROVENANCE.md)。

## Commands

Static and data preflight only:

```bash
cd /root/onereason_final_reproduction_20260901
bash run.sh --preflight
```

Run the full four-stage chain in the foreground:

```bash
cd /root/onereason_final_reproduction_20260901
bash run.sh
```

The full run starts a separate read-only reproduction-quality dashboard at
`http://127.0.0.1:8891`. It polls append-only metrics, stage markers, checkpoint
structure, adapter hashes, and optional external-evaluation records. It does not
import the training runners, allocate CUDA memory, or alter RNG state. Start only
the dashboard after preflight with:

```bash
bash run.sh --monitor-only
```

Use `--no-monitor` only when a submission environment forbids local listeners.

Continue after already verified PASS stages:

```bash
bash run.sh --from-stage 3
```

`--from-stage` does not bypass parent verification. A completed stage is skipped only when both its PASS marker and checkpoint contract pass. Existing incomplete output causes a hard stop and is never overwritten or auto-resumed.

## Frozen seeds

| Stage | Seed |
| --- | ---: |
| BETA SFT | `20260806` |
| GR_REC_v1 training | `20260816` |
| Recommendation fixed probe | `20260818` |
| GRPO-TK training | `20260816` |
| MC_USER selection | `20260823` |
| MC_USER candidate | deterministic SHA256 of selection seed, prompt step, sample ID, and candidate index |

The Recommendation runners additionally apply their historical rank seed rule,
`torch.manual_seed(training_seed + local_rank)`. SFT uses the same LLaMAFactory
seed contract as the original run. The MC runner resets the candidate seed before
each candidate generation, so candidate order does not depend on ambient RNG state.

## Preserved checkpoints

- BETA SFT: epochs `checkpoint-553` and `checkpoint-1106`.
- GR_REC_v1: `checkpoint-500/1000/1500`.
- GRPO-TK: `checkpoint-50/100/150/200/250`.
- MC_USER: `prompt-step-0025/0050/0075/0100`.

Each selected parent is validated as adapter-only with exactly 504/504 LoRA tensors. Trainer checkpoints must also contain optimizer, scheduler, trainer state, training arguments, and four rank RNG files. MC_USER is intentionally non-resumable and stores its formal state with each adapter checkpoint.

The pretrained competition base at `/data/models/onereason-8b-pretrain-competition`
is an organizer-provided base model, not a teacher model and not a contestant-created
dataset. Every fine-tuning dataset consumed by this bundle is registered below
`/root/reproduce_datasets`; no teacher or expert checkpoint is required.
`BASE_MODEL_SHA256SUMS` freezes all four model shards and every tokenizer/config
file, and the launcher verifies all 16 files before any stage can use a GPU.

The runtime source is also isolated: `dependencies/llamafactory` is a clean
snapshot based on upstream `01398eb...` plus the exact native argument fields
used by BETA. `ENVIRONMENT_LOCK.json` freezes both Python package sets, the
LLaMAFactory snapshot commit, the four-A800 topology, compute capability, and
driver version. The preflight rejects any mismatch before training.

## Reproducibility boundary

The script freezes data bytes, data order, seeds, sampling settings, optimizer settings, checkpoint boundaries, code commit, four-GPU topology, and parent handoffs. Exact historical weight SHA equality cannot be guaranteed by seed alone because FlashAttention, NCCL reductions, CUDA kernels, and stochastic GPU sampling may be nondeterministic at floating-point level. The final report therefore verifies the complete training contract and adapter structure without falsely treating historical tensor hashes as a portable guarantee.

## Four-stage quality comparison

`historical_reference.json` freezes selected historical adapter SHA values and
matched training windows at every retained checkpoint. `repro_quality.py`
compares reproduced metrics only after the matching milestone exists. Contract
failures are hard failures; trajectory-band and adapter-SHA differences are
separate review signals because distributed stochastic training is not bitwise
portable.

The dashboard also loads `historical_curves.json`, a compact snapshot exported
from each original run's complete metric log. Stage charts draw the historical
and reproduced trajectories together and show both values on hover. Curves are
aggregated by training step and downsampled only for rendering; milestone
window checks retain their original contract.

Use the chart `+`, `-`, and reset controls or the mouse wheel to zoom around the
cursor. Drag horizontally to pan. The selected step window and its auto-scaled
Y axis survive the dashboard's five-second refresh cycle.

Create a machine-readable snapshot:

```bash
python repro_quality.py --root /root/onereason_final_reproduction_20260901 \
  --output state/reproduction_quality.json
```

For an optional CPU-only tensor comparison against a historical adapter:

```bash
python compare_adapters.py \
  --reference /path/to/historical/adapter_model.safetensors \
  --reproduced /path/to/reproduced/adapter_model.safetensors \
  --output results/stage_adapter_gap.json
```

To populate every available same-step comparison automatically, run the
low-priority CPU watcher:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 19 ionice -c3 python \
  build_adapter_comparisons.py \
  --root /root/onereason_final_reproduction_20260901 --watch
```

These values compare LoRA adapter tensors only, never the 8B base model. The
watcher writes cosine similarity, relative L2, and maximum absolute delta
atomically under `evidence/adapter_comparisons/` for the read-only monitor.

External evaluation is intentionally not inferred from online reward. Put a
small JSON file at `evaluations/<stage-id>.json`, for example:

```json
{"aggregate": 1.3510, "evaluator": "official-11-metric", "repeat": 1}
```

The dashboard will then display the reproduced score and delta without changing
the training run. See `docs/REPRODUCTION_QUALITY.md` for status semantics and
`docs/GPT_REVIEW_PROMPT.md` for the independent review request.
