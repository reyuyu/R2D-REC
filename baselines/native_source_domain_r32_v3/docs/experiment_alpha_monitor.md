# Experiment Alpha Monitor Optimization

## Phase 2: 98/2 leak-safe validation split

The source dataset `alpha-jiankong` remains read-only.  The derived version
`alpha-jiankong-split-v1` uses deterministic SHA256 selection with seed
`20260812`; `train98` is the only future training split and `dev2` is held out.

- Recommendation is split by connected components of `recommendation_group_id`
  and canonical ordered SID-history plus target domain.
- User samples are split by normalized input/history family, co-locating paired
  CoT and NoThink variants.
- Material is split by normalized complete prompt family.
- A global canonical-prompt union prevents cross-task identical-prompt leakage.
- Rows are copied verbatim; recommendation multi-positive metadata remains
  unchanged.

Artifacts in the derived data directory are `train.jsonl`, `dev.jsonl`,
`dataset_info.json`, `manifest.json`, `split_audit.json`, and `split_audit.md`.
The phase performs no model forward, validation forward, training, loss change,
sampler change, or monitor change.

### Completed audit

- Source / train / dev rows: 221,252 / 216,732 / 4,520 (dev 2.042919%).
- Recommendation groups: 20,531 / 20,060 / 471.  All four target domains are
  present in dev.
- Required cross-side intersections are all zero: recommendation group,
  recommendation SID-history + domain, user family, canonical prompt, and exact
  full row.  The source has 244 pre-existing exact duplicate instances; output
  has the same 244, with no cross-side duplicate.
- Data-only 8K neat packing: train98 33,810 packs and dev2 733 packs.  With
  four GPUs and GA16 (global batch 64 packs), train98 is 529 optimizer
  steps/epoch; a future full dev pass would process all 733 packed sequences
  (184 four-GPU batches).

## Phase 3: lightweight teacher-forced validation sidecar

The training path remains unchanged.  A callback performs validation only after
an optimizer step, enters `model.eval()` and `torch.inference_mode()`, restores
the model mode and Python/CPU/CUDA RNG state, and does not call backward,
optimizer/scheduler step, or alter the global training step.

- The deterministic `alpha_jiankong_dev_probe_v1` is a strict dev subset:
  176 rows from 85 whole recommendation groups (50 singleton and 35
  multi-row groups), packed into 42 sequences.  It contains 101 CoT and 75
  NoThink recommendation segments; domain rows are video/prod/ad/living =
  123/26/20/7.  SHA-256:
  `664098072283f45566b5d0c614eb8506a3683bede39786c1464aa437682aeee4`.
- The formal cadence is one probe per 100 optimizer steps and one full pass at
  each epoch end.  Its main log emits only `va`–`vo`, `ga`–`ge`, and the
  conservative overfit-warning flag.  Detailed metrics, gold probabilities,
  domain metrics, source-domain reweighting, and duplicate metadata are in
  `alpha_validation_metrics.jsonl`.
- Evaluation uses an exact rank-stride sampler, so the 42-pack probe shards as
  11/11/10/10 and the 733-pack full dev shards as 184/183/183/183: no padding
  or duplicated statistics.

### Final 4-GPU smoke result

The final 12-step smoke (probe at steps 5 and 10, then one full dev pass) is
PASS.  Both probes emitted all `va`–`vo` and `ga`–`ge`, had no missing or
invalid recommendation route, and did not raise the conservative warning.
The full dev processed exactly 733 packs / 1,061 recommendation segments
(621 CoT, 440 NoThink; 3,183 final SID positions) in 172.60 seconds.

- Probe mean wall time: 9.82 seconds (42 packs).
- Full-dev wall time: 172.60 seconds (733 packs).
- Steady training time: about 37.69 seconds/optimizer step.
- Estimated validation overhead per 529-step epoch: about 1.1% (five probes
  plus one full pass); probe-only amortized cost is about 0.26%.
- Peak allocator memory recorded by the callback: 69.29 GiB/rank; the highest
  observed driver memory during full dev was about 76.5 GiB.  There was no
  OOM, NaN/Inf, DDP mismatch, packed-boundary error, or metadata error.

CPU regression covers metric parity, exact-shard semantics, domain reweighting,
and preservation of gradients, optimizer state, scheduler state, RNG, and
training mode.  The formal configuration remains unlaunched.

## Phase 4: formal configuration freeze

The formal entrypoint is intentionally separate from all smoke outputs.  It
copies the frozen YAML into a fresh run directory, rewrites only the run-local
output and validation JSONL paths, snapshots the split/probe/material manifests,
and runs a fail-closed preflight before it allocates any GPU memory.

The preflight rejects a wrong dataset registry, row or cache count, train/dev
leakage, non-fixed probe SHA, stale `max_steps`, non-null resume checkpoint,
REC-PU/PackRatio activation, disabled monitor fields, or a material/SID route
that differs from the native baseline contract.  It also prints the epoch plan
derived from the actual packed cache: 33,810 train packs, four GPUs, batch one,
GA16, 529 optimizer steps per epoch, 1,058 total steps, and 32 warmup steps.

This is not a strict single-variable score ablation against the old BATA
baseline.  Alpha uses the cleaned `alpha-jiankong` source and the leak-safe
98/2 held-out split; only the objective and optimizer recipe are held to the
ordinary native SID8 baseline.  Monitor OFF/ON loss and gradient parity show
that the monitoring itself does not alter training gradients, but future score
differences must not be attributed to monitoring alone.

## Formal training result: alpha_jiankong (mother experiment)

The stages above (leak-safe split, validation sidecar, SID/domain weight
contract fix, formal config freeze) are milestones of the single mother
experiment `alpha_jiankong`, not separate experiments.  After the SID8 cache
weight-contract fix (plain token=1, canonical=4, non-canonical SID/domain=8),
the formal 2-epoch run scored:

| Epoch | Total | Breakdown |
| --- | ---: | --- |
| 1 | `1.2605` | material `0.0465, 0.0379, 0.0441, 0.0426`; user `0.1502, 0.0915`; recommendation `0.1204, 0.1394, 0.2016, 0.1521`; world `0.2342` |
| 2 | **`1.2992`** | material `0.0490, 0.0369, 0.0516, 0.0420`; user `0.1556, 0.0955`; recommendation `0.1241, 0.1394, 0.2002, 0.1683`; world `0.2364` |

Epoch 1 → 2 improvement `+0.0387`, driven mainly by user and recommendation
subscores; world stays stable.  Material subscores follow the historical
convention of not being comparable across runs.
