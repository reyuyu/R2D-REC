# REC-PU

## Formal Run

- Run: `REC-PU-BETA-MATERIAL-ALIGNED-R32-B005-2E`
- Parent: `BETA-MATERIAL-ALIGNED-SID8-R32-2E-GC04-4GPU`
- Dataset version: `BETA_material_aligned_v1`
- Dataset: `onereason_beta_material_aligned`
- Manifest: `/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json`
- Training: 4 GPUs, LoRA r32/alpha64/dropout0.05, 8K neat packing, FA2,
  global batch 64, LR 2e-4 cosine, warmup 0.03, 2 epochs, GC 0.4,
  bf16/pure_bf16/Liger, seed 20260806.

## REC-PU Objective

REC-PU replaces, rather than adds to, the one-hot CE of the final
recommendation SID `a/b/c` components. It uses observed positives as `P`,
same-level unobserved SIDs as `U`, and all wrong-level or ordinary tokens as
`O`. `U` denominator gradients are attenuated with beta `0.05`; `O` gradients
remain 1.0. The positive target is mean-positive. SID and domain token weight
is 8 and the baseline SID8 denominator is unchanged.

Debug probes, candidate dumps, reference comparisons, profiling, and full-vocab
top-k are disabled in the formal run.

## Locked Material Contract

`material_sample` has 100000 rows with ordinary response weight 1, SID/domain
weight 8, and the locked four-domain multiplier. `sid_bucket_canonical_no_think`
has 11298 rows with every response token weight 4 and no domain multiplier.
`sid_bucket_reverse` has 29586 rows with ordinary weight 1, SID/domain weight 8,
and no domain multiplier.

Domain counts and weights: video 30092 / 1.1791795483099141; prod 29180 /
0.7738397930531297; ad 22768 / 1.133452723686895; living 17960 /
0.8980530210503628.

## Preflight and Regression

The launch script runs the immutable material-aligned preflight before creating
GPU processes. REC-PU Phase 1 through 3 and the Phase 4.5 optimized/reference
CPU regression suite must also pass before launch.

Phase 4.5 checks: U gradient ratio 0.05, O gradient ratio 1.0, no double
counting, unchanged denominator, and optimized/reference loss and gradient
equivalence. The earlier vectorized 20-step four-GPU run was stable with a
0.84% slowdown. The final fused custom-autograd implementation is selected for
the formal run; by decision, it is not separately re-smoked.
