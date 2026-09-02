# BATA step554 FlashAttention deterministic-backward replay

## Verdict

`FADET_STEP554_FULLY_REPEATABLE`

With `FLASH_ATTENTION_DETERMINISTIC=1`, two fresh four-rank replays of the
historical checkpoint-553 produced exact equality through the step554 update.
The narrow supported conclusion is:

`FA2_BACKWARD_WAS_A_MATERIAL_NONDETERMINISM_SOURCE`

This does not establish that FlashAttention was the only source of divergence
in the full historical 2026-08-12 run, nor does one repeatable update prove a
full 1106-step recipe is effect-reproducible.

## Frozen contract

- Historical checkpoint: BATA checkpoint-553; all nine adapter, optimizer,
  scheduler, trainer-state, training-args, and rank RNG files matched their
  pinned SHA256 values.
- Recovered trainer source SHA before forensic integration:
  `91b9bfb360cfe2b0bfef8d333dcace433a99098c1facc0c898f1c7db117f8a10`.
- SID8 integration SHA:
  `b272923dbbc6e744ade2d5e5f1e3007520f27ceaf1b563a32e5fb355a32944f4`.
- LLaMAFactory commit: `01398eb18dd475a6e27c36f15b970aeacf0d4a60`.
- Four A800 ranks, GA16, 8K packed cache, original batch/LR/AdamW/cosine
  schedule, scheduler horizon 1106, diagnostic stop after optimizer step554.
- FA2 and Liger remained enabled. Strict deterministic algorithms and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` remained enabled.

## Actual input identity

The actual consumed collator outputs were fingerprinted before Accelerator
device placement. The eight-field contract covers `input_ids`, `labels`,
`loss_weights`, `sample_ids`, `sample_task_ids`, `sample_domain_weights`,
`attention_mask`, and `position_ids`. FA2 neat packing intentionally represents
`attention_mask` as explicit `None`; this is encoded with the same `ABSENT`
marker used by the prior exact DET contract. No CUDA tensor was copied back for
input hashing, no random API was called, and no model forward was added.

| Rank | 16-microbatch ordered fingerprint | A/B exact |
|---:|---|:---:|
| 0 | `acec704d63052dfeaf9464944c3c1a2904adc8bbc67af3e9580386642ac46095` | yes |
| 1 | `045f5ddbc0e6d7d5c9b443bb1f2f6d2a66917d49e6b6d0bd3524a74ad4b0a034` | yes |
| 2 | `b3d474278bda352f74334dcaaf4956f8e6c779a0d565b24c1fe3f9159c1a05ad` | yes |
| 3 | `365d252daed708c22f23d3039917dc19112bfefdc57fb091fb35d51414d95f3f` | yes |

The four fingerprints also equal the previously frozen DET contract exactly.
Restored Python, NumPy, CPU Torch, and CUDA RNG fingerprints matched A/B on
every rank.

## FA2 runtime confirmation

- `flash-attn`: `2.7.4.post1`.
- Environment: `FLASH_ATTENTION_DETERMINISTIC=1`, exported before Python.
- A low-interference wrapper observed the arguments of actual API calls; it
  did not execute an extra forward.
- Both `flash_attn_func` and `flash_attn_varlen_func` were observed on every
  rank in both runs, always with `deterministic=True`.

## Step554 evidence

| Rank | Local loss A/B | PRE_ALLREDUCE | POST_ALLREDUCE | PRE/POST clip | Grad norm A/B |
|---:|---:|:---:|:---:|:---:|---:|
| 0 | 2.0197828598320484 | exact | exact | exact | 0.742214977741241 |
| 1 | 1.59639323502779 | exact | exact | exact | 0.742214977741241 |
| 2 | 2.560986466705799 | exact | exact | exact | 0.742214977741241 |
| 3 | 1.3222751496359706 | exact | exact | exact | 0.742214977741241 |

All 16 rank-local microbatch losses are exactly equal A/B, not only their
means. The single DDP bucket layout is identical: float32, 87,293,952 values,
504 LoRA parameters. Parameter-offset comparison found 0/504 divergent LoRA
gradient tensors; attention projections were 0/288 divergent and MLP
projections were 0/216 divergent.

Final public fingerprints are also exact:

| State | A/B fingerprint |
|---|---|
| LoRA | `3632a80ff4f4505557fb1bf407c5de6741ef36c2ec96f884833bbeaa2276766d` |
| effective B@A | `031fbe3376a9c59cd1e4b9d9405d023de9c56c76d27f51865901afdec356ae15` |
| optimizer | `ee9136f9c9d53235606b5a2ea038c15ce7e452d58e8f4e04671371a47973fcbd` |

## Scope and artifacts

The first two FADET launch attempts were preserved server-side and stopped
before the first model forward because the new input probe initially rejected
the real `BatchEncoding` type and then the intentional FA2 packed `None`
attention mask. Targeted regressions fixed both evidence-layer bugs. Neither
failed attempt reached backward or an optimizer update. The successful pair
used a third fresh root and fresh checkpoint copies.

No dataset row, raw token value, checkpoint, model weight, optimizer/RNG byte,
or credential is included. The full public-safe evidence is in
`fadet_step554_result.json`; its SHA256 is
`b568e976a2715da2a460b0f2a3a26e7b7fc7802ed87cfd2ba37b1b221487a34a`.
The complete forensic CPU/static suite passed 45/45 tests.
No 553-to-560 or long training, downstream GRPO stage, or external evaluation
was run.
