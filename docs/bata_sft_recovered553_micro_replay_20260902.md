# BATA SFT historical checkpoint-553 micro-replay audit

Date: 2026-09-02

## Conclusion

Classification: **CASE C**.

Two independent 4-GPU replays started from byte-identical copies of the real
historical checkpoint-553, used the same recovered 06:33 source, packed Arrow
cache, batches, restored RNG states, scheduler horizon, package environment,
and launch environment. They nevertheless diverged on the first replayed
optimizer update (step 554). The first observed loss difference was
`1.4901161193847656e-08` on rank 0 and the global grad-norm difference was
`-6.92605972290039e-05`. By step 555, LoRA, effective B@A, and optimizer
fingerprints were different.

This supports actual runtime/kernel nondeterminism. It does not support a data
ordering, checkpoint input, RNG restore, scheduler, or dataloader-skip mismatch
between Replay A and Replay B.

## Retry1 stop

- Preserved root: `/root/onereason_final_reproduction_20260902_sft_retry1`
- Last completed optimizer step: `670`
- The four SFT ranks were stopped gracefully after confirming no checkpoint
  save was active.
- The outer `run.sh` process remains suspended (`STAT=T`) so downstream stages
  cannot start.
- GPU 0-3 after stop and after both replays: no compute processes, `5 MiB`
  used per GPU, `0%` utilization.
- No retry1 logs, state, monitor data, or checkpoint were deleted.

## Historical source contract

Pristine recovered source:
`/root/bata_sft_micro_replay_20260902/source_pristine`

Instrumented runtime copy:
`/root/bata_sft_micro_replay_20260902/source_runtime`

| Runtime file | SHA256 |
|---|---|
| `scripts/train_native_source_domain_r32_v3.py` (pristine) | `91b9bfb360cfe2b0bfef8d333dcace433a99098c1facc0c898f1c7db117f8a10` |
| `rec_pu/sid8_rec_pu_integration.py` | `b272923dbbc6e744ade2d5e5f1e3007520f27ceaf1b563a32e5fb355a32944f4` |
| `scripts/run_native_source_domain_r32_v3.sh` | `28534965b0b3e96fb77401306bc73d440702512bd0edf152f428d4cad66b7f2b` |
| `scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh` | `e6c35e53850533b23cc771d187177763984b2a0ea1d142003ddf13d039ea4230` |
| `config/train_bata_baseline_4gpu_gc04_2epoch.yaml` | `ff6b589b16c075b354569033159ddfa1c1cf0978dc55a5fe909a2d3e068d8445` |
| `scripts/validate_bata_baseline.py` | `ba48cc4e7253a48b4218290e4876a62dda71d259dcbf2d84a0c7314abc5e9787` |
| `pack_ratio_sampler.py` | `9776f0a55e4c10439f8f2a5bf7a33e960562c507e66bacdd5cfa1ee96b5554ed` |
| LLaMAFactory Git commit | `01398eb18dd475a6e27c36f15b970aeacf0d4a60` |
| LLaMAFactory diff | `1eabb66650106e6e56586544e5b330e8ff6a1e4863da889304e52e72c81f5ec3` |
| LLaMAFactory Python tree | `6f7bce5478bc164279cc3aff25e4f3c1e4e665bb0bb3b9ebcdd0279c74766e8e` |

The runtime trainer differs from pristine only by the replay instrumentation.
Its SHA is `9766197d6f512a4686c600944613c473fe84ed1f3ad8fa7568cf1d6d8f513a4f`.
The complete runtime tree SHA is
`8e709af79072c72ad208c5be3ad2cd65f09a0972d1d5d3f21601ed6c4c629a5d`.

## Historical checkpoint-553 input

Source checkpoint:
`/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-553`

| File | SHA256 |
|---|---|
| `adapter_model.safetensors` | `9c5426a5c329e9ec19a5dddc351c51c498dab4b4ed3a81ce80cfa07dc84cab9e` |
| `optimizer.pt` | `4806181f0754d954c4c9f30fe38d5b33587582dbb2a710e9571e7d7493213d98` |
| `scheduler.pt` | `b951d51e69752e0946c62e098ed98dc80e281b252d1eca4f0c996ae08dc7dccb` |
| `trainer_state.json` | `bacd549ddeaadde8d065d9d6184c68aa1fda88083a1577005fae8a79839e29d8` |
| `training_args.bin` | `2007c7967597a348c3f8ff68991fff0ab399ce019501a55dbbe3c741da6bb7d2` |
| `rng_state_0.pth` | `5d7293a18c2d02ee3cd516babc094c680cf849faba4e4352053bccf44a760996` |
| `rng_state_1.pth` | `b1d4c3eebf0e573130bb904de7c972893cb4fadc5f6a5c4d33a4548714f67196` |
| `rng_state_2.pth` | `e148a4b7c58775628765cd1617cad9edee5fcddab223723c80dc47ce79f9fe9a` |
| `rng_state_3.pth` | `e31d3245179273a4f709cfc7b6529dc141494bab2b36d977ac14370970dde64d` |

The checkpoint contract is `global_step=553`, `epoch=1.0`, scheduler
`last_epoch=553`. Both replay copies matched every expected SHA before load.

## Packed cache contract

Both runs directly loaded the same read-only Arrow cache mount:
`/root/bata_sft_micro_replay_20260902/historical_packed_cache_mount`.

It maps the 16 historical Arrow shards (35,380 packed rows, sequence length
8,192) and did not retokenize or repack data. All four ranks recorded identical
ordered batch fingerprints between A and B for every step 554-560; every step
contained 16 microbatches per rank.

## Replay summaries

| Replay | Updates | Final step | Runtime | Final adapter file SHA |
|---|---:|---:|---:|---|
| A | 7 | 560 | 278.034 s | `b7727509c7b09017713dc149e299c09faadee10b5da98ba5973141c255155799` |
| B | 7 | 560 | 277.608 s | `472b57f71342beeb78e14248d55c54c5a22321cae6f3b69a8052b66d3489a0f9` |

The configured scheduler horizon remained `max_steps=1106`. A callback stopped
training only after global step 560. Both outputs are adapter-only.

## Heavy fingerprints

| Point | Replay | LoRA | Effective B@A | Optimizer |
|---|---|---|---|---|
| initial553 | A/B | `277923713f1aa771fb6baffdd469f2fa3cc3f267aa2e0bd5908c2bc00ae5a98c` | `9c88e40868ae02d18a53e9356fe0bab0a56e9b72f88855096ff598a3c5bbfe7b` | `f6e496860b212d1702afa86923fbc473f5a0d323b0085aab6b68b82fe1283507` |
| step555 | A | `b7931124213ef19ff83be329fab6ef555fa79906da35f6d3f89fa6a6864cb68c` | `dd60881908d6033c5ac296c847d88bf591b8258f721e37744e4b92039699a310` | `fa138653a23fd3708f147a0b8362872cdf7a2761d53da125c5501eb17566c527` |
| step555 | B | `fbb8ecf88b7a3f5783161308c359b5c0b881fe9e409e77e5243c0598fbc1672e` | `718e19aea7407d81759e859e3843746b57f0c978a4742c3109ab40773a76a603` | `6c34924eb3851bb05e4a28a6839510cabfdd7d7d3b5c27d5fc45e07cd82b4b6c` |
| step560 | A | `bc9a1894c7fcb69e80d50f541d41168ed1cd0e9bc1b72293c250c6011d492743` | `76f9b0d018e89e0ef7e1d2684cce061c66e272d2d277a55284093d7709644c1f` | `1e63b2c108f712026bb399953d65626b626ac6a86bdaf185efb443aa3faa428f` |
| step560 | B | `778acce15fc4a4b9cc18908ecf529f63174f92c4d44f974d457ae73073faf8bb` | `a76b0e90505d7177fab67653feeb41a44c642daddd5809098b36e173bb6523d2` | `06737c07f0b9ad64e33519c0d6a246579894d5bfa7711185a3f4ff64d09888f5` |

At step 560, direct A-vs-B tensor comparison gives:

- Raw LoRA: cosine `0.9999841728620636`, L2 `0.38046419624103267`,
  relative L2 vs A `0.005626194852755128` (504 tensors).
- Effective B@A: cosine `0.9999449366721759`, L2
  `0.6545380213844452`, relative L2 vs A `0.010493999827034738`
  (252 modules).

The compact RNG hashes below are identical in Replay A and Replay B. Batch
fingerprints are likewise identical for the corresponding rank and step.

| Step | Rank | Ordered batch fingerprint | Compact RNG hash |
|---:|---:|---|---|
| 555 | 0 | `13c1cb5893c132e16d1904293797cb60e3001b23c280df7ed48d88e022e78e62` | `50edca55d145837b3c8bdec71380ead707efab69ea32a3432f69167e6f71629c` |
| 555 | 1 | `f5dd39dd2d229a31253808523fd81d431a020cec5a11e232beb7463dd44e88d5` | `ee2c27565b2cb4c7b0602414f966beb23851e4787b5a0cd4e6a41a228b6c28e2` |
| 555 | 2 | `2fad9c0817163a0b5937eebd13d251f6afac387a922e84f42639ff19207a7cc7` | `191ffbbc278aad2d8def6dae5db4a6d1b2b72776adcc6b659dc6f4912f45c2e8` |
| 555 | 3 | `9a6e491b91ca60264ae519171c148f96d04571bcf958813a38b229ecac871fca` | `56ebe902ac133b22cec528627281d8ca37c54015f4b744ddbce1023bea511721` |
| 560 | 0 | `73f803ec315360224b191b8b57beaee98d61e525dbe5fa50907a856c063e94e5` | `c9e6e68e9d6fe8da01cc697c27e59a6fda38e54a7ac3cd8329f56b407fd4978f` |
| 560 | 1 | `b34c8ac5227adc3bb8a23bb712b89473d5ab012734ef21ad6812758204304f0d` | `2f50adf96166dc22008663a134ad0d0d5a2d1d100d0d95abe271ad3b989e76c8` |
| 560 | 2 | `2296bc201985fc1b7bd0fddfb7253654ac932fb60732de2b203a6bcb0a573666` | `7353c3c7be7440374eac52d7b93f58719169be355f4b0a9b38e50b81520d9420` |
| 560 | 3 | `115b90c8a9f0b97682b6bb1bd7932542c6a6d8d469371291ee4e3a9959c61c6a` | `1e2c4a8ea97608b3dde5c7f3ccc907f81448558f85a49684d65f92ef868c704b` |

## Metrics against historical trajectory

| Step | Run | Loss | Grad norm | LR | Epoch |
|---:|---|---:|---:|---:|---:|
| 555 | Historical | 30.4055877686 | 0.6057028174 | 0.000104687226247 | 1.0036178632 |
| 555 | A | 29.8909378052 | 0.6030939817 | 0.000104687226247 | 1.0036178632 |
| 555 | B | 29.8971786499 | 0.8010533452 | 0.000104687226247 | 1.0036178632 |
| 560 | Historical | 26.7383972168 | 0.9542764425 | 0.000103223090876 | 1.0126625212 |
| 560 | A | 26.7387847900 | 0.9227518439 | 0.000103223090876 | 1.0126625212 |
| 560 | B | 26.7228332520 | 1.0998361111 | 0.000103223090876 | 1.0126625212 |

The step-555 logged loss is not directly comparable: the historical value is a
five-step logging window (551-555), while each replay resumed at 553 and its
first logging window contains only 554-555. Grad norm, LR, and epoch remain
pointwise comparable. Step 560 uses the same 556-560 window in all three runs;
Replay A differs from history by `+0.0003875732` loss and `-0.0315245986` grad
norm, while Replay B differs by `-0.0155639648` loss and `+0.1455596685` grad
norm.

The equality of LR, epoch, task mixture, and batch fingerprints confirms that
the replay reached the intended historical location. Replay A being closer to
the historical scalar loss at step 560 is not repeatable evidence because A
and B diverged from each other under the same inputs.

## Restore and runtime audit

- Adapter weights are loaded before Trainer execution.
- Optimizer and scheduler are created and then restored from checkpoint.
- `on_train_begin` precedes epoch RNG restoration in Transformers.
- Because checkpoint-553 is exactly at the epoch boundary,
  `steps_trained_in_current_epoch=0`; no dataloader batch is skipped.
- RNG is restored before constructing/iterating the epoch dataloader.
- Instrumentation performs hashes only after restore and does not call Python,
  NumPy, CPU Torch, or CUDA random APIs.
- No restore-order or dataloader-skip difference was found between A and B.

Replay environment:

- Python 3.11.14; PyTorch 2.5.1+cu124; Transformers 5.3.0;
  Accelerate 1.11.0; PEFT 0.18.1.
- CUDA 12.4; cuDNN 9.1.0; NVIDIA driver 535.129.03.
- FlashAttention 2.7.4.post1; Liger enabled.
- `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_IB_DISABLE=1`,
  `NCCL_SOCKET_IFNAME=lo`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.
- `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` were unset.
- TF32 matmul false, TF32 cuDNN true, cuDNN benchmark false, cuDNN
  deterministic false, deterministic algorithms false.

Transformers 5.3.0 blocks checkpoint loading with PyTorch below 2.6. The replay
therefore used a narrow, recorded compatibility bypass after verifying every
historical checkpoint SHA; `torch.load(weights_only=True)` was retained, with
explicit NumPy safe globals only for RNG-state deserialization. The historical
job trained continuously through step 553 and did not exercise this resume
path. This is a replay-control-path difference, while the exact historical
PyTorch/CUDA/NCCL runtime build remains unproven because it was not recorded.

Likely numerical divergence sources are the nondeterministic 4-GPU execution
stack: DDP/NCCL reductions, FlashAttention, Liger kernels, and scatter/reduction
kernels. This audit does not isolate one kernel, and no deterministic replay
was run as required.

## Preserved evidence

Successful runs and all failed pre-update attempts remain under:
`/root/bata_sft_micro_replay_20260902/runs`.

The failed attempts stopped before any optimizer update and document the cache
mount, manifest-path, PyTorch safety-gate, NumPy RNG allowlist, and scalar
fingerprint issues. They were not deleted or overwritten.

No 1106-step replay, deterministic replay, GR_REC, GRPO-TK, MC_USER, new
external evaluation, or downstream stage was executed.
