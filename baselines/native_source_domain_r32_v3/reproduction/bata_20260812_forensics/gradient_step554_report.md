# BATA step554 gradient localization

## Verdict

ROOT_CAUSE_LEVEL = LOCAL_BACKWARD

GRAD-A2 and GRAD-B2 restore the same historical checkpoint-553 and have the
same restored RNG, frozen step554 input contract, runtime microbatch metadata,
16 rank-local microbatch losses, rank-local mean losses, and DDP bucket
layout. Their only DDP bucket already differs at PRE_ALLREDUCE on every rank.
Therefore local backward is the first observed gradient divergence; DDP/NCCL,
gradient clipping, and optimizer update are downstream of it.

## Inputs

| Contract | Value |
| --- | --- |
| Historical adapter SHA256 | 9c5426a5c329e9ec19a5dddc351c51c498dab4b4ed3a81ce80cfa07dc84cab9e |
| Recovered trainer source SHA256 | 91b9bfb360cfe2b0bfef8d333dcace433a99098c1facc0c898f1c7db117f8a10 |
| SID8 integration SHA256 | b272923dbbc6e744ade2d5e5f1e3007520f27ceaf1b563a32e5fb355a32944f4 |
| Launcher SHA256 | 28534965b0b3e96fb77401306bc73d440702512bd0edf152f428d4cad66b7f2b |
| LLaMAFactory | 01398eb18dd475a6e27c36f15b970aeacf0d4a60 |
| PyTorch / CUDA | 2.5.1+cu124 / 12.4 |
| Topology | world size 4, GA16, one optimizer update |
| Training horizon / stop | 1106 / 554 |
| Kernels | FA2 ON, Liger ON |
| Determinism | strict algorithms, warn_only=False, CUBLAS_WORKSPACE_CONFIG=:4096:8 |

The exact step554 input values were established by DET-A/DET-B and frozen as
frozen_step554_batch_contract.json. GRAD-A2/B2 use the same checkpoint, cache,
sampler/RNG contract and validate the same ordered runtime field metadata and
count. This avoids the previous eight full CUDA-to-CPU tensor copies on every
microbatch. It adds no model forward and calls no random API.

## Per-rank evidence

All ranks have one float32 bucket with 87,293,952 elements, 504 parameters,
and layout SHA c9b95dab8bd5cd73d17c74fccde68fc34bbdcd2d7c868446d5959592d6d9a93b.

| Rank | Local loss A2 / B2 | PRE_ALLREDUCE A2 / B2 | Equal | POST_ALLREDUCE A2 / B2 | Equal |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.0197828598320484 / same | 54455989be71... / f6aa9bfc9ebf... | No | ed2768ea18c2... / 6bc8a97d8ccb... | No |
| 1 | 1.59639323502779 / same | 6b04db242ccf... / 894268077dc9... | No | ed2768ea18c2... / 6bc8a97d8ccb... | No |
| 2 | 2.560986466705799 / same | c4cf36bb265e... / a161c9ea72fe... | No | ed2768ea18c2... / 6bc8a97d8ccb... | No |
| 3 | 1.3222751496359706 / same | c7711620f4e3... / eb4fbdf56cf1... | No | ed2768ea18c2... / 6bc8a97d8ccb... | No |

The pre-allreduce fingerprints differ across ranks as expected for rank-local
data, but the decisive comparison is each rank A2 versus the same rank B2.
Every one of those repeat comparisons differs before DDP reduction.

## Clip and update

| Evidence | GRAD-A2 | GRAD-B2 | Equal |
| --- | --- | --- | --- |
| PRE_CLIP LoRA gradient | a5ab818aca1ad1a0779a78e412c6899c03b9825d2eadfee8014cfb8876c969fb | c7cf24c75985199b94f6552b50453144557b06cc1584e0071ea4caa69c5632c8 | No |
| POST_CLIP LoRA gradient | same as A2 PRE_CLIP | same as B2 PRE_CLIP | No |
| Returned grad norm | 0.7422494292259216 | 0.7421417236328125 | No |
| Step554 LoRA | ec09693231c2... | 9301b6a924e3... | No |
| Step554 effective B@A | a14db24d6591... | 2fc58c039e73... | No |
| Step554 optimizer | d973a8d4846a... | c5df2a5b0480... | No |

Within each run PRE_CLIP equals POST_CLIP exactly, so clipping did not change
the recorded LoRA gradients. The clip and update differences inherit the
already-observed local-backward divergence.

## DDP hook semantics

The hook delegates to PyTorch 2.5.1's official default_hooks.allreduce_hook.
Its installed implementation divides the bucket by process-group size before
asynchronous all_reduce, matching the default averaged-gradient semantics.
The source fingerprint is
af26f3355821ec45eb8976b8278d67f91c1e0edcdbf15b2692835c216df12444.
Bucket pairing uses ordered call index plus parameter layout SHA, parameter
count/offset metadata, dtype, shape, and size rather than bucket index alone.

## FA2 static audit

Installed FlashAttention 2.7.4.post1 exposes deterministic=False on both fixed
and varlen APIs. Installed Transformers 5.3.0 accepts a deterministic argument
and can populate it from FLASH_ATTENTION_DETERMINISTIC=1. Thus the current call
chain can enable FA2's deterministic backward path. This round did not enable
it, as required; the evidence does not distinguish FA2 backward from Liger or
another local backward kernel.

## Run safety

The first GRAD-A attempt was rejected by PyTorch's hook annotation validator
before any forward/backward or optimizer update. It remains preserved. The
signature-only fix was committed, and fresh GRAD-A2/GRAD-B2 directories were
used. No FA2-off or Liger-off run, long training, downstream stage, model
evaluation, or historical checkpoint modification occurred.
