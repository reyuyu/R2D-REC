# GR_USER_v1 Trainer Objective Contract

## Frozen objective

- Routes: `action` and `chain`, processed in route-homogeneous batches.
- Group size: `G=4`; each contiguous group must contain one `sample_id`.
- Generation: temperature `0.9`, top-p `0.95`, maximum 512 new tokens.
- Action reward: Set-F1.
- Chain reward: `0.5 * ActionAlignmentF1 + 0.5 * LogicAlignmentF1`.
- Group advantage: population standard deviation (`correction=0`) and epsilon `1e-4`.
- GRPO: clipped objective, epsilon `0.2`, beta `0`, `loss_type=grpo`.
- Local penalty: sqrt-normalized with base lambda `0.50`.

For candidate `i` in group `g`:

```text
A_i = (R_i - mean(R_g)) / (population_std(R_g) + 1e-4)
```

If the group standard deviation is zero, `A_i=0`. For an included violation
span containing `n` generated tokens:

```text
lambda_eff = 0.50 / sqrt(n)
A_i,t = min(A_i, -lambda_eff)
```

Unmasked tokens retain `A_i,t=A_i`. Overlapping violations use the strongest
effective penalty and are never summed. The whitelists remain owned by the
Phase 3A contract; `wrong_selection_sid` and format/schema violations do not
receive a local penalty.

## Data and token contract

The trainer calls the existing Action/Chain scorer and Phase 3A penalty-mask
compiler. It does not duplicate parser, reward, or constraint logic.

The compiler works on canonical text tokens, while the policy loss consumes the
model's original `completion_ids`. The shared generated-token projector checks
that both token sequences decode to the scored completion and then projects
records and per-kind masks onto the original IDs. Insertions, deletions, empty
included spans, and partial intersections with replacement blocks fail closed.
The final hard contract is:

```text
token_advantages.shape == penalty_mask.shape == completion_ids.shape == [B, T]
```

## Loss and monitoring

The standard clipped GRPO coefficients are unchanged. Only the broadcast
sequence advantage is replaced by the `[B,T]` token-advantage tensor. Loss is
averaged over valid completion tokens per candidate, then over candidates.

Monitoring keeps task and constraint signals separate. It reports reward and
zero-std statistics, route reward components and violation counts, sequence and
token advantage statistics, masked candidate/token rates, per-kind masked token
counts and incremental negative mass, positive-sequence token flips, policy
ratio, clipping fraction, loss, and finiteness.

## Correctness evidence

CPU regression: 100 tests passed across Phase 1, Phase 2, Phase 3A, Phase 4A,
rollout projection, the token-local objective, and the User trainer.

The no-penalty parity audit used 32 randomized float64 trials:

| Check | Maximum error |
| --- | ---: |
| Loss | 0.0 |
| Per-token loss | 0.0 |
| Log-prob gradient | 0.0 |

Synthetic tests also verify positive sequence/local negative gradient direction,
preservation of stronger negative sequence advantages, zero-std local signal,
sqrt values for spans of 1/4/10/100 tokens, strongest-only overlap,
`wrong_selection_sid` exclusion, route dispatch, and group isolation.

## GPU correctness smoke

Final run: `GR-USER-TRAINER-SMOKE-20260819-193832`, physical GPU 0.

- Four unique prompts: two Action and two Chain.
- `G=4`, 16 generated candidates total.
- Forward-only losses: Action `-9.536743e-07`; Chain `0.005949602`.
- Combined task reward population std: `0.1751844761`.
- Combined masked-token fraction: `0.01879825445`.
- Forward-only peak allocated VRAM: `36094.67 MiB`.
- Action local masks: none; its only observed constraint was non-whitelisted
  `wrong_selection_sid`.
- Chain local masks: 30 `date_mismatch` tokens and 26 `action_mismatch` tokens.

The optimizer smoke used non-reentrant activation checkpointing, learning rate
`1e-6`, and exactly two steps (Action then Chain). Peak allocated VRAM was
`26175.27 MiB`.

| Check | Result |
| --- | ---: |
| Maximum LoRA grad norm | 1.169086627 |
| Maximum clip fraction | 0.01629072614 |
| LoRA L2 delta | 0.01430002980 |
| LoRA maximum absolute delta | 2.002343535e-06 |
| Base tensor delta | 0.0 |
| NaN/Inf | false |

All 399 hashed base tensors had identical SHA-256 before and after. The 504
hashed LoRA tensors changed. In the OFF/ON local-gradient contrast, unmasked
token advantages and gradients both had maximum error `0.0`; 10 masked tokens
changed gradient, with maximum absolute change `0.0009143238328`.

An initial BF16 gradient-comparison run stopped before optimization; the strict
mathematical comparison was moved to float64. A subsequent run stopped on
single-GPU activation OOM before backward. Activation checkpointing resolved it
without changing G, generation length, objective, or optimizer batch semantics.

The final smoke did not run the 600-sample Pilot, full training, checkpoint
saving, or external evaluation. Frozen GR_USER data hashes were unchanged and
GR_REC_v1 was not modified.
