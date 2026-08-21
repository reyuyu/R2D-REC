# GR_REC_NoThinkOnly_Frontier_v1 Formal E1 Forensic

## Completion and contract

- Run: GR-REC-NOTHINK-ONLY-FRONTIER-G8BASE-E1-20260822
- Launch commit: 3b0e434f8a603fac882a1e7485fd14d5dee2c551
- Parent: Fresh original BATA adapter
- Completed: 1,544/1,544 optimizer steps, 772 stochastic rollouts
- Training population: 1,544 real NoThink G8 groups, 12,352 candidates
- Runtime: 7,014.32 seconds end to end; trainer reported 6,887 seconds
- Checkpoints: 250, 500, 666, 750, 1000, 1250, 1544
- Think optimizer updates: zero; Think was fixed-probe inference only
- External benchmark: not run

The run completed without OOM, non-finite metrics, runtime exception, or
checkpoint failure. The final adapter differs from Fresh BATA (LoRA L2
0.98008, maximum absolute delta 0.0014143). The checkpoint contains only
LoRA adapter tensors; the frozen base delta is zero by the optimizer and
checkpoint contract.

## Optimization metrics

| Window | Loss mean | Grad norm mean | Approx KL mean | Clip fraction mean | Reward mean | Zero-std ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-250 | 0.04995 | 0.59051 | 0.000325 | 0.002377 | 0.15888 | 0.148 |
| 647-896 | 0.07863 | 0.63595 | 0.000239 | 0.001508 | 0.25488 | 0.340 |
| 1295-1544 | 0.08120 | 0.48629 | 0.000309 | 0.001613 | 0.15425 | 0.348 |
| Full epoch | 0.07141 | 0.57673 | 0.000269 | 0.001858 | 0.20829 | 0.271 |

The optimizer remained active but conservative: KL and clipping stayed small,
gradients stayed finite, and there is no numerical-instability signature. The
late reward mean fell from the middle window while the zero-std fraction stayed
high. Because each step uses a different on-policy training group, this is not
a matched evaluation curve; it is evidence of noisier and less diverse late
training signal, not by itself proof of benchmark regression or improvement.

## Frontier and format evidence

Observed scalar reward counts were -1: 53, -0.25: 507, 0: 9,137,
0.5: 2,221, 2: 305, and 8: 129. Strict-format violations were 53/12,352
(0.4291%): 44 unexpected prose cases and 9 invalid SIDs.

Frontier candidate exposure was:

| Stage | Count | Candidate rate |
| --- | ---: | ---: |
| Domain | 507 | 4.1046% |
| A | 9,137 | 73.9718% |
| B | 2,221 | 17.9809% |
| C | 305 | 2.4692% |

The Gold-A bridge activated in 415/1,544 groups (26.8782%). C-frontier
rollouts had a larger rollout maximum gradient (mean 1.1555, median 1.0220,
maximum 3.1950, n=197) than rollouts without C-frontier (mean 0.4030,
median 0.3574, maximum 1.5160, n=575). This confirms C is the strongest
Frontier event in practice. It remained finite, so the formal run does not show
a hard gradient failure.

## Fixed-probe evidence

The four fixed groups are longitudinal diagnostics, not a statistically
representative external benchmark. NoThink probe reward was noisy and
non-monotonic: 0.0000 at step 0, 0.3594 at 200, 0.4375 at 600, 0.0781
at 1000, and 0.1719 at 1544. This does not support selecting the final
checkpoint from probe reward alone.

Think was never optimized, but it shares the same LoRA adapter. Its fixed-probe
mean reward declined from 3.7305 at step 0 to 0.7891 at step 1544; mean
exact count declined from 1.75 to 0.25, and mean invalid Beam32 count rose
from 0 to 11. This is material cross-route interference evidence. It does
not invalidate the NoThink-only ablation, but it means the resulting adapter is
not route-neutral and should not be described as preserving Think quality.

## Initial conclusion

Engineering acceptance passes: the frozen Frontier formula ran for a complete
epoch, all required artifacts exist, LoRA changed, base stayed frozen, and the
monitor streams are complete. Algorithmic efficacy is unresolved. Frontier
credit was exercised at scale and remained numerically bounded, but the late
training signal did not improve monotonically, the small NoThink probe was
noisy, and the untouched Think route degraded strongly on its fixed probe.

Checkpoint choice and a performance claim require the separately authorized
held-out external benchmark. No such benchmark was started in this phase.

Structured evidence:
gr_rec_nothink_frontier_v1_formal_e1_forensic_20260822.json
