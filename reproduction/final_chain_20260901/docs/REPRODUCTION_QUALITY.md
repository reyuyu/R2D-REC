# Four-stage reproduction quality contract

## Purpose

The monitor answers four different questions without collapsing them into one
green/red label:

1. **Contract parity**: are data bytes, code, environment, parent handoff,
   optimizer settings, step counts, and checkpoint structure valid?
2. **Training-trajectory parity**: at the same retained milestone, are rolling
   metric means inside a deliberately broad historical band?
3. **Parameter parity**: is the selected adapter bitwise identical, and if not,
   what are its CPU-computed relative L2, cosine, and maximum absolute delta?
4. **Effect parity**: does a separately recorded fixed external evaluation match
   the selected historical score?

Only item 1 is a hard reproduction gate. Items 2-4 are evidence requiring the
correct interpretation; none is silently converted into a contract result.

The training chart uses two actual step-indexed series: the immutable original
run snapshot and the live reproduction log. Both series are aggregated by step
before display, so duplicate distributed-rank records cannot visually weight a
step more heavily. The existing milestone window bands remain the decision
surface for trajectory review.

Checkpoint cosine, relative L2, and max-absolute delta compare the historical
and reproduced LoRA adapter at the same stage and same step. They are not
full-model metrics and are not comparable across different LoRA structures.

## Stage references

| Stage | Selected checkpoint | Retained comparison points | Historical adapter SHA prefix | Historical score |
| --- | ---: | --- | --- | ---: |
| BETA SFT | 1106 | 553, 1106 | `4c077d9b0865b883` | 1.3313 primary; 1.3246/1.3313 recorded |
| GR_REC V1 | 1500 | 500, 1000, 1500 | `a5e92db011662799` | 1.3510 |
| GRPO-TK | 250 | 50, 100, 150, 200, 250 | `64e1a85500b68f48` | 1.3579 |
| MC_USER Hybrid K4 | 100 | 25, 50, 75, 100 | `cca496b3ec4139fe` | 1.3617 repeat mean; 1.356 project headline |

The dashboard uses the exact metric schemas emitted by each trainer. Absolute
loss values are not compared across stages.

## Status semantics

- `contract pass`: the stage PASS marker exists and the selected adapter-only
  checkpoint has all required state files.
- `contract fail`: a STOPPED marker or invalid checkpoint structure exists.
- `within reference band`: all available milestone window means are inside the
  historical min/max range expanded by a documented numerical margin.
- `review`: at least one available window mean is outside that band. This is not
  proof of a broken run because on-policy samples can diverge.
- `bitwise match`: reproduced and historical adapter SHA256 are identical.
- `SHA different`: expected under nondeterministic GPU trajectories; run
  `compare_adapters.py` before making a parameter-level claim.
- `external pending`: no fixed evaluation JSON was supplied. Online reward must
  not be used as a substitute.

## Read-only guarantee

`repro_monitor_server.py` uses only the Python standard library. API refreshes
read metrics, state markers, small JSON summaries, and checkpoint directory
metadata. Adapter SHA is calculated only after the selected adapter exists;
large tensor numerical comparison is never executed by the web server.

The frontend runs on port 8891 and is separate from the experiment monitor on
8878. It does not write files or expose deletion/training controls.

## External evaluation records

Store only small records in the isolated bundle:

```text
evaluations/01_sft_beta.json
evaluations/02_gr_rec_v1.json
evaluations/03_grpo_tk.json
evaluations/04_mc_user.json
```

Each needs an `aggregate`, while evaluator identity, repeat number, and the 11
component metrics should be included when available. These files can be synced
to Git because they contain no data samples or model weights.

## Security and publication boundary

Safe to publish: scripts, configs, exact hashes, row counts, metric summaries,
checkpoint contracts, environment versions, and review documentation.

Do not publish: dataset rows, prompts, rollouts, raw logs containing samples,
base or adapter weights, optimizer states, credentials, host connection details,
or private absolute paths beyond historical audit labels already documented.
