# GR_REC_Think_CompositeInterest_v1

This directory contains the audited metric and the production-shaped Think-only
training integration. The trainer inherits the stable `RecGRPOTrainer`, reuses
its generation and Beam32 path, and replaces only the gathered G4 advantages.

Current audited state:
- eligible Think groups: 1446 of 1549
- fixed probes excluded: 4
- training groups after G4 tail drop: 1440
- fresh G4 rollouts / optimizer steps at two iterations: 360 / 720
- selected text metric: character bigram multiset F1
- matching threshold: .30
- match-quality floor: .60
- DATA_PROVENANCE_READY: YES under eligible-only policy
- LEXICAL_METRIC_READY: YES
- CPU_DRY_RUN_PASS: YES

`run_gr_rec_think_composite_interest_v1.py --dry-run` validates actual data,
paths, cohort, probes, sampler topology, frozen hyperparameters and checkpoint
milestones without loading a model. GPU training remains separately authorized.
