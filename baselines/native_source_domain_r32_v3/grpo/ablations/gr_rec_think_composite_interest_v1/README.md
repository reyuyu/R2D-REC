# GR_REC_Think_CompositeInterest_v1

This directory contains CPU-only shared-parser extensions, deterministic CompositeInterest reward primitives, exact Gold provenance and eligible-cohort audit, M1-M5 real positive/negative lexical calibration, a fail-closed read-only data adapter, offline synthetic audit, and tests.

Current audited state:
- eligible Think groups: 1446 of 1549
- selected text metric: character bigram multiset F1
- matching threshold: .30
- match-quality floor: .60
- DATA_PROVENANCE_READY: YES under eligible-only policy
- LEXICAL_METRIC_READY: YES

Formal runner construction is intentionally absent. No code in this directory starts generation, loads model weights, touches CUDA, or performs optimizer or scheduler steps.
