# GR_REC_Think_CompositeInterest_v1

This directory contains CPU-only shared parser extensions, deterministic CompositeInterest reward primitives, a strict provenance audit, a fail-closed read-only data adapter, offline metric audit, and tests.

Formal runner construction is intentionally absent. Gold CoT provenance matched 1458 of 1549 Think groups, so DATA_PROVENANCE_READY is NO. The data adapter raises DataProvenanceError unless complete provenance is explicitly supplied.

No code in this directory starts generation, loads a model, touches CUDA, or performs optimizer or scheduler steps.
