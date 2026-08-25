# TrueRec-GRPO

TrueRec-GRPO is a recommendation-only GRPO data and training track. Phase 0.1 is intentionally limited to CPU-only source provenance and canonical recommendation-group construction.

## Phase 0.1 contract

- Canonical source: the complete BATA baseline JSONL.
- Canonical key: `recommendation_group_id` from `aux_metadata_json`.
- Recommendation rows: `data_source == "recommend"`.
- Routes: `recommendation_cot` and `recommendation_nocot` from `source_segment`.
- Target domain: parsed from `recommendation_current_gold_sid`; the source has no separate target-domain field.
- History: extracted from `instruction + input`, excluding the terminal task paragraph and `/think` or `/no_think` marker.
- Beta-Gamma is audited against its run-local dataset manifest and complete transformed JSONL. It must preserve Beta's row count, recommendation groups, and Gold contracts while removing only the recommendation answer bridge.

Phase 0.1 does not create splits, novelty classes, rendered GRPO data, rewards, or training jobs.
