# Dataset Layout

`dataset/` is the active, immutable training snapshot for this baseline. Its content is identified by `manifest.json`, including source counts and SHA256. Do not overwrite this directory during a later data experiment; create another version directory and change the training YAML explicitly.

## Active Version

`native_source_domain_r32_v3` has 219,370 records and a uniform JSONL schema:

```text
instruction, input, output, history, data_source, source_segment, aux_metadata_json
```

`data_source` classifies the loss route: `sid_bucket_canonical_no_think`, `sid_bucket_reverse`, `material_sample`, `understand_user`, or `recommend`. `source_segment` keeps the original fine-grained route. Both are training-side metadata and are not forwarded to the model.

For Recommendation V3 records, `aux_metadata_json` preserves `group_id`, `group_size`, `all_gold_sids`, and `current_gold_sid`. The present baseline does not use it in its loss. A future multi-positive loss can consume this field to construct token-aligned targets without re-parsing prompts or changing the dataset schema.

## Version Policy

`versions.json` is the registry. `dataset_schema_v1_no_aux_20260810/` and `dataset_pre_schema_20260810/` are archived provenance snapshots, not fallback defaults. Every new dataset version must have a directory, `manifest.json`, stable name, parent version list, source counts, and an explicit YAML reference.
