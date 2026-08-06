# OneReason Dataset Versions

## Policy

- `/data/lf_data/onereason_*.jsonl` is the immutable, full-training raw source. It is the canonical **all-train** source; no validation split is consumed by new experiments.
- Historical `/data/lf_data_splits/*_train98.jsonl` and the old versions derived from them are retained only so past experiments remain reproducible. They are not inputs to new dataset versions.
- Managed versions live under `/data/lf_data_versions/alltrain/<version>/`. A version stores only the subdatasets it changes and declares a parent version for all untouched subdatasets.
- The machine-readable version graph is `data/onereason_dataset_versions.json`; it records file path, count, SHA-256 and the registered dataset alias for each override.
- Dataset files are immutable after registration. A correction creates a new version; it never overwrites V1/V2 or the raw source.

## Version Graph

```text
raw_all
  -> v1_thought_prompt_all
       -> v2_recommendation_cot_complete_all
            -> v3_material_clean        # example: only material cot/nocot overrides
```

`v3_material_clean` therefore resolves material from V3, recommendation from V2, and user/world from V1. This is the key property that prevents copying all eight JSONL files for a one-subdataset cleaning run.

## Create the Initial Full-Training Versions

```bash
cd /app/LLaMA-Factory

python scripts/manage_onereason_datasets.py init-raw-all
python scripts/manage_onereason_datasets.py create-thought-prompts \
  --version v1_thought_prompt_all --parent raw_all
python scripts/manage_onereason_datasets.py create-recommendation-cot-complete \
  --version v2_recommendation_cot_complete_all --parent v1_thought_prompt_all
python scripts/manage_onereason_datasets.py audit
```

The thought-prompt build appends `/think` to CoT prompts and `/no_think` to no-CoT prompts only when absent. The recommendation build retains samples whose `output` contains all of `【兴趣归纳】`, `【行为模式】`, and `【预测总结】`.

## Clean Only One Subdataset

Write the cleaned JSONL outside the managed tree first, inspect it, then register it as a patch:

```bash
python scripts/manage_onereason_datasets.py register-version \
  --version v3_material_clean \
  --parent v2_recommendation_cot_complete_all \
  --description "Remove material records failing the V3 audit." \
  --patch onereason_material_cot=/data/clean/onereason_material_cot.jsonl \
  --patch onereason_material_nocot=/data/clean/onereason_material_nocot.jsonl
```

The command validates JSONL, copies only those patch files into the immutable managed directory, records counts/digests, and registers only the changed aliases in `dataset_info.json`.

## Training Selection

New all-training configurations use the eight unsuffixed logical names and no split suffix:

```yaml
dataset: onereason_material_cot,onereason_material_nocot,onereason_user_action_nocot,onereason_user_chain_cot,onereason_user_chain_nocot,onereason_recommendation_cot,onereason_world_cot,onereason_world_nocot
multitask_train_dataset_suffix: ""
multitask_dataset_version: v2_recommendation_cot_complete_all
multitask_dataset_version_overrides: {}
multitask_dataset_version_manifest: data/onereason_dataset_versions.json
```

The Trainer resolves each logical dataset through the parent chain before loading it. For an ablation that deliberately mixes versions, `multitask_dataset_version_overrides` still has precedence for the named logical dataset.

Legacy configs that use `_train98` plus `raw`, `v1_thought_prompt`, or `v2_recommendation_cot_complete` retain their existing string-based resolution and are not silently redirected.
