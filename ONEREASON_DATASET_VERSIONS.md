# OneReason Dataset Versions

## Layout

- Raw sources remain immutable in /data/lf_data_splits/onereason_*_train98.jsonl.
- Each managed version lives in /data/lf_data_versions/<version>/.
- Every version must contain MANIFEST.json with input/output digests, counts and its transformation.
- Dataset registry names follow onereason_<subtask>_<version>_train98. The legacy raw name remains onereason_<subtask>_train98.

## v1_thought_prompt

v1_thought_prompt preserves every field except input. It appends a newline plus /think to all cot prompts and a newline plus /no_think to all nocot prompts when that exact suffix is absent. The world datasets already carried the matching marker, so their records are semantically unchanged.

Creation is reproducible with:

    cd /app/LLaMA-Factory
    python scripts/create_onereason_v1_thought_prompt.py

The command intentionally refuses to overwrite an existing version or registry entry.

## Macro Training Selection

Keep the normal eight raw logical names in dataset. Select a full version with:

    multitask_train_dataset_suffix: _train98
    multitask_dataset_version: v1_thought_prompt
    multitask_dataset_version_overrides: {}

For a single-subtask ablation, choose a base version then override only one logical dataset:

    multitask_dataset_version: v1_thought_prompt
    multitask_dataset_version_overrides:
      onereason_user_action_nocot: raw

This resolves that subtask to its raw registry name and resolves the other seven to their v1 registry names. Task IDs, subtask ratios, balanced_40 and sampler order remain unchanged.
