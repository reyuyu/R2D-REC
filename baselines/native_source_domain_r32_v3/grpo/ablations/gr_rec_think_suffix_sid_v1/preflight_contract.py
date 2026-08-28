"""Pure hard-gate evaluation for the Think suffix GPU preflight."""

from __future__ import annotations


HARD_CONDITIONS = (
    "world_size_4",
    "runtime_import_provenance",
    "ddp_g8_alignment",
    "synthetic_zero_std_retry_sync",
    "real_resample_sync",
    "real_resample_contract",
    "online_advantage_parity",
    "full_completion_attention",
    "suffix_mask_subset",
    "cot_action_gradient_zero",
    "padding_action_gradient_zero",
    "suffix_action_gradient_nonzero",
    "finite_loss_and_gradient",
    "lora_has_gradient",
    "base_has_no_gradient",
    "base_requires_grad_zero",
    "lora_checksum_unchanged",
    "base_version_unchanged",
    "optimizer_step_absent",
    "scheduler_step_absent",
)


def evaluate_preflight_conditions(observations: dict) -> dict:
    conditions = {name: bool(observations.get(name)) for name in HARD_CONDITIONS}
    failures = [name for name in HARD_CONDITIONS if not conditions[name]]
    return {
        "conditions": conditions,
        "failure_reasons": failures,
        "preflight_pass": not failures,
    }
