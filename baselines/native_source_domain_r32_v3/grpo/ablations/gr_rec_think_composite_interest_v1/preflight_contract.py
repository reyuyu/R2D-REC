"""Pure preflight condition evaluation; importing this module never touches CUDA."""
from __future__ import annotations

from typing import Any


HARD_CONDITIONS = (
    "raw_decode_runtime",
    "online_reward_parity",
    "online_advantage_parity",
    "ddp_g4_alignment",
    "gold_leakage_absent",
    "finite_loss_and_grad",
    "lora_has_gradient",
    "base_has_no_gradient",
    "trainable_checksum_unchanged",
    "base_unchanged",
    "nccl_error_absent",
    "oom_absent",
    "nan_absent",
    "inf_absent",
)


def sid_runtime_observation(generated_sid_candidate_count: int) -> dict[str, Any]:
    """Describe this batch without turning SID generation into a hard gate."""
    count = int(generated_sid_candidate_count)
    if count < 0:
        raise ValueError("generated SID candidate count cannot be negative")
    return {
        "generated_sid_candidate_count": count,
        "sid_evidence_runtime_visible": count > 0,
        "sid_evidence_runtime_status": (
            "VISIBLE" if count > 0 else "NO_SID_GENERATED_IN_THIS_BATCH"
        ),
    }


def evaluate_preflight_conditions(**observations: Any) -> dict[str, Any]:
    """Evaluate only frozen hard gates; SID visibility is intentionally excluded."""
    conditions = {
        "raw_decode_runtime": bool(observations.get("raw_decode_runtime")),
        "online_reward_parity": bool(observations.get("online_reward_parity")),
        "online_advantage_parity": bool(observations.get("online_advantage_parity")),
        "ddp_g4_alignment": bool(observations.get("ddp_g4_alignment")),
        "gold_leakage_absent": not bool(observations.get("gold_leakage")),
        "finite_loss_and_grad": (
            bool(observations.get("loss_finite"))
            and bool(observations.get("grad_finite"))
        ),
        "lora_has_gradient": bool(observations.get("lora_has_gradient")),
        "base_has_no_gradient": not bool(observations.get("base_has_gradient")),
        "trainable_checksum_unchanged": bool(observations.get("checksum_unchanged")),
        "base_unchanged": not bool(observations.get("base_changed")),
        "nccl_error_absent": not bool(observations.get("nccl_error")),
        "oom_absent": not bool(observations.get("oom")),
        "nan_absent": not bool(observations.get("nan")),
        "inf_absent": not bool(observations.get("inf")),
    }
    failure_reasons = [name for name in HARD_CONDITIONS if not conditions[name]]
    return {
        "conditions": conditions,
        "preflight_pass": not failure_reasons,
        "failure_reasons": failure_reasons,
    }
