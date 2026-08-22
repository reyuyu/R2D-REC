"""Pure preflight condition evaluation; importing this module never touches CUDA."""
from __future__ import annotations

from collections import Counter
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


def prepare_preflight_loss_context(trainer: Any) -> int:
    """Mirror the frozen Trainer loop state needed by a direct loss-only audit."""
    value = getattr(getattr(trainer, "args", None), "gradient_accumulation_steps", None)
    if isinstance(value, bool):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient_accumulation_steps must be a positive integer") from exc
    if steps <= 0 or value != steps:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if steps != 1:
        raise AssertionError(
            f"frozen Composite gradient_accumulation_steps must be 1, got {steps}"
        )
    trainer.current_gradient_accumulation_steps = steps
    return steps


def masked_token_tuples(token_ids: Any, token_mask: Any) -> list[tuple[int, ...]]:
    """Return each sequence with mask-zero padding removed."""
    ids_rows = token_ids.detach().cpu().tolist() if hasattr(token_ids, "detach") else token_ids
    mask_rows = token_mask.detach().cpu().tolist() if hasattr(token_mask, "detach") else token_mask
    if len(ids_rows) != len(mask_rows):
        raise ValueError("token ids and mask batch sizes differ")
    output = []
    for ids, mask in zip(ids_rows, mask_rows):
        if len(ids) != len(mask):
            raise ValueError("token ids and mask sequence lengths differ")
        output.append(tuple(int(token) for token, keep in zip(ids, mask) if bool(keep)))
    return output


def raw_decode_diagnostics(
    tokenizer: Any,
    completion_ids: Any,
    completion_mask: Any,
    runtime_candidates: list[dict[str, Any]],
    rank: int,
) -> dict[str, Any]:
    """Compare rank-local reward text with masked, unpadded completion ids."""
    token_rows = masked_token_tuples(completion_ids, completion_mask)
    candidates = sorted(
        (row for row in runtime_candidates if int(row["rank"]) == int(rank)),
        key=lambda row: int(row["local_index"]),
    )
    if len(candidates) != len(token_rows):
        raise ValueError("rank-local runtime candidates do not match completion batch")
    mismatches = []
    for expected_index, (candidate, token_row) in enumerate(zip(candidates, token_rows)):
        if int(candidate["local_index"]) != expected_index:
            raise ValueError("rank-local runtime candidate indices are not contiguous")
        decoded = tokenizer.decode(list(token_row), skip_special_tokens=False)
        if decoded != candidate["completion"]:
            mismatches.append({
                "local_index": expected_index,
                "runtime_completion": candidate["completion"],
                "masked_decode": decoded,
            })
    return {
        "candidate_count": len(candidates),
        "mismatch_count": len(mismatches),
        "pass": not mismatches,
        "mismatches": mismatches,
    }


def float_vectors_close(left: list[float], right: list[float], tolerance: float = 2e-5) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) < tolerance for a, b in zip(left, right)
    )


def sequence_advantage_signatures(batch: dict[str, Any]) -> Counter:
    """Bind every advantage to its unpadded prompt and completion token ids."""
    prompts = masked_token_tuples(batch["prompt_ids"], batch["prompt_mask"])
    completions = masked_token_tuples(batch["completion_ids"], batch["completion_mask"])
    advantages = batch["advantages"]
    if hasattr(advantages, "detach"):
        advantages = advantages.detach().float().cpu().tolist()
    if not (len(prompts) == len(completions) == len(advantages)):
        raise ValueError("sequence/advantage batch sizes differ")
    return Counter(
        (prompt, completion, float(advantage))
        for prompt, completion, advantage in zip(prompts, completions, advantages)
    )


def post_shuffle_association_parity(
    pre_shuffle: dict[str, Any], post_shuffle: dict[str, Any]
) -> bool:
    """Accept order changes only when sequence-to-advantage associations survive."""
    return sequence_advantage_signatures(pre_shuffle) == sequence_advantage_signatures(post_shuffle)


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
