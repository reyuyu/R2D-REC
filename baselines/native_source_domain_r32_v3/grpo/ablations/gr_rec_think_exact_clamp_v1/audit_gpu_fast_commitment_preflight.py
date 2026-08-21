"""Minimal zero-update GPU preflight for formal Domain commitment credit."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from audit_gpu_gradient_scale import (
    ADAPTER,
    BASE,
    DATA,
    M_NO,
    build_route_dataset,
    capture_rng_state,
    clipped_policy_loss,
    completion_logps,
    hierarchical_token_advantages,
    isolated_gradient,
    legacy_advantages,
    legacy_text_domain_advantages,
    load_model,
    make_rollout_batch,
    pairing_status,
    trainable_parameter_checksum,
    trainable_parameters,
)


DOMAIN_ONLY_INDICES = (0, 1, 3, 4, 5)


def stable_vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    dot = left_norm = right_norm = 0.0
    for start in range(0, left.numel(), 1_000_000):
        lpart = left[start:start + 1_000_000].double()
        rpart = right[start:start + 1_000_000].double()
        dot += float(torch.dot(lpart, rpart))
        left_norm += float(torch.dot(lpart, lpart))
        right_norm += float(torch.dot(rpart, rpart))
    denominator = math.sqrt(left_norm * right_norm)
    if denominator == 0.0:
        return None
    return max(-1.0, min(1.0, dot / denominator))


def abc_parity(reference_group: dict, credits: list[tuple]) -> bool:
    current = [list(row[1:]) for row in credits]
    return current == reference_group["abc_credits_reconstructed_from_immutable_rollout"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Formal Domain commitment fast preflight")
    parser.add_argument("--execute-zero-step-gpu-preflight", action="store_true")
    parser.add_argument("--cpu-regression-passed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--paired-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.execute_zero_step_gpu_preflight:
        parser.error("explicit zero-step GPU preflight authorization flag is required")
    if not args.cpu_regression_passed:
        parser.error("CPU regression must pass before GPU preflight")
    if args.groups != 8 or args.seed != 20260816:
        parser.error("fast paired preflight is fixed to groups=8 seed=20260816")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("preflight requires an explicitly authorized CUDA device")
    reference = json.loads(Path(args.paired_reference).read_text(encoding="utf-8"))
    if len(reference.get("groups", [])) != 8:
        raise RuntimeError("paired reference must contain exactly 8 groups")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, _template = load_model(args.device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    parameters = trainable_parameters(model)
    if any("lora" not in name.lower() for name, _parameter in parameters):
        raise RuntimeError("non-LoRA trainable parameters found")
    checksum_before = trainable_parameter_checksum(parameters)
    dataset = build_route_dataset(DATA, n_groups=8, seed=args.seed, chunk=8)
    records = [record for record in dataset if record["route"] == "no_think"]
    if len(records) != 8:
        raise RuntimeError("NoThink paired dataset did not contain 8 groups")

    parity = []
    results = []
    for index, record in enumerate(records):
        batch = make_rollout_batch(
            record, model, tokenizer, args.device, args.max_new_tokens
        )
        pair = pairing_status(index, batch, reference["groups"][index])
        parity.append(pair)
        if not pair["valid"]:
            model.zero_grad(set_to_none=True)
            invalid = {
                "preflight": "FAST_PREFLIGHT_REVIEW",
                "flags": ["PAIRED_AUDIT_INVALID"],
                "parity": parity,
                "failed_audit_index": index,
                "checksum_before": checksum_before,
                "checksum_after": trainable_parameter_checksum(parameters),
            }
            Path(args.output).write_text(
                json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("PAIRED_AUDIT_INVALID")

        shared_rng_state = capture_rng_state(batch.input_ids.device)
        with torch.no_grad():
            old_logps = completion_logps(model, batch).detach()
        token_advantages, credits, diagnostics = hierarchical_token_advantages(
            batch, tokenizer
        )
        columns = list(zip(*credits))
        row = {
            "audit_index": index,
            "group_id": batch.group_id,
            "rollout_fingerprint": batch.fingerprint,
            "rewards": list(batch.rewards),
            "eligible_count": diagnostics["eligible_count"],
            "branch_count": diagnostics["branch_count"],
            "direct_sid_fallback_count": diagnostics["direct_sid_fallback_count"],
            "unresolved_count": diagnostics["unresolved_count"],
            "hier_grad_norm": None,
            "hier_over_legacy": None,
            "matched_support_cosine": None,
            "abc_stage_active": {
                "a": any(value != 0 for value in columns[1]),
                "b": any(value != 0 for value in columns[2]),
                "c": any(value != 0 for value in columns[3]),
            },
            "abc_credited_token_count": {
                "a": sum(value != 0 for value in columns[1]),
                "b": sum(value != 0 for value in columns[2]),
                "c": sum(value != 0 for value in columns[3]),
            },
            "abc_credit_parity": abc_parity(reference["groups"][index], credits),
        }
        if index in DOMAIN_ONLY_INDICES:
            legacy_scalar = legacy_advantages(batch.rewards, batch.input_ids.device)
            legacy_tokens = legacy_scalar.unsqueeze(1).expand_as(batch.completion_mask)
            matched_tokens = legacy_text_domain_advantages(
                batch, legacy_scalar, diagnostics
            )
            legacy_norm, legacy_vector = isolated_gradient(
                model,
                parameters,
                lambda: clipped_policy_loss(
                    model, batch, old_logps, legacy_tokens, "legacy_mean"
                ),
                shared_rng_state,
                batch.input_ids.device,
            )
            hier_norm, hier_vector = isolated_gradient(
                model,
                parameters,
                lambda: clipped_policy_loss(
                    model, batch, old_logps, token_advantages, "hierarchical_sum"
                ),
                shared_rng_state,
                batch.input_ids.device,
            )
            _matched_norm, matched_vector = isolated_gradient(
                model,
                parameters,
                lambda: clipped_policy_loss(
                    model, batch, old_logps, matched_tokens, "hierarchical_sum"
                ),
                shared_rng_state,
                batch.input_ids.device,
            )
            row["hier_grad_norm"] = hier_norm
            row["hier_over_legacy"] = hier_norm / legacy_norm if legacy_norm > 0 else None
            row["matched_support_cosine"] = stable_vector_cosine(
                matched_vector, hier_vector
            )
            del legacy_vector, hier_vector, matched_vector
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    model.zero_grad(set_to_none=True)
    checksum_after = trainable_parameter_checksum(parameters)
    parameter_change = checksum_before != checksum_after
    by_index = {row["audit_index"]: row for row in results}
    g0 = by_index[0]
    strong_groups = [by_index[index] for index in (1, 3, 4, 5)]
    abc_67_parity = all(by_index[index]["abc_credit_parity"] for index in (6, 7))
    matched = [by_index[index]["matched_support_cosine"] for index in DOMAIN_ONLY_INDICES]
    conditions = {
        "fingerprint_parity_8_of_8": sum(p["rollout_fingerprint"] for p in parity) == 8,
        "reward_parity_8_of_8": sum(p["rewards"] for p in parity) == 8,
        "parameter_checksum_unchanged": not parameter_change,
        "g0_candidate_fallback_active": (
            g0["eligible_count"] == 8
            and g0["branch_count"] == 7
            and g0["direct_sid_fallback_count"] == 1
            and g0["unresolved_count"] == 0
            and g0["hier_grad_norm"] is not None
            and g0["hier_grad_norm"] > 0
        ),
        "aligned_domain_groups_nonzero_above_saturation_floor": all(
            row["hier_grad_norm"] is not None and row["hier_grad_norm"] > 1e-4
            for row in strong_groups
        ),
        "matched_support_cosine_near_one": all(
            value is not None and value > 0.99 for value in matched
        ),
        "groups_6_7_abc_parity": abc_67_parity,
        "cpu_regression_passed": True,
    }
    preflight = "FAST_PREFLIGHT_PASS" if all(conditions.values()) else "FAST_PREFLIGHT_REVIEW"
    payload = {
        "type": "gr_rec_think_exact_clamp_v1_fast_commitment_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "adapter": ADAPTER,
        "groups": 8,
        "seed": args.seed,
        "zero_update": True,
        "trainable_parameter_checksum_before": checksum_before,
        "trainable_parameter_checksum_after": checksum_after,
        "parameter_change": parameter_change,
        "fingerprint_parity_count": sum(p["rollout_fingerprint"] for p in parity),
        "reward_parity_count": sum(p["rewards"] for p in parity),
        "conditions": conditions,
        "preflight": preflight,
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if parameter_change:
        raise RuntimeError("trainable parameters changed during zero-step preflight")
    return payload


if __name__ == "__main__":
    main()
