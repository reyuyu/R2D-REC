"""Zero-update diagnostic for the earliest NoThink Domain template branch."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from audit_gpu_gradient_scale import (
    RolloutBatch,
    capture_rng_state,
    clipped_policy_loss,
    completion_logps,
    isolated_gradient,
    legacy_advantages,
    make_rollout_batch,
    pairing_status,
    trainable_parameter_checksum,
    trainable_parameters,
)
from audit_gpu_gradient_scale import BASE, ADAPTER, DATA, M_NO, build_route_dataset, load_model
from nothink_hierarchical_credit import (
    conditional_hierarchical_credits,
    find_final_sid_token_positions,
    hierarchy_state,
    locate_text_domain_token,
)


DECLARATIONS = {
    "video": "该用户最近喜欢的视频有: ",
    "prod": "该用户最近点击了商品: ",
    "ad": "该用户最近感兴趣的广告有: ",
    "living": "该用户最近首次打赏了主播: ",
}


@dataclass(frozen=True)
class BranchAlignment:
    valid: bool
    declaration_domain: str | None
    sid_domain: str | None
    declaration_start: int | None
    branch_position: int | None
    branch_token_id: int | None
    failure: str | None


def token_record(tokenizer, token_id: int) -> dict:
    return {
        "token_id": int(token_id),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        "decoded": tokenizer.decode([int(token_id)]),
    }


def longest_common_prefix(sequences: list[list[int]]) -> list[int]:
    prefix = []
    for column in zip(*sequences):
        if len(set(column)) != 1:
            break
        prefix.append(int(column[0]))
    return prefix


def build_template_spec(tokenizer) -> dict:
    templates = {}
    sequences = []
    for domain, declaration in DECLARATIONS.items():
        ids = [int(value) for value in tokenizer.encode(declaration, add_special_tokens=False)]
        if not ids:
            raise RuntimeError(f"empty declaration tokenization for {domain}")
        sequences.append(ids)
        templates[domain] = {
            "declaration": declaration,
            "token_ids": ids,
            "tokens": [token_record(tokenizer, value) for value in ids],
        }
    prefix = longest_common_prefix(sequences)
    branch_index = len(prefix)
    if any(branch_index >= len(ids) for ids in sequences):
        raise RuntimeError("one declaration is a token-prefix of another")
    branch_tokens = {
        domain: token_record(tokenizer, item["token_ids"][branch_index])
        for domain, item in templates.items()
    }
    return {
        "type": "gr_rec_think_exact_clamp_v1_earliest_domain_branch_tokenizer_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": BASE,
        "model_weights_loaded": False,
        "templates": templates,
        "longest_common_prefix_token_ids": prefix,
        "longest_common_prefix_tokens": [token_record(tokenizer, value) for value in prefix],
        "earliest_branch_declaration_token_index": branch_index,
        "earliest_branch_tokens": branch_tokens,
    }


def subsequence_starts(values: list[int], pattern: list[int], start: int, stop: int) -> list[int]:
    return [
        index
        for index in range(start, stop - len(pattern) + 1)
        if values[index:index + len(pattern)] == pattern
    ]


def locate_earliest_branch(
    completion_ids, final_sid, tokenizer, template_spec: dict,
) -> BranchAlignment:
    if final_sid is None:
        return BranchAlignment(False, None, None, None, None, None, "invalid_sid")
    sid_domain = str(final_sid[0])
    sid_positions = find_final_sid_token_positions(completion_ids, final_sid, tokenizer)
    ids = [int(value) for value in completion_ids]
    close_ids = [int(value) for value in tokenizer.encode("</think>", add_special_tokens=False)]
    close_starts = subsequence_starts(ids, close_ids, 0, sid_positions[0])
    if not close_starts:
        return BranchAlignment(
            False, None, sid_domain, None, None, None,
            "missing_think_close_before_final_sid",
        )
    search_start = close_starts[-1] + len(close_ids)
    matches = []
    for domain, definition in template_spec["templates"].items():
        pattern = [int(value) for value in definition["token_ids"]]
        matches.extend(
            (domain, start)
            for start in subsequence_starts(ids, pattern, search_start, sid_positions[0])
        )
    if not matches:
        return BranchAlignment(
            False, None, sid_domain, None, None, None,
            "missing_exact_declaration_template",
        )
    if len(matches) != 1:
        return BranchAlignment(
            False, None, sid_domain, None, None, None,
            "multiple_declaration_template_matches",
        )
    declaration_domain, declaration_start = matches[0]
    branch_position = (
        declaration_start + template_spec["earliest_branch_declaration_token_index"]
    )
    branch_token_id = ids[branch_position]
    expected_id = int(
        template_spec["earliest_branch_tokens"][declaration_domain]["token_id"]
    )
    if branch_token_id != expected_id:
        raise RuntimeError("located branch token disagrees with tokenizer audit")
    valid = declaration_domain == sid_domain
    return BranchAlignment(
        valid,
        declaration_domain,
        sid_domain,
        declaration_start,
        branch_position,
        branch_token_id,
        None if valid else "declaration_sid_domain_mismatch",
    )


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


def probability_record(old_logps, row: int, position: int | None, token_id: int | None) -> dict:
    if position is None:
        return {"token_id": token_id, "token_position": None, "old_logp": None, "old_probability": None}
    logp = float(old_logps[row, position])
    return {
        "token_id": int(token_id),
        "token_position": int(position),
        "old_logp": logp,
        "old_probability": math.exp(logp),
    }


def reference_group_is_domain_only(reference_group: dict) -> bool:
    old_stage = reference_group.get("old_stage_active", {})
    return bool(
        old_stage.get("domain")
        and not any(old_stage.get(stage) for stage in ("a", "b", "c"))
    )


def audit_rollout(
    model, tokenizer, parameters, batch: RolloutBatch, template_spec: dict,
    domain_only: bool,
) -> dict:
    shared_rng_state = capture_rng_state(batch.input_ids.device)
    with torch.no_grad():
        old_logps = completion_logps(model, batch).detach()
    states = [
        hierarchy_state(sid, batch.gold_sids, batch.target_domain)
        for sid in batch.predicted_sids
    ]
    credits = conditional_hierarchical_credits(states)
    domain_advantages = [float(row[0]) for row in credits]
    branch_alignments = [
        locate_earliest_branch(ids, sid, tokenizer, template_spec)
        for ids, sid in zip(batch.completion_ids_list, batch.predicted_sids)
    ]
    branch_group_valid = all(
        not state.valid or alignment.valid
        for state, alignment in zip(states, branch_alignments)
    )
    branch_hier_tokens = torch.zeros_like(batch.completion_ids, dtype=torch.float32)
    legacy_branch_tokens = torch.zeros_like(batch.completion_ids, dtype=torch.float32)
    legacy_scalar = legacy_advantages(batch.rewards, batch.input_ids.device)
    if branch_group_valid:
        for row, (state, alignment, domain_advantage) in enumerate(
            zip(states, branch_alignments, domain_advantages)
        ):
            if state.valid and alignment.branch_position is not None:
                branch_hier_tokens[row, alignment.branch_position] = domain_advantage
                legacy_branch_tokens[row, alignment.branch_position] = legacy_scalar[row]

    branch_norm = legacy_branch_norm = matched_cosine = None
    if domain_only:
        branch_norm, branch_vector = isolated_gradient(
            model,
            parameters,
            lambda: clipped_policy_loss(
                model, batch, old_logps, branch_hier_tokens, "hierarchical_sum"
            ),
            shared_rng_state,
            batch.input_ids.device,
        )
        legacy_branch_norm, legacy_branch_vector = isolated_gradient(
            model,
            parameters,
            lambda: clipped_policy_loss(
                model, batch, old_logps, legacy_branch_tokens, "hierarchical_sum"
            ),
            shared_rng_state,
            batch.input_ids.device,
        )
        matched_cosine = stable_vector_cosine(legacy_branch_vector, branch_vector)
        del branch_vector, legacy_branch_vector

    candidates = []
    for row, (ids, sid, alignment) in enumerate(
        zip(batch.completion_ids_list, batch.predicted_sids, branch_alignments)
    ):
        text_alignment = locate_text_domain_token(ids, sid, tokenizer) if sid is not None else None
        branch = probability_record(
            old_logps, row, alignment.branch_position, alignment.branch_token_id
        )
        noun_position = text_alignment.text_domain_token_position if text_alignment else None
        noun_id = int(ids[noun_position]) if noun_position is not None else None
        noun = probability_record(old_logps, row, noun_position, noun_id)
        sid_position = text_alignment.sid_domain_token_position if text_alignment else None
        sid_id = int(ids[sid_position]) if sid_position is not None else None
        sid_record = probability_record(old_logps, row, sid_position, sid_id)
        candidates.append({
            "candidate_index": row,
            "domain_advantage": domain_advantages[row],
            "branch_alignment": asdict(alignment),
            "earliest_branch": branch,
            "text_domain_noun": noun,
            "sid_domain": sid_record,
        })
    return {
        "group_id": batch.group_id,
        "rollout_fingerprint": batch.fingerprint,
        "rewards": list(batch.rewards),
        "domain_only": domain_only,
        "branch_alignment_valid": branch_group_valid,
        "valid_branch_candidate_count": sum(item.valid for item in branch_alignments),
        "branch_hier_grad_norm": branch_norm,
        "legacy_branch_grad_norm": legacy_branch_norm,
        "legacy_branch_hier_cosine": matched_cosine,
        "domain_advantages": domain_advantages,
        "candidates": candidates,
    }


def numeric_summary(values) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def build_paired_comparison(reference: dict, current: dict, parity: list[dict]) -> dict:
    groups = []
    domain_only_groups = []
    for old, new in zip(reference["groups"], current["groups"]):
        legacy = float(old["legacy_grad_norm"])
        row = {
            "audit_index": new["audit_index"],
            "group_id": new["group_id"],
            "rollout_fingerprint": new["rollout_fingerprint"],
            "rewards": new["rewards"],
            "domain_only": new["domain_only"],
            "branch_alignment_valid": new["branch_alignment_valid"],
            "legacy_grad_norm": legacy,
            "old_sid_domain_hier_grad_norm": old["old_sid_domain_hier_grad_norm"],
            "text_domain_hier_grad_norm": old["new_text_domain_hier_grad_norm"],
            "branch_domain_hier_grad_norm": new["branch_hier_grad_norm"],
            "branch_hier_over_legacy": (
                new["branch_hier_grad_norm"] / legacy
                if new["branch_hier_grad_norm"] is not None and legacy > 0 else None
            ),
            "legacy_branch_hier_cosine": new["legacy_branch_hier_cosine"],
            "candidates": new["candidates"],
        }
        groups.append(row)
        if row["domain_only"]:
            domain_only_groups.append(row)

    branch_probs = [
        c["earliest_branch"]["old_probability"]
        for g in domain_only_groups for c in g["candidates"]
    ]
    noun_probs = [
        c["text_domain_noun"]["old_probability"]
        for g in domain_only_groups for c in g["candidates"]
    ]
    sid_probs = [
        c["sid_domain"]["old_probability"]
        for g in domain_only_groups for c in g["candidates"]
    ]
    branch_summary = numeric_summary(branch_probs)
    noun_summary = numeric_summary(noun_probs)
    sid_summary = numeric_summary(sid_probs)
    ratio_summary = numeric_summary(g["branch_hier_over_legacy"] for g in domain_only_groups)
    all_domain_only_aligned = all(g["branch_alignment_valid"] for g in domain_only_groups)
    if not all_domain_only_aligned:
        verdict = "BRANCH_POINT_AMBIGUOUS"
    elif (
        branch_summary["median"] is not None
        and noun_summary["median"] is not None
        and sid_summary["median"] is not None
        and branch_summary["median"] < min(noun_summary["median"], sid_summary["median"]) - 0.01
        and ratio_summary["median"] is not None
        and ratio_summary["median"] >= 0.25
    ):
        verdict = "EARLIEST_BRANCH_RECOVERS_SIGNAL"
    elif (
        branch_summary["median"] is not None
        and branch_summary["median"] >= 0.99
        and ratio_summary["median"] is not None
        and ratio_summary["median"] < 0.25
    ):
        verdict = "DOMAIN_DECISION_ALREADY_SATURATED"
    else:
        verdict = "BRANCH_POINT_AMBIGUOUS"
    return {
        "type": "gr_rec_think_exact_clamp_v1_earliest_domain_branch_paired_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paired_audit_valid": all(item["valid"] for item in parity),
        "parity": parity,
        "summary": {
            "fingerprint_parity_count": sum(item["rollout_fingerprint"] for item in parity),
            "reward_parity_count": sum(item["rewards"] for item in parity),
            "domain_only_group_count": len(domain_only_groups),
            "domain_only_aligned_group_count": sum(
                group["branch_alignment_valid"] for group in domain_only_groups
            ),
            "branch_probability": branch_summary,
            "noun_probability": noun_summary,
            "sid_probability": sid_summary,
            "branch_hier_over_legacy": ratio_summary,
            "matched_support_cosine": numeric_summary(
                g["legacy_branch_hier_cosine"] for g in domain_only_groups
            ),
            "verdict": verdict,
        },
        "checksum": {
            "before": current["trainable_parameter_checksum_before"],
            "after": current["trainable_parameter_checksum_after"],
            "parameter_change": current["parameter_change"],
        },
        "groups": groups,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Earliest Domain branch zero-step diagnostic")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer-only", action="store_true")
    mode.add_argument("--execute-zero-step-gpu-audit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--template-result")
    parser.add_argument("--paired-reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-output")
    args = parser.parse_args(argv)
    if args.execute_zero_step_gpu_audit:
        required = (args.template_result, args.paired_reference, args.paired_output)
        if not all(required):
            parser.error("GPU audit requires template result, paired reference and paired output")
        if args.groups != 8 or args.seed != 20260816:
            parser.error("this paired diagnostic is fixed to groups=8 seed=20260816")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.tokenizer_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        payload = build_template_spec(tokenizer)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GPU diagnostic requires an explicitly authorized CUDA device")
    template_spec = json.loads(Path(args.template_result).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.paired_reference).read_text(encoding="utf-8"))
    if len(reference.get("groups", [])) != 8:
        raise RuntimeError("paired reference must contain exactly 8 groups")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, _template = load_model(args.device)
    if build_template_spec(tokenizer)["templates"] != template_spec["templates"]:
        raise RuntimeError("GPU tokenizer does not match CPU tokenizer audit")
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

    groups = []
    parity = []
    for index, record in enumerate(records):
        batch = make_rollout_batch(
            record, model, tokenizer, args.device, args.max_new_tokens
        )
        status = pairing_status(index, batch, reference["groups"][index])
        parity.append(status)
        if not status["valid"]:
            model.zero_grad(set_to_none=True)
            invalid = {
                "paired_audit_valid": False,
                "flags": ["PAIRED_AUDIT_INVALID"],
                "parity": parity,
                "failed_audit_index": index,
                "checksum_before": checksum_before,
                "checksum_after": trainable_parameter_checksum(parameters),
            }
            Path(args.paired_output).write_text(
                json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("PAIRED_AUDIT_INVALID")
        domain_only = reference_group_is_domain_only(reference["groups"][index])
        result = audit_rollout(
            model, tokenizer, parameters, batch, template_spec, domain_only
        )
        result["audit_index"] = index
        groups.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    model.zero_grad(set_to_none=True)
    checksum_after = trainable_parameter_checksum(parameters)
    payload = {
        "type": "gr_rec_think_exact_clamp_v1_gpu_earliest_domain_branch_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "zero_update": True,
        "diagnostic_only": True,
        "base": BASE,
        "adapter": ADAPTER,
        "groups_requested": 8,
        "seed": args.seed,
        "trainable_parameter_checksum_before": checksum_before,
        "trainable_parameter_checksum_after": checksum_after,
        "parameter_change": checksum_before != checksum_after,
        "template_result": str(Path(args.template_result).resolve()),
        "groups": groups,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paired = build_paired_comparison(reference, payload, parity)
    paired["reference_path"] = str(Path(args.paired_reference).resolve())
    paired["current_path"] = str(Path(args.output).resolve())
    Path(args.paired_output).write_text(
        json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if payload["parameter_change"]:
        raise RuntimeError("trainable parameters changed during zero-step diagnostic")
    return payload


if __name__ == "__main__":
    main()
