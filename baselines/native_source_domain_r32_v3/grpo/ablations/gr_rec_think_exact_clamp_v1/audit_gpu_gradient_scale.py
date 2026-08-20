"""Zero-update GPU gradient-scale audit for the current NoThink objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from grpo_model import ADAPTER, BASE, encode_prompt, generate_batch, load_model
from grpo_sid import final_sid, parse_sid, q_reward
from grpo_trl_trainer import M_NO, ROUTE_LOSS_W, build_route_dataset
from nothink_bridge import BRIDGE_LAMBDA, require_single_token, uniform_multi_positive_ce
from nothink_hierarchical_credit import (
    conditional_hierarchical_credits,
    hierarchy_state,
    locate_domain_commitment_token,
)
from run_grpo_trl_smoke import DATA


EPSILON = 0.2


@dataclass
class RolloutBatch:
    group_id: str
    prompt_ids: tuple[int, ...]
    completion_ids_list: tuple[tuple[int, ...], ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    rewards: tuple[float, ...]
    predicted_sids: tuple[tuple | None, ...]
    gold_sids: tuple[tuple, ...]
    target_domain: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            [self.prompt_ids, self.completion_ids_list], separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def reward_pattern(rewards) -> str:
    values = tuple(float(value) for value in rewards)
    if values == (0.0,) * M_NO:
        return "ALL_ZERO"
    if values == (-0.25,) * M_NO:
        return "ALL_WRONG_DOMAIN_ZERO_SIGNAL"
    if any(value >= 8.0 for value in values):
        return "EXACT_SIGNAL"
    if any(value >= 2.0 for value in values):
        return "AB_SIGNAL"
    if any(value >= 0.5 for value in values):
        return "A_ONLY"
    return "MIXED"


def legacy_advantages(rewards, device) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device=device)
    if values.numel() != M_NO:
        raise ValueError("legacy audit objective requires exactly one G8")
    return (values - values.mean()) / (values.std(correction=0) + 1e-4)


def hierarchical_token_advantages(batch: RolloutBatch, tokenizer) -> tuple[torch.Tensor, list, dict]:
    states = [
        hierarchy_state(sid, batch.gold_sids, batch.target_domain)
        for sid in batch.predicted_sids
    ]
    alignments = []
    for sid, candidate_ids in zip(batch.predicted_sids, batch.completion_ids_list):
        try:
            alignment = (
                locate_domain_commitment_token(candidate_ids, sid, tokenizer)
                if sid is not None else None
            )
        except (RuntimeError, ValueError):
            alignment = None
        alignments.append(alignment)
    eligibility = [alignment is not None and alignment.eligible for alignment in alignments]
    credits = conditional_hierarchical_credits(states, domain_eligible=eligibility)
    token_advantages = torch.zeros_like(batch.completion_ids, dtype=torch.float32)
    for row, (sid, alignment, candidate_credit) in enumerate(
        zip(batch.predicted_sids, alignments, credits)
    ):
        if sid is None:
            if any(value != 0 for value in candidate_credit):
                raise RuntimeError("invalid SID received nonzero hierarchical credit")
            continue
        positions = alignment.hierarchy_token_positions
        for value, position in zip(candidate_credit, positions):
            if position is None:
                if value != 0:
                    raise RuntimeError("nonzero hierarchy credit has no token position")
                continue
            token_advantages[row, position] = value
    diagnostics = {
        "domain_text_alignment_valid": all(
            not state.valid or eligible for state, eligible in zip(states, eligibility)
        ),
        "domain_commitment_eligible": eligibility,
        "domain_commitment_modes": [alignment.mode if alignment else None for alignment in alignments],
        "text_domains": [alignment.commitment_domain if alignment else None for alignment in alignments],
        "text_domain_token_positions": [
            alignment.commitment_token_position if alignment else None
            for alignment in alignments
        ],
        "sid_domain_token_positions": [
            alignment.sid_domain_token_position if alignment else None
            for alignment in alignments
        ],
        "alignment_failures": [alignment.failure if alignment else "unresolved" for alignment in alignments],
        "eligible_count": sum(eligibility),
        "branch_count": sum(alignment is not None and alignment.mode == "branch" for alignment in alignments),
        "direct_sid_fallback_count": sum(
            alignment is not None and alignment.mode == "direct_sid_fallback"
            for alignment in alignments
        ),
        "unresolved_count": sum(not eligible for eligible in eligibility),
    }
    return token_advantages, credits, diagnostics


def legacy_text_domain_advantages(
    batch: RolloutBatch, legacy_scalar: torch.Tensor, alignment: dict,
) -> torch.Tensor:
    """Diagnostic-only legacy scalar advantage on matched text-Domain support."""
    token_advantages = torch.zeros_like(batch.completion_ids, dtype=torch.float32)
    for row, (sid, position, eligible) in enumerate(
        zip(
            batch.predicted_sids,
            alignment["text_domain_token_positions"],
            alignment["domain_commitment_eligible"],
        )
    ):
        if sid is not None and position is not None and eligible:
            token_advantages[row, position] = legacy_scalar[row]
    return token_advantages


def sampled_domain_diagnostics(
    batch: RolloutBatch, old_logps: torch.Tensor, credits: list, alignment: dict,
) -> list[dict]:
    records = []
    for row, sid in enumerate(batch.predicted_sids):
        text_position = alignment["text_domain_token_positions"][row]
        sid_position = alignment["sid_domain_token_positions"][row]
        text_logp = float(old_logps[row, text_position]) if text_position is not None else None
        sid_logp = float(old_logps[row, sid_position]) if sid_position is not None else None
        records.append({
            "candidate_index": row,
            "text_domain": alignment["text_domains"][row],
            "sid_domain": sid[0] if sid is not None else None,
            "text_domain_token_position": text_position,
            "sid_domain_token_position": sid_position,
            "text_domain_old_logp": text_logp,
            "text_domain_old_probability": math.exp(text_logp) if text_logp is not None else None,
            "sid_domain_old_logp": sid_logp,
            "sid_domain_old_probability": math.exp(sid_logp) if sid_logp is not None else None,
            "domain_advantage": credits[row][0],
            "alignment_failure": alignment["alignment_failures"][row],
        })
    return records


def completion_logps(model, batch: RolloutBatch) -> torch.Tensor:
    width = batch.completion_ids.size(1)
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        logits_to_keep=width + 1,
    )
    logits = output.logits
    if logits.size(1) < width + 1:
        raise RuntimeError("model did not return enough next-token logits")
    next_logits = logits[:, -(width + 1):-1, :]
    return torch.log_softmax(next_logits.float(), dim=-1).gather(
        2, batch.completion_ids.unsqueeze(-1)
    ).squeeze(-1)


def clipped_policy_loss(
    model, batch: RolloutBatch, old_logps: torch.Tensor,
    token_advantages: torch.Tensor, reduction: str,
) -> torch.Tensor:
    current_logps = completion_logps(model, batch)
    ratio = torch.exp(current_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
    per_token = -torch.min(ratio * token_advantages, clipped_ratio * token_advantages)
    masked_sum = (per_token * batch.completion_mask).sum(-1)
    if reduction == "legacy_mean":
        per_sample = masked_sum / batch.completion_mask.sum(-1).clamp(min=1.0)
    elif reduction == "hierarchical_sum":
        per_sample = masked_sum
    else:
        raise ValueError(f"unknown reduction: {reduction}")
    return (per_sample * ROUTE_LOSS_W["no_think"]).mean()


def trainable_parameters(model):
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable adapter parameters found")
    return parameters


def trainable_parameter_checksum(parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(json.dumps(list(parameter.shape)).encode("ascii"))
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw))
        del raw
    return digest.hexdigest()


def copy_trainable_gradient(parameters) -> torch.Tensor:
    parts = []
    for _name, parameter in parameters:
        if parameter.grad is None:
            parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            parts.append(parameter.grad.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def capture_rng_state(device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None
    return cpu_state, cuda_state


def restore_rng_state(state, device) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def isolated_gradient(
    model, parameters, objective, rng_state=None, device=None,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    if rng_state is not None:
        restore_rng_state(rng_state, device)
    loss = objective()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite audit objective")
    loss.backward()
    vector = copy_trainable_gradient(parameters)
    norm = float(torch.linalg.vector_norm(vector))
    model.zero_grad(set_to_none=True)
    return norm, vector


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def dead_zero_bridge_loss(model, batch: RolloutBatch, tokenizer) -> torch.Tensor:
    if batch.rewards != (0.0,) * M_NO:
        raise ValueError("bridge gradient is allowed only for a real exact all-zero rollout")
    gold_a = sorted({sid[1] for sid in batch.gold_sids if sid[0] == batch.target_domain})
    if not gold_a:
        raise RuntimeError("all-zero bridge group has no target-domain Gold A")
    domain_id = require_single_token(tokenizer, f"<|{batch.target_domain}_begin|>")
    target_ids = [require_single_token(tokenizer, f"<s_a_{value}>") for value in gold_a]
    query = torch.tensor(
        [batch.prompt_ids + (domain_id,)], dtype=torch.long, device=batch.input_ids.device
    )
    logits = model(
        input_ids=query,
        attention_mask=torch.ones_like(query),
        use_cache=False,
        logits_to_keep=1,
    ).logits[0, -1]
    return uniform_multi_positive_ce(logits, target_ids)


def audit_rollout(model, tokenizer, parameters, batch: RolloutBatch) -> dict:
    shared_rng_state = capture_rng_state(batch.input_ids.device)
    with torch.no_grad():
        old_logps = completion_logps(model, batch).detach()
    legacy_scalar = legacy_advantages(batch.rewards, batch.input_ids.device)
    legacy_tokens = legacy_scalar.unsqueeze(1).expand_as(batch.completion_mask)
    hierarchical_tokens, credits, alignment = hierarchical_token_advantages(batch, tokenizer)
    columns = list(zip(*credits))
    domain_only = (
        any(value != 0 for value in columns[0])
        and not any(value != 0 for column in columns[1:] for value in column)
    )

    legacy_norm, legacy_vector = isolated_gradient(
        model, parameters,
        lambda: clipped_policy_loss(
            model, batch, old_logps, legacy_tokens, "legacy_mean"
        ), shared_rng_state, batch.input_ids.device,
    )
    hierarchical_norm, hierarchical_vector = isolated_gradient(
        model, parameters,
        lambda: clipped_policy_loss(
            model, batch, old_logps, hierarchical_tokens, "hierarchical_sum"
        ), shared_rng_state, batch.input_ids.device,
    )
    legacy_text_norm = None
    legacy_text_hier_cosine = None
    if domain_only:
        legacy_text_tokens = legacy_text_domain_advantages(batch, legacy_scalar, alignment)
        legacy_text_norm, legacy_text_vector = isolated_gradient(
            model, parameters,
            lambda: clipped_policy_loss(
                model, batch, old_logps, legacy_text_tokens, "hierarchical_sum"
            ), shared_rng_state, batch.input_ids.device,
        )
        legacy_text_hier_cosine = vector_cosine(
            legacy_text_vector, hierarchical_vector
        )
        del legacy_text_vector
    ratio = hierarchical_norm / legacy_norm if legacy_norm > 0 else None
    result = {
        "group_id": batch.group_id,
        "rollout_fingerprint": batch.fingerprint,
        "reward_pattern": reward_pattern(batch.rewards),
        "rewards": list(batch.rewards),
        "legacy_grad_norm": legacy_norm,
        "hier_grad_norm": hierarchical_norm,
        "hier_over_legacy": ratio,
        "legacy_hier_cosine": vector_cosine(legacy_vector, hierarchical_vector),
        "legacy_text_domain_grad_norm": legacy_text_norm,
        "legacy_text_domain_hier_cosine": legacy_text_hier_cosine,
        "hierarchical_credits": [list(row) for row in credits],
        "stage_active": {
            "domain": any(value != 0 for value in columns[0]),
            "a": any(value != 0 for value in columns[1]),
            "b": any(value != 0 for value in columns[2]),
            "c": any(value != 0 for value in columns[3]),
        },
        "credited_token_count": {
            "domain": sum(value != 0 for value in columns[0]),
            "a": sum(value != 0 for value in columns[1]),
            "b": sum(value != 0 for value in columns[2]),
            "c": sum(value != 0 for value in columns[3]),
        },
        **alignment,
        "candidate_domain_diagnostics": sampled_domain_diagnostics(
            batch, old_logps, credits, alignment
        ),
        "dead_zero_bridge_active": batch.rewards == (0.0,) * M_NO,
        "bridge_raw_grad_norm": None,
        "bridge_weighted_grad_norm": None,
        "bridge_over_hier": None,
    }
    del legacy_vector, hierarchical_vector
    if result["dead_zero_bridge_active"]:
        raw_norm, bridge_vector = isolated_gradient(
            model, parameters, lambda: dead_zero_bridge_loss(model, batch, tokenizer)
        )
        result["bridge_raw_grad_norm"] = raw_norm
        result["bridge_weighted_grad_norm"] = BRIDGE_LAMBDA * raw_norm
        del bridge_vector
    return result


def numeric_summary(values) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(records: list[dict]) -> dict:
    active_hier = [record["hier_grad_norm"] for record in records if record["hier_grad_norm"] > 0]
    median_active_hier = statistics.median(active_hier) if active_hier else None
    bridge_ratios = []
    for record in records:
        weighted = record["bridge_weighted_grad_norm"]
        if weighted is not None and median_active_hier:
            record["bridge_over_hier"] = weighted / median_active_hier
            bridge_ratios.append(record["bridge_over_hier"])
    ratio_summary = numeric_summary(record["hier_over_legacy"] for record in records)
    bridge_summary = numeric_summary(bridge_ratios)
    flags = []
    median_ratio = ratio_summary["median"]
    if median_ratio is not None and median_ratio < 0.25:
        flags.append("HIER_GRAD_TOO_SMALL_REVIEW")
    if median_ratio is not None and median_ratio > 4.0:
        flags.append("HIER_GRAD_TOO_LARGE_REVIEW")
    if any(value > 0.20 for value in bridge_ratios):
        flags.append("REVIEW_BRIDGE_LAMBDA")
    return {
        "group_count": len(records),
        "legacy_grad_norm": numeric_summary(r["legacy_grad_norm"] for r in records),
        "hier_grad_norm": numeric_summary(r["hier_grad_norm"] for r in records),
        "hier_over_legacy": ratio_summary,
        "legacy_hier_cosine": numeric_summary(r["legacy_hier_cosine"] for r in records),
        "legacy_text_domain_grad_norm": numeric_summary(
            r["legacy_text_domain_grad_norm"] for r in records
        ),
        "legacy_text_domain_hier_cosine": numeric_summary(
            r["legacy_text_domain_hier_cosine"] for r in records
        ),
        "median_active_hier_grad_norm": median_active_hier,
        "bridge_weighted_over_median_active_hier": bridge_summary,
        "flags": flags,
    }


def make_rollout_batch(record, model, tokenizer, device, max_new_tokens) -> RolloutBatch:
    prompt_ids = tuple(encode_prompt(tokenizer, record["prompt"]))
    was_training = model.training
    model.eval()
    try:
        texts, completion_ids = generate_batch(
            model, tokenizer, [list(prompt_ids)], max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=1.0, num_beams=1,
            num_return_sequences=M_NO, return_ids=True,
        )
    finally:
        model.train(was_training)
    if len(completion_ids) != M_NO or any(not ids for ids in completion_ids):
        raise RuntimeError("generation did not produce one complete G8")
    predicted = tuple(final_sid(text) for text in texts)
    gold = tuple(
        sid for value in record["all_gold_sids"] if (sid := parse_sid(value)) is not None
    )
    rewards = tuple(q_reward(sid, gold) for sid in predicted)
    max_completion = max(map(len, completion_ids))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    completion_tensor = torch.full(
        (M_NO, max_completion), pad_id, dtype=torch.long, device=device
    )
    completion_mask = torch.zeros(
        (M_NO, max_completion), dtype=torch.long, device=device
    )
    for row, ids in enumerate(completion_ids):
        completion_tensor[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        completion_mask[row, :len(ids)] = 1
    prompt_tensor = torch.tensor([prompt_ids] * M_NO, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_tensor, completion_tensor], dim=1)
    attention_mask = torch.cat([torch.ones_like(prompt_tensor), completion_mask], dim=1)
    return RolloutBatch(
        group_id=record["recommendation_group_id"],
        prompt_ids=prompt_ids,
        completion_ids_list=tuple(tuple(ids) for ids in completion_ids),
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_ids=completion_tensor,
        completion_mask=completion_mask,
        rewards=rewards,
        predicted_sids=predicted,
        gold_sids=gold,
        target_domain=record["target_domain"],
    )


def pairing_status(index: int, batch: RolloutBatch, reference_group: dict) -> dict:
    checks = {
        "group_id": batch.group_id == reference_group.get("group_id"),
        "rollout_fingerprint": batch.fingerprint == reference_group.get("rollout_fingerprint"),
        "rewards": list(batch.rewards) == reference_group.get("rewards"),
    }
    return {"audit_index": index, **checks, "valid": all(checks.values())}


def build_paired_comparison(reference: dict, current: dict, parity: list[dict]) -> dict:
    groups = []
    domain_only_indices = []
    for index, (old, new) in enumerate(zip(reference["groups"], current["groups"])):
        old_norm = old["hier_grad_norm"]
        new_norm = new["hier_grad_norm"]
        old_stage = old["stage_active"]
        old_count = old["credited_token_count"]
        abc_stage_parity = all(old_stage[key] == new["stage_active"][key] for key in "abc")
        abc_count_parity = all(old_count[key] == new["credited_token_count"][key] for key in "abc")
        old_domain_only = old_stage["domain"] and not any(old_stage[key] for key in "abc")
        if old_domain_only:
            domain_only_indices.append(index)
        groups.append({
            "audit_index": index,
            "group_id": new["group_id"],
            "rollout_fingerprint": new["rollout_fingerprint"],
            "rewards": new["rewards"],
            "legacy_grad_norm": new["legacy_grad_norm"],
            "old_sid_domain_hier_grad_norm": old_norm,
            "new_text_domain_hier_grad_norm": new_norm,
            "new_text_hier_over_legacy": (
                new_norm / new["legacy_grad_norm"] if new["legacy_grad_norm"] > 0 else None
            ),
            "new_text_hier_over_old_sid_hier": new_norm / old_norm if old_norm > 0 else None,
            "legacy_full_text_hier_cosine": new["legacy_hier_cosine"],
            "legacy_text_domain_grad_norm": new["legacy_text_domain_grad_norm"],
            "legacy_text_domain_hier_cosine": new["legacy_text_domain_hier_cosine"],
            "old_stage_active": old_stage,
            "new_stage_active": new["stage_active"],
            "old_credited_token_count": old_count,
            "new_credited_token_count": new["credited_token_count"],
            "abc_stage_parity": abc_stage_parity,
            "abc_credited_count_parity": abc_count_parity,
            "abc_credits_reconstructed_from_immutable_rollout": [
                row[1:] for row in new["hierarchical_credits"]
            ],
            "domain_text_alignment_valid": new["domain_text_alignment_valid"],
            "candidate_domain_diagnostics": new["candidate_domain_diagnostics"],
        })

    domain_groups = [groups[index] for index in domain_only_indices]
    candidates = [
        candidate
        for group in domain_groups
        for candidate in group["candidate_domain_diagnostics"]
    ]
    return {
        "type": "gr_rec_think_exact_clamp_v1_text_domain_paired_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paired_audit_valid": all(item["valid"] for item in parity),
        "reference_type": reference.get("type"),
        "reference_created_at": reference.get("created_at"),
        "current_created_at": current.get("created_at"),
        "parity": parity,
        "domain_only_indices": domain_only_indices,
        "summary": {
            "group_count": len(groups),
            "fingerprint_parity_count": sum(item["rollout_fingerprint"] for item in parity),
            "reward_parity_count": sum(item["rewards"] for item in parity),
            "alignment_valid_group_count": sum(
                group["domain_text_alignment_valid"] for group in groups
            ),
            "domain_only_old_sid_hier_grad_norm": numeric_summary(
                group["old_sid_domain_hier_grad_norm"] for group in domain_groups
            ),
            "domain_only_new_text_hier_grad_norm": numeric_summary(
                group["new_text_domain_hier_grad_norm"] for group in domain_groups
            ),
            "domain_only_new_over_old": numeric_summary(
                group["new_text_hier_over_old_sid_hier"] for group in domain_groups
            ),
            "domain_only_new_over_legacy": numeric_summary(
                group["new_text_hier_over_legacy"] for group in domain_groups
            ),
            "domain_only_matched_support_cosine": numeric_summary(
                group["legacy_text_domain_hier_cosine"] for group in domain_groups
            ),
            "domain_only_text_probability": numeric_summary(
                item["text_domain_old_probability"] for item in candidates
            ),
            "domain_only_sid_probability": numeric_summary(
                item["sid_domain_old_probability"] for item in candidates
            ),
            "group_6_7_abc_parity": all(
                groups[index]["abc_stage_parity"]
                and groups[index]["abc_credited_count_parity"]
                for index in (6, 7)
            ),
        },
        "checksum": {
            "before": current["trainable_parameter_checksum_before"],
            "after": current["trainable_parameter_checksum_after"],
            "parameter_change": current["parameter_change"],
        },
        "groups": groups,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Zero-update NoThink GPU gradient audit")
    parser.add_argument("--execute-zero-step-gpu-audit", action="store_true")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default="gpu_gradient_scale_audit.json")
    parser.add_argument("--paired-reference")
    parser.add_argument("--paired-output")
    args = parser.parse_args(argv)
    if not args.execute_zero_step_gpu_audit:
        parser.error("explicit --execute-zero-step-gpu-audit authorization flag is required")
    if not 1 <= args.groups <= 16:
        parser.error("--groups must be in [1, 16]")
    if bool(args.paired_reference) != bool(args.paired_output):
        parser.error("--paired-reference and --paired-output must be provided together")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this harness requires an explicitly authorized CUDA device")
    paired_reference = None
    if args.paired_reference:
        paired_reference = json.loads(Path(args.paired_reference).read_text(encoding="utf-8"))
        reference_groups = paired_reference.get("groups", [])
        if len(reference_groups) != args.groups:
            raise RuntimeError(
                "paired reference group count does not match --groups: "
                f"{len(reference_groups)} != {args.groups}"
            )
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
    unexpected = [name for name, _parameter in parameters if "lora" not in name.lower()]
    if unexpected:
        raise RuntimeError(f"non-LoRA trainable parameters found: {unexpected[:5]}")
    parameter_checksum_before = trainable_parameter_checksum(parameters)

    dataset = build_route_dataset(DATA, n_groups=args.groups, seed=args.seed, chunk=8)
    records = [record for record in dataset if record["route"] == "no_think"]
    if len(records) != args.groups:
        raise RuntimeError("NoThink audit dataset selection did not match --groups")
    group_results = []
    parity = []
    for index, record in enumerate(records):
        batch = make_rollout_batch(
            record, model, tokenizer, args.device, args.max_new_tokens
        )
        if paired_reference is not None:
            status = pairing_status(index, batch, paired_reference["groups"][index])
            parity.append(status)
            if not status["valid"]:
                model.zero_grad(set_to_none=True)
                checksum_after_invalid_pair = trainable_parameter_checksum(parameters)
                invalid_payload = {
                    "type": "gr_rec_think_exact_clamp_v1_text_domain_paired_audit",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "paired_audit_valid": False,
                    "flags": ["PAIRED_AUDIT_INVALID"],
                    "parity": parity,
                    "failed_audit_index": index,
                    "checksum": {
                        "before": parameter_checksum_before,
                        "after": checksum_after_invalid_pair,
                        "parameter_change": (
                            parameter_checksum_before != checksum_after_invalid_pair
                        ),
                    },
                }
                Path(args.paired_output).write_text(
                    json.dumps(invalid_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError("PAIRED_AUDIT_INVALID")
        result = audit_rollout(model, tokenizer, parameters, batch)
        result["audit_index"] = index
        group_results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    model.zero_grad(set_to_none=True)
    parameter_checksum_after = trainable_parameter_checksum(parameters)
    parameter_change = parameter_checksum_before != parameter_checksum_after
    payload = {
        "type": "gr_rec_think_exact_clamp_v1_gpu_gradient_scale_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "zero_update": True,
        "model_mode": "eval_generation_train_gradient_with_shared_rng",
        "activation_memory_mode": "non_reentrant_gradient_checkpointing",
        "gradient_forward_rng": "shared_old_legacy_hier_matched_text",
        "domain_credit_placement": "natural_language_domain_decision_token",
        "matched_support_objective": "diagnostic_only_legacy_text_domain",
        "base": BASE,
        "adapter": ADAPTER,
        "seed": args.seed,
        "trainable_parameter_count": sum(value.numel() for _name, value in parameters),
        "trainable_tensor_count": len(parameters),
        "trainable_parameter_checksum_before": parameter_checksum_before,
        "trainable_parameter_checksum_after": parameter_checksum_after,
        "parameter_change": parameter_change,
        "bridge_lambda": BRIDGE_LAMBDA,
        "summary": summarize(group_results),
        "groups": group_results,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if paired_reference is not None:
        paired_payload = build_paired_comparison(paired_reference, payload, parity)
        paired_payload["reference_path"] = str(Path(args.paired_reference).resolve())
        paired_payload["current_path"] = str(Path(args.output).resolve())
        if not paired_payload["paired_audit_valid"]:
            paired_payload["flags"] = ["PAIRED_AUDIT_INVALID"]
        Path(args.paired_output).write_text(
            json.dumps(paired_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not paired_payload["paired_audit_valid"]:
            raise RuntimeError("PAIRED_AUDIT_INVALID")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    model.zero_grad(set_to_none=True)
    if parameter_change:
        raise RuntimeError("trainable parameters changed during zero-step audit")
    return payload


if __name__ == "__main__":
    main()
