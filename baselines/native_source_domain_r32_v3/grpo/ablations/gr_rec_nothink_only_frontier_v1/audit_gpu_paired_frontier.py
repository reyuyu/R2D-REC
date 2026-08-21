#!/usr/bin/env python3
"""Zero-update paired OLD_HIER vs FRONTIER_V1 gradient audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

GRPO_ROOT = Path(__file__).resolve().parents[2]
THINK_PARENT = GRPO_ROOT / "ablations" / "gr_rec_think_exact_clamp_v1"
sys.path.insert(0, str(GRPO_ROOT / "scripts"))
sys.path.insert(0, str(THINK_PARENT))

from grpo_model import BASE, encode_prompt, generate_batch
from grpo_sid import final_sid, parse_sid, q_reward
from grpo_trl_trainer import ROUTE_LOSS_W
from nothink_hierarchical_credit import (
    conditional_hierarchical_credits,
    hierarchy_state,
    locate_domain_commitment_token,
)

from format_validator import validate_nothink_completion
from frontier_credit import FORMAT_ADV_TOTAL, STAGE_NAMES, plan_frontier_credits

FRESH_ADAPTER = (
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333"
)
HIER_CHECKPOINT = (
    "/data/GRPO/outputs/formal/"
    "GR-REC-NOTHINK-ONLY-HIER-G8BASE-E1-20260821/checkpoint-1250"
)
SELECTION_FILE = (
    GRPO_ROOT / "results" /
    "gr_rec_nothink_frontier_v1_historical_forensic_20260822.json"
)
M_NO = 8
EPSILON = 0.2
REPRESENTATIVE_LABELS = {
    "format": "format_violation",
    "a": "a_frontier_heavy",
    "b": "b_frontier_heavy",
    "c": "c_frontier_heavy",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-zero-step-gpu-audit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--selection-file", type=Path, default=SELECTION_FILE)
    parser.add_argument("--hier-checkpoint", type=Path, default=Path(HIER_CHECKPOINT))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute_zero_step_gpu_audit:
        parser.error("explicit zero-step authorization flag is required")
    if not args.device.startswith("cuda"):
        parser.error("an explicit CUDA device is required")
    return args


def load_policy(adapter: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    model.train()
    return model, tokenizer


def parameter_sets(model):
    lora = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    base = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    if not lora or any("lora" not in name.lower() for name, _ in lora):
        raise RuntimeError("trainable parameters are not exclusively LoRA")
    return lora, base


def tensor_checksum(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw))
        del raw
    return digest.hexdigest()


def base_version_fingerprint(parameters):
    return hashlib.sha256(
        json.dumps([(name, value._version) for name, value in parameters]).encode()
    ).hexdigest()


def copy_gradient(parameters):
    chunks = []
    for _name, parameter in parameters:
        chunks.append(
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().float().reshape(-1).cpu()
        )
    return torch.cat(chunks)


def completion_logps(model, batch):
    width = batch["completion_ids"].size(1)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        logits_to_keep=width + 1,
    )
    logits = output.logits[:, -(width + 1):-1, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(
        2, batch["completion_ids"].unsqueeze(-1)
    ).squeeze(-1)


def policy_loss(model, batch, old_logps, token_advantages):
    current_logps = completion_logps(model, batch)
    ratio = torch.exp(current_logps - old_logps)
    clipped = ratio.clamp(1 - EPSILON, 1 + EPSILON)
    per_token = -torch.minimum(
        ratio * token_advantages, clipped * token_advantages
    )
    per_sample = (per_token * batch["completion_mask"]).sum(-1)
    return (per_sample * ROUTE_LOSS_W["no_think"]).mean()


def isolated_gradient(model, parameters, objective):
    model.zero_grad(set_to_none=True)
    rng_cpu = torch.random.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(model.device)
    loss = objective()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite audit loss")
    loss.backward()
    vector = copy_gradient(parameters)
    norm = float(torch.linalg.vector_norm(vector))
    finite = bool(torch.isfinite(vector).all())
    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda, model.device)
    return float(loss.detach()), norm, vector, finite


def cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return None if float(denominator) == 0 else float(torch.dot(left, right) / denominator)


def make_batch(record, model, tokenizer, max_new_tokens):
    prompt_ids = tuple(encode_prompt(tokenizer, record["prompt"]))
    model.eval()
    try:
        texts, generated = generate_batch(
            model,
            tokenizer,
            [list(prompt_ids)],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
            num_return_sequences=M_NO,
            return_ids=True,
        )
    finally:
        model.train()
    if len(generated) != M_NO or any(not ids for ids in generated):
        raise RuntimeError("generation did not return one complete G8")
    width = max(map(len, generated))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    completions = torch.full((M_NO, width), pad, dtype=torch.long, device=model.device)
    mask = torch.zeros_like(completions)
    for row, ids in enumerate(generated):
        completions[row, : len(ids)] = torch.tensor(ids, device=model.device)
        mask[row, : len(ids)] = 1
    prompts = torch.tensor([prompt_ids] * M_NO, dtype=torch.long, device=model.device)
    gold = tuple(filter(None, (parse_sid(value) for value in record["gold_sids"])))
    predicted = tuple(final_sid(text) for text in texts)
    return {
        "group_id": record["group_id"],
        "selection_label": record["selection_label"],
        "prompt_ids": prompt_ids,
        "texts": tuple(texts),
        "completion_ids_list": tuple(tuple(ids) for ids in generated),
        "completion_ids": completions,
        "completion_mask": mask,
        "input_ids": torch.cat((prompts, completions), dim=1),
        "attention_mask": torch.cat((torch.ones_like(prompts), mask), dim=1),
        "gold_sids": gold,
        "predicted_sids": predicted,
        "rewards": tuple(q_reward(sid, gold) for sid in predicted),
        "target_domain": record["target_domain"],
    }


def old_hier_tokens(batch, tokenizer):
    states = [
        hierarchy_state(sid, batch["gold_sids"], batch["target_domain"])
        for sid in batch["predicted_sids"]
    ]
    alignments = []
    for sid, ids in zip(batch["predicted_sids"], batch["completion_ids_list"]):
        try:
            alignment = locate_domain_commitment_token(ids, sid, tokenizer) if sid else None
        except (RuntimeError, ValueError):
            alignment = None
        alignments.append(alignment)
    eligible = [alignment is not None and alignment.eligible for alignment in alignments]
    credits = conditional_hierarchical_credits(states, domain_eligible=eligible)
    tokens = torch.zeros_like(batch["completion_ids"], dtype=torch.float32)
    for row, (alignment, row_credits) in enumerate(zip(alignments, credits)):
        if alignment is None:
            if any(value != 0 for value in row_credits):
                raise RuntimeError("OLD_HIER token alignment error")
            continue
        for value, position in zip(row_credits, alignment.hierarchy_token_positions):
            if position is None and value != 0:
                raise RuntimeError("OLD_HIER nonzero credit has no token")
            if position is not None:
                tokens[row, position] = value
    return tokens, credits


def frontier_tokens(batch, tokenizer):
    validations = [validate_nothink_completion(text) for text in batch["texts"]]
    states = [
        hierarchy_state(item.parsed_sid, batch["gold_sids"], batch["target_domain"])
        for item in validations
    ]
    plan = plan_frontier_credits(states, [item.valid for item in validations])
    tokens = torch.zeros_like(batch["completion_ids"], dtype=torch.float32)
    stage_tokens = {
        "format": torch.zeros_like(tokens),
        **{name: torch.zeros_like(tokens) for name in STAGE_NAMES},
    }
    alignments = []
    for row, (validation, ids, candidate) in enumerate(
        zip(validations, batch["completion_ids_list"], plan.candidates)
    ):
        if not validation.valid:
            length = int(batch["completion_mask"][row].sum())
            if length <= 0:
                raise RuntimeError("format violation has empty completion")
            value = FORMAT_ADV_TOTAL / length
            tokens[row, :length] = value
            stage_tokens["format"][row, :length] = value
            alignments.append(None)
            continue
        alignment = locate_domain_commitment_token(ids, validation.parsed_sid, tokenizer)
        alignments.append(alignment)
        for column, (name, value, position) in enumerate(
            zip(STAGE_NAMES, candidate.credits, alignment.hierarchy_token_positions)
        ):
            if value != 0:
                tokens[row, position] = value
                stage_tokens[name][row, position] = value
    return tokens, stage_tokens, plan, validations, alignments


def serialize_matrix(tensor):
    return tensor.detach().float().cpu().tolist()


def audit_group(model, tokenizer, parameters, batch):
    with torch.no_grad():
        old_logps = completion_logps(model, batch).detach()
        current_logps = completion_logps(model, batch).detach()
    old_tokens, old_credits = old_hier_tokens(batch, tokenizer)
    frontier, stage_tokens, plan, validations, alignments = frontier_tokens(batch, tokenizer)
    old_loss, old_norm, old_vector, old_finite = isolated_gradient(
        model, parameters, lambda: policy_loss(model, batch, old_logps, old_tokens)
    )
    frontier_loss, frontier_norm, frontier_vector, frontier_finite = isolated_gradient(
        model, parameters, lambda: policy_loss(model, batch, old_logps, frontier)
    )
    result = {
        "group_id": batch["group_id"],
        "selection_label": batch["selection_label"],
        "completion_ids": [list(ids) for ids in batch["completion_ids_list"]],
        "completion_mask": serialize_matrix(batch["completion_mask"]),
        "rewards": list(batch["rewards"]),
        "old_logp": serialize_matrix(old_logps),
        "current_logp": serialize_matrix(current_logps),
        "paired_logp_max_abs_diff": float((old_logps - current_logps).abs().max()),
        "old_loss": old_loss,
        "frontier_loss": frontier_loss,
        "old_grad_norm": old_norm,
        "frontier_grad_norm": frontier_norm,
        "frontier_over_old": frontier_norm / old_norm if old_norm else None,
        "cosine_old_frontier": cosine(old_vector, frontier_vector),
        "old_grad_finite": old_finite,
        "frontier_grad_finite": frontier_finite,
        "old_hier_credits": [list(row) for row in old_credits],
        "frontier_credits": [list(item.credits) for item in plan.candidates],
        "frontier_kinds": [list(item.kinds) for item in plan.candidates],
        "frontier_counts": dict(zip(STAGE_NAMES, plan.frontier_negative_counts)),
        "positive_active": dict(zip(STAGE_NAMES, plan.positive_active)),
        "format_violation_count": sum(not item.valid for item in validations),
        "format_reasons": [item.reason for item in validations],
        "format_credit_total_per_candidate": [
            FORMAT_ADV_TOTAL if not item.valid else None for item in validations
        ],
        "hierarchy_gated_for_format": all(
            item.valid or candidate.credits == (0.0, 0.0, 0.0, 0.0)
            for item, candidate in zip(validations, plan.candidates)
        ),
        "alignment_valid": all(
            not item.valid or alignment is not None
            for item, alignment in zip(validations, alignments)
        ),
        "stage_tokens": stage_tokens,
    }
    del old_vector, frontier_vector
    return result


def add_stage_decomposition(model, parameters, batch, old_logps, result):
    stages = {}
    for name, tokens in result.pop("stage_tokens").items():
        if not bool((tokens != 0).any()):
            stages[name] = {"active": False, "grad_norm": None, "over_full": None}
            continue
        loss, norm, vector, finite = isolated_gradient(
            model, parameters, lambda values=tokens: policy_loss(model, batch, old_logps, values)
        )
        stages[name] = {
            "active": True,
            "loss": loss,
            "grad_norm": norm,
            "over_full": norm / result["frontier_grad_norm"] if result["frontier_grad_norm"] else None,
            "finite": finite,
        }
        del vector
    result["stage_decomposition"] = stages


def numeric(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "n": len(values), "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "median": statistics.median(values),
    }


def audit_policy(policy_name, adapter, selected, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_policy(adapter, args.device)
    lora, base = parameter_sets(model)
    lora_before = tensor_checksum(lora)
    base_version_before = base_version_fingerprint(base)
    base_checksum_before = tensor_checksum(base)
    batches = []
    groups = []
    hard_failures = []
    for index, record in enumerate(selected):
        batch = make_batch(record, model, tokenizer, args.max_new_tokens)
        result = audit_group(model, tokenizer, lora, batch)
        result["audit_index"] = index
        batches.append(batch)
        groups.append(result)
        if not result["old_grad_finite"] or not result["frontier_grad_finite"]:
            hard_failures.append(f"NONFINITE_GRAD_GROUP_{index}")
        if not result["alignment_valid"]:
            hard_failures.append(f"TOKEN_ALIGNMENT_GROUP_{index}")
        if not result["hierarchy_gated_for_format"]:
            hard_failures.append(f"FORMAT_HIERARCHY_OVERLAP_GROUP_{index}")
        print(json.dumps({
            "policy": policy_name,
            "group": index,
            "label": result["selection_label"],
            "old_grad": result["old_grad_norm"],
            "frontier_grad": result["frontier_grad_norm"],
            "frontier_counts": result["frontier_counts"],
            "format": result["format_violation_count"],
        }), flush=True)

    representative_indices = set()
    actual_format = next(
        (i for i, result in enumerate(groups) if result["format_violation_count"] > 0),
        None,
    )
    if actual_format is not None:
        representative_indices.add(actual_format)
    for stage, label in REPRESENTATIVE_LABELS.items():
        if stage == "format":
            continue
        matches = [i for i, result in enumerate(groups) if result["selection_label"] == label]
        if matches:
            representative_indices.add(matches[0])
    for index, (batch, result) in enumerate(zip(batches, groups)):
        if index in representative_indices:
            with torch.no_grad():
                old_logps = completion_logps(model, batch).detach()
            add_stage_decomposition(model, lora, batch, old_logps, result)
        else:
            result.pop("stage_tokens")

    model.zero_grad(set_to_none=True)
    base_grad_names = [name for name, value in base if value.grad is not None]
    lora_after = tensor_checksum(lora)
    base_version_after = base_version_fingerprint(base)
    base_checksum_after = tensor_checksum(base)
    if base_grad_names:
        hard_failures.append("BASE_GRADIENT_PRESENT")
    if lora_before != lora_after:
        hard_failures.append("LORA_PARAMETER_CHANGED")
    if base_version_before != base_version_after or base_checksum_before != base_checksum_after:
        hard_failures.append("BASE_PARAMETER_CHANGED")
    payload = {
        "policy_state": policy_name,
        "adapter": adapter,
        "groups_completed": len(groups),
        "optimizer_steps": 0,
        "lora_checksum_before": lora_before,
        "lora_checksum_after": lora_after,
        "base_version_before": base_version_before,
        "base_version_after": base_version_after,
        "base_checksum_before": base_checksum_before,
        "base_checksum_after": base_checksum_after,
        "base_gradient_names": base_grad_names,
        "parameters_changed": bool(
            lora_before != lora_after
            or base_version_before != base_version_after
            or base_checksum_before != base_checksum_after
        ),
        "hard_failures": hard_failures,
        "summary": {
            "old_grad_norm": numeric(item["old_grad_norm"] for item in groups),
            "frontier_grad_norm": numeric(item["frontier_grad_norm"] for item in groups),
            "frontier_over_old": numeric(item["frontier_over_old"] for item in groups),
            "cosine_old_frontier": numeric(item["cosine_old_frontier"] for item in groups),
        },
        "groups": groups,
    }
    del batches, groups, lora, base, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    selection = json.loads(args.selection_file.read_text(encoding="utf-8"))[
        "selected_paired_audit_groups"
    ]
    if len(selection) != 10 or len({item["group_id"] for item in selection}) != 10:
        raise RuntimeError("fixed paired-audit selection must contain 10 unique groups")
    if not args.hier_checkpoint.is_dir():
        raise FileNotFoundError(args.hier_checkpoint)
    policies = [
        audit_policy("fresh_original_BATA", FRESH_ADAPTER, selection, args),
        audit_policy("hier_v1_checkpoint_1250", str(args.hier_checkpoint), selection, args),
    ]
    hard_failures = [
        f"{policy['policy_state']}:{failure}"
        for policy in policies
        for failure in policy["hard_failures"]
    ]
    payload = {
        "type": "gr_rec_nothink_frontier_v1_paired_gradient_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": "12240c1bf45c22ac6222b1bbe792b8332428beea",
        "base": BASE,
        "seed": args.seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "g": 8,
        "fixed_group_count": 10,
        "generation_contract": "one generation per policy_state x G8; shared immutable completion for both objectives",
        "optimizer_steps": 0,
        "parameters_changed": any(policy["parameters_changed"] for policy in policies),
        "hard_fail": bool(hard_failures),
        "hard_failures": hard_failures,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "hard_fail": payload["hard_fail"],
        "hard_failures": hard_failures,
        "optimizer_steps": 0,
        "parameters_changed": payload["parameters_changed"],
    }, ensure_ascii=False, indent=2))
    if payload["hard_fail"]:
        raise RuntimeError("FRONTIER_PAIRED_AUDIT_HARD_FAIL")
    return payload


if __name__ == "__main__":
    main()
