#!/usr/bin/env python3
"""Complete format-only decomposition from saved immutable audit completions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import audit_gpu_paired_frontier as audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-zero-step-gpu-audit", action="store_true")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not args.execute_zero_step_gpu_audit:
        parser.error("explicit zero-step authorization flag is required")
    return args


def rebuild_batch(record, saved, model, tokenizer):
    prompt_ids = tuple(audit.encode_prompt(tokenizer, record["prompt"]))
    generated = [list(ids) for ids in saved["completion_ids"]]
    texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in generated]
    width = max(map(len, generated))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    completions = torch.full((audit.M_NO, width), pad, dtype=torch.long, device=model.device)
    mask = torch.zeros_like(completions)
    for row, ids in enumerate(generated):
        completions[row, : len(ids)] = torch.tensor(ids, device=model.device)
        mask[row, : len(ids)] = 1
    prompts = torch.tensor([prompt_ids] * audit.M_NO, dtype=torch.long, device=model.device)
    gold = tuple(filter(None, (audit.parse_sid(value) for value in record["gold_sids"])))
    predicted = tuple(audit.final_sid(text) for text in texts)
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
        "rewards": tuple(audit.q_reward(sid, gold) for sid in predicted),
        "target_domain": record["target_domain"],
    }


def main(argv=None):
    args = parse_args(argv)
    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    selection = json.loads(audit.SELECTION_FILE.read_text(encoding="utf-8"))[
        "selected_paired_audit_groups"
    ]
    policy = next(
        item for item in payload["policies"]
        if item["policy_state"] == "hier_v1_checkpoint_1250"
    )
    index = next(
        i for i, group in enumerate(policy["groups"])
        if group["format_violation_count"] > 0
    )
    saved = policy["groups"][index]
    model, tokenizer = audit.load_policy(policy["adapter"], args.device)
    lora, base = audit.parameter_sets(model)
    lora_before = audit.tensor_checksum(lora)
    base_version_before = audit.base_version_fingerprint(base)
    batch = rebuild_batch(selection[index], saved, model, tokenizer)
    if [list(ids) for ids in batch["completion_ids_list"]] != saved["completion_ids"]:
        raise RuntimeError("saved completion parity failed")
    frontier, stage_tokens, _plan, validations, _alignments = audit.frontier_tokens(
        batch, tokenizer
    )
    if sum(not item.valid for item in validations) != saved["format_violation_count"]:
        raise RuntimeError("format validation parity failed")
    with torch.no_grad():
        old_logps = audit.completion_logps(model, batch).detach()
    loss, norm, vector, finite = audit.isolated_gradient(
        model,
        lora,
        lambda: audit.policy_loss(model, batch, old_logps, stage_tokens["format"]),
    )
    saved.setdefault("stage_decomposition", {})["format"] = {
        "active": True,
        "loss": loss,
        "grad_norm": norm,
        "over_full": norm / saved["frontier_grad_norm"] if saved["frontier_grad_norm"] else None,
        "finite": finite,
        "reused_saved_completion": True,
    }
    model.zero_grad(set_to_none=True)
    lora_after = audit.tensor_checksum(lora)
    base_version_after = audit.base_version_fingerprint(base)
    base_grad_names = [name for name, value in base if value.grad is not None]
    supplement = {
        "policy_state": policy["policy_state"],
        "audit_index": index,
        "group_id": saved["group_id"],
        "format_violation_count": saved["format_violation_count"],
        "format_advantage_total": audit.FORMAT_ADV_TOTAL,
        "format_only_grad_norm": norm,
        "format_only_over_full": saved["stage_decomposition"]["format"]["over_full"],
        "completion_regenerated": False,
        "optimizer_steps": 0,
        "lora_checksum_before": lora_before,
        "lora_checksum_after": lora_after,
        "base_version_before": base_version_before,
        "base_version_after": base_version_after,
        "base_gradient_names": base_grad_names,
        "parameters_changed": lora_before != lora_after or base_version_before != base_version_after,
    }
    payload["format_stage_supplement"] = supplement
    if not finite or base_grad_names or supplement["parameters_changed"]:
        payload["hard_fail"] = True
        payload["hard_failures"].append("FORMAT_STAGE_SUPPLEMENT_HARD_FAIL")
    args.audit.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(supplement, ensure_ascii=False, indent=2))
    if payload["hard_fail"]:
        raise RuntimeError("FRONTIER_PAIRED_AUDIT_HARD_FAIL")


if __name__ == "__main__":
    main()
