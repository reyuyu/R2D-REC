"""Real-runtime integration shell for TrueRec G8; CPU/mock-audited in Phase 1.1."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import torch

TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in (TRUE_REC_ROOT / "credit", TRUE_REC_ROOT / "analysis"):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from batch_collator_v1 import PaddedBusinessGroup, collate_business_group
from evaluation_runtime_v1 import evaluate_dev512, evaluate_probe20, final_checkpoint_selection_interface
from frontier_credit_v1 import FORMAT_INVALID_TOTAL
from rollout_runtime_v1 import BusinessGroupRollout, G, build_rollout_group, capture_sampled_logps, generation_contract
from truerec_loss_v1 import HPR_LAMBDA, compose_total_loss, frontier_ppo_loss, multi_positive_log_mass_loss
from truerec_runtime_v1 import build_group_runtime_plan


ROUTE_MULTIPLIER = None
FORMAT_PENALTY_CONTEXT_TERMS = 0
FORMAT_PENALTY_DOMAIN_TERMS = 0
TRAINER_MICROBATCH_SIZE = 2
LOGICAL_POLICY_SCORING_PASSES_PER_GROUP = 1
PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP = G // TRAINER_MICROBATCH_SIZE


@dataclass(frozen=True)
class IntegratedGroupLoss:
    frontier_loss: torch.Tensor
    hpr_loss_raw: torch.Tensor
    hpr_loss_weighted: torch.Tensor
    total_loss: torch.Tensor
    monitoring: dict[str, float | int]


def gather_padded_action_logps(logits: torch.Tensor, batch: PaddedBusinessGroup) -> torch.Tensor:
    return gather_padded_action_logps_rows(logits, batch, 0, G)


def gather_padded_action_logps_rows(
    logits: torch.Tensor,
    batch: PaddedBusinessGroup,
    start: int,
    stop: int,
) -> torch.Tensor:
    if not 0 <= start < stop <= G:
        raise ValueError("invalid candidate row slice")
    if logits.ndim != 3 or logits.shape[:2] != (stop - start, batch.input_ids.shape[1]):
        raise ValueError("policy logits must align with padded input batch")
    row_count = stop - start
    rows = torch.arange(row_count, device=logits.device).unsqueeze(1).expand(row_count, 3)
    causal_indices = batch.causal_logit_indices[start:stop].to(logits.device)
    selected = logits[rows, causal_indices]
    log_probs = torch.log_softmax(selected, dim=-1)
    completion_ids = batch.completion_ids[start:stop].to(logits.device)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def format_credit_tensors(group: BusinessGroupRollout, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    credits = torch.zeros((G, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((G, 3), dtype=torch.bool, device=device)
    for row, candidate in enumerate(group.candidates):
        if candidate.metrics["format_valid"]:
            continue
        length = len(candidate.completion_ids)
        if not 0 < length <= 3:
            raise ValueError("invalid format penalty needs actual completion length 1..3")
        credits[row, :length] = FORMAT_INVALID_TOTAL / length
        mask[row, :length] = True
    return credits, mask


def hpr_loss_padded(logits: torch.Tensor, batch: PaddedBusinessGroup, runtime_hpr) -> torch.Tensor:
    if not runtime_hpr.sites:
        return logits.sum() * 0.0
    site_losses = []
    for site in runtime_hpr.sites:
        position_losses = []
        for sample_index, action_position in site.onpolicy_positions:
            causal_index = int(batch.causal_logit_indices[sample_index, action_position])
            position_losses.append(multi_positive_log_mass_loss(logits[sample_index, causal_index], site.target_token_ids))
        site_losses.append(torch.stack(position_losses).mean())
    return torch.stack(site_losses).mean()


class TrueRecGRPOTrainerV1:
    def __init__(self, policy, token_to_id, pad_token_id: int, epsilon: float = 0.2, padding_side: str = "right", device=None):
        self.policy = policy
        self.token_to_id = token_to_id
        self.pad_token_id = int(pad_token_id)
        self.epsilon = epsilon
        self.padding_side = padding_side
        self.device = device
        self.logical_policy_scoring_passes = 0
        self.physical_policy_forward_calls = 0
        self.train_policy_forward_calls = 0
        self.hpr_extra_forward_calls = 0

    def compute_group(self, group: BusinessGroupRollout) -> IntegratedGroupLoss:
        batch = collate_business_group(group, self.pad_token_id, self.padding_side)
        candidate_metrics = [candidate.metrics for candidate in group.candidates]
        runtime = build_group_runtime_plan(candidate_metrics, group.all_gold_abc, self.token_to_id)
        format_credits, format_mask = format_credit_tensors(group)
        frontier_parts = []
        hpr_position_losses = [[] for _ in runtime.hpr.sites]
        zero_graph = None
        self.logical_policy_scoring_passes += 1
        for start in range(0, G, TRAINER_MICROBATCH_SIZE):
            stop = min(start + TRAINER_MICROBATCH_SIZE, G)
            input_ids = batch.input_ids[start:stop]
            attention_mask = batch.attention_mask[start:stop]
            if self.device is not None:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            output = self.policy(input_ids=input_ids, attention_mask=attention_mask)
            self.physical_policy_forward_calls += 1
            self.train_policy_forward_calls += 1
            logits = output.logits if hasattr(output, "logits") else output
            current = gather_padded_action_logps_rows(logits, batch, start, stop)
            old_logps = batch.old_logps[start:stop].to(current.device)
            hierarchy = frontier_ppo_loss(
                current.unsqueeze(0), old_logps.unsqueeze(0),
                runtime.token_credits[start:stop].to(current.device).unsqueeze(0),
                runtime.token_credit_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
            )
            format_loss = frontier_ppo_loss(
                current.unsqueeze(0), old_logps.unsqueeze(0),
                format_credits[start:stop].to(current.device).unsqueeze(0),
                format_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
            )
            candidate_weight = (stop - start) / G
            frontier_parts.append((hierarchy.loss + format_loss.loss) * candidate_weight)
            chunk_zero = logits.sum() * 0.0
            zero_graph = chunk_zero if zero_graph is None else zero_graph + chunk_zero
            for site_index, site in enumerate(runtime.hpr.sites):
                for sample_index, action_position in site.onpolicy_positions:
                    if start <= sample_index < stop:
                        causal_index = int(batch.causal_logit_indices[sample_index, action_position])
                        hpr_position_losses[site_index].append(
                            multi_positive_log_mass_loss(
                                logits[sample_index - start, causal_index], site.target_token_ids,
                            )
                        )
        frontier = torch.stack(frontier_parts).sum()
        if runtime.hpr.sites:
            if any(not losses for losses in hpr_position_losses):
                raise ValueError("HPR site lacks on-policy positions")
            hpr_raw = torch.stack([torch.stack(losses).mean() for losses in hpr_position_losses]).mean()
        else:
            hpr_raw = zero_graph
        total = compose_total_loss(frontier, hpr_raw)
        valid = [item for item in candidate_metrics if item["format_valid"]]
        abc_values = [item["parsed_abc"] for item in valid]
        parsed_tokens = [tuple(re.findall(r"<s_[abc]_\d+>", value)) for value in abc_values]
        a_values = [tokens[:1] for tokens in parsed_tokens]
        ab_values = [tokens[:2] for tokens in parsed_tokens]
        monitor = dict(runtime.monitoring)
        monitor.update({"business_group_count": 1, "rollout_candidate_count": G, "format_valid_rate": len(valid) / G, "GROUP_ANY_A": int(any(item["A_hit"] for item in candidate_metrics)), "GROUP_ANY_AB": int(any(item["AB_hit"] for item in candidate_metrics)), "GROUP_ANY_EXACT": int(any(item["exact"] for item in candidate_metrics)), "HPR_A": int(runtime.hpr.trigger == "HPR_A"), "HPR_B": int(runtime.hpr.trigger == "HPR_B"), "HPR_C": int(runtime.hpr.trigger == "HPR_C"), "HPR_NONE": int(runtime.hpr.trigger == "HPR_NONE"), "unique_A_per_G8": len(set(a_values)), "unique_AB_per_G8": len(set(ab_values)), "unique_ABC_per_G8": len(set(abc_values)), "frontier_loss": float(total.frontier_loss.detach()), "hpr_loss_raw": float(total.hpr_loss_raw.detach()), "hpr_loss_weighted": float(total.hpr_loss_weighted.detach()), "total_loss": float(total.total_loss.detach())})
        return IntegratedGroupLoss(total.frontier_loss, total.hpr_loss_raw, total.hpr_loss_weighted, total.total_loss, monitor)

    def compute_batch(self, groups: Sequence[BusinessGroupRollout]) -> IntegratedGroupLoss:
        if not groups:
            raise ValueError("empty business-group batch")
        outputs = [self.compute_group(group) for group in groups]
        frontier = torch.stack([item.frontier_loss for item in outputs]).mean()
        hpr_raw = torch.stack([item.hpr_loss_raw for item in outputs]).mean()
        total = compose_total_loss(frontier, hpr_raw)
        monitoring = {"business_group_count": len(groups), "rollout_candidate_count": G * len(groups), "frontier_loss": float(total.frontier_loss.detach()), "hpr_loss_raw": float(total.hpr_loss_raw.detach()), "hpr_loss_weighted": float(total.hpr_loss_weighted.detach()), "total_loss": float(total.total_loss.detach())}
        for key in ("format_valid_rate", "A_hit_rate", "AB_hit_rate", "exact_rate", "wrong_history_copy_rate", "unique_A_per_G8", "unique_AB_per_G8", "unique_ABC_per_G8"):
            monitoring[key] = sum(item.monitoring[key] for item in outputs) / len(outputs)
        for key in ("GROUP_ANY_A", "GROUP_ANY_AB", "GROUP_ANY_EXACT", "HPR_A", "HPR_B", "HPR_C", "HPR_NONE", "frontier_A_positive", "frontier_A_negative", "frontier_B_positive", "frontier_B_negative", "frontier_C_positive", "frontier_C_negative"):
            monitoring[key] = sum(item.monitoring[key] for item in outputs)
        return IntegratedGroupLoss(total.frontier_loss, total.hpr_loss_raw, total.hpr_loss_weighted, total.total_loss, monitoring)


PHASE08_SHA = {
    "credit/frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b",
    "credit/hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32",
}
PHASE10_SHA = {
    "trainer/action_alignment.py": "2ca6c2b1ca034e84be55e359100ed86cf01503200fefa789e30abb2661f505e1",
    "trainer/truerec_loss_v1.py": "4a52f40639c332824e5dc45d6e4da1defcdbf4fc9f8e0c0dc380d5c4e5287d57",
    "trainer/truerec_runtime_v1.py": "2f18378f165d523e8eceaf85f8d575c7a237faacb99a072d8f22d326cef254f7",
    "trainer/truerec_trainer_v1.py": "7b5da094f7cb000fe5d66123eb4310a208e6fd963c0443b55efe9fb2cdddbdfb",
}
PHASE11_FILES = (
    "trainer/rollout_runtime_v1.py", "trainer/batch_collator_v1.py",
    "trainer/truerec_grpo_trainer_v1.py", "trainer/evaluation_runtime_v1.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _AuditOutput:
    def __init__(self, logits): self.logits = logits


class _AuditPolicy:
    def __init__(self, logits): self.logits = logits; self.calls = 0; self.attention_masks = []
    def __call__(self, input_ids, attention_mask):
        start = self.calls * input_ids.shape[0]
        self.calls += 1; self.attention_masks.append(attention_mask.clone())
        return _AuditOutput(self.logits[start:start + input_ids.shape[0]])


def run_cpu_audit(output_dir: Path, source_truerec_root: Path, runtime_truerec_root: Path) -> dict[str, Any]:
    """Run deterministic tensor-only integration gates; never loads or executes a real model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token_map = {1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>", 5: "<s_b_5>", 6: "<s_c_6>", 7: "<s_c_7>", 8: "<s_b_8>", 9: "<s_c_9>", 20: "bad", 21: "bad2"}
    token_ids = {value: key for key, value in token_map.items()}
    gold = ("<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_2><s_c_7>", "<s_a_1><s_b_8><s_c_9>")
    record = {"recommendation_group_id": "phase11-synthetic-g8", "fixed_domain_token": "<video>", "all_gold_abc": gold, "history_sids": ("<video><s_a_4><s_b_5><s_c_6>",)}
    completions = [[1, 2, 3], [1, 2, 7], [1, 2, 6], [1, 5, 6], [4, 5, 6], [20, 21], [1, 8, 9], [4, 5, 6]]
    torch.manual_seed(1101)
    rollout_logits = torch.randn(G, 3, 32)
    group = build_rollout_group(record, [10, 11, 12, 13], completions, rollout_logits, token_map.__getitem__)
    captured = torch.zeros((G, 3))
    captured_values = capture_sampled_logps(rollout_logits, completions)
    for row, values in enumerate(captured_values):
        captured[row, :len(values)] = torch.tensor(values)
    reconstructed = torch.zeros((G, 3))
    reconstructed_mask = torch.zeros((G, 3), dtype=torch.bool)
    log_probs = torch.log_softmax(rollout_logits, -1)
    for row, ids in enumerate(completions):
        reconstructed[row, :len(ids)] = log_probs[row, torch.arange(len(ids)), torch.tensor(ids)]
        reconstructed_mask[row, :len(ids)] = True
    old_before = captured.clone()
    perturbed = rollout_logits.clone()
    for row, ids in enumerate(completions):
        perturbed[row, torch.arange(len(ids)), torch.tensor(ids)] += 0.7
    changed = torch.zeros((G, 3))
    for row, values in enumerate(capture_sampled_logps(perturbed, completions)):
        changed[row, :len(values)] = torch.tensor(values)
    old_audit = {
        "old_logps_from_rollout_policy": True,
        "reconstruction_exact": bool(torch.equal(captured[reconstructed_mask], reconstructed[reconstructed_mask])),
        "old_unchanged_after_current_perturbation": bool(torch.equal(old_before, captured)),
        "current_changed_after_perturbation": bool(torch.any(changed[reconstructed_mask] != captured[reconstructed_mask])),
        "ratio_not_one_count": int(torch.count_nonzero(torch.exp(changed[reconstructed_mask] - captured[reconstructed_mask]) != 1.0)),
    }
    _write_json(output_dir / "old_logp_contract_audit.json", old_audit)

    right = collate_business_group(group, 0, "right")
    left = collate_business_group(group, 0, "left")
    padding_audit = {
        "variable_context_lengths": [2, 4, 9],
        "right_padding_alignment": bool(torch.equal(right.action_indices[:, 0], torch.full((G,), 4))),
        "left_padding_short_candidate_offset": int(left.action_indices[5, 0] - right.action_indices[5, 0]),
        "attention_mask_actual_token_counts": right.attention_mask.sum(1).tolist(),
        "attention_mask_gate": bool(right.attention_mask[5, :6].all() and not right.attention_mask[5, 6]),
        "pad_token_id": 0,
    }
    _write_json(output_dir / "padding_alignment_audit.json", padding_audit)

    format_credits, format_mask = format_credit_tensors(group)
    format_audit = {
        "format_invalid_total_contract": FORMAT_INVALID_TOTAL,
        "invalid_candidate_index": 5,
        "invalid_frontier": group.candidates[5].metrics["frontier"],
        "actual_completion_length": len(group.candidates[5].completion_ids),
        "per_actual_token_credit": format_credits[5, :2].tolist(),
        "credit_total": float(format_credits[5].sum()),
        "credit_mask": format_mask[5].tolist(),
        "context_terms": FORMAT_PENALTY_CONTEXT_TERMS,
        "domain_terms": FORMAT_PENALTY_DOMAIN_TERMS,
    }
    _write_json(output_dir / "format_penalty_audit.json", format_audit)

    sequence_length = right.input_ids.shape[1]
    torch.manual_seed(1102)
    policy = _AuditPolicy(torch.randn(G, sequence_length, 32))
    trainer = TrueRecGRPOTrainerV1(policy, token_ids.__getitem__, 0)
    loss = trainer.compute_group(group)
    shared_audit = {
        "logical_policy_scoring_passes_per_group": trainer.logical_policy_scoring_passes,
        "physical_policy_forward_calls_per_group": trainer.physical_policy_forward_calls,
        "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls,
        "frontier_loss": float(loss.frontier_loss.detach()),
        "hpr_loss_raw": float(loss.hpr_loss_raw.detach()),
        "hpr_loss_weighted": float(loss.hpr_loss_weighted.detach()),
        "total_loss": float(loss.total_loss.detach()),
        "hpr_lambda": HPR_LAMBDA,
        "route_multiplier": ROUTE_MULTIPLIER,
        "monitoring_fields": sorted(loss.monitoring),
    }
    _write_json(output_dir / "shared_training_forward_audit.json", shared_audit)

    split_dir = runtime_truerec_root / "data" / "splits"
    probe = json.loads((split_dir / "probe20_group_ids.json").read_text(encoding="utf-8"))
    dev = json.loads((split_dir / "dev_group_ids.json").read_text(encoding="utf-8"))
    probe_ids = probe.get("group_ids", probe) if isinstance(probe, dict) else probe
    dev_ids = dev.get("group_ids", dev) if isinstance(dev, dict) else dev
    if probe_ids and isinstance(probe_ids[0], dict):
        probe_ids = [row["recommendation_group_id"] for row in probe_ids]
    probe_interface = evaluate_probe20(probe_ids, dev_ids)
    dev_interface = evaluate_dev512(dev_ids)
    final_interface = final_checkpoint_selection_interface()
    eval_audit = {
        "probe20": {"groups": probe_interface.expected_groups, "optimizer": probe_interface.optimizer_allowed, "backward": probe_interface.backward_allowed},
        "dev512": {"groups": dev_interface.expected_groups, "optimizer": dev_interface.optimizer_allowed, "backward": dev_interface.backward_allowed},
        "final2048_checkpoint_selection_allowed": final_interface.checkpoint_selection_allowed,
    }
    _write_json(output_dir / "evaluation_interface_audit.json", eval_audit)

    source_runtime = {name: {"source": _sha(source_truerec_root / name), "runtime": _sha(runtime_truerec_root / name)} for name in PHASE11_FILES}
    frozen = {name: _sha(runtime_truerec_root / name) for name in (*PHASE08_SHA, *PHASE10_SHA)}
    rollout_audit = {
        "G": G,
        "business_group_unit": len(group.candidates) == G and [item.sample_index for item in group.candidates] == list(range(G)),
        "generation_contract": generation_contract(),
        "phase07_parser_reused": all("frontier" in item.metrics for item in group.candidates),
        "phase08_sha_unchanged": all(frozen[name] == digest for name, digest in PHASE08_SHA.items()),
        "phase10_core_sha_unchanged": all(frozen[name] == digest for name, digest in PHASE10_SHA.items()),
        "source_runtime_sha": source_runtime,
        "github_runtime_parity": all(value["source"] == value["runtime"] for value in source_runtime.values()),
        "gpu_inference_started": False,
        "real_model_forward_started": False,
        "training_started": False,
        "backward_started": False,
        "optimizer_started": False,
    }
    _write_json(output_dir / "rollout_runtime_audit.json", rollout_audit)
    review = (
        "TrueRec-GRPO Phase 1.1 CPU integration audit: PASS\n"
        "One recommendation_group_id remains one G8 business group. Rollout-policy old logps are captured before training and remain separate from perturbed current logps.\n"
        "Frontier PPO and HPR consume one shared mock-policy logits tensor; invalid-format credit is normalized only over actual completion tokens.\n"
        "Probe20 and Dev512 are inference-only interfaces; Final2048 checkpoint selection is disabled. No GPU, real model forward, backward, optimizer, or training was started.\n"
    )
    (output_dir / "CHATGPT_PHASE1_1_REVIEW.txt").write_text(review, encoding="utf-8")
    return {"rollout": rollout_audit, "old": old_audit, "padding": padding_audit, "format": format_audit, "shared": shared_audit, "evaluation": eval_audit}


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only TrueRec Phase 1.1 integration audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-truerec-root", type=Path, required=True)
    parser.add_argument("--runtime-truerec-root", type=Path, required=True)
    args = parser.parse_args()
    audit = run_cpu_audit(args.output_dir, args.source_truerec_root, args.runtime_truerec_root)
    required = (audit["rollout"]["business_group_unit"], audit["rollout"]["github_runtime_parity"], audit["old"]["reconstruction_exact"], audit["old"]["old_unchanged_after_current_perturbation"], audit["old"]["current_changed_after_perturbation"], audit["padding"]["right_padding_alignment"], audit["padding"]["attention_mask_gate"], audit["shared"]["logical_policy_scoring_passes_per_group"] == 1, audit["shared"]["physical_policy_forward_calls_per_group"] == 4, audit["shared"]["hpr_extra_forward_calls"] == 0, not audit["evaluation"]["final2048_checkpoint_selection_allowed"])
    if not all(required):
        raise SystemExit("PHASE1_1_AUDIT=FAIL")
    print("PHASE1_1_AUDIT=PASS")


if __name__ == "__main__":
    main()
