"""CPU-only full-G8 versus microbatch=2 Trainer equivalence audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "credit", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from batch_collator_v1 import collate_business_group  # noqa: E402
from rollout_runtime_v1 import G, PPO_OLD_LOGP_SOURCE, build_rollout_group  # noqa: E402
from truerec_grpo_trainer_v1 import (  # noqa: E402
    LOGICAL_POLICY_SCORING_PASSES_PER_GROUP,
    PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP,
    TRAINER_MICROBATCH_SIZE,
    TrueRecGRPOTrainerV1,
    format_credit_tensors,
    gather_padded_action_logps,
    hpr_loss_padded,
)
from truerec_loss_v1 import HPR_LAMBDA, compose_total_loss, frontier_ppo_loss  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


VOCAB_SIZE = 32
TOKEN_MAP = {
    1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>",
    5: "<s_b_5>", 6: "<s_c_6>", 7: "<s_c_7>", 8: "<s_b_8>",
    9: "<s_c_9>", 20: "bad", 21: "also_bad",
}
TOKEN_IDS = {value: key for key, value in TOKEN_MAP.items()}
GOLD = ("<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_2><s_c_7>", "<s_a_1><s_b_8><s_c_9>")
CASES = {
    "HPR_A": [[4, 5, 6]] * G,
    "HPR_B": [[1, 5, 6]] * G,
    "HPR_C": [[1, 2, 6]] * G,
    "HPR_NONE": [[1, 2, 3]] * G,
    "FORMAT_INVALID": [[1, 2, 3]] * (G - 1) + [[20, 21]],
}


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, 7, dtype=torch.float64)
        self.output = torch.nn.Linear(7, VOCAB_SIZE, bias=True, dtype=torch.float64)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _id_to_token(token_id: int) -> str:
    return TOKEN_MAP.get(int(token_id), f"token_{token_id}")


def _group(case_name: str, completions: list[list[int]]):
    record = {
        "recommendation_group_id": f"phase1-2c2-{case_name.lower()}",
        "fixed_domain_token": "<video>",
        "all_gold_abc": GOLD,
        "history_sids": ("<video><s_a_4><s_b_5><s_c_6>",),
    }
    generator = torch.Generator().manual_seed(1202)
    rollout_logits = torch.randn(G, 3, VOCAB_SIZE, generator=generator)
    return build_rollout_group(record, [10, 11, 12, 13], completions, rollout_logits, _id_to_token)


def full_g8_reference(policy: TinyPolicy, group):
    """The pre-microbatch Trainer expression, composed only from frozen helpers."""
    batch = collate_business_group(group, 0, "right")
    runtime = build_group_runtime_plan(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc, TOKEN_IDS.__getitem__,
    )
    logits = policy(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
    current = gather_padded_action_logps(logits, batch)
    hierarchy = frontier_ppo_loss(
        current.unsqueeze(0), batch.old_logps.to(current).unsqueeze(0),
        runtime.token_credits.to(current).unsqueeze(0), runtime.token_credit_mask.to(current.device).unsqueeze(0), 0.2,
    )
    format_credits, format_mask = format_credit_tensors(group, current.device)
    format_loss = frontier_ppo_loss(
        current.unsqueeze(0), batch.old_logps.to(current).unsqueeze(0),
        format_credits.to(current).unsqueeze(0), format_mask.unsqueeze(0), 0.2,
    )
    hpr_raw = hpr_loss_padded(logits, batch, runtime.hpr)
    total = compose_total_loss(hierarchy.loss + format_loss.loss, hpr_raw)
    return total, runtime


def _max_gradient_difference(reference: TinyPolicy, candidate: TinyPolicy) -> float:
    maximum = 0.0
    for (name_a, parameter_a), (name_b, parameter_b) in zip(reference.named_parameters(), candidate.named_parameters()):
        if name_a != name_b or parameter_a.grad is None or parameter_b.grad is None:
            raise AssertionError("trainable gradient inventory mismatch")
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad)
        maximum = max(maximum, float((parameter_a.grad - parameter_b.grad).abs().max()))
    return maximum


def run_audit(output_dir: Path | None = None) -> dict[str, Any]:
    torch.manual_seed(12021)
    seed_policy = TinyPolicy()
    case_results = {}
    max_frontier = max_hpr = max_total = max_gradient = 0.0
    for case_name, completions in CASES.items():
        group = _group(case_name, completions)
        reference_policy = copy.deepcopy(seed_policy)
        candidate_policy = copy.deepcopy(seed_policy)
        reference, runtime = full_g8_reference(reference_policy, group)
        trainer = TrueRecGRPOTrainerV1(candidate_policy, TOKEN_IDS.__getitem__, 0)
        candidate = trainer.compute_group(group)
        torch.testing.assert_close(reference.frontier_loss, candidate.frontier_loss)
        torch.testing.assert_close(reference.hpr_loss_raw, candidate.hpr_loss_raw)
        torch.testing.assert_close(reference.total_loss, candidate.total_loss)
        reference.total_loss.backward()
        candidate.total_loss.backward()
        gradient_difference = _max_gradient_difference(reference_policy, candidate_policy)
        differences = {
            "frontier": float((reference.frontier_loss - candidate.frontier_loss).abs()),
            "hpr": float((reference.hpr_loss_raw - candidate.hpr_loss_raw).abs()),
            "total": float((reference.total_loss - candidate.total_loss).abs()),
            "gradient": gradient_difference,
        }
        max_frontier = max(max_frontier, differences["frontier"])
        max_hpr = max(max_hpr, differences["hpr"])
        max_total = max(max_total, differences["total"])
        max_gradient = max(max_gradient, differences["gradient"])
        case_results[case_name] = {
            "status": "PASS",
            "runtime_trigger": runtime.hpr.trigger,
            "differences": differences,
            "logical_policy_scoring_passes": trainer.logical_policy_scoring_passes,
            "physical_policy_forward_calls": trainer.physical_policy_forward_calls,
            "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls,
        }
        if (reference_policy.forward_calls, candidate_policy.forward_calls) != (1, 4):
            raise AssertionError("full/microbatch forward-count contract failed")

    audit = {
        "status": "PASS",
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "G": G,
        "business_group_unit_preserved": True,
        "logical_policy_scoring_passes_per_group": LOGICAL_POLICY_SCORING_PASSES_PER_GROUP,
        "physical_policy_forward_calls_per_group": PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP,
        "hpr_extra_forward_calls": 0,
        "frontier_reduction_preserved": True,
        "hpr_site_reduction_preserved": True,
        "hpr_group_reduction_preserved": True,
        "cases": case_results,
        "max_abs_differences": {
            "frontier_loss": max_frontier,
            "hpr_loss": max_hpr,
            "total_loss": max_total,
            "gradient": max_gradient,
        },
        "frontier_loss_equivalent": True,
        "hpr_loss_equivalent": True,
        "total_loss_equivalent": True,
        "gradient_equivalence": True,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "hpr_lambda": HPR_LAMBDA,
        "execution": {
            "gpu_inference_started": False,
            "real_model_backward_started": False,
            "optimizer_steps": 0,
            "next_experiment_started": False,
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trainer_g8_microbatch_equivalence.json").write_text(
            json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (output_dir / "CHATGPT_PHASE1_2C2_REVIEW.txt").write_text(
            "TrueRec-GRPO Phase 1.2C2 CPU mathematical equivalence: PASS\n"
            "The logical unit remains one complete G8 business group. The full runtime plan is built once, while policy execution is physically split into four 2-row forwards.\n"
            "Frontier, HPR site/group reductions, total loss, and tiny-policy gradients match the full-G8 reference for HPR_A/B/C/NONE and invalid-format cases. No GPU, real-model backward, optimizer, or training ran.\n",
            encoding="utf-8",
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = run_audit(args.output_dir)
    print(f"PHASE1_2C2_CPU_AUDIT={audit['status']}")


if __name__ == "__main__":
    main()
