"""Deterministic CPU/autograd audit for the Phase-3A Think runtime contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for dependency in (Path(__file__).resolve().parent, ROOT / "credit", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from frontier_credit_v1 import FORMAT_INVALID_TOTAL  # noqa: E402
from hpr_plan_v1 import split_abc  # noqa: E402
from rollout_runtime_v1 import BusinessGroupRollout, RolloutCandidate  # noqa: E402
from think_trainer_v1 import A_RESCUE_LAMBDA, BC_HPR_LAMBDA, ThinkGRPOTrainerV1  # noqa: E402


G = 8
EPSILON = 0.2
PAD_ID = 0
PAIRED_RECORDS_SHA256 = "0b5c561a97810783d43dd7e41450d48295a7e1ae7e6466c6d59c81b3a2ecd357"
TOKEN_TO_ID = {
    "<s_a_1>": 1, "<s_a_2>": 2, "<s_a_3>": 3, "<s_a_9>": 9,
    "<s_b_1>": 11, "<s_b_2>": 12, "<s_b_3>": 13, "<s_b_9>": 19,
    "<s_c_1>": 21, "<s_c_2>": 22, "<s_c_3>": 23, "<s_c_9>": 29,
}
GOLD3 = (
    "<s_a_1><s_b_1><s_c_1>", "<s_a_2><s_b_2><s_c_2>",
    "<s_a_3><s_b_3><s_c_3>",
)


class TinyCausalPolicy(nn.Module):
    """A real causal mock: logits[t] depend only on non-padding IDs through t."""
    def __init__(self, vocab_size: int = 80, hidden_size: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_ID, dtype=torch.float64)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=True, dtype=torch.float64)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        embedded = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        causal = torch.tanh(torch.cumsum(embedded, dim=1))
        return SimpleNamespace(logits=self.projection(causal))


def abc(a: int, b: int = 1, c: int = 1) -> str:
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def candidate_metrics(value: str | None, gold: Sequence[str], valid: bool = True) -> dict[str, Any]:
    parsed = split_abc(value) if value else None
    gold_parts = [split_abc(item) for item in gold]
    return {
        "format_valid": bool(valid),
        "parsed_abc": value if valid else None,
        "A_hit": bool(valid and parsed and parsed[0] in {item[0] for item in gold_parts}),
        "AB_hit": bool(valid and parsed and parsed[:2] in {item[:2] for item in gold_parts}),
        "exact": bool(valid and parsed and parsed in set(gold_parts)),
        "wrong_history_copy": False,
    }


def make_group(name: str, values: Sequence[str | None], gold: Sequence[str] = GOLD3) -> BusinessGroupRollout:
    if len(values) != G:
        raise ValueError("synthetic group requires G8")
    candidates = []
    for index, value in enumerate(values):
        if value is None:
            completion = (50, 51)
            metrics = candidate_metrics(None, gold, False)
        else:
            completion = tuple(TOKEN_TO_ID[token] for token in split_abc(value))
            metrics = candidate_metrics(value, gold)
        candidates.append(RolloutCandidate(index, completion, tuple(0.0 for _ in completion), metrics))
    return BusinessGroupRollout(name, (60, 61, 62, 63), "<video>", tuple(gold), tuple(candidates))


def deterministic_groups() -> dict[str, BusinessGroupRollout]:
    return {
        "NO_GOLD_A": make_group("no-gold-a", [abc(9, 9, 9)] * 8),
        "A_MODE_COLLAPSE_HPR_B": make_group("collapse-b", [abc(1, 9, 9)] * 5 + [abc(9, 9, 9)] * 3),
        "A_MODE_COLLAPSE_HPR_C": make_group("collapse-c", [abc(1, 1, 9)] * 5 + [abc(9, 9, 9)] * 3),
        "A_MODE_COLLAPSE_HPR_NONE": make_group("collapse-none", [abc(1, 1, 1)] * 5 + [abc(9, 9, 9)] * 3),
        "NO_RESCUE_EXACT": make_group(
            "no-rescue-exact",
            [abc(1, 1, 1), abc(2, 2, 2), abc(3, 3, 3)] + [abc(9, 9, 9)] * 5,
        ),
        "CANDIDATE0_FORMAT_INVALID_RESCUE": make_group(
            "candidate0-invalid", [None] + [abc(1, 9, 9)] * 5 + [abc(9, 9, 9)] * 2,
        ),
    }


def fresh_policy(state=None) -> TinyCausalPolicy:
    torch.manual_seed(3301)
    policy = TinyCausalPolicy()
    if state is not None:
        policy.load_state_dict(state)
    return policy


def gradients(policy: nn.Module) -> torch.Tensor:
    return torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter in policy.parameters()
    ])


def _loss_values(output) -> dict[str, float]:
    names = (
        "semantic_frontier_loss", "format_loss", "frontier_total_loss",
        "a_rescue_loss_raw", "a_rescue_loss_weighted", "bc_hpr_loss_raw",
        "bc_hpr_loss_weighted", "total_loss",
    )
    values = {}
    for name in names:
        value = getattr(output, name, getattr(output, name.replace("loss", "value"), None))
        values[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
    return values


def parity_case(name: str, group: BusinessGroupRollout) -> dict[str, Any]:
    initial = fresh_policy().state_dict()
    full_policy = fresh_policy(initial)
    full_trainer = ThinkGRPOTrainerV1(full_policy, TOKEN_TO_ID.__getitem__, PAD_ID, streaming_microbatch_size=2)
    full = full_trainer.compute_group(group)
    full.total_loss.backward()
    full_grad = gradients(full_policy)
    full_values = _loss_values(full)
    result: dict[str, Any] = {"case": name, "full": full_values, "full_grad_norm": float(full_grad.norm())}
    for microbatch in (1, 2):
        policy = fresh_policy(initial)
        trainer = ThinkGRPOTrainerV1(policy, TOKEN_TO_ID.__getitem__, PAD_ID, streaming_microbatch_size=microbatch)
        streamed = trainer.backward_group_streaming(group)
        grad = gradients(policy)
        values = _loss_values(streamed)
        delta = full_grad - grad
        prefix = f"mb{microbatch}"
        result[prefix] = values
        result[f"{prefix}_grad_norm"] = float(grad.norm())
        result[f"full_vs_{prefix}_loss_max_abs_diff"] = max(
            abs(full_values[key] - values[key]) for key in full_values
        )
        result[f"full_vs_{prefix}_grad_max_abs_diff"] = float(delta.abs().max())
        result[f"full_vs_{prefix}_grad_relative_norm_diff"] = float(
            delta.norm() / full_grad.norm().clamp_min(torch.finfo(torch.float64).tiny)
        )
        result[f"{prefix}_forward_calls"] = trainer.physical_policy_forward_calls
        result[f"{prefix}_backward_calls"] = trainer.streaming_backward_calls
    return result


def run_audit(source_commit: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cases = [parity_case(name, group) for name, group in deterministic_groups().items()]
    loss_mb1 = max(item["full_vs_mb1_loss_max_abs_diff"] for item in cases)
    loss_mb2 = max(item["full_vs_mb2_loss_max_abs_diff"] for item in cases)
    grad_mb1 = max(item["full_vs_mb1_grad_max_abs_diff"] for item in cases)
    grad_mb2 = max(item["full_vs_mb2_grad_max_abs_diff"] for item in cases)
    tolerance = 1e-10
    parity = {
        "dtype": "float64", "loss_tolerance": tolerance, "gradient_tolerance": tolerance,
        "cases": cases,
        "max_loss_abs_diff_mb1": loss_mb1, "max_loss_abs_diff_mb2": loss_mb2,
        "max_gradient_abs_diff_mb1": grad_mb1, "max_gradient_abs_diff_mb2": grad_mb2,
        "full_vs_streaming_mb1_loss_parity": "PASS" if loss_mb1 <= tolerance else "FAIL",
        "full_vs_streaming_mb2_loss_parity": "PASS" if loss_mb2 <= tolerance else "FAIL",
        "full_vs_streaming_mb1_grad_parity": "PASS" if grad_mb1 <= tolerance else "FAIL",
        "full_vs_streaming_mb2_grad_parity": "PASS" if grad_mb2 <= tolerance else "FAIL",
    }
    passed = all(value == "PASS" for key, value in parity.items() if key.endswith("parity"))
    contract = {
        "source_main_commit": source_commit,
        "paired_records_sha256": PAIRED_RECORDS_SHA256,
        "G": G, "epsilon": EPSILON,
        "a_rescue_lambda": A_RESCUE_LAMBDA, "bc_hpr_lambda": BC_HPR_LAMBDA,
        "think_semantic_negative_credit": "DISABLED",
        "format_penalty": "REUSED_FROM_TRUEREC",
        "format_invalid_total": FORMAT_INVALID_TOTAL,
        "a_rescue_reduction": "ONE_GROUP_LEVEL_A_DISTRIBUTION",
        "a_rescue_sample_row": 0,
        "hpr_a": "DISABLED",
        "hpr_bc_target_source": "FROZEN_HPR_PLAN_V1",
        "a_rescue_extra_forward_calls": 0, "bc_hpr_extra_forward_calls": 0,
        "real_model_loaded": False, "gpu_started": False,
        "real_generation_started": False, "mock_cpu_forward": True,
        "mock_cpu_backward": True, "optimizer_steps": 0,
        "phase3a_runtime_contract": "PASS" if passed else "FAIL",
    }
    summaries = [{
        "case": item["case"],
        "full_loss": item["full"]["total_loss"],
        "a_rescue_loss_raw": item["full"]["a_rescue_loss_raw"],
        "bc_hpr_loss_raw": item["full"]["bc_hpr_loss_raw"],
        "format_loss": item["full"]["format_loss"],
    } for item in cases]
    return contract, summaries, parity


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def review(contract: dict[str, Any], parity: dict[str, Any]) -> str:
    fields = {
        "SOURCE_MAIN_COMMIT": contract["source_main_commit"],
        "PAIRED_RECORDS_SHA256": contract["paired_records_sha256"],
        "G": 8, "EPSILON": 0.2, "A_RESCUE_LAMBDA": 0.02, "BC_HPR_LAMBDA": 0.02,
        "THINK_SEMANTIC_NEGATIVE_CREDIT": "DISABLED",
        "FORMAT_PENALTY": "REUSED_FROM_TRUEREC",
        "A_RESCUE_REDUCTION": "ONE_GROUP_LEVEL_A_DISTRIBUTION",
        "A_RESCUE_SAMPLE_ROW": 0, "HPR_A": "DISABLED",
        "HPR_BC_TARGET_SOURCE": "FROZEN_HPR_PLAN_V1",
        "FULL_VS_STREAMING_MB1_LOSS_PARITY": parity["full_vs_streaming_mb1_loss_parity"],
        "FULL_VS_STREAMING_MB2_LOSS_PARITY": parity["full_vs_streaming_mb2_loss_parity"],
        "FULL_VS_STREAMING_MB1_GRAD_PARITY": parity["full_vs_streaming_mb1_grad_parity"],
        "FULL_VS_STREAMING_MB2_GRAD_PARITY": parity["full_vs_streaming_mb2_grad_parity"],
        "A_RESCUE_EXTRA_FORWARD_CALLS": 0, "BC_HPR_EXTRA_FORWARD_CALLS": 0,
        "REAL_MODEL_LOADED": "NO", "GPU_STARTED": "NO", "REAL_GENERATION_STARTED": "NO",
        "MOCK_CPU_FORWARD": "YES", "MOCK_CPU_BACKWARD": "YES", "OPTIMIZER_STEPS": 0,
        "PHASE3A_RUNTIME_CONTRACT": contract["phase3a_runtime_contract"],
    }
    return "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract, cases, parity = run_audit(args.source_commit)
    write_json(args.output_dir / "contract.json", contract)
    write_json(args.output_dir / "deterministic_cases.json", cases)
    write_json(args.output_dir / "gradient_parity.json", parity)
    (args.output_dir / "REVIEW.txt").write_text(review(contract, parity), encoding="utf-8")
    print(review(contract, parity), end="")
    if contract["phase3a_runtime_contract"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
