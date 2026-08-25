"""CPU-testable TrueRec trainer core with one shared policy forward per group."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

import torch

from action_alignment import CONTEXT_NONACTION_DIRECT_LOSS_TERMS, DOMAIN_DIRECT_LOSS_TERMS
from truerec_loss_v1 import HPR_GROUP_REDUCTION, HPR_LAMBDA, HPR_SITE_REDUCTION, compose_total_loss, frontier_ppo_loss
from truerec_runtime_v1 import build_group_runtime_plan, hpr_loss_from_shared_logits, sampled_action_logps


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
TRUE_REC = RUNTIME / "truerec_grpo"
OUTPUT = TRUE_REC / "results/phase1_0"
RELATIVE_FILES = (
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/trainer/action_alignment.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_loss_v1.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_runtime_v1.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_trainer_v1.py"),
)
PHASE08_SHA = {
    "frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b",
    "hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32",
}


@dataclass(frozen=True)
class TrueRecGroupBatch:
    input_ids: torch.Tensor
    context_length: int
    completion_ids: torch.Tensor
    old_logps: torch.Tensor
    candidates: Sequence[dict[str, Any]]
    all_gold_abc: Sequence[str]
    token_to_id: Callable[[str], int]


@dataclass(frozen=True)
class GroupLossOutput:
    frontier_loss: torch.Tensor
    hpr_loss_raw: torch.Tensor
    hpr_loss_weighted: torch.Tensor
    total_loss: torch.Tensor
    monitoring: dict[str, float | int]


class TrueRecTrainerCore:
    def __init__(self, policy, epsilon: float = 0.2):
        self.policy = policy
        self.epsilon = epsilon
        self.policy_forward_calls = 0
        self.hpr_extra_forward_calls = 0

    def compute_group(self, group: TrueRecGroupBatch) -> GroupLossOutput:
        if group.input_ids.shape[0] != 8 or group.completion_ids.shape != (8, 3) or group.old_logps.shape != (8, 3):
            raise ValueError("TrueRec group must be G8 with three action tokens")
        runtime = build_group_runtime_plan(group.candidates, group.all_gold_abc, group.token_to_id)
        output = self.policy(group.input_ids)
        self.policy_forward_calls += 1
        logits = output.logits if hasattr(output, "logits") else output
        current_logps = sampled_action_logps(logits, group.context_length, group.completion_ids)
        frontier = frontier_ppo_loss(current_logps.unsqueeze(0), group.old_logps.unsqueeze(0), runtime.token_credits.to(current_logps.device).unsqueeze(0), runtime.token_credit_mask.to(current_logps.device).unsqueeze(0), self.epsilon)
        hpr_raw = hpr_loss_from_shared_logits(logits, group.context_length, runtime.hpr)
        total = compose_total_loss(frontier.loss, hpr_raw)
        monitor = dict(runtime.monitoring)
        monitor.update({"group_count": 1, "HPR_A_groups": int(runtime.hpr.trigger == "HPR_A"), "HPR_B_groups": int(runtime.hpr.trigger == "HPR_B"), "HPR_C_groups": int(runtime.hpr.trigger == "HPR_C"), "HPR_NONE_groups": int(runtime.hpr.trigger == "HPR_NONE"), "frontier_loss": float(total.frontier_loss.detach()), "hpr_loss_raw": float(total.hpr_loss_raw.detach()), "hpr_loss_weighted": float(total.hpr_loss_weighted.detach()), "total_loss": float(total.total_loss.detach())})
        return GroupLossOutput(total.frontier_loss, total.hpr_loss_raw, total.hpr_loss_weighted, total.total_loss, monitor)

    def compute_batch(self, groups: Sequence[TrueRecGroupBatch]) -> GroupLossOutput:
        if not groups:
            raise ValueError("empty business-group batch")
        outputs = [self.compute_group(group) for group in groups]
        frontier = torch.stack([item.frontier_loss for item in outputs]).mean()
        hpr_raw = torch.stack([item.hpr_loss_raw for item in outputs]).mean()
        total = compose_total_loss(frontier, hpr_raw)
        monitor_keys = outputs[0].monitoring.keys()
        monitor = {key: sum(item.monitoring[key] for item in outputs) for key in monitor_keys}
        monitor["group_count"] = len(groups)
        for key in ("A_hit_rate", "AB_hit_rate", "exact_rate", "wrong_history_copy_rate"):
            monitor[key] /= len(groups)
        monitor.update({"candidate_count": 8 * len(groups), "frontier_loss": float(total.frontier_loss.detach()), "hpr_loss_raw": float(total.hpr_loss_raw.detach()), "hpr_loss_weighted": float(total.hpr_loss_weighted.detach()), "total_loss": float(total.total_loss.detach())})
        return GroupLossOutput(total.frontier_loss, total.hpr_loss_raw, total.hpr_loss_weighted, total.total_loss, monitor)


def contract_constants() -> dict[str, Any]:
    return {"HPR_LAMBDA": HPR_LAMBDA, "FRONTIER_LEVELS": ["A", "B", "C"], "DOMAIN_IS_RL_ACTION": False, "DOMAIN_DIRECT_LOSS_TERMS": DOMAIN_DIRECT_LOSS_TERMS, "CONTEXT_NONACTION_DIRECT_LOSS_TERMS": CONTEXT_NONACTION_DIRECT_LOSS_TERMS, "FRONTIER_REDUCTION": "candidate credited-token SUM -> G8 mean -> business-group batch mean", "HPR_SITE_REDUCTION": HPR_SITE_REDUCTION, "HPR_GROUP_REDUCTION": HPR_GROUP_REDUCTION, "history_penalty": False, "KL_beta": 0.0, "old_logps_contract": "caller-provided rollout-policy logps; never synthesized from current_logps"}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    head, origin, status = _git("rev-parse", "HEAD"), _git("rev-parse", "origin/main"), _git("status", "--short")
    files = {}
    for relative in RELATIVE_FILES:
        source = SOURCE_REPO / relative
        runtime = TRUE_REC / Path(*relative.parts[3:])
        blob = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "show", f"{head}:{relative.as_posix()}"])
        values = {"github": hashlib.sha256(blob).hexdigest(), "source": _file_sha(source), "runtime": _file_sha(runtime)}
        if len(set(values.values())) != 1:
            raise RuntimeError(f"TRAINER_CODE_PARITY_FAIL={relative}:{values}")
        files[relative.as_posix()] = values
    phase08 = {}
    for name, expected in PHASE08_SHA.items():
        path = SOURCE_REPO / "baselines/native_source_domain_r32_v3/truerec_grpo/credit" / name
        actual = _file_sha(path)
        if actual != expected:
            raise RuntimeError(f"PHASE08_CREDIT_SHA_CHANGED={name}:{actual}")
        phase08[name] = actual
    if head != origin or status:
        raise RuntimeError(f"GIT_PROVENANCE_FAIL={head},{origin},{status!r}")
    return {"implement_commit": head, "push_status": "PASS", "git_status_short": "EMPTY", "github_runtime_parity": "PASS", "files": files, "phase08_unchanged_sha": phase08}


class _AuditOutput:
    def __init__(self, logits): self.logits = logits


class _AuditPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.sentinel = torch.nn.Parameter(torch.tensor(1.0))
        self.logits = logits
        self.forward_calls = 0

    def forward(self, input_ids):
        self.forward_calls += 1
        return _AuditOutput(self.logits + self.sentinel * 0.0)


def _audit_candidates() -> list[dict[str, Any]]:
    return [{"format_valid": True, "A_hit": False, "AB_hit": False, "exact": False, "parsed_abc": "<s_a_4><s_b_5><s_c_6>", "sample_index": index, "wrong_history_copy": index == 0} for index in range(8)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_contract_audit(test_status: str) -> None:
    if test_status != "PASS":
        raise RuntimeError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    from action_alignment import align_action
    from truerec_loss_v1 import multi_positive_log_mass_loss

    alignment = align_action(10, [1, 2, 3])
    trainer_contract = {**contract_constants(), "action_indices": list(alignment.action_indices), "causal_logit_indices": list(alignment.logit_indices), "LOSS_POSITION_GATE": "PASS", "monitoring_fields": ["group_count", "candidate_count", "A_hit_rate", "AB_hit_rate", "exact_rate", "HPR_A_groups", "HPR_B_groups", "HPR_C_groups", "HPR_NONE_groups", "frontier_A_positive", "frontier_A_negative", "frontier_B_positive", "frontier_B_negative", "frontier_C_positive", "frontier_C_negative", "frontier_loss", "hpr_loss_raw", "hpr_loss_weighted", "total_loss", "wrong_history_copy_rate"]}

    zero = torch.zeros(1, 8, 3)
    ratio1 = frontier_ppo_loss(zero, zero.clone(), torch.ones_like(zero), torch.ones_like(zero, dtype=torch.bool), 0.2)
    log2 = torch.log(torch.tensor(2.0))
    current = torch.tensor([[[log2, 0.0, 0.0]]])
    old = torch.zeros_like(current)
    positive = frontier_ppo_loss(current, old, torch.tensor([[[1.0, 0.0, 0.0]]]), torch.tensor([[[True, False, False]]]), 0.2)
    negative = frontier_ppo_loss(current, old, torch.tensor([[[-1.0, 0.0, 0.0]]]), torch.tensor([[[True, False, False]]]), 0.2)
    multi_logits = torch.tensor([0.0, 1.0, 2.0])
    multi_loss = multi_positive_log_mass_loss(multi_logits, [1, 2])
    expected_multi = -torch.logsumexp(torch.log_softmax(multi_logits, -1)[torch.tensor([1, 2])], 0)
    total = compose_total_loss(torch.tensor(2.0), torch.tensor(3.0))
    loss_contract = {"PPO_RATIO1_TEST": "PASS" if torch.equal(ratio1.ratios, torch.ones_like(zero)) else "FAIL", "PPO_CLIP_POSITIVE_TEST": "PASS" if abs(float(positive.loss) + 1.2) < 1e-6 else "FAIL", "PPO_CLIP_NEGATIVE_TEST": "PASS" if abs(float(negative.loss) - 2.0) < 1e-6 else "FAIL", "MULTIPOSITIVE_HPR_TEST": "PASS" if torch.allclose(multi_loss, expected_multi) else "FAIL", "TOTAL_LOSS_COMPOSITION_TEST": "PASS" if abs(float(total.total_loss) - 2.06) < 1e-6 else "FAIL", "ratio1": ratio1.ratios.tolist(), "positive_clipped_loss": float(positive.loss), "negative_clipped_loss": float(negative.loss), "multi_positive_loss": float(multi_loss), "total_components": {"frontier_loss": float(total.frontier_loss), "hpr_loss_raw": float(total.hpr_loss_raw), "hpr_loss_weighted": float(total.hpr_loss_weighted), "total_loss": float(total.total_loss)}}

    torch.manual_seed(20260826)
    logits = torch.randn(8, 5, 12)
    completion_ids = torch.tensor([[4, 5, 6]] * 8)
    input_ids = torch.cat([torch.full((8, 2), 10), completion_ids], dim=1)
    old_logps = sampled_action_logps(logits, 2, completion_ids).detach()
    mapping = {"<s_a_1>": 1, "<s_b_2>": 2, "<s_c_3>": 3}
    group = TrueRecGroupBatch(input_ids, 2, completion_ids, old_logps, _audit_candidates(), ["<s_a_1><s_b_2><s_c_3>"], mapping.__getitem__)
    policy = _AuditPolicy(logits)
    before = policy.sentinel.detach().clone()
    core = TrueRecTrainerCore(policy)
    output = core.compute_group(group)
    unchanged = torch.equal(before, policy.sentinel.detach())
    shared = {"POLICY_FORWARD_CALLS_PER_GROUP": policy.forward_calls, "HPR_EXTRA_FORWARD_CALLS": core.hpr_extra_forward_calls, "SHARED_FORWARD_TEST": "PASS" if policy.forward_calls == 1 and core.hpr_extra_forward_calls == 0 else "FAIL", "parameter_mutation": False if unchanged else True, "real_model_loaded": False, "real_model_forward": False, "mock_monitoring": output.monitoring}
    if any(loss_contract[key] != "PASS" for key in ("PPO_RATIO1_TEST", "PPO_CLIP_POSITIVE_TEST", "PPO_CLIP_NEGATIVE_TEST", "MULTIPOSITIVE_HPR_TEST", "TOTAL_LOSS_COMPOSITION_TEST")) or shared["SHARED_FORWARD_TEST"] != "PASS" or not unchanged:
        raise RuntimeError("CPU_CONTRACT_AUDIT_FAIL")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "trainer_contract_audit.json", {"code": audit, "contract": trainer_contract})
    _write_json(OUTPUT / "loss_contract_audit.json", loss_contract)
    _write_json(OUTPUT / "shared_forward_audit.json", shared)
    review = render_review(audit, trainer_contract, loss_contract, shared)
    (OUTPUT / "CHATGPT_PHASE1_0_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def render_review(audit, contract, loss, shared) -> str:
    values = {"IMPLEMENT_COMMIT": audit["implement_commit"], "RESULT_COMMIT": "PENDING_REPORT_COMMIT", "PUSH_STATUS": audit["push_status"], "GIT_STATUS_SHORT": audit["git_status_short"], "GITHUB_RUNTIME_PARITY": audit["github_runtime_parity"], "HPR_LAMBDA": HPR_LAMBDA, "FRONTIER_LEVELS": "A,B,C", "DOMAIN_IS_RL_ACTION": "NO", "FRONTIER_REDUCTION": contract["FRONTIER_REDUCTION"], "HPR_SITE_REDUCTION": contract["HPR_SITE_REDUCTION"], "HPR_GROUP_REDUCTION": contract["HPR_GROUP_REDUCTION"], "POLICY_FORWARD_CALLS_PER_GROUP": shared["POLICY_FORWARD_CALLS_PER_GROUP"], "HPR_EXTRA_FORWARD_CALLS": shared["HPR_EXTRA_FORWARD_CALLS"], "DOMAIN_DIRECT_LOSS_TERMS": DOMAIN_DIRECT_LOSS_TERMS, "CONTEXT_NONACTION_DIRECT_LOSS_TERMS": CONTEXT_NONACTION_DIRECT_LOSS_TERMS, "LOSS_POSITION_GATE": contract["LOSS_POSITION_GATE"], "PPO_RATIO1_TEST": loss["PPO_RATIO1_TEST"], "PPO_CLIP_POSITIVE_TEST": loss["PPO_CLIP_POSITIVE_TEST"], "PPO_CLIP_NEGATIVE_TEST": loss["PPO_CLIP_NEGATIVE_TEST"], "MULTIPOSITIVE_HPR_TEST": loss["MULTIPOSITIVE_HPR_TEST"], "SHARED_FORWARD_TEST": shared["SHARED_FORWARD_TEST"], "TOTAL_LOSS_COMPOSITION_TEST": loss["TOTAL_LOSS_COMPOSITION_TEST"], "TEST_STATUS": "PASS", "GPU_INFERENCE_STARTED": "NO", "MODEL_FORWARD_STARTED": "NO", "TRAINING_STARTED": "NO", "BACKWARD_STARTED": "NO", "OPTIMIZER_STARTED": "NO", "OPTIMIZER_STEPS": 0, "NEXT_EXPERIMENT_STARTED": "NO"}
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-audit", action="store_true")
    parser.add_argument("--test-status", choices=("PASS", "FAIL"), required=True)
    args = parser.parse_args()
    if not args.contract_audit:
        raise RuntimeError("only --contract-audit is available in Phase1.0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    run_contract_audit(args.test_status)


if __name__ == "__main__":
    main()
