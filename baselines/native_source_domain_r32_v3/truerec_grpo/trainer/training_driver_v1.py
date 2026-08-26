"""Single-process lifecycle driver for the frozen TrueRec-GRPO V1 contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from rollout_runtime_v1 import G, PPO_OLD_LOGP_SOURCE
from truerec_grpo_trainer_v1 import (
    LONG_CONTEXT_STREAMING_MICROBATCH_SIZE,
    TRAINER_MICROBATCH_SIZE,
)
from truerec_loss_v1 import HPR_LAMBDA


GRPO_INIT_FAMILY = "BETA_BASELINE"
BUSINESS_UNIT = "RECOMMENDATION_GROUP_ID"
POLICY_MODE = "EVAL"
TRAIN_BRIDGE = False
TRAIN_FIXED_DOMAIN_IN_CONTEXT = True
TRAIN_ACTION = ("A", "B", "C")
KL_BETA = 0.0
OPTIMIZER_FAMILY = "AdamW"
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
DEFAULT_GROUPS_PER_OPTIMIZER_STEP = 1
ADAPTIVE_STREAMING_MICROBATCH = True
LONG_CONTEXT_THRESHOLD_TOKENS = 2200


class DriverContractError(RuntimeError):
    pass


class PolicyMutationError(DriverContractError):
    pass


@dataclass
class DriverState:
    business_groups_seen: int = 0
    optimizer_steps: int = 0
    global_step: int = 0
    rollouts_completed: int = 0
    old_rescores_completed: int = 0
    streaming_backwards_completed: int = 0
    groups_in_accumulation_window: int = 0
    failed: bool = False


@dataclass(frozen=True)
class DriverStepResult:
    group_id: str
    optimizer_step_performed: bool
    backward_result: Any
    state: dict[str, int | bool]


class TrueRecTrainingDriverV1:
    """Orchestrate one logical business group without owning model/runtime details."""

    def __init__(
        self,
        *,
        rollout_fn: Callable[[dict[str, Any]], Any],
        old_rescore_fn: Callable[[dict[str, Any], Any], Any],
        trainer: Any,
        optimizer: Any,
        gradient_finite_fn: Callable[[], bool],
        policy_fingerprint_fn: Callable[[], Any],
        monitor: Callable[[str, dict[str, Any]], None] | None = None,
        groups_per_optimizer_step: int = DEFAULT_GROUPS_PER_OPTIMIZER_STEP,
    ) -> None:
        if groups_per_optimizer_step < 1:
            raise ValueError("groups_per_optimizer_step must be positive")
        self.rollout_fn = rollout_fn
        self.old_rescore_fn = old_rescore_fn
        self.trainer = trainer
        self.optimizer = optimizer
        self.gradient_finite_fn = gradient_finite_fn
        self.policy_fingerprint_fn = policy_fingerprint_fn
        self.monitor = monitor
        self.groups_per_optimizer_step = groups_per_optimizer_step
        self.state = DriverState()
        self.lifecycle_events: list[str] = []

    def _event(self, name: str) -> None:
        self.lifecycle_events.append(name)

    def _fail(self) -> None:
        self.state.failed = True

    def run_group(self, record: dict[str, Any]) -> DriverStepResult:
        if self.state.failed:
            raise DriverContractError("driver is terminal after a previous failure")
        group_id = str(record.get("recommendation_group_id", ""))
        if not group_id:
            raise DriverContractError("missing recommendation_group_id")
        try:
            if self.state.groups_in_accumulation_window == 0:
                self.optimizer.zero_grad(set_to_none=True)
                self._event("zero_grad")

            policy_before_rollout = self.policy_fingerprint_fn()
            rollout = self.rollout_fn(record)
            self.state.rollouts_completed += 1
            self._event("rollout")

            policy_after_rollout = self.policy_fingerprint_fn()
            if policy_after_rollout != policy_before_rollout:
                raise PolicyMutationError("policy changed during rollout before old-logp rescore")

            group = self.old_rescore_fn(record, rollout)
            self.state.old_rescores_completed += 1
            self._event("old_rescore")
            if self.policy_fingerprint_fn() != policy_before_rollout:
                raise PolicyMutationError("policy changed between rollout and old-logp rescore")

            backward_result = self.trainer.backward_group_streaming(group)
            self.state.streaming_backwards_completed += 1
            self._event("streaming_backward")

            gradients_finite = bool(self.gradient_finite_fn())
            self._event("gradient_gate")
            if not gradients_finite:
                raise FloatingPointError("non-finite gradient")

            self.state.business_groups_seen += 1
            self.state.groups_in_accumulation_window += 1
            stepped = False
            if self.state.groups_in_accumulation_window == self.groups_per_optimizer_step:
                self.optimizer.step()
                self._event("optimizer_step")
                self.state.optimizer_steps += 1
                self.state.global_step += 1
                self.state.groups_in_accumulation_window = 0
                self._event("global_step_increment")
                stepped = True

            result = DriverStepResult(group_id, stepped, backward_result, asdict(self.state))
            if self.monitor is not None:
                self.monitor("group_complete", {"group_id": group_id, **result.state})
            return result
        except Exception:
            self._fail()
            raise

    def run(self, records: Iterable[dict[str, Any]]) -> list[DriverStepResult]:
        return [self.run_group(record) for record in records]

    def export_state(self) -> dict[str, int | bool]:
        return asdict(self.state)

    def import_state(self, value: dict[str, Any]) -> None:
        expected = set(DriverState.__dataclass_fields__)
        if set(value) != expected:
            raise DriverContractError("driver checkpoint state schema mismatch")
        restored = DriverState(**value)
        if restored.failed or restored.groups_in_accumulation_window != 0:
            raise DriverContractError("driver checkpoint is not at a healthy optimizer boundary")
        if restored.optimizer_steps != restored.global_step:
            raise DriverContractError("optimizer/global step mismatch")
        self.state = restored


def audit_pilot4096_admission(records_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Read-only admission gate for the frozen Pilot4096 business-group records."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = int(manifest["groups"])
    expected_sha = str(manifest["pilot_records_sha256"])
    digest = hashlib.sha256()
    group_ids: set[str] = set()
    record_count = 0
    fixed_domain_count = 0
    nonempty_gold_count = 0
    context_contract_count = 0
    with records_path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            record = json.loads(raw)
            record_count += 1
            group_id = str(record.get("recommendation_group_id", ""))
            if not group_id or group_id in group_ids:
                raise DriverContractError(f"duplicate or empty Pilot group at row {record_count}")
            group_ids.add(group_id)
            domain = record.get("target_domain")
            fixed = record.get("fixed_domain_token")
            if domain in {"video", "prod", "ad", "living"} and fixed == f"<|{domain}_begin|>":
                fixed_domain_count += 1
            gold = record.get("all_gold_abc")
            if isinstance(gold, list) and gold and all(isinstance(item, str) and item for item in gold):
                nonempty_gold_count += 1
            if (
                isinstance(record.get("system"), str) and record["system"]
                and isinstance(record.get("user_content_nothink"), str) and record["user_content_nothink"]
                and isinstance(fixed, str) and fixed
            ):
                context_contract_count += 1
    actual_sha = digest.hexdigest()
    gates = {
        "record_count": record_count,
        "unique_recommendation_group_id": len(group_ids),
        "fixed_domain_present_count": fixed_domain_count,
        "all_gold_abc_nonempty_count": nonempty_gold_count,
        "context_contract_present_count": context_contract_count,
        "records_sha256": actual_sha,
        "manifest_records_sha256": expected_sha,
    }
    if not (
        record_count == len(group_ids) == expected_count == 4096
        and fixed_domain_count == nonempty_gold_count == context_contract_count == expected_count
        and actual_sha == expected_sha
    ):
        raise DriverContractError(f"Pilot4096 admission failed: {gates}")
    gates["pilot_admission"] = "PASS"
    return gates


def frozen_contract() -> dict[str, Any]:
    return {
        "phase12_closed": True,
        "grpo_init_family": GRPO_INIT_FAMILY,
        "business_unit": BUSINESS_UNIT,
        "G": G,
        "fixed_target_domain": TRAIN_FIXED_DOMAIN_IN_CONTEXT,
        "train_bridge": TRAIN_BRIDGE,
        "train_action": list(TRAIN_ACTION),
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "policy_mode": POLICY_MODE,
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "adaptive_streaming_microbatch": ADAPTIVE_STREAMING_MICROBATCH,
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "long_context_microbatch_size": LONG_CONTEXT_STREAMING_MICROBATCH_SIZE,
        "hpr_lambda": HPR_LAMBDA,
        "kl_beta": KL_BETA,
        "optimizer_family": OPTIMIZER_FAMILY,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }
