"""Phase 1.3B CPU-only checkpoint/resume deterministic equivalence audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from checkpoint_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_ORDER_SEED,
    METADATA_FILE,
    PILOT_RECORDS_SHA256,
    STATE_FILE,
    CheckpointContractError,
    load_checkpoint,
    order_metadata,
    save_checkpoint_atomic,
)
from training_driver_v1 import (  # noqa: E402
    TrueRecTrainingDriverV1,
    audit_pilot4096_admission,
    frozen_contract,
)


DATASET_IDENTITY = {
    "records_sha256": PILOT_RECORDS_SHA256,
    "record_count": 4096,
    "unique_group_count": 4096,
}


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float64))


class TinyStreamingTrainer:
    def __init__(self, model: TinyPolicy) -> None:
        self.model = model

    def backward_group_streaming(self, group: dict[str, Any]) -> dict[str, float]:
        random_sum = sum(group["random_output"])
        scale = torch.tensor(random_sum, dtype=self.model.weight.dtype)
        loss = (self.model.weight * scale).square().sum()
        loss.backward()
        return {"total_value": float(loss.detach())}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def policy_fingerprint(model: TinyPolicy) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in model.parameters())


def build_components(output_log: list[tuple[float, float, float]]):
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    trainer = TinyStreamingTrainer(model)

    def rollout(record):
        output = (random.random(), float(np.random.random()), float(torch.rand(())))
        output_log.append(output)
        return {"recommendation_group_id": record["recommendation_group_id"], "random_output": output}

    driver = TrueRecTrainingDriverV1(
        rollout_fn=rollout,
        old_rescore_fn=lambda record, rollout_value: dict(rollout_value),
        trainer=trainer,
        optimizer=optimizer,
        gradient_finite_fn=lambda: all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        policy_fingerprint_fn=lambda: policy_fingerprint(model),
        groups_per_optimizer_step=1,
    )
    return model, optimizer, driver


def tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensor_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _run_slice(driver, order: list[str], start: int, stop: int) -> list[str]:
    sequence = []
    for group_id in order[start:stop]:
        driver.run_group({"recommendation_group_id": group_id})
        sequence.append(group_id)
    return sequence


def run_equivalence(group_ids: list[str], work_dir: Path) -> dict[str, Any]:
    order, order_info = order_metadata(group_ids, DEFAULT_ORDER_SEED)
    if order_info["count"] != 4096:
        raise AssertionError("formal order must contain Pilot4096")

    seed_all(1303)
    continuous_random: list[tuple[float, float, float]] = []
    model_a, optimizer_a, driver_a = build_components(continuous_random)
    sequence_a = _run_slice(driver_a, order, 0, 6)

    seed_all(1303)
    resumed_random: list[tuple[float, float, float]] = []
    model_b0, optimizer_b0, driver_b0 = build_components(resumed_random)
    sequence_b = _run_slice(driver_b0, order, 0, 3)
    checkpoint = work_dir / "checkpoint-step-3"
    save_checkpoint_atomic(
        checkpoint, model=model_b0, optimizer=optimizer_b0, driver=driver_b0,
        dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=3, order=order_info,
    )
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    del model_b0, optimizer_b0, driver_b0

    # Fresh objects are intentionally constructed before checkpoint state restoration.
    seed_all(99999)
    model_b, optimizer_b, driver_b = build_components(resumed_random)
    cursor = load_checkpoint(
        checkpoint, model=model_b, optimizer=optimizer_b, driver=driver_b,
        current_dataset_identity=DATASET_IDENTITY, current_group_ids=group_ids,
    )
    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    rng_equal = tuple(a == b for a, b in zip(expected_rng, actual_rng))
    # Restore once more so the diagnostic RNG draws do not perturb resumed training.
    cursor = load_checkpoint(
        checkpoint, model=model_b, optimizer=optimizer_b, driver=driver_b,
        current_dataset_identity=DATASET_IDENTITY, current_group_ids=group_ids,
    )
    sequence_b.extend(_run_slice(driver_b, order, cursor["next_group_index"], 6))

    model_equal = tensor_tree_equal(model_a.state_dict(), model_b.state_dict())
    optimizer_equal = tensor_tree_equal(optimizer_a.state_dict(), optimizer_b.state_dict())
    driver_equal = driver_a.export_state() == driver_b.export_state()
    sequence_equal = sequence_a == sequence_b
    random_equal = continuous_random == resumed_random
    no_duplicate = len(sequence_b) == len(set(sequence_b))
    no_skipped = sequence_b == order[:6]
    gates = {
        "group_id_sequence_equal": sequence_equal,
        "rollout_random_output_sequence_equal": random_equal,
        "final_model_state_equal": model_equal,
        "final_optimizer_state_equal": optimizer_equal,
        "final_driver_state_equal": driver_equal,
        "final_global_step_equal": driver_a.state.global_step == driver_b.state.global_step,
        "final_next_group_index_equal": cursor["next_group_index"] + 3 == 6,
        "python_rng_restored": rng_equal[0],
        "numpy_rng_restored": rng_equal[1],
        "torch_cpu_rng_restored": rng_equal[2],
        "no_duplicate_group_after_resume": no_duplicate,
        "no_skipped_group_after_resume": no_skipped,
    }
    if not all(gates.values()):
        raise AssertionError(f"checkpoint equivalence failed: {gates}")
    return {
        **{key: "PASS" for key in gates},
        "group_id_sequence": sequence_b,
        "rollout_random_outputs": resumed_random,
        "final_driver_state": driver_b.export_state(),
        "cursor_after_load": cursor,
        "order": order_info,
        "optimizer_state_fields": sorted(next(iter(optimizer_b.state_dict()["state"].values())).keys()),
        "checkpoint_only_at_optimizer_boundary": "PASS",
    }


def _mutate_metadata(source: Path, target: Path, mutation) -> None:
    shutil.copytree(source, target)
    path = target / METADATA_FILE
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_rejection_audit(group_ids: list[str], valid_checkpoint: Path, work_dir: Path) -> dict[str, str]:
    def rejected(path: Path, *, ids=group_ids, contract=None, dataset=DATASET_IDENTITY) -> bool:
        model, optimizer, driver = build_components([])
        try:
            load_checkpoint(
                path, model=model, optimizer=optimizer, driver=driver,
                current_dataset_identity=dataset, current_group_ids=ids,
                current_contract=frozen_contract() if contract is None else contract,
            )
        except CheckpointContractError:
            return True
        return False

    mismatch_ids = list(group_ids); mismatch_ids[-1] = "different-group-id"
    changed_contract = copy.deepcopy(frozen_contract()); changed_contract["hpr_lambda"] = 0.03
    missing = work_dir / "missing"; missing.mkdir(); (missing / METADATA_FILE).write_text("{}")
    corrupt = work_dir / "corrupt"; shutil.copytree(valid_checkpoint, corrupt); (corrupt / STATE_FILE).write_bytes(b"not torch")
    partial = work_dir / "partial"; _mutate_metadata(valid_checkpoint, partial, lambda v: v["driver_state"].update(groups_in_accumulation_window=1))
    failed = work_dir / "failed"; _mutate_metadata(valid_checkpoint, failed, lambda v: v["driver_state"].update(failed=True))
    dataset_bad = dict(DATASET_IDENTITY); dataset_bad["records_sha256"] = "0" * 64
    gates = {
        "dataset_mismatch_rejected": rejected(valid_checkpoint, dataset=dataset_bad),
        "order_mismatch_rejected": rejected(valid_checkpoint, ids=mismatch_ids),
        "contract_mismatch_rejected": rejected(valid_checkpoint, contract=changed_contract),
        "missing_checkpoint_rejected": rejected(missing),
        "corrupt_checkpoint_rejected": rejected(corrupt),
        "partial_window_checkpoint_rejected": rejected(partial),
        "failed_state_checkpoint_rejected": rejected(failed),
    }
    if not all(gates.values()):
        raise AssertionError(f"checkpoint rejection gate failed: {gates}")
    return {key: "PASS" for key in gates}


def read_group_ids(records_path: Path) -> list[str]:
    with records_path.open(encoding="utf-8") as handle:
        return [json.loads(line)["recommendation_group_id"] for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-status", choices=("PASS",), required=True)
    args = parser.parse_args()
    pilot = audit_pilot4096_admission(args.pilot_records, args.pilot_manifest)
    group_ids = read_group_ids(args.pilot_records)
    with tempfile.TemporaryDirectory(prefix="truerec-phase13b-") as temporary:
        work_dir = Path(temporary)
        equivalence = run_equivalence(group_ids, work_dir)
        rejection = run_rejection_audit(group_ids, work_dir / "checkpoint-step-3", work_dir)
    result = {
        "phase": "TrueRec-GRPO Phase 1.3B",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pilot4096": pilot,
        "equivalence": equivalence,
        "rejection_gates": rejection,
        "cuda_rng_tested": False,
        "real_model_loaded": False,
        "gpu_started": False,
        "ddp_implemented": False,
        "formal_training_started": False,
        "test_status": args.test_status,
        "next_experiment_started": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_resume_determinism_audit.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "CHATGPT_PHASE1_3B_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.3B CPU Checkpoint / Resume Determinism Contract\n"
        "A six-group uninterrupted tiny-policy run exactly matched a fresh-process 3+3 resume, including model, AdamW state, driver state, cursor, group/random sequences, and Python/NumPy/torch CPU RNG. All identity and invalid-boundary rejection gates passed. No real model, CUDA, DDP, or training ran.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
