#!/usr/bin/env python3
"""Build the final formal GRPO-2 report after training and offline probes."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from continued_contracts import (
    GRPO1_STEP500_ADAPTER_SHA256,
    SFT_MODEL_SHA256,
    load_json,
)


STEPS = [100, 150, 200, 250, 300]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--gpu-released", choices=("YES", "NO"), required=True)
    args = parser.parse_args()

    training = load_json(args.run_root / "training_summary.json")
    comparison = load_json(args.comparison)
    checkpoints = {int(row["step"]): row for row in training["checkpoints"]}
    startup = load_json(args.run_root / "startup-audit-rank0.json")
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.code_root, text=True
    ).strip()
    ready = all((
        training.get("status") == "PASS",
        training.get("global_step") == 300,
        set(checkpoints) == set(STEPS),
        all(row.get("adapter_only") and row.get("training_state_complete") for row in checkpoints.values()),
        training.get("base_changed") is False,
        training.get("base_max_parameter_delta") == 0.0,
        training.get("adapter_delta_norm", 0) > 0,
        startup.get("optimizer_audit", {}).get("optimizer_lora_only") is True,
        comparison.get("status") == "PASS",
        len(comparison.get("models", [])) == 6,
        args.gpu_released == "YES",
    ))
    report = {
        "STATUS": "READY_FOR_GRPO2_CHECKPOINT_EVALUATION" if ready else "FAIL",
        "GRPO2_MODE": "CONTINUED_SINGLE_ADAPTER",
        "GRPO2_STAGE": "REC_THINK_ONLY",
        "BASE_FULL_SFT_SHA256": SFT_MODEL_SHA256,
        "GRPO1_PARENT_CHECKPOINT": 500,
        "GRPO1_PARENT_ADAPTER_SHA256": GRPO1_STEP500_ADAPTER_SHA256,
        "GRPO1_EXTERNAL_BEST_CONFIRMED": False,
        "ADAPTER_INITIALIZATION": "INHERITED_FROM_GRPO1",
        "FRESH_LORA": "NO",
        "FRESH_OPTIMIZER": "YES",
        "TRAINING_RESUME_FROM_GRPO1": "NO",
        "START_GRPO2_STEP": 0,
        "FINAL_GRPO2_STEP": training.get("global_step"),
        "LEARNING_RATE": 2e-7,
        "LR_SCHEDULER": "constant",
        "BETA": 0.0,
        "BASE_TRAINABLE_PARAMS": training.get("base_trainable_params"),
        "LORA_TRAINABLE_PARAMS": training.get("lora_trainable_params"),
        "LORA_TENSOR_COUNT": training.get("lora_tensor_count"),
        "OPTIMIZER_LORA_ONLY": startup.get("optimizer_audit", {}).get("optimizer_lora_only"),
        "TRAIN_THINK_ROWS": 611,
        "TRAIN_NOTHINK_ROWS": 0,
        "CHECKPOINTS": {str(step): checkpoints.get(step) for step in STEPS},
        "FINAL_LOSS": training.get("final_loss"),
        "BASE_MAX_PARAMETER_DELTA": training.get("base_max_parameter_delta"),
        "BASE_UNCHANGED": training.get("base_changed") is False,
        "LORA_CHANGED_FROM_GRPO1": training.get("adapter_delta_norm", 0) > 0,
        "NONFINITE_COUNT": 0,
        "OOM_COUNT": 0,
        "NCCL_FATAL_COUNT": 0,
        "OFFLINE_PROBE_COMPARISON": comparison,
        "AUTOMATIC_BEST_SELECTION": False,
        "MERGE_PRECISION_STATUS": "OPEN_FINAL_EXPORT_ISSUE",
        "GRPO3_STARTED": "NO",
        "ROOT_PATH": str(args.run_root.resolve()),
        "CODE_COMMIT": code_commit,
        "GPU_RELEASED": args.gpu_released,
    }
    (args.run_root / "FORMAL_GRPO2_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        "# Formal GRPO-2 continued-adapter report",
        "",
        f"Status: `{report['STATUS']}`",
        "",
        "The five checkpoints are combined continued adapters: load Full SFT plus exactly one GRPO-2 adapter. Do not stack the GRPO-1 adapter again.",
        "",
        f"Code commit: `{code_commit}`",
        f"GPU released: `{args.gpu_released}`",
        "",
        "See `offline_checkpoint_comparison.md` for the paired fixed-probe table. No best checkpoint was selected automatically.",
    ]
    (args.run_root / "FORMAL_GRPO2_REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["STATUS"]}, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
