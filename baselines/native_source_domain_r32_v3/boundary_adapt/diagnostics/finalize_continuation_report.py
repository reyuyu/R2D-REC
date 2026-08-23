"""Create the compact, provenance-preserving Boundary continuation report."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

RESULTS = Path("/data/GRPO/boundary_adapt/results")
TARGET = RESULTS / "boundary_adapt_continuation_from300_20260824.json"
RAW = RESULTS / "boundary_adapt_continuation_from300_20260824_train_raw.json"
PHASE_A = RESULTS / "boundary_adapt_self_cot_health_gate_20260824.json"
FIXED = RESULTS / "boundary_adapt_continuation_fixed_cot_sweep_20260824.json"
SELF_PARTS = RESULTS / "self_cot_health_20260824_parts"
FORMAL = RESULTS / "boundary_adapt_formal_train_300step_20260824.json"
OUTPUT = Path("/data/outputs/boundary_adapt/continuation_from300_to1500")
TRAIN_LOG = RESULTS / "boundary_adapt_continuation_from300_20260824.log"
TRAINER_COMMIT = "b44fab34bd92fe85065d5707da10e7cb7f3ea317"
DIAGNOSTIC_COMMIT = "d296f4a017a8802fa6e8f2ba08cf6c7f12e26a4c"


def compact_self(item):
    keys = (
        "self_cot_count", "closed_count", "closed_rate", "empty_count", "valid_cot_count",
        "completion_token_mean", "completion_token_min", "completion_token_max",
        "literal_sid_occurrences", "bare_raw_mean", "exact", "ab", "a", "invalid",
        "abc_invalid_rate", "history_exact_copy_rate_valid", "history_not_gold_rate",
    )
    return {key: item[key] for key in keys}


def main():
    result_mtime = TARGET.stat().st_mtime
    train = json.loads(TARGET.read_text(encoding="utf-8"))
    if "ranks" not in train:
        train = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        RAW.write_text(json.dumps(train, indent=2) + "\n", encoding="utf-8")
    ranks = train["ranks"]
    phase_a = json.loads(PHASE_A.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    best = str(fixed["best_fixed_cot_checkpoint"])
    self300 = json.loads((SELF_PARTS / "checkpoint_300.json").read_text(encoding="utf-8"))
    self_best = json.loads((SELF_PARTS / f"checkpoint_{best}.json").read_text(encoding="utf-8"))
    table = [{
        "checkpoint": row["checkpoint"],
        "bare_raw_mean": row["bare"]["beam_raw_mean"],
        "bridge_raw_mean": row["bridge"]["beam_raw_mean"],
        "bridge_dependency_gap": row["bridge_dependency_gap"],
        "interface_recovery_ratio": row["interface_recovery_ratio"],
        "bare_exact": row["bare"]["exact"], "bare_ab": row["bare"]["ab"],
        "bare_a": row["bare"]["a"], "bare_invalid": row["bare"]["invalid"],
        "bare_history_exact_copy_rate_valid": row["bare"]["exact_copy_rate_valid"],
        "bare_history_not_gold": row["bare"]["history_not_gold"],
    } for row in fixed["table"]]
    checkpoints = sorted(int(path.name.split("-")[-1]) for path in OUTPUT.glob("checkpoint-*") if (path / "adapter_model.safetensors").is_file())
    report = {
        "type": "boundary_adaptation_continuation_from_step300",
        "continuation_trainer_commit": TRAINER_COMMIT,
        "diagnostic_commit": DIAGNOSTIC_COMMIT,
        "phase_a_pass": phase_a["phase_a_pass"],
        "phase_a_hard_stop_reasons": phase_a["hard_stop_reasons"],
        "phase_a_step0": compact_self(phase_a["checkpoints"]["0"]),
        "phase_a_step300": compact_self(phase_a["checkpoints"]["300"]),
        "total_groups": 15943, "total_paths": 42799,
        "paths_per_group_scale": 42799 / 15943,
        "sampler_seed": 20260824, "sampler_epoch": 0,
        "continuation_local_offset": 300,
        "continuation_allowed_steps": 1200,
        "continuation_actual_steps": min(item["optimizer_steps"] for item in ranks),
        "final_total_step": min(item["final_total_step"] for item in ranks),
        "continuation_wall_min": (result_mtime - TRAIN_LOG.stat().st_mtime) / 60,
        "active_step_wall_sec": (
            (OUTPUT / "checkpoint-1500/adapter_model.safetensors").stat().st_mtime
            - (OUTPUT / "checkpoint-400/adapter_model.safetensors").stat().st_mtime
        ) / 1100,
        "checkpoints_saved": checkpoints,
        "major_checkpoints": [450, 600, 900, 1200, 1500],
        "init_model_parity_pass": all(item["init_parity"]["max_abs_diff"] == 0 for item in ranks),
        "only_lora_trainable": all(item["only_lora_trainable"] for item in ranks),
        "base_changed": any(item["base_changed"] for item in ranks),
        "lora_changed": all(item["lora_changed"] for item in ranks),
        "second_lora_created": any(item["second_lora_created"] for item in ranks),
        "old_optimizer_resumed": any(item["old_optimizer_resumed"] for item in ranks),
        "abc_supervised_tokens_per_path": sorted({item["labels"] for item in ranks}),
        "final_rank_losses": [item["loss"] for item in ranks],
        "final_loss_mean_across_ranks": statistics.fmean(item["loss"] for item in ranks),
        "step300_final_loss_mean_across_ranks": formal["final_loss_mean_across_ranks"],
        "pure_ce_loss_trend": "ENDPOINT_MICROBATCHES_NOT_COMPARABLE; lightweight trainer did not log a same-sample CE curve",
        "fixed_cot_table": table,
        "best_fixed_cot_checkpoint": int(best),
        "best_fixed_cot_adapter": str(OUTPUT / f"checkpoint-{best}"),
        "self_cot_eval_checkpoints": [300, int(best)],
        "self_cot_step300": compact_self(self300),
        "self_cot_best": compact_self(self_best),
        "self_cot_trend": "BARE_RAW_AND_CLOSURE_SLIGHTLY_WORSE_AT_BEST_FIXED_CHECKPOINT; HISTORY_COPY_HEALTH_IMPROVED",
        "fixed_bare_plateau": fixed["fixed_bare_plateau"],
        "bridge_gap_remains": fixed["bridge_gap_remains"],
        "kd_candidate": fixed["kd_candidate"],
        "next_action": "SELECT_STEP900_FOR_FIXED_COT_INTERFACE; do not prefer Step1500; investigate Self-CoT objective mismatch before more CE",
        "blockers": [],
        "raw_training_result": str(RAW),
        "phase_a_result": str(PHASE_A),
        "fixed_cot_result": str(FIXED),
    }
    TARGET.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FINAL_RESULT={TARGET}")
    print(f"BEST_FIXED_COT_CHECKPOINT={best}")
    print(f"KD_CANDIDATE={'YES' if report['kd_candidate'] else 'NO'}")


if __name__ == "__main__":
    main()
