"""Generate a deterministic CPU-only User GRPO monitor demo."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone

try:
    from .writer import MonitorWriter
except ImportError:  # direct script execution
    from writer import MonitorWriter


RUN_ID = "demo-user-grpo"
STEPS = 40
KINDS = (
    "hallucinated_sid",
    "duplicate_sid",
    "date_mismatch",
    "action_mismatch",
    "duplicate_event",
    "chronology_violation",
    "excess_event",
)


def iso(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


def lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _masked_span(completion: str, text: str, kind: str) -> dict:
    start = completion.index(text)
    return {"start": start, "end": start + len(text), "kind": kind}


def _action_candidates(step: int) -> list[dict]:
    candidates = []
    gold = ["<|video_begin|><s_a_12><s_b_35><s_c_90>"]
    for index in range(4):
        sid = gold[0] if index == 0 else f"<|video_begin|><s_a_12><s_b_35><s_c_{90 + index}>"
        completion = '{"predicted_sids":["' + sid + '"]}'
        violations = [] if index == 0 else ["wrong_selection_sid"]
        spans = []
        if index == 3:
            violations = ["hallucinated_sid"]
            spans = [_masked_span(completion, f"<s_c_{90 + index}>", "hallucinated_sid")]
        candidates.append({
            "candidate_id": index,
            "route": "action",
            "completion": completion,
            "completion_length": 21 + index,
            "reward": 1.0 if index == 0 else 0.5 if index == 1 else 0.0,
            "sequence_advantage": (1.3, 0.3, -0.7, -0.9)[index],
            "f1": 1.0 if index == 0 else 0.5 if index == 1 else 0.0,
            "precision": 1.0 if index == 0 else 0.5,
            "recall": 1.0 if index == 0 else 0.5 if index == 1 else 0.0,
            "gold_sids": gold,
            "pred_sids": [sid],
            "violations": violations,
            "masked_spans": spans,
            "penalty_kinds": sorted(set(violations) & {"hallucinated_sid", "duplicate_sid"}),
            "match_spans": [_masked_span(completion, sid, "correct_sid")] if sid in gold else [],
            "masked_token_count": len(spans),
        })
    return candidates


def _chain_candidates(step: int) -> list[dict]:
    candidates = []
    for index in range(4):
        date = "2025-06-18" if index < 2 else "2025-06-19"
        action = "click" if index != 3 else "purchase"
        completion = (
            '{"events":[{"date":"' + date + '","action":"' + action
            + '","sid":"<|prod_begin|><s_a_8><s_b_4><s_c_2>"}]}'
        )
        violations = []
        spans = []
        if index == 2:
            violations = ["date_mismatch"]
            spans = [_masked_span(completion, date, "date_mismatch")]
        elif index == 3:
            violations = ["action_mismatch"]
            spans = [_masked_span(completion, action, "action_mismatch")]
        candidates.append({
            "candidate_id": index,
            "route": "chain",
            "completion": completion,
            "completion_length": 54 + index * 2,
            "reward": (0.82, 0.67, 0.45, 0.28)[index],
            "sequence_advantage": (1.2, 0.4, -0.4, -1.2)[index],
            "action_alignment": (1.0, 0.9, 0.8, 0.6)[index],
            "logic_alignment": (0.64, 0.44, 0.1, 0.0)[index],
            "violations": violations,
            "masked_spans": spans,
            "penalty_kinds": list(violations),
            "match_spans": [_masked_span(completion, "<|prod_begin|><s_a_8><s_b_4><s_c_2>", "correct_sid")],
            "masked_token_count": len(spans),
        })
    return candidates


def write_probe_demo(writer: MonitorWriter, steps: int = STEPS) -> None:
    for probe_step, reason in ((0, "baseline"), (steps // 2, "periodic"), (steps, "final")):
        progress = probe_step / max(1, steps)
        for route, count in (("action", 12), ("chain", 8)):
            for index in range(count):
                candidates = _action_candidates(probe_step) if route == "action" else _chain_candidates(probe_step)
                if route == "action":
                    summary = {
                        "f1_mean": lerp(0.57, 0.66, progress),
                        "precision_mean": lerp(0.65, 0.73, progress),
                        "recall_mean": lerp(0.54, 0.63, progress),
                        "exact_match_rate": lerp(0.08, 0.16, progress),
                    }
                    for candidate in candidates:
                        candidate["reward"] = min(1.0, candidate["reward"] + 0.08 * progress)
                        candidate["f1"] = candidate["reward"]
                else:
                    summary = {
                        "total_reward_mean": lerp(0.36, 0.43, progress),
                        "action_alignment_mean": lerp(0.54, 0.62, progress),
                        "logic_alignment_mean": lerp(0.18, 0.24, progress),
                    }
                    for candidate in candidates:
                        candidate["reward"] = min(1.0, candidate["reward"] + 0.06 * progress)
                writer.write_probe({
                    "step": probe_step,
                    "reason": reason,
                    "seed": 20260820,
                    "group_id": f"demo-{route}-probe-{index:02d}",
                    "route": route,
                    "bucket": index,
                    "gold_sids": candidates[0].get("gold_sids", ["<|prod_begin|><s_a_8><s_b_4><s_c_2>"]),
                    "reward_mean": sum(candidate["reward"] for candidate in candidates) / 4,
                    "reward_std": 0.2,
                    "probe_wall_sec": 42.0,
                    "candidates": candidates,
                    route: summary,
                })


def generate(output_dir: str, run_id: str = RUN_ID, steps: int = STEPS) -> str:
    writer = MonitorWriter(True, output_dir, run_id, rank=0, trace_every=1)
    base = datetime(2026, 8, 19, 20, 0, 0, tzinfo=timezone.utc)
    writer.write_manifest({
        "run_id": run_id,
        "run_kind": "user_grpo",
        "experiment": "GR_USER_v1",
        "parent": "BATA baseline Epoch2 checkpoint-1106",
        "G": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "reward": {"action": "set_f1", "chain": "action_logic_alignment"},
        "token_penalty": {"strategy": "sqrt", "lambda": 0.5},
        "dataset": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
        "world_size": 4,
        "learning_rate": 1e-6,
        "max_steps": steps,
        "start_time": iso(base, 0),
        "git_commit": "demo-cpu-only",
        "demo": True,
        "demo_note": "Synthetic UI data only; not a training result.",
        "fixed_probe": {
            "enabled": True,
            "dataset": "/data/GRPO_USER/data/gr_user_v1/probe_v1.jsonl",
            "seed": 20260820,
            "every_steps": 20,
            "action_prompts": 12,
            "chain_prompts": 8,
            "G": 4,
        },
    })

    write_probe_demo(writer, steps)

    rollout_id = 0
    for step in range(1, steps + 1):
        progress = (step - 1) / max(1, steps - 1)
        route = "action" if step % 2 else "chain"
        action_f1 = lerp(0.62, 0.70, progress)
        wrong_selection = lerp(0.73, 0.60, progress)
        chain_reward = lerp(0.46, 0.53, progress)
        logic_alignment = lerp(0.25, 0.33, progress)
        date_mismatch = lerp(0.30, 0.20, progress)
        masked_rate = 0.012 + 0.004 * abs(math.sin(step / 6))
        common = {
            "step": step,
            "route": route,
            "loss": round(0.018 - 0.00022 * step + 0.001 * math.sin(step / 4), 7),
            "grad_norm": round(0.72 + 0.09 * abs(math.sin(step / 5)), 5),
            "learning_rate": 1e-6,
            "ratio_mean": round(1.0 + 0.003 * math.sin(step / 7), 7),
            "clip_fraction": round(0.012 * abs(math.sin(step / 8)), 7),
            "task_reward_mean": round(action_f1 if route == "action" else chain_reward, 5),
            "task_reward_std": round(0.16 - 0.025 * progress + 0.01 * abs(math.sin(step / 5)), 5),
            "zero_std_ratio": round(0.12 - 0.035 * progress, 5),
            "sequence_advantage_mean": round(0.01 * math.sin(step / 3), 6),
            "sequence_advantage_std": round(0.99 + 0.01 * math.sin(step / 8), 6),
            "token_advantage_mean": round(-0.018 - 0.006 * progress, 6),
            "token_advantage_std": round(1.01 + 0.02 * math.sin(step / 7), 6),
            "masked_candidate_rate": round(0.18 + 0.03 * abs(math.sin(step / 9)), 5),
            "masked_token_rate": round(masked_rate, 6),
            "positive_sequence_masked_token_flip_count": 2 + step % 4,
            "timestamp": iso(base, step * 42),
        }
        if route == "action":
            common.update({
                "f1_mean": round(action_f1, 5),
                "precision_mean": round(lerp(0.68, 0.75, progress), 5),
                "recall_mean": round(lerp(0.59, 0.67, progress), 5),
                "exact_match_rate": round(lerp(0.31, 0.39, progress), 5),
                "wrong_selection_candidate_rate": round(wrong_selection, 5),
                "hallucination_candidate_rate": round(lerp(0.08, 0.045, progress), 5),
                "duplicate_candidate_rate": round(lerp(0.05, 0.028, progress), 5),
                "per_kind_masked_token_count": {"hallucinated_sid": 5, "duplicate_sid": 8},
                "per_kind_incremental_negative_mass": {"hallucinated_sid": 2.5, "duplicate_sid": 2.0},
                "violation_counts": {"wrong_selection_sid": 23, "hallucinated_sid": 3, "duplicate_sid": 2},
            })
        else:
            common.update({
                "total_reward_mean": round(chain_reward, 5),
                "action_alignment_mean": round(lerp(0.67, 0.74, progress), 5),
                "logic_alignment_mean": round(logic_alignment, 5),
                "grounded_rate": round(lerp(0.78, 0.84, progress), 5),
                "partially_grounded_rate": round(lerp(0.17, 0.13, progress), 5),
                "ungrounded_rate": round(lerp(0.05, 0.03, progress), 5),
                "per_kind_masked_token_count": {
                    "hallucinated_sid": 3, "date_mismatch": 12, "action_mismatch": 9,
                    "duplicate_event": 15, "chronology_violation": 5, "excess_event": 4,
                },
                "per_kind_incremental_negative_mass": {
                    "hallucinated_sid": 1.5, "date_mismatch": 1.9, "action_mismatch": 1.6,
                    "duplicate_event": 1.1, "chronology_violation": 0.9, "excess_event": 0.7,
                },
                "violation_counts": {
                    "hallucinated_sid": 2, "date_mismatch": round(32 * date_mismatch),
                    "action_mismatch": 6, "duplicate_event": 3,
                    "chronology_violation": 2, "excess_event": 1,
                },
                "date_mismatch_candidate_rate": round(date_mismatch, 5),
                "action_mismatch_candidate_rate": round(lerp(0.22, 0.15, progress), 5),
            })
        writer.write_step(common)

        if step % 5 == 0:
            rollout_id += 1
            candidates = _action_candidates(step) if route == "action" else _chain_candidates(step)
            rewards = [candidate["reward"] for candidate in candidates]
            writer.write_rollout({
                "rollout_id": rollout_id,
                "step": step,
                "route": route,
                "g": 4,
                "group_ids": [f"demo-user-group-{rollout_id:02d}"],
                "reward_mean": sum(rewards) / len(rewards),
                "reward_std": 0.36 if route == "action" else 0.21,
                "zero_std_ratio": common["zero_std_ratio"],
                "completion_length_mean": sum(item["completion_length"] for item in candidates) / 4,
                "masked_candidate_rate": common["masked_candidate_rate"],
                "masked_token_rate": common["masked_token_rate"],
                "timestamp": common["timestamp"],
            })
            writer.write_trace({
                "rollout_id": rollout_id,
                "step": step,
                "route": route,
                "group_id": f"demo-user-group-{rollout_id:02d}",
                "candidates": candidates,
                "timestamp": common["timestamp"],
            })
    return str(writer.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/GRPO/runs")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--probes-only", action="store_true")
    args = parser.parse_args()
    if args.probes_only:
        writer = MonitorWriter(True, args.output_dir, args.run_id, rank=0, trace_every=1)
        probe_path = writer.run_dir / "probes.jsonl"
        if probe_path.exists():
            raise FileExistsError(f"demo probes already exist: {probe_path}")
        write_probe_demo(writer, args.steps)
        print(probe_path)
    else:
        print(generate(args.output_dir, args.run_id, args.steps))


if __name__ == "__main__":
    main()
