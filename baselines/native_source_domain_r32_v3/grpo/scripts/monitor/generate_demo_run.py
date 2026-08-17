"""Generate a deterministic, CPU-only GRPO monitor demo run."""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta, timezone

try:
    from .writer import MonitorWriter
except ImportError:  # direct script execution
    from writer import MonitorWriter


def iso(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


def generate(output_dir: str, run_id: str = "demo-phase1") -> str:
    rng = random.Random(20260817)
    writers = [MonitorWriter(True, output_dir, run_id, rank, trace_every=10) for rank in range(4)]
    base = datetime(2026, 8, 17, 12, 26, 40, 922100, tzinfo=timezone.utc)
    writers[0].write_manifest({
        "run_id": run_id,
        "start_time": iso(base, 0),
        "git_commit": "demo-cpu-only",
        "seed": 20260817,
        "world_size": 4,
        "model_path": "/models/Qwen3-8B",
        "adapter_path": "/adapters/BATA-BASELINE-R32-2E",
        "dataset_path": "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl",
        "dataset_version": "rec_mp_grpo_v2",
        "learning_rate": 1e-6,
        "max_steps": 100,
        "num_iterations": 2,
        "think_g": 4,
        "nothink_g": 8,
        "temperature": {"think": 0.9, "no_think": 1.0},
        "top_p": {"think": 0.95, "no_think": 1.0},
        "beta": 0.0,
        "epsilon": 0.2,
        "max_prompt_length": 8192,
        "max_completion_length": 2048,
        "beam32": {"num_beams": 32, "num_return_sequences": 32, "max_new_tokens": 128, "batch": 1},
        "beam_rank_balance": True,
    })

    elapsed = 0.0
    rollout_id = 0
    for step in range(1, 101):
        is_new = step % 2 == 1
        if is_new:
            rollout_id += 1
        route = "think" if rollout_id % 3 != 0 else "no_think"
        think = route == "think"
        reward = (2.3 + 0.018 * step + 0.35 * math.sin(step / 8)) if think else (-0.19 + 0.0014 * step)
        zero_std = max(0.03, 0.5 - step * 0.0035) if think else max(0.01, 0.17 - step * 0.0012)
        generation = (65 + 7 * math.sin(rollout_id / 4)) if think else (2.4 + 0.2 * math.sin(rollout_id))
        beam = (39 + 2 * math.cos(rollout_id / 5)) if think else 0.0
        policy = 0.92 if think else 0.36
        rollout_wall = generation + beam + policy + 1.4
        if is_new:
            elapsed += rollout_wall
            lengths = [int(rng.gauss(760 if think else 19, 100 if think else 1.2)) for _ in range(16)]
            rollout = {
                "rollout_id": rollout_id,
                "step": step,
                "route": route,
                "g": 4 if think else 8,
                "recommendation_group_ids": [f"group-{rollout_id:03d}-{i}" for i in range(4 if think else 2)],
                "reward_mean": round(reward, 5),
                "reward_std": round(0.8 + abs(math.sin(step / 11)), 5),
                "zero_std_ratio": round(zero_std, 5),
                "completion_length_mean": round(sum(lengths) / len(lengths), 2),
                "completion_length_min": min(lengths),
                "completion_length_max": max(lengths),
                "generation_wall_sec": round(generation, 3),
                "beam_wall_sec": round(beam, 3),
                "rollout_wall_sec": round(rollout_wall, 3),
                "timestamp": iso(base, elapsed),
            }
            if think:
                rollout.update({"closure_rate": round(0.92 + 0.07 * math.sin(step / 14), 4), "exact_count": 5 + rollout_id % 4, "ab_count": 2, "a_count": 3, "invalid_count": 0})
            else:
                rollout["reward_level_dist"] = {"-1": 0, "-0.25": 3, "0": 11, "0.5": 2, "2": 0, "8": 0}
            writers[0].write_rollout(rollout)

            for rank, writer in enumerate(writers):
                rank_beam = max(0.0, beam + (rank - 1.5) * 0.24 + rng.uniform(-0.12, 0.12))
                writer.write_rank({
                    "rollout_id": rollout_id,
                    "step": step,
                    "route": route,
                    "generation_wall_sec": round(generation + rng.uniform(-2, 2), 3),
                    "beam_exec_wall_sec": round(rank_beam, 3),
                    "beam_task_count": 4 if think else 0,
                    "beam_input_gather_wall_sec": 0.004 if think else 0.0,
                    "beam_result_gather_wall_sec": 0.003 if think else 0.0,
                    "peak_allocated_mb": 20100 + rank * 180,
                    "peak_reserved_mb": 47000 + rank * 240,
                    "timestamp": iso(base, elapsed),
                })

            if writers[0].trace_due(rollout_id):
                candidates = []
                for candidate in range(4 if think else 8):
                    beam_sids = None
                    if think:
                        beam_sids = []
                        for beam_index in range(32):
                            if candidate == 0 and beam_index == 0:
                                beam_sids.append(["prod", 1000, 42, 7])
                            elif candidate == 1 and beam_index == 0:
                                beam_sids.append(["prod", 1000, 42, 99])
                            elif candidate == 2 and beam_index == 0:
                                beam_sids.append(["prod", 1000, 77, 3])
                            else:
                                beam_sids.append(["prod", 2000 + candidate, 80 + beam_index, beam_index])
                    candidates.append({
                        "candidate_id": candidate,
                        "completion": (f"Consider user history and material fit for candidate {candidate}. </think>" if think else f"<s_a_{1000 + candidate}><s_b_42><s_c_7>"),
                        "completion_length": lengths[candidate],
                        "closed": True if think else None,
                        "parsed_sid": None if think else ["prod", 1000 + candidate, 42, 7],
                        "reward": 8.0 if candidate == 0 else (2.0 if candidate == 1 else 0.0),
                        "reward_level": None if think else (8.0 if candidate == 0 else (2.0 if candidate == 1 else 0.0)),
                        "exact": 1 if think and candidate == 0 else 0,
                        "ab": 1 if think and candidate == 1 else 0,
                        "a": 0,
                        "beam_sids": beam_sids,
                    })
                writers[0].write_trace({
                    "rollout_id": rollout_id,
                    "step": step,
                    "route": route,
                    "group_id": f"group-{rollout_id:03d}-0",
                    "gold_sids": [["prod", 1000, 42, 7]],
                    "candidates": candidates,
                    "timestamp": iso(base, elapsed),
                })

        ratio = 1.0 if is_new else 1.0 + 0.002 * math.sin(step / 7)
        writers[0].write_step({
            "step": step,
            "epoch": round(step / 100, 4),
            "rollout_id": rollout_id,
            "route": route,
            "loss": round(-0.0007 * math.sin(step / 9), 7),
            "grad_norm": round(0.28 + 0.11 * abs(math.sin(step / 13)), 5),
            "learning_rate": 1e-6,
            "ratio_mean": round(ratio, 7),
            "clip_fraction": 0.0 if is_new else round(0.015 * abs(math.sin(step / 5)), 6),
            "approx_kl": 0.0 if is_new else round(0.0017 * abs(math.sin(step / 6)), 7),
            "reward_mean": round(reward, 5),
            "reward_std": round(0.8 + abs(math.sin(step / 11)), 5),
            "zero_std_ratio": round(zero_std, 5),
            "completion_mean_length": round(760 + 80 * math.sin(step / 10), 2) if think else 18.2,
            "generation_wall_sec": round(generation, 3) if is_new else None,
            "rollout_wall_sec": round(rollout_wall, 3) if is_new else None,
            "policy_wall_sec": policy,
            "timestamp": iso(base, elapsed + (0 if is_new else policy)),
        })
    return str(writers[0].run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../../runs")
    parser.add_argument("--run-id", default="demo-phase1")
    args = parser.parse_args()
    print(generate(args.output_dir, args.run_id))


if __name__ == "__main__":
    main()
