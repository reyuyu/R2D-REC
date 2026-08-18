"""Generate a deterministic, CPU-only GRPO monitor demo run."""
from __future__ import annotations

import argparse
import math
import random
import statistics
from datetime import datetime, timedelta, timezone

try:
    from .writer import MonitorWriter
except ImportError:  # direct script execution
    from writer import MonitorWriter


def iso(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


DOMAINS = ("video", "living", "prod", "ad")


def _probe_dsr(think_candidates, nothink_candidates, step):
    branch = "dead_zero" if step == 0 else "prefix_rescue" if step == 200 else "primary_variance"
    grounded = 2.0 + step / 400
    think_details = [{
        "raw_interest_n": 4,
        "grounded_n": int(grounded),
        "parser_success": True,
        "grounded_interests": [{
            "title": "Verified interest",
            "verified_sids": [think_candidates[index]["beam_sids"][0]],
        }],
        "ungrounded_or_fake_sids": [],
        "d_cot": 0.72 + 0.08 * index,
        "s_cot": 0.80 + 0.04 * index,
        "s_prefix": 0.03 + step / 2000,
        "s_explore": 0.42 - step / 2000,
        "s_dead": 0.34,
        "s_aux": 0.55 + 0.05 * index,
        "a_aux": (-1.3, -0.4, 0.4, 1.3)[index],
    } for index in range(4)]
    predicted = [item["parsed_sid"][1] for item in nothink_candidates]
    frequencies = {value: predicted.count(value) for value in set(predicted)}
    concentration = max(frequencies.values()) / 8
    rewards = [item["reward"] for item in nothink_candidates]
    active = all(value == 0 for value in rewards)
    no_details = [{
        "predicted_a": value,
        "a_frequency": frequencies[value],
        "frequency_weight": frequencies[value] / 8,
        "gold_a": reward >= 0.5,
        "gold_ab": reward >= 2,
        "exact": reward == 8,
        "rescue_target": active,
    } for value, reward in zip(predicted, rewards)]
    return {
        "think": {
            "parser_success_rate": 1.0,
            "grounded_n_mean": grounded,
            "cot_group_similarity": 0.78 - step / 1000,
            "s_cot_mean": statistics.fmean(item["s_cot"] for item in think_details),
            "s_prefix_mean": statistics.fmean(item["s_prefix"] for item in think_details),
            "s_explore_mean": statistics.fmean(item["s_explore"] for item in think_details),
            "s_dead_mean": 0.34,
            "branch": branch,
            "aux_std": 0.056,
            "primary_zero_std": step < 400,
            "candidates": think_details,
        },
        "nothink": {
            "primary_zero_std": len(set(rewards)) == 1,
            "all_zero": active,
            "predicted_as": predicted,
            "a_frequencies": {str(key): value for key, value in frequencies.items()},
            "concentration": concentration,
            "gold_unique_a": 1,
            "would_rescue": active,
            "lambda_a": 0.10,
            "coefficient": 0.10 * concentration if active else 0.0,
            "any_gold_a": any(value >= 0.5 for value in rewards),
            "any_gold_ab": any(value >= 2 for value in rewards),
            "any_exact": any(value == 8 for value in rewards),
            "candidates": no_details,
        },
    }


def generate(output_dir: str, run_id: str = "demo-phase1", dsr: bool = False) -> str:
    rng = random.Random(20260817)
    writers = [MonitorWriter(True, output_dir, run_id, rank, trace_every=10) for rank in range(4)]
    base = datetime(2026, 8, 17, 12, 26, 40, 922100, tzinfo=timezone.utc)
    manifest = {
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
        "max_steps": 400 if dsr else 100,
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
        "fixed_probe": {
            "enabled": True,
            "group_ids": [f"probe-group-{index}" for index in range(4)],
            "every_steps": 200 if dsr else 50,
            "seed": 20260818,
            "excluded_from_training": True,
        },
    }
    if dsr:
        manifest.update({
            "experiment": "GR_REC_DSR_Ablation_v1",
            "parent": "BATA baseline",
            "baseline": "GR_REC_v1",
            "dsr": {"think_lambda": 0.10, "nothink_scale": 1.0, "monitor_schema": 2},
        })
    writers[0].write_manifest(manifest)

    for step in ((0, 200, 400) if dsr else (0, 50, 100)):
        for group_index in range(4):
            domain = DOMAINS[group_index]
            gold = [domain, 1000 + group_index, 42, 7]
            think_candidates = []
            for candidate_index in range(4):
                if dsr and step == 0:
                    reward = 0.0
                elif dsr and step == 200:
                    reward = 0.5
                elif dsr:
                    reward = 2.0 + candidate_index * 1.5 + group_index * 0.1
                else:
                    reward = min(8.0, candidate_index * 0.5 + step * 0.025 + group_index * 0.1)
                beams = [gold if beam_index == 0 and step >= 50 else
                         [domain, 2000 + candidate_index, 80 + beam_index, beam_index]
                         for beam_index in range(32)]
                think_candidates.append({
                    "completion": f"Fixed probe reasoning {group_index}/{candidate_index} at step {step}. </think>",
                    "completion_length": 520 + candidate_index * 31,
                    "completion_sha256": f"think-{step}-{group_index}-{candidate_index}",
                    "closed": True,
                    "reward": reward,
                    "exact": 1 if step >= 50 else 0,
                    "ab": candidate_index % 2,
                    "a": 1,
                    "invalid": 0,
                    "beam_sids": beams,
                })
            nothink_candidates = []
            for candidate_index in range(8):
                exact = step >= (400 if dsr else 100) and candidate_index == 0
                a_hit = dsr and step >= 200 and candidate_index == 0
                if exact:
                    sid = gold
                    reward = 8.0
                elif a_hit:
                    sid = [domain, gold[1], 99, candidate_index]
                    reward = 0.5
                else:
                    wrong_a = 3000 if dsr and step == 0 and candidate_index < 6 else 3000 + candidate_index
                    sid = [domain, wrong_a, 90, candidate_index]
                    reward = 0.0
                nothink_candidates.append({
                    "completion": f"<|{domain}_begin|><s_a_{sid[1]}><s_b_{sid[2]}><s_c_{sid[3]}>",
                    "completion_length": 18,
                    "completion_sha256": f"nothink-{step}-{group_index}-{candidate_index}",
                    "parsed_sid": sid,
                    "reward": reward,
                })
            probe_event = {
                "step": step,
                "reason": "baseline" if step == 0 else "final" if step == (400 if dsr else 100) else "interval",
                "group_id": f"probe-group-{group_index}",
                "target_domain": domain,
                "gold_sids": [gold],
                "think_prompt": f"Think fixed prompt {group_index}",
                "nothink_prompt": f"NoThink fixed prompt {group_index}",
                "think": {
                    "reward_mean": statistics.fmean(item["reward"] for item in think_candidates),
                    "reward_std": statistics.pstdev(item["reward"] for item in think_candidates),
                    "closure_rate": 1.0,
                    "exact_count": sum(item["exact"] for item in think_candidates),
                    "ab_count": sum(item["ab"] for item in think_candidates),
                    "a_count": sum(item["a"] for item in think_candidates),
                    "invalid_count": 0,
                    "candidates": think_candidates,
                },
                "nothink": {
                    "reward_mean": statistics.fmean(item["reward"] for item in nothink_candidates),
                    "reward_std": statistics.pstdev(item["reward"] for item in nothink_candidates),
                    "candidates": nothink_candidates,
                },
                "probe_wall_sec": 108.0,
                "seed": 20260818,
            }
            if dsr:
                probe_event["dsr"] = _probe_dsr(think_candidates, nothink_candidates, step)
            writers[0].write_probe(probe_event)

    elapsed = 0.0
    rollout_id = 0
    for step in range(1, (401 if dsr else 101)):
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

            if dsr:
                if think:
                    primary_zero = max(0.05, 0.48 - step * 0.003)
                    dsr_rollout = {
                        "type": "dsr_rollout", "rollout_id": rollout_id, "step": step,
                        "route": route, "parser_success_rate": min(0.995, 0.91 + 0.0002 * step),
                        "grounded_interest_count_mean": 2.1 + 0.003 * step,
                        "grounded_interest_count_distribution": {
                            "0": 0, "1": 1, "2": 5, "3": 8, "4": 2, "5+": 0,
                        },
                        "s_cot_mean": 0.72 + 0.0005 * step, "s_cot_std": 0.12,
                        "d_cot_mean": 0.66 + 0.0004 * step, "s_prefix_mean": 0.04 + 0.0006 * step,
                        "s_explore_mean": 0.31, "s_dead_mean": 0.24,
                        "primary_zero_std_rate": primary_zero, "all_zero_rate": primary_zero * 0.62,
                        "aux_zero_std_rate": 0.18, "think_aux_zero_std_rate": 0.18,
                        "signal_rescue_rate": primary_zero * 0.73,
                        "unique_a_mean": 5.0 + 0.005 * step, "a_entropy_mean": 0.61 + 0.0007 * step,
                        "correct_a_support": 0.08 + 0.001 * step,
                        "correct_ab_support": 0.025 + 0.0005 * step,
                        "branches": {"primary_variance": 2, "dead_zero": 1, "prefix_rescue": 1, "cot_only_saturated_or_other": 0},
                        "dsr_record_gather_wall_sec": 0.006 + 0.001 * math.sin(step),
                    }
                else:
                    phase = "healthy" if step <= 50 else "wrong_a_rotation"
                    concentration = max(0.2, 0.82 - 0.006 * step)
                    gold_a_hit = min(0.65, 0.12 + 0.009 * step) if phase == "healthy" else 0.34
                    dsr_rollout = {
                        "type": "dsr_rollout", "rollout_id": rollout_id, "step": step,
                        "route": route, "primary_zero_std_rate": 0.42, "all_zero_group_rate": 0.31,
                        "rescue_active_group_rate": 0.31, "all_zero_same_a_rate": 0.08,
                        "group_a_concentration_mean": concentration,
                        "group_a_concentration_p50": concentration,
                        "group_a_concentration_p90": min(1.0, concentration + 0.14),
                        "group_a_concentration_max": min(1.0, concentration + 0.2),
                        "any_gold_a_hit_rate": gold_a_hit, "any_gold_ab_hit_rate": gold_a_hit * 0.35,
                        "any_exact_hit_rate": gold_a_hit * 0.08, "valid_sid_rate": 0.96,
                        "wrong_domain_rate": 0.04,
                        "candidate_reward_rate_distribution": {"-1": 0.04, "-0.25": 0.04, "0": 0.58, "0.5": 0.22, "2": 0.09, "8": 0.03},
                        "gold_a_strata": {
                            "sparse": {"group_count": 1, "all_zero_rate": 0.38, "rescue_active_rate": 0.38, "a_concentration_mean": concentration + 0.05, "any_gold_a_hit_rate": gold_a_hit, "reward_mean": 0.21},
                            "dense": {"group_count": 1, "all_zero_rate": 0.24, "rescue_active_rate": 0.24, "a_concentration_mean": concentration - 0.05, "any_gold_a_hit_rate": gold_a_hit + 0.05, "reward_mean": 0.34},
                        },
                        "rotation_scenario": phase,
                        "dsr_record_gather_wall_sec": 0.005,
                    }
                writers[0]._append("dsr_metrics.jsonl", dsr_rollout)
                if rollout_id % 10 == 0:
                    writers[0]._append("dsr_traces.jsonl", {
                        "type": "dsr_trace", "rollout_id": rollout_id, "step": step,
                        "route": route, "candidates": [],
                    })

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
        step_event = {
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
        }
        writers[0].write_step(step_event)
        if dsr:
            raw_aux = 0.0012 * math.sin(step / 8) if think else 0.0
            rescue = 0.0016 * abs(math.sin(step / 11)) if not think else 0.0
            primary = step_event["loss"] - 0.1 * raw_aux - rescue
            writers[0]._append("dsr_steps.jsonl", {
                "type": "dsr_step", "step": step, "rollout_id": rollout_id, "route": route,
                "primary_loss": primary, "think_aux_loss_raw": raw_aux,
                "think_aux_contribution": 0.1 * raw_aux,
                "nothink_rescue_loss": rescue, "dsr_total_loss": step_event["loss"],
                "ratio_mean": step_event["ratio_mean"], "clip_fraction": step_event["clip_fraction"],
                "approx_kl": step_event["approx_kl"], "grad_norm": step_event["grad_norm"],
            })
    return str(writers[0].run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../../runs")
    parser.add_argument("--run-id", default="demo-phase1")
    parser.add_argument("--mode", choices=("legacy", "dsr"), default="legacy")
    args = parser.parse_args()
    print(generate(args.output_dir, args.run_id, dsr=args.mode == "dsr"))


if __name__ == "__main__":
    main()
