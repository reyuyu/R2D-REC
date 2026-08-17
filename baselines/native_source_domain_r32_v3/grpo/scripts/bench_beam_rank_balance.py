# -*- coding: utf-8 -*-
"""4-GPU Beam32 determinism and rank load-balancing benchmark.

Diagnostic only: every Beam task keeps production input construction and
generation parameters. Only the rank executing a frozen task changes.
"""
import hashlib
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from grpo_model import encode_prompt, generate_batch, load_model, render_prompt
from grpo_sid import final_sid, parse_sid, think_reward
from grpo_trl_trainer import build_route_dataset
from run_grpo_trl_smoke import DATA


SEED = 20260816
WORLD = 4
ROLLOUTS = 2
G = 4
RESULT = "/data/GRPO/logs/beam_rank_balance.json"


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


def think_generation_config(tokenizer, max_new_tokens=2048):
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_p=0.95,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        cache_implementation=None,
        stop_strings=None,
    )


def hash_nested_ids(sequences):
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(",".join(str(token_id) for token_id in sequence).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def hash_ids(sequence):
    return hashlib.sha256(",".join(str(token_id) for token_id in sequence).encode("ascii")).hexdigest()


def parse_gold(raw_sids):
    return [
        list(parsed) for raw in raw_sids
        if (parsed := parse_sid(raw)) is not None
    ]


def generate_think(model, tokenizer, prompt, seed, think_token_id):
    rendered = [render_prompt(tokenizer, prompt)] * G
    inputs = tokenizer(
        text=rendered,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        max_length=8192,
        truncation=True,
        add_special_tokens=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    criteria = StoppingCriteriaList([ThinkTokenStop(think_token_id)])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=think_generation_config(tokenizer),
            disable_compile=True,
            tokenizer=tokenizer,
            stopping_criteria=criteria,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    prompt_width = inputs["input_ids"].size(1)
    completions = []
    closures = []
    for ids in output[:, prompt_width:].tolist():
        try:
            closure = ids.index(think_token_id)
            ids = ids[: closure + 1]
        except ValueError:
            closure = None
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
        completions.append(ids)
        closures.append(closure)
    return {
        "completion_ids": completions,
        "completion_lengths": [len(ids) for ids in completions],
        "closure_positions": closures,
        "prompt_tokens": int(inputs["attention_mask"][0].sum()),
        "wall_sec": wall,
    }


def production_beam_context(tokenizer, prompt, completion_ids, prompt_ids_cache):
    """Exact current production decode/trim/encode Beam context path."""
    cot = tokenizer.decode(completion_ids, skip_special_tokens=False)
    close = cot.find("</think>")
    cot_trim = cot[: close + len("</think>")] if close >= 0 else cot
    if prompt not in prompt_ids_cache:
        prompt_ids_cache[prompt] = encode_prompt(tokenizer, prompt)
    cot_ids = tokenizer.encode(cot_trim, add_special_tokens=False)
    return prompt_ids_cache[prompt] + cot_ids


def build_local_tasks(tokenizer, row, generation, rollout_id, rank):
    cache = {}
    gold = parse_gold(row["all_gold_sids"])
    tasks = []
    for local_index, completion_ids in enumerate(generation["completion_ids"]):
        input_ids = production_beam_context(tokenizer, row["prompt"], completion_ids, cache)
        tasks.append({
            "task_id": f"rollout{rollout_id}:rank{rank}:local{local_index}",
            "rollout_id": rollout_id,
            "origin_rank": rank,
            "local_index": local_index,
            "original_index": rank * G + local_index,
            "recommendation_group_id": row["recommendation_group_id"],
            "input_ids": input_ids,
            "input_sha256": hash_ids(input_ids),
            "context_length": len(input_ids),
            "gold": gold,
        })
    return tasks


def execute_task(model, tokenizer, task, rank):
    torch.cuda.synchronize()
    started = time.perf_counter()
    texts, output_ids = generate_batch(
        model,
        tokenizer,
        [task["input_ids"]],
        max_new_tokens=128,
        do_sample=False,
        num_beams=32,
        num_return_sequences=32,
        return_ids=True,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    sids = [final_sid(text) for text in texts]
    gold = {tuple(item) for item in task["gold"]}
    reward, exact, ab, a = think_reward(sids, gold)
    return {
        "task_id": task["task_id"],
        "rollout_id": task["rollout_id"],
        "origin_rank": task["origin_rank"],
        "local_index": task["local_index"],
        "original_index": task["original_index"],
        "executed_rank": rank,
        "context_length": task["context_length"],
        "input_sha256": task["input_sha256"],
        "beam_wall_sec": wall,
        "output_ids": output_ids,
        "output_sha256": hash_nested_ids(output_ids),
        "sids": [list(sid) if sid is not None else None for sid in sids],
        "reward": reward,
        "exact": exact,
        "ab": ab,
        "a": a,
        "invalid": sum(sid is None for sid in sids),
    }


def all_gather_objects(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def timed_all_gather(value):
    dist.barrier()
    started = time.perf_counter()
    gathered = all_gather_objects(value)
    elapsed = time.perf_counter() - started
    times = all_gather_objects(elapsed)
    return gathered, times


def task_order(task):
    return task["rollout_id"], task["original_index"]


def cost(task, proxy):
    length = task["context_length"]
    if proxy == "length":
        return float(length)
    if proxy == "quadratic":
        return float(length * length + 128 * length)
    raise ValueError(proxy)


def schedule_lpt(tasks, proxy, world):
    assignments = {rank: [] for rank in range(world)}
    loads = [0.0] * world
    for task in sorted(tasks, key=lambda item: (-cost(item, proxy), item["task_id"])):
        target = min(range(world), key=lambda rank: (loads[rank], rank))
        assignments[target].append(task["task_id"])
        loads[target] += cost(task, proxy)
    return assignments, loads


def baseline_assignment(tasks, world):
    return {
        rank: [task["task_id"] for task in tasks if task["origin_rank"] == rank]
        for rank in range(world)
    }


def run_phase(model, tokenizer, tasks, assignment, rank, phase):
    by_id = {task["task_id"]: task for task in tasks}
    assigned = assignment[rank]
    dist.barrier()
    phase_started = time.perf_counter()
    local_results = [execute_task(model, tokenizer, by_id[task_id], rank) for task_id in assigned]
    torch.cuda.synchronize()
    local_compute_sec = time.perf_counter() - phase_started
    dist.barrier()
    critical_sec = time.perf_counter() - phase_started

    gather_started = time.perf_counter()
    gathered_results = all_gather_objects(local_results)
    result_gather_sec = time.perf_counter() - gather_started
    timing = {
        "rank": rank,
        "phase": phase,
        "task_count": len(assigned),
        "task_ids": assigned,
        "task_wall_sum_sec": sum(item["beam_wall_sec"] for item in local_results),
        "local_compute_sec": local_compute_sec,
        "critical_barrier_sec": critical_sec,
        "result_gather_sec": result_gather_sec,
    }
    gathered_timing = all_gather_objects(timing)
    flat_results = [item for rank_items in gathered_results for item in rank_items]
    flat_results.sort(key=lambda item: (item["rollout_id"], item["original_index"]))
    if len(flat_results) != len(tasks) or len({item["task_id"] for item in flat_results}) != len(tasks):
        raise RuntimeError(f"{phase} did not execute every task exactly once")
    return flat_results, gathered_timing


def compare_results(reference, candidate):
    by_id = {item["task_id"]: item for item in candidate}
    task_checks = []
    for expected in reference:
        actual = by_id[expected["task_id"]]
        output_equal = expected["output_ids"] == actual["output_ids"]
        sid_equal = expected["sids"] == actual["sids"]
        metrics_equal = all(
            expected[key] == actual[key]
            for key in ("reward", "exact", "ab", "a", "invalid")
        )
        task_checks.append({
            "task_id": expected["task_id"],
            "input_equal": expected["input_sha256"] == actual["input_sha256"],
            "output_equal": output_equal,
            "output_sequences_equal": sum(
                left == right for left, right in zip(expected["output_ids"], actual["output_ids"])
            ),
            "sid_equal": sid_equal,
            "sid_count_equal": sum(left == right for left, right in zip(expected["sids"], actual["sids"])),
            "metrics_equal": metrics_equal,
            "expected_hash": expected["output_sha256"],
            "candidate_hash": actual["output_sha256"],
        })
    return {
        "tasks": task_checks,
        "input_ids_equal": [sum(item["input_equal"] for item in task_checks), len(task_checks)],
        "output_tasks_equal": [sum(item["output_equal"] for item in task_checks), len(task_checks)],
        "output_sequences_equal": [sum(item["output_sequences_equal"] for item in task_checks), len(task_checks) * 32],
        "sid_tasks_equal": [sum(item["sid_equal"] for item in task_checks), len(task_checks)],
        "sid_sequences_equal": [sum(item["sid_count_equal"] for item in task_checks), len(task_checks) * 32],
        "metrics_equal": [sum(item["metrics_equal"] for item in task_checks), len(task_checks)],
    }


def summarize_phase(timing, input_gather_times=None, scheduler_times=None):
    task_walls = [item["task_wall_sum_sec"] for item in timing]
    local_compute = [item["local_compute_sec"] for item in timing]
    gather = [item["result_gather_sec"] for item in timing]
    summary = {
        "rank_task_wall_sec": task_walls,
        "rank_local_compute_sec": local_compute,
        "rank_task_counts": [item["task_count"] for item in timing],
        "critical_path_sec": max(local_compute),
        "rank_wall_min_sec": min(local_compute),
        "rank_wall_mean_sec": statistics.mean(local_compute),
        "rank_wall_max_sec": max(local_compute),
        "result_gather_sec_by_rank": gather,
        "result_gather_max_sec": max(gather),
    }
    if input_gather_times is not None:
        summary["input_gather_sec_by_rank"] = input_gather_times
        summary["input_gather_max_sec"] = max(input_gather_times)
    if scheduler_times is not None:
        summary["scheduler_sec_by_rank"] = scheduler_times
        summary["scheduler_max_sec"] = max(scheduler_times)
    summary["candidate_e2e_sec"] = summary["critical_path_sec"]
    if input_gather_times is not None:
        summary["candidate_e2e_sec"] += max(input_gather_times)
    if scheduler_times is not None:
        summary["candidate_e2e_sec"] += max(scheduler_times)
    summary["candidate_e2e_sec"] += max(gather)
    return summary


def pearson(left, right):
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def ranks(values):
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for position in range(start, end):
            result[ordered[position]] = average_rank
        start = end
    return result


def cross_rank_determinism(model, tokenizer, tasks, rank):
    ordered = sorted(tasks, key=lambda task: (task["context_length"], task["task_id"]))
    indices = sorted({0, len(ordered) // 5, 2 * len(ordered) // 5,
                      3 * len(ordered) // 5, 4 * len(ordered) // 5, len(ordered) - 1})
    selected = [ordered[index] for index in indices]
    dist.barrier()
    local = [execute_task(model, tokenizer, task, rank) for task in selected]
    gathered = all_gather_objects(local)
    checks = []
    for task_index, task in enumerate(selected):
        rank_results = [gathered[other_rank][task_index] for other_rank in range(dist.get_world_size())]
        reference = rank_results[0]
        checks.append({
            "task_id": task["task_id"],
            "context_length": task["context_length"],
            "rank_walls_sec": [item["beam_wall_sec"] for item in rank_results],
            "rank_hashes": [item["output_sha256"] for item in rank_results],
            "output_ids_equal": all(item["output_ids"] == reference["output_ids"] for item in rank_results),
            "output_sequences_equal": sum(
                sequence == reference["output_ids"][sequence_index]
                for item in rank_results
                for sequence_index, sequence in enumerate(item["output_ids"])
            ),
            "sids_equal": all(item["sids"] == reference["sids"] for item in rank_results),
            "reward_equal": all(item["reward"] == reference["reward"] for item in rank_results),
            "metrics_equal": all(
                (item["exact"], item["ab"], item["a"], item["invalid"])
                == (reference["exact"], reference["ab"], reference["a"], reference["invalid"])
                for item in rank_results
            ),
            "rank_results": rank_results,
        })
    return {
        "contexts": len(selected),
        "context_lengths": [task["context_length"] for task in selected],
        "checks": checks,
        "output_contexts_equal": [sum(item["output_ids_equal"] for item in checks), len(checks)],
        "output_sequences_equal": [sum(item["output_sequences_equal"] for item in checks), len(checks) * WORLD * 32],
        "sid_equal": [sum(item["sids_equal"] for item in checks), len(checks)],
        "reward_equal": [sum(item["reward_equal"] for item in checks), len(checks)],
        "metrics_equal": [sum(item["metrics_equal"] for item in checks), len(checks)],
    }


def save_result(result):
    temporary = RESULT + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, RESULT)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != WORLD:
        raise RuntimeError(f"expected {WORLD} ranks, got {world}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = f"cuda:{rank}"

    model, tokenizer, _ = load_model(device)
    model.eval()
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) != 1:
        raise RuntimeError(f"</think> must be one token, got {think_tokens}")
    think_token_id = think_tokens[0]

    dataset = build_route_dataset(DATA, n_groups=8, seed=SEED, chunk=8)
    think_rows = [row for row in dataset if row["route"] == "think"]
    if len(think_rows) != ROLLOUTS * WORLD:
        raise RuntimeError(f"expected {ROLLOUTS * WORLD} Think rows, got {len(think_rows)}")

    local_rollout_tasks = []
    local_generation = []
    for rollout_id in range(ROLLOUTS):
        row = think_rows[rollout_id * WORLD + rank]
        generation = generate_think(
            model, tokenizer, row["prompt"], SEED + rollout_id * WORLD + rank, think_token_id,
        )
        tasks = build_local_tasks(tokenizer, row, generation, rollout_id, rank)
        local_rollout_tasks.append(tasks)
        local_generation.append({
            "rollout_id": rollout_id,
            "rank": rank,
            "recommendation_group_id": row["recommendation_group_id"],
            **{key: value for key, value in generation.items() if key != "completion_ids"},
        })
        print(
            f"rank={rank} rollout={rollout_id} generation={generation['wall_sec']:.3f}s "
            f"prompt={generation['prompt_tokens']} completion={generation['completion_lengths']}",
            flush=True,
        )

    gathered_tasks = []
    input_gather_times = []
    for rollout_id in range(ROLLOUTS):
        gathered, times = timed_all_gather(local_rollout_tasks[rollout_id])
        tasks = [task for rank_tasks in gathered for task in rank_tasks]
        tasks.sort(key=task_order)
        gathered_tasks.append(tasks)
        input_gather_times.append(times)

    all_tasks = [task for rollout in gathered_tasks for task in rollout]
    # One excluded production-shape Beam warmup per rank.
    execute_task(model, tokenizer, all_tasks[rank], rank)
    determinism = cross_rank_determinism(model, tokenizer, all_tasks, rank)
    if not all(item["output_ids_equal"] for item in determinism["checks"]):
        if rank == 0:
            save_result({"status": "cross_rank_determinism_failed", "determinism": determinism})
        dist.barrier()
        raise RuntimeError("cross-rank Beam output token parity failed")

    gathered_generation = all_gather_objects(local_generation)
    generation = [item for rank_items in gathered_generation for item in rank_items]
    generation.sort(key=lambda item: (item["rollout_id"], item["rank"]))

    rollout_results = []
    for rollout_id, tasks in enumerate(gathered_tasks):
        baseline = baseline_assignment(tasks, world)
        assignments = {"baseline": baseline}
        predicted_loads = {"baseline": [sum(task["context_length"] for task in tasks if task["origin_rank"] == r)
                                          for r in range(world)]}
        scheduler_times = {}
        for proxy in ("length", "quadratic"):
            started = time.perf_counter()
            assignment, loads = schedule_lpt(tasks, proxy, world)
            scheduler_elapsed = time.perf_counter() - started
            assignments[proxy] = assignment
            predicted_loads[proxy] = loads
            scheduler_times[proxy] = all_gather_objects(scheduler_elapsed)

        phase_order = ("baseline", "length", "quadratic") if rollout_id == 0 else ("quadratic", "length", "baseline")
        phases = {}
        for phase in phase_order:
            results, timing = run_phase(model, tokenizer, tasks, assignments[phase], rank, phase)
            phases[phase] = {
                "results": results,
                "timing": timing,
                "summary": summarize_phase(
                    timing,
                    input_gather_times[rollout_id] if phase != "baseline" else None,
                    scheduler_times.get(phase),
                ),
            }
            print(
                f"rank={rank} rollout={rollout_id} phase={phase} "
                f"tasks={timing[rank]['task_count']} local={timing[rank]['local_compute_sec']:.3f}s",
                flush=True,
            )

        for phase in ("length", "quadratic"):
            phases[phase]["parity"] = compare_results(phases["baseline"]["results"], phases[phase]["results"])
            baseline_critical = phases["baseline"]["summary"]["critical_path_sec"]
            candidate_e2e = phases[phase]["summary"]["candidate_e2e_sec"]
            phases[phase]["critical_speedup_pct"] = (1 - phases[phase]["summary"]["critical_path_sec"] / baseline_critical) * 100
            phases[phase]["e2e_speedup_pct"] = (1 - candidate_e2e / baseline_critical) * 100

        rollout_results.append({
            "rollout_id": rollout_id,
            "tasks": tasks,
            "input_gather_sec_by_rank": input_gather_times[rollout_id],
            "assignments": assignments,
            "predicted_loads": predicted_loads,
            "scheduler_sec_by_rank": scheduler_times,
            "phase_order": phase_order,
            "phases": phases,
        })

    if rank == 0:
        baseline_task_results = [
            item for rollout in rollout_results for item in rollout["phases"]["baseline"]["results"]
        ]
        lengths = [item["context_length"] for item in baseline_task_results]
        walls = [item["beam_wall_sec"] for item in baseline_task_results]
        correlation = {
            "tasks": len(lengths),
            "pearson_context_length_vs_wall": pearson(lengths, walls),
            "spearman_context_length_vs_wall": pearson(ranks(lengths), ranks(walls)),
        }
        aggregate = {}
        for phase in ("baseline", "length", "quadratic"):
            critical = [rollout["phases"][phase]["summary"]["critical_path_sec"] for rollout in rollout_results]
            e2e = [
                rollout["phases"][phase]["summary"]["candidate_e2e_sec"]
                if phase != "baseline" else rollout["phases"][phase]["summary"]["critical_path_sec"]
                for rollout in rollout_results
            ]
            aggregate[phase] = {
                "critical_path_values_sec": critical,
                "critical_path_mean_sec": statistics.mean(critical),
                "critical_path_median_sec": statistics.median(critical),
                "e2e_values_sec": e2e,
                "e2e_mean_sec": statistics.mean(e2e),
                "e2e_median_sec": statistics.median(e2e),
            }
        baseline_mean = aggregate["baseline"]["critical_path_mean_sec"]
        for phase in ("length", "quadratic"):
            aggregate[phase]["critical_speedup_pct"] = (
                1 - aggregate[phase]["critical_path_mean_sec"] / baseline_mean
            ) * 100
            aggregate[phase]["e2e_speedup_pct"] = (
                1 - aggregate[phase]["e2e_mean_sec"] / baseline_mean
            ) * 100

        generation_critical = [
            max(item["wall_sec"] for item in generation if item["rollout_id"] == rollout_id)
            for rollout_id in range(ROLLOUTS)
        ]
        for phase in ("baseline", "length", "quadratic"):
            beam = aggregate[phase]["e2e_values_sec"]
            rollout_wall = [gen + beam_wall for gen, beam_wall in zip(generation_critical, beam)]
            aggregate[phase]["think_rollout_values_sec"] = rollout_wall
            aggregate[phase]["think_rollout_mean_sec"] = statistics.mean(rollout_wall)
        baseline_rollout_mean = aggregate["baseline"]["think_rollout_mean_sec"]
        for phase in ("length", "quadratic"):
            aggregate[phase]["think_rollout_speedup_pct"] = (
                1 - aggregate[phase]["think_rollout_mean_sec"] / baseline_rollout_mean
            ) * 100

        result = {
            "status": "completed",
            "world": world,
            "rollouts": ROLLOUTS,
            "tasks": len(all_tasks),
            "beam_contract": {
                "context_batch": 1,
                "num_beams": 32,
                "num_return_sequences": 32,
                "max_new_tokens": 128,
                "do_sample": False,
                "cache_implementation": None,
            },
            "input_construction": "decode -> text </think> trim -> encode -> prompt_ids + cot_ids",
            "generation": generation,
            "generation_critical_sec": generation_critical,
            "determinism": determinism,
            "correlation": correlation,
            "rollout_results": rollout_results,
            "aggregate": aggregate,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
        }
        save_result(result)
        print(json.dumps({
            "determinism": {
                key: determinism[key] for key in (
                    "contexts", "context_lengths", "output_contexts_equal",
                    "output_sequences_equal", "sid_equal", "reward_equal", "metrics_equal",
                )
            },
            "correlation": correlation,
            "aggregate": aggregate,
        }, ensure_ascii=False, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
