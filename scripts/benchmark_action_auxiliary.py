"""Benchmark the legacy and vectorized Action Select auxiliary backends.

This isolates the work performed after the model forward, using the same logits,
labels, metadata, dtype, warmup, cap, and device for both implementations.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import torch

from llamafactory.train.sft.user_action_auxiliary import (
    ACTION_STAT_SIZE,
    UserActionAuxiliaryController,
)


class BenchmarkTokenizer:
    def __init__(self, domain_count: int = 4, type_count: int = 8192) -> None:
        vocab: dict[str, int] = {}
        domains = ("video", "prod", "ad", "living")
        for index in range(domain_count):
            vocab[f"<|{domains[index]}_begin|>"] = 10 + index
        start = 32
        for group in ("a", "b", "c"):
            for index in range(type_count):
                vocab[f"<s_{group}_{index}>"] = start + index
            start += type_count
        self._vocab = vocab

    def get_added_vocab(self):
        return self._vocab


def make_args(vectorized: bool, chunk_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        user_action_history_trie_enabled=True,
        user_action_history_trie_weight=0.06,
        user_action_length_guard_enabled=True,
        user_action_continue_domain_extra=0.75,
        user_action_continue_separator_extra=0.20,
        user_action_no_early_stop_weight=0.02,
        user_action_stop_domain_weight=0.05,
        user_action_stop_tail_extra=1.0,
        user_action_max_stop_tail_positions=4,
        user_action_aux_cap_ratio=0.08,
        user_action_aux_warmup_steps=100,
        user_action_aux_vectorized_enabled=vectorized,
        user_action_aux_full_vocab_chunk_size=chunk_size,
    )


@dataclass(frozen=True)
class Case:
    name: str
    segment_sid_counts: tuple[int, ...]


def build_inputs(
    case: Case,
    tokenizer: BenchmarkTokenizer,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sequence_length: int = 0,
):
    semantic = UserActionAuxiliaryController(tokenizer, make_args(False, 64))._semantic_cpu
    domains, a_ids, b_ids, c_ids = (semantic[name] for name in ("domain", "a", "b", "c"))
    candidates = list(product(domains, a_ids[:8], b_ids[:8], c_ids[:8]))
    metadata = []
    labels: list[int] = [-100]
    cursor = 1
    candidate_offset = 0
    for sid_count in case.segment_sid_counts:
        answer = candidates[candidate_offset : candidate_offset + sid_count]
        candidate_offset += sid_count
        history = answer + candidates[candidate_offset : candidate_offset + max(4, sid_count // 2)]
        candidate_offset += max(4, sid_count // 2)
        units = []
        boundaries = []
        for sid_index, sid in enumerate(answer):
            positions = tuple(range(cursor, cursor + 4))
            labels.extend(sid)
            units.append(
                {
                    "value": list(sid),
                    "pos_domain": positions[0],
                    "pos_a": positions[1],
                    "pos_b": positions[2],
                    "pos_c": positions[3],
                }
            )
            cursor += 4
            if sid_index + 1 < sid_count:
                labels.append(5)
                boundaries.append(
                    {
                        "next_domain_pos": cursor + 1,
                        "continue_token_pos": cursor,
                        "stop_token_id": 6,
                    }
                )
                cursor += 1
        tail_positions = list(range(cursor, cursor + 4))
        labels.extend((7, 8, 9, 15))
        cursor += 4
        metadata.append(
            {
                "action_select": True,
                "parse_valid": True,
                "parse_ms": 0.0,
                "history_sids": [list(sid) for sid in history],
                "answer_sid_units": units,
                "continuation_boundaries": boundaries,
                "final_tail_positions": tail_positions,
                "gold_duplicate": False,
            }
        )
    if sequence_length:
        if sequence_length < len(labels):
            raise ValueError(
                f"sequence_length={sequence_length} is smaller than the generated case length={len(labels)}."
            )
        labels.extend([-100] * (sequence_length - len(labels)))
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(20260803)
    logits = torch.randn(
        1,
        len(labels),
        vocab_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return logits, labels_tensor, metadata


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def measure_cuda(fn, warmup: int, repeats: int, backward: bool) -> tuple[dict[str, float], float]:
    for _ in range(warmup):
        loss = fn()
        if backward:
            loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    timings = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        loss = fn()
        if backward:
            loss.backward()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    peak_delta = float(torch.cuda.max_memory_allocated() - baseline) / (1024**2)
    return {
        "median_ms": statistics.median(timings),
        "p90_ms": percentile(timings, 0.90),
        "mean_ms": statistics.mean(timings),
    }, peak_delta


def benchmark_backend(
    controller: UserActionAuxiliaryController,
    logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: list[dict],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    action_ce = torch.tensor(2.0, dtype=torch.float32, device=logits.device)

    def forward_only():
        return controller.compute(logits, labels, metadata, action_ce, 100).loss

    forward, forward_peak = measure_cuda(forward_only, warmup, repeats, backward=False)
    grad_logits = logits.detach().requires_grad_(True)

    def forward_backward():
        grad_logits.grad = None
        return controller.compute(grad_logits, labels, metadata, action_ce, 100).loss

    combined, combined_peak = measure_cuda(forward_backward, warmup, repeats, backward=True)
    return {
        "aux_forward": forward,
        "aux_forward_backward": combined,
        "forward_peak_memory_mib": forward_peak,
        "forward_backward_peak_memory_mib": combined_peak,
    }


def benchmark_disabled(logits: torch.Tensor, warmup: int, repeats: int) -> dict[str, object]:
    grad_logits = logits.detach().requires_grad_(True)

    def disabled_forward():
        return logits.reshape(-1)[0].float() * 0.0

    def disabled_forward_backward():
        grad_logits.grad = None
        return grad_logits.reshape(-1)[0].float() * 0.0

    forward, forward_peak = measure_cuda(disabled_forward, warmup, repeats, backward=False)
    combined, combined_peak = measure_cuda(
        disabled_forward_backward, warmup, repeats, backward=True
    )
    return {
        "aux_forward": forward,
        "aux_forward_backward": combined,
        "forward_peak_memory_mib": forward_peak,
        "forward_backward_peak_memory_mib": combined_peak,
    }


def profile_backend(controller, logits, labels, metadata) -> list[dict[str, object]]:
    action_ce = torch.tensor(2.0, device=logits.device)
    profiled_logits = logits.detach().requires_grad_(True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        result = controller.compute(profiled_logits, labels, metadata, action_ce, 100)
        result.loss.backward()
    wanted = {
        "aten::logsumexp",
        "aten::_log_softmax",
        "aten::cross_entropy_loss",
        "aten::index_select",
        "aten::gather",
    }
    summary = []
    for event in profile.key_averages():
        if event.key in wanted:
            summary.append(
                {
                    "op": event.key,
                    "calls": event.count,
                    "cpu_self_ms": event.self_cpu_time_total / 1000.0,
                    "cuda_total_ms": event.device_time_total / 1000.0,
                }
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--vocab-size", type=int, default=176255)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Pad logits/labels to this length; 0 uses the compact generated length.",
    )
    parser.add_argument(
        "--cases",
        default="short,medium,long,multi_segment",
        help="Comma-separated subset of short, medium, long, multi_segment.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA for synchronized timings and memory metrics.")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = BenchmarkTokenizer()
    all_cases = (
        Case("short", (2,)),
        Case("medium", (8,)),
        Case("long", (24,)),
        Case("multi_segment", (8, 8)),
    )
    selected = {name.strip() for name in args.cases.split(",") if name.strip()}
    known = {case.name for case in all_cases}
    if not selected or not selected <= known:
        raise ValueError(f"cases must be a non-empty subset of {sorted(known)}, got {sorted(selected)}")
    cases = tuple(case for case in all_cases if case.name in selected)
    report = {
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "vocab_size": args.vocab_size,
        "chunk_size": args.chunk_size,
        "sequence_length": args.sequence_length,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": {},
    }
    for case in cases:
        logits, labels, metadata = build_inputs(
            case, tokenizer, args.vocab_size, device, dtype, args.sequence_length
        )
        controllers = {
            "legacy": UserActionAuxiliaryController(tokenizer, make_args(False, args.chunk_size)),
            "vectorized": UserActionAuxiliaryController(tokenizer, make_args(True, args.chunk_size)),
        }
        zero = torch.zeros(ACTION_STAT_SIZE, dtype=torch.float32, device=device)
        plan = controllers["vectorized"]._build_vectorized_plan(logits, metadata, zero)
        full_positions = torch.cat(
            (
                plan.continue_ce_positions,
                plan.early_stop_positions,
                plan.tail_positions,
                plan.tail_positions,
            )
        )
        case_report = {
            "segments": len(metadata),
            "sid_count": sum(case.segment_sid_counts),
            "trie_positions": sum(group.positions.numel() for group in plan.trie_groups.values()),
            "full_vocab_unique_positions": int(torch.unique(full_positions).numel()),
            "disabled": benchmark_disabled(logits, args.warmup, args.repeats),
        }
        torch.cuda.empty_cache()
        for name, controller in controllers.items():
            case_report[name] = benchmark_backend(
                controller, logits, labels, metadata, args.warmup, args.repeats
            )
        legacy_ms = case_report["legacy"]["aux_forward_backward"]["median_ms"]
        vector_ms = case_report["vectorized"]["aux_forward_backward"]["median_ms"]
        case_report["forward_backward_speedup"] = legacy_ms / vector_ms
        if args.profile and case.name == "long":
            case_report["profiler"] = {
                name: profile_backend(controller, logits, labels, metadata)
                for name, controller in controllers.items()
            }
        report["cases"][case.name] = case_report
        del logits, labels, metadata, controllers
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
