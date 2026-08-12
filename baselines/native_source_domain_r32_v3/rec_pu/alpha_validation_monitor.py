"""Sidecar teacher-forced validation for the alpha monitor experiment.

The module deliberately owns no optimisation logic.  It consumes packed rows
already produced by the native training pipeline, evaluates exact disjoint DDP
shards under ``inference_mode``, and emits compact sum/count tensors only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from rec_pu.alpha_recommendation_monitor import (
    ALL_METRIC_NAMES,
    CORE_METRIC_NAMES,
    _topk_current_hits,
    collect_alpha_recommendation_monitor,
)
from rec_pu.sid8_rec_pu_integration import RecPUConfig, SIDComponentVocab, coerce_packed_target, compute_native_sid8_loss


DOMAINS = ("video", "prod", "ad", "living")
# Keep a~o -> va~vo one-to-one, e.g. b_rec_* becomes vb_rec_*.
VALIDATION_METRIC_NAMES = tuple("v" + name for name in CORE_METRIC_NAMES)
ROLLING_TRAIN_METRIC_NAMES = CORE_METRIC_NAMES[:6]
EXTRA_METRIC_NAMES = (
    "val_rec_gold_prob_a",
    "val_rec_gold_prob_b",
    "val_rec_gold_prob_c",
    "val_rec_gold_path_nll",
    *(f"val_rec_{domain}_gold_sid_ce" for domain in DOMAINS),
    *(f"val_rec_{domain}_tf_chain" for domain in DOMAINS),
    "val_rec_segments",
    "val_rec_cot_segments",
    "val_rec_nocot_segments",
    "val_rec_positions",
)
EXTRA_INDEX = {name: index for index, name in enumerate(EXTRA_METRIC_NAMES)}


@dataclass(frozen=True)
class AlphaValidationConfig:
    enabled: bool = False
    probe_enabled: bool = False
    probe_interval: int = 100
    full_dev_enabled: bool = False
    full_dev_at_epoch_end: bool = False
    split_dir: str = "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1"
    dev_cache: str = "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev2_8k"
    probe_cache: str = "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev_probe_v1_8k"
    metrics_path: str = "/data/logs/baselines/native_source_domain_r32_v3/alpha_validation_metrics.jsonl"

    def __post_init__(self) -> None:
        if self.probe_interval < 1:
            raise ValueError("alpha_dev_probe_interval must be >= 1.")


class ExactShardSampler(Sampler[int]):
    """A no-padding DDP sampler: every packed row is evaluated exactly once."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size, self.rank, self.world_size = int(size), int(rank), int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return (self.size - self.rank + self.world_size - 1) // self.world_size if self.rank < self.size else 0


def exact_shard_indices(size: int, rank: int, world_size: int) -> list[int]:
    return list(ExactShardSampler(size, rank, world_size))


def _dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank_world() -> tuple[int, int]:
    return (torch.distributed.get_rank(), torch.distributed.get_world_size()) if _dist_ready() else (0, 1)


def _all_reduce(stats: torch.Tensor) -> torch.Tensor:
    result = stats.detach().clone()
    if _dist_ready():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def _add_pair(stats: torch.Tensor, name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.float64).reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel():
        stats[EXTRA_INDEX[name], 0].add_(values.sum())
        stats[EXTRA_INDEX[name], 1].add_(values.numel())


def _add_count(stats: torch.Tensor, name: str, value: int) -> None:
    stats[EXTRA_INDEX[name], 0].add_(float(value))


def build_domain_token_map(tokenizer: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for domain in DOMAINS:
        token = f"<|{domain}_begin|>"
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0 or tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Missing exact domain token: {token}")
        result[token_id] = domain
    return result


def collect_alpha_validation_extras(
    *,
    base_per_token_ce: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rec_targets: Sequence[Sequence[Mapping[str, Any]]] | None,
    sid_component_vocab: SIDComponentVocab,
    domain_token_map: Mapping[int, str],
) -> torch.Tensor:
    """Secondary validation values using exactly the Phase-1 target geometry."""

    stats = torch.zeros((len(EXTRA_METRIC_NAMES), 2), device=logits.device, dtype=torch.float64)
    if not rec_targets:
        return stats
    shift_labels = labels[:, 1:]
    shift_logits = logits[:, :-1]
    with torch.no_grad():
        for row, raw_targets in enumerate(rec_targets):
            for raw in raw_targets:
                target = coerce_packed_target(raw)
                route = target.source_segment
                if route not in {"recommendation_cot", "recommendation_nocot"}:
                    continue
                positions = torch.tensor(
                    [target.a_logit_position, target.b_logit_position, target.c_logit_position], device=logits.device
                )
                if int(positions.min()) < 0 or int(positions.max()) >= shift_labels.size(1):
                    raise ValueError("Alpha validation final-SID position is outside shifted labels.")
                if not bool((shift_labels[row, positions] != -100).all()):
                    raise ValueError("Alpha validation final-SID position is not supervised.")
                domain_position = int(target.a_label_position) - 1
                domain_id = int(labels[row, domain_position].item()) if domain_position >= 0 else -1
                domain = domain_token_map.get(domain_id)
                if domain is None:
                    raise ValueError("Alpha validation could not recover final-SID target domain.")
                ce = base_per_token_ce[row, positions]
                _add_pair(stats, "val_rec_gold_prob_a", torch.exp(-ce[0]))
                _add_pair(stats, "val_rec_gold_prob_b", torch.exp(-ce[1]))
                _add_pair(stats, "val_rec_gold_prob_c", torch.exp(-ce[2]))
                _add_pair(stats, "val_rec_gold_path_nll", ce.sum())
                _add_pair(stats, f"val_rec_{domain}_gold_sid_ce", ce)
                selected = [shift_logits[row, int(position)].unsqueeze(0) for position in positions]
                current = [shift_labels[row, int(position)].view(1) for position in positions]
                a32 = _topk_current_hits(selected[0], current[0], sid_component_vocab.a, 32).bool()
                b8 = _topk_current_hits(selected[1], current[1], sid_component_vocab.b, 8).bool()
                c8 = _topk_current_hits(selected[2], current[2], sid_component_vocab.c, 8).bool()
                _add_pair(stats, f"val_rec_{domain}_tf_chain", (a32 & b8 & c8).float())
                _add_count(stats, "val_rec_segments", 1)
                _add_count(stats, "val_rec_cot_segments" if route == "recommendation_cot" else "val_rec_nocot_segments", 1)
                _add_count(stats, "val_rec_positions", 3)
    return stats


def validation_metric_parity(
    *,
    base_per_token_ce: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rec_targets: Sequence[Sequence[Mapping[str, Any]]] | None,
    sid_component_vocab: SIDComponentVocab,
) -> torch.Tensor:
    """The validation primary helper is intentionally the train helper itself."""

    return collect_alpha_recommendation_monitor(
        base_per_token_ce=base_per_token_ce,
        logits=logits,
        labels=labels,
        rec_targets=rec_targets,
        sid_component_vocab=sid_component_vocab,
        collect_tf=True,
    )


@contextmanager
def preserved_eval_state(model: torch.nn.Module) -> Iterable[None]:
    """Restore train mode and CPU/current-device RNG even if validation raises."""

    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state()
    python_rng = random.getstate()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    model.eval()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng)
        random.setstate(python_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        model.train(was_training)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _mean_pairs(stats: torch.Tensor, names: Sequence[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index, name in enumerate(names):
        count = stats[index, 1]
        result[name] = (stats[index, 0] / count).item() if count.item() else None
    return result


def _counter_value(stats: torch.Tensor, name: str) -> float:
    return float(stats[EXTRA_INDEX[name], 0].item())


def source_domain_weights(split_dir: str | Path) -> dict[str, float]:
    audit = json.loads((Path(split_dir) / "split_audit.json").read_text(encoding="utf-8"))
    rows = audit["recommendation"]["domain_rows"]
    total = sum(int(rows[domain]["original"]) for domain in DOMAINS)
    return {domain: int(rows[domain]["original"]) / total for domain in DOMAINS}


def count_exact_duplicates(path: str | Path) -> tuple[int, int]:
    values: Counter[bytes] = Counter()
    with Path(path).open("rb") as handle:
        for line in handle:
            values[line.rstrip(b"\n")] += 1
    total = sum(values.values())
    return total - len(values), len(values)


class AlphaValidationRunner:
    """Reusable sidecar evaluator for a probe or the entire packed dev cache."""

    def __init__(self, config: AlphaValidationConfig) -> None:
        self.config = config
        self._datasets: dict[str, Any] = {}
        self._domain_weights = source_domain_weights(config.split_dir)
        self._dev_duplicate_extra, self._dev_unique_rows = count_exact_duplicates(Path(config.split_dir) / "dev.jsonl")
        self._probe_history: list[dict[str, float | None]] = []

    def _dataset(self, kind: str):
        if kind not in self._datasets:
            from datasets import load_from_disk

            path = self.config.probe_cache if kind == "probe" else self.config.dev_cache
            loaded = load_from_disk(path)
            self._datasets[kind] = loaded["train"] if hasattr(loaded, "keys") else loaded
        return self._datasets[kind]

    def run(self, trainer: Any, model: torch.nn.Module, *, kind: str, global_step: int, epoch: float | None) -> dict[str, Any]:
        if kind not in {"probe", "full"}:
            raise ValueError(f"Unknown alpha validation kind: {kind}")
        dataset = self._dataset(kind)
        rank, world = _rank_world()
        device = next(model.parameters()).device
        sampler = ExactShardSampler(len(dataset), rank, world)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=trainer.data_collator, num_workers=0)
        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        vocab = getattr(trainer, "_rec_pu_component_vocab", None)
        if vocab is None:
            from rec_pu.sid8_rec_pu_integration import build_sid_component_vocab

            vocab = build_sid_component_vocab(tokenizer)
            trainer._rec_pu_component_vocab = vocab
        domain_map = build_domain_token_map(tokenizer)
        primary = torch.zeros((len(ALL_METRIC_NAMES), 2), device=device, dtype=torch.float64)
        extra = torch.zeros((len(EXTRA_METRIC_NAMES), 2), device=device, dtype=torch.float64)
        local_packs = 0
        start = time.perf_counter()
        if _dist_ready():
            torch.distributed.barrier()
        with preserved_eval_state(model), torch.inference_mode():
            for batch in loader:
                batch = _move_to_device(batch, device)
                labels = batch.pop("labels")
                weights = batch.pop("loss_weights")
                sample_ids = batch.pop("sample_ids")
                sample_task_ids = batch.pop("sample_task_ids")
                sample_domain_weights = batch.pop("sample_domain_weights")
                rec_targets = batch.pop("rec_pu_targets", None)
                outputs = model(**batch)
                _, details = compute_native_sid8_loss(
                    logits=outputs.logits,
                    labels=labels,
                    loss_weights=weights,
                    sample_ids=sample_ids,
                    sample_task_ids=sample_task_ids,
                    sample_domain_weights=sample_domain_weights,
                    rec_pu_targets=rec_targets,
                    rec_pu_config=RecPUConfig(False, 0.05),
                    sid_component_vocab=vocab,
                )
                primary.add_(validation_metric_parity(
                    base_per_token_ce=details.base_per_token_ce,
                    logits=outputs.logits,
                    labels=labels,
                    rec_targets=rec_targets,
                    sid_component_vocab=vocab,
                ))
                extra.add_(collect_alpha_validation_extras(
                    base_per_token_ce=details.base_per_token_ce,
                    logits=outputs.logits,
                    labels=labels,
                    rec_targets=rec_targets,
                    sid_component_vocab=vocab,
                    domain_token_map=domain_map,
                ))
                local_packs += 1
                del outputs, details
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        timing = torch.tensor([elapsed, float(local_packs), torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0.0], device=device, dtype=torch.float64)
        if _dist_ready():
            torch.distributed.all_reduce(primary, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(extra, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(timing[:1], op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(timing[1:2], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(timing[2:], op=torch.distributed.ReduceOp.MAX)
        if int(timing[1].item()) != len(dataset):
            raise RuntimeError(f"Alpha validation exact-once failure: expected {len(dataset)}, got {int(timing[1].item())}")
        metrics = _mean_pairs(primary[:len(CORE_METRIC_NAMES)], VALIDATION_METRIC_NAMES)
        metrics.update(_mean_pairs(extra, EXTRA_METRIC_NAMES))
        # Segment/position fields are counters by definition, not means.
        for name in ("val_rec_segments", "val_rec_cot_segments", "val_rec_nocot_segments", "val_rec_positions"):
            metrics[name] = _counter_value(extra, name)
        domain_values = {domain: metrics[f"val_rec_{domain}_gold_sid_ce"] for domain in DOMAINS}
        domain_chain = {domain: metrics[f"val_rec_{domain}_tf_chain"] for domain in DOMAINS}
        if all(value is not None for value in domain_values.values()):
            metrics["val_rec_reweighted_gold_sid_ce"] = sum(self._domain_weights[d] * float(domain_values[d]) for d in DOMAINS)
        if all(value is not None for value in domain_chain.values()):
            metrics["val_rec_reweighted_tf_chain"] = sum(self._domain_weights[d] * float(domain_chain[d]) for d in DOMAINS)
        payload: dict[str, Any] = {
            "global_step": int(global_step), "epoch": None if epoch is None else float(epoch), "type": kind,
            "wall_time_sec": float(timing[0].item()), "packed_sequences": int(timing[1].item()),
            "world_size": world, "exact_once": True,
            "peak_cuda_gib": float(timing[2].item() / (1024 ** 3)),
            "dev_exact_duplicate_extra_instances": self._dev_duplicate_extra,
            "dev_unique_exact_rows": self._dev_unique_rows,
            "source_domain_weights": self._domain_weights,
            "metrics": metrics,
        }
        return payload

    def record_probe_gap(self, trainer: Any, result: dict[str, Any]) -> dict[str, float | int]:
        stats = getattr(trainer, "_alpha_train_gap_stats", None)
        if stats is None:
            return {"alpha_rec_overfit_warning": 0}
        global_stats = _all_reduce(stats)
        stats.zero_()
        train = _mean_pairs(global_stats, ROLLING_TRAIN_METRIC_NAMES)
        validation = result["metrics"]
        gaps = {
            "ga_rec_cot_gold_gap": _gap(validation.get("vb_rec_cot_gold_sid_ce"), train.get("b_rec_cot_gold_sid_ce")),
            "gb_rec_nocot_gold_gap": _gap(validation.get("vc_rec_nocot_gold_sid_ce"), train.get("c_rec_nocot_gold_sid_ce")),
            "gc_rec_gold_a_gap": _gap(validation.get("vd_rec_gold_a_ce"), train.get("d_rec_gold_a_ce")),
            "gd_rec_gold_b_gap": _gap(validation.get("ve_rec_gold_b_ce"), train.get("e_rec_gold_b_ce")),
            "ge_rec_gold_c_gap": _gap(validation.get("vf_rec_gold_c_ce"), train.get("f_rec_gold_c_ce")),
        }
        gold_train = _average([train[key] for key in ("b_rec_cot_gold_sid_ce", "c_rec_nocot_gold_sid_ce", "d_rec_gold_a_ce", "e_rec_gold_b_ce", "f_rec_gold_c_ce")])
        gold_dev = _average([validation[key] for key in ("vb_rec_cot_gold_sid_ce", "vc_rec_nocot_gold_sid_ce", "vd_rec_gold_a_ce", "ve_rec_gold_b_ce", "vf_rec_gold_c_ce")])
        chain = validation.get("vk_rec_tf_chain_32_8_8")
        self._probe_history.append({"train_gold": gold_train, "dev_gold": gold_dev, "chain": chain})
        warning = 0
        if len(self._probe_history) >= 3:
            first, middle, last = self._probe_history[-3:]
            if all(value is not None for point in (first, middle, last) for value in point.values()):
                warning = int(
                    float(first["train_gold"]) > float(middle["train_gold"]) > float(last["train_gold"])
                    and float(first["dev_gold"]) < float(middle["dev_gold"]) < float(last["dev_gold"])
                    and float(first["chain"]) > float(middle["chain"]) > float(last["chain"])
                )
        gaps["alpha_rec_overfit_warning"] = warning
        result["rolling_train_metrics"] = train
        result["gaps"] = gaps
        return gaps

    def append_record(self, result: Mapping[str, Any]) -> None:
        rank, _ = _rank_world()
        if rank:
            return
        path = Path(self.config.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


def _gap(value: float | None, train: float | None) -> float | None:
    return None if value is None or train is None else float(value) - float(train)


def _average(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
