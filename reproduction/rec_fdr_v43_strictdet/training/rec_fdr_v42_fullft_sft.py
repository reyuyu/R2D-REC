#!/usr/bin/env python3
"""V4.2 Full-FT trainer for Interest-Only and domain-ratio CoT/unCoT.

FDR makes one vectorised pairwise comparison at the first wrong SID token of a
legal train-only tuple.  It never launches a candidate-specific Transformer
forward during training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import torch
import torch.distributed.fsdp as fsdp
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

from omegaconf import OmegaConf

from legal_sid_universe_v5 import LegalSidUniverse


# Transformers 5.3 expects this public FSDP alias; Torch 2.5 keeps it under
# composable FSDP. Apply the same compatibility shim as the established jobs.
if not hasattr(fsdp, "register_fsdp_forward_method"):
    from torch.distributed._composable.fsdp import register_fsdp_forward_method

    fsdp.register_fsdp_forward_method = register_fsdp_forward_method


SID_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TASK_IDS = {"base": 1, "full_cot": 2, "full_sid": 3, "r1": 4, "r2": 5}
CORE_JOINT_METRICS = (
    "rec", "sid_set", "a_set", "ab_set", "fd_total", "fd_loss_a", "fd_loss_b", "fd_loss_c",
    "fd_pair_count_a", "fd_pair_count_b", "fd_pair_count_c", "fd_pair_accuracy_a", "fd_pair_accuracy_b", "fd_pair_accuracy_c",
    "fd_margin_a", "fd_margin_b", "fd_margin_c", "fd_violation_rate_a", "fd_violation_rate_b", "fd_violation_rate_c",
    "anchor_nll_a", "anchor_nll_b", "anchor_nll_c", "anchor_nll_sid_sum",
)
MODE_METRICS = tuple(
    f"{mode}_{name}"
    for mode in ("cot", "uncot")
    for name in CORE_JOINT_METRICS
)
DOMAIN_MODE_METRICS = tuple(
    f"{domain}_{mode}_{name}"
    for domain in ("video", "prod", "ad", "living")
    for mode in ("cot", "uncot")
    for name in ("rec", "anchor_nll_sid_sum", "fd_total")
)
JOINT_METRIC_NAMES = CORE_JOINT_METRICS + MODE_METRICS + DOMAIN_MODE_METRICS
MODE_IDS = {"none": 0, "cot": 1, "uncot": 2}
LAST_CURRICULUM_METRICS: dict[str, Any] | None = None
REC_JOINT_CONFIG: dict[str, Any] = {"enabled": False}
LEGAL_SID_UNIVERSE: LegalSidUniverse | None = None
CURRICULUM_PLAN_REPORT: dict[str, Any] = {}
STOP_AT_PROGRESS: float | None = None
# Optional V5.1 branch-only contract.  It is deliberately parsed outside
# LLaMA-Factory's dataclasses so the normal/shared/control curriculum remains
# byte-for-byte unchanged.
DOMAIN_CONDITIONED_FULL_C2: dict[str, Any] | None = None
V42_CONFIG: dict[str, Any] = {}
BASE_TRAIN = [
    "material_no_think_semantic_to_sid_train",
    "material_no_think_sid_to_semantic_train",
    "material_think_semantic_to_sid_train",
    "material_think_sid_to_semantic_train",
    "user_clean_train",
]
BASE_VAL = [name.removesuffix("_train") + "_val" for name in BASE_TRAIN]


class SmokeComplete(RuntimeError):
    """Internal clean stop after a real optimizer step, before model saving."""
POOL_DATASETS = {
    "base": BASE_TRAIN,
    "full_k1": ["rec_full_k1_train"],
    "full_k2": ["rec_full_k2_train"],
    "full_k3": ["rec_full_k3_train"],
    "r1": ["rec_r1_train"],
    "r2_k1": [f"rec_r2_k1_pack_seed_{seed}_train" for seed in range(4)],
    "r2_k2": ["rec_r2_k2_train"],
    "r2_k3": ["rec_r2_k3_train"],
    "val_full": ["rec_full_val"],
    "val_r1": ["rec_r1_val"],
    "val_r2": ["rec_r2_val"],
    "val_base": BASE_VAL,
}


def active_pool_datasets() -> dict[str, list[str]]:
    """Return normal pools plus V5.1's branch-only domain Full pools."""
    pools = {name: list(values) for name, values in POOL_DATASETS.items()}
    if DOMAIN_CONDITIONED_FULL_C2 is not None:
        dataset_names = DOMAIN_CONDITIONED_FULL_C2.get("dataset_names", {})
        for domain in DOMAIN_CONDITIONED_FULL_C2.get("domains", []):
            values = dataset_names.get(str(domain), {})
            for bucket in ("k1", "k2", "k3"):
                name = values.get(bucket)
                if not isinstance(name, str) or not name:
                    raise ValueError(f"missing dataset_name for domain C2 pool {domain}/{bucket}")
                pools[f"full_{domain}_{bucket}"] = [name]
    return pools


def metric_reduction_keys() -> list[str]:
    """Return the rank-invariant schema used by every metric collective."""
    return sorted({
        *(task for task in ("base", "full", "r1", "r2")),
        *(f"objective_{part}/{task}" for part in ("num", "den") for task in ("base", "full", "r1", "r2")),
        *(f"component_{part}/{group_id}" for part in ("num", "den") for group_id in TASK_IDS.values()),
        *(f"joint_{part}/{task}/{name}" for part in ("num", "den") for task in ("full", "r2") for name in JOINT_METRIC_NAMES),
        *(f"packs/{task}" for task in ("base", "full", "r1", "r2")),
        *(f"tokens/{task}" for task in ("base", "full", "r1", "r2")),
        *(f"tokens/{task}/{bucket}" for task in ("full", "r2") for bucket in ("k1", "k2", "k3")),
        *(f"mode_count/{mode}" for mode in ("cot", "uncot")),
        *(f"mode_tokens/{mode}" for mode in ("cot", "uncot")),
        *(f"domain_mode_count/{domain}/{mode}" for domain in ("video", "prod", "ad", "living") for mode in ("cot", "uncot")),
        *(f"domain_mode_tokens/{domain}/{mode}" for domain in ("video", "prod", "ad", "living") for mode in ("cot", "uncot")),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def task_for_dataset(name: str) -> str:
    if name.startswith("rec_full_"):
        return "full"
    if name.startswith("rec_r1_"):
        return "r1"
    if name.startswith("rec_r2_"):
        return "r2"
    return "base"


def _patch_dataset_metadata() -> None:
    from llamafactory.data import converter

    original = converter.get_dataset_converter

    class MetadataConverter:
        def __init__(self, base: Any, source_name: str):
            self.base = base
            self.source_name = source_name

        def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
            output = self.base(example)
            output["_curriculum_task"] = task_for_dataset(self.source_name)
            output["_sample_weight"] = float(example.get("sample_weight", 1.0))
            output["_rec_group_index"] = int(example.get("rec_group_index", 0))
            default_mode = "cot" if output["_curriculum_task"] == "full" else "none"
            output["_response_mode"] = str(example.get("response_mode", default_mode))
            return output

    def get_converter(name: str, dataset_attr: Any, data_args: Any) -> Any:
        source_name = Path(dataset_attr.dataset_name).stem
        return MetadataConverter(original(name, dataset_attr, data_args), source_name)

    converter.get_dataset_converter = get_converter


def _content_token_fields(
    processor: Any, content: str, task: str, sample_weight: float, response_mode: str
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Map task objectives onto exact assistant-content token offsets."""
    formatted_slots = processor.template.format_assistant.apply(content=content)
    formatted_text = "".join(slot for slot in formatted_slots if isinstance(slot, str))
    target_ids = processor.template._convert_elements_to_ids(processor.tokenizer, formatted_slots)
    encoded = processor.tokenizer(formatted_text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != target_ids:
        raise ValueError("assistant formatter IDs differ from tokenizer IDs")
    if not formatted_text.startswith(content):
        raise ValueError("assistant response content must precede the formatter suffix")

    objective = [0.0] * len(target_ids)
    metric = [0.0] * len(target_ids)
    groups = [0] * len(target_ids)
    content_offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    if task == "base":
        positions = list(range(len(target_ids)))
        for i in positions:
            objective[i] = 1.0
            metric[i] = 1.0
            groups[i] = TASK_IDS["base"]
        return target_ids, objective, metric, groups

    think = THINK_RE.search(content)
    if think is None:
        raise ValueError(f"{task}: assistant response has no complete think block")

    if task in {"full", "r1"} and response_mode != "uncot":
        cot_positions = [
            i for i, (a, b) in enumerate(content_offsets)
            if b > a and a < think.end() and b > think.start()
        ]
        if not cot_positions:
            raise ValueError(f"{task}: no CoT tokens mapped")
        cot_fraction = 0.5 if task == "full" else 1.0
        group = TASK_IDS["full_cot"] if task == "full" else TASK_IDS["r1"]
        for i in cot_positions:
            objective[i] = cot_fraction / len(cot_positions)
            metric[i] = 1.0 / len(cot_positions)
            groups[i] = group

    if task in {"full", "r2"}:
        sid_matches = list(SID_RE.finditer(content, think.end()))
        if len(sid_matches) != 1:
            raise ValueError(f"{task}: expected exactly one final SID, found {len(sid_matches)}")
        sid = sid_matches[0]
        sid_positions = [
            i for i, (a, b) in enumerate(content_offsets)
            if b > a and a < sid.end() and b > sid.start()
        ]
        if not sid_positions:
            raise ValueError(f"{task}: no SID tokens mapped")
        # V2 replaces single-positive SID CE with the normalized multi-positive
        # recommendation objective. Metric weights remain for diagnostics.
        fraction = 0.0
        group = TASK_IDS["full_sid"] if task == "full" else TASK_IDS["r2"]
        for i in sid_positions:
            objective[i] = fraction / len(sid_positions)
            metric[i] = 1.0 / len(sid_positions)
            groups[i] = group

    if not any(value > 0 for value in objective) and task not in {"r2", "full"}:
        raise ValueError(f"{task}: no objective tokens")
    return target_ids, objective, metric, groups


def _patch_packed_processor() -> None:
    from llamafactory.data.processor import supervised as supervised_processor
    from llamafactory.data.processor.processor_utils import greedy_knapsack
    from llamafactory.data.processor.supervised import MAX_SU_SEQ_IDX, PackingParams
    from llamafactory.extras import logging
    from llamafactory.extras.constants import IGNORE_INDEX

    logger = logging.get_logger(__name__)

    def encode(
        processor: Any,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
        tools: str | None,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        task: str,
        sample_weight: float,
        rec_group_index: int,
        rec_instance_index: int,
        response_mode: str,
    ) -> tuple[list[int], list[int], list[float], list[float], list[int], list[int], list[int], list[int], list[int]]:
        messages = processor.template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, processor.processor
        )
        input_ids, labels = processor.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, processor.tokenizer, processor.processor
        )
        discard_history_cot = processor.data_args.mask_history and not processor.template.preserve_thinking
        pairs = processor.template.encode_multiturn(
            processor.tokenizer, messages, system, tools, discard_history_cot
        )
        if processor.data_args.mask_history:
            pairs = pairs[::-1]
        objective = [0.0] * len(input_ids)
        metric = [0.0] * len(input_ids)
        groups = [0] * len(input_ids)
        rec_group_ids = [0] * len(input_ids)
        rec_instance_ids = [0] * len(input_ids)
        mode_ids = [0] * len(input_ids)
        sample_group_ids = [0] * len(input_ids)
        sample_mode_id = MODE_IDS.get(response_mode)
        if sample_mode_id is None:
            raise ValueError(f"unknown response_mode: {response_mode}")
        for turn, (source_ids, target_ids) in enumerate(pairs):
            source_length = len(source_ids)
            if processor.data_args.train_on_prompt:
                source_labels = source_ids
                source_objective = [1.0] * source_length
                source_metric = [1.0] * source_length
                source_groups = [TASK_IDS["base"]] * source_length
            elif processor.template.efficient_eos and turn != 0:
                source_labels = [processor.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_length - 1)
                source_objective = [1.0] + [0.0] * (source_length - 1)
                source_metric = [1.0] + [0.0] * (source_length - 1)
                source_groups = [TASK_IDS["base"]] + [0] * (source_length - 1)
            else:
                source_labels = [IGNORE_INDEX] * source_length
                source_objective = [0.0] * source_length
                source_metric = [0.0] * source_length
                source_groups = [0] * source_length

            if processor.data_args.mask_history and turn != 0:
                target_labels = [IGNORE_INDEX] * len(target_ids)
                target_objective = [0.0] * len(target_ids)
                target_metric = [0.0] * len(target_ids)
                target_groups = [0] * len(target_ids)
                target_rec_groups = [0] * len(target_ids)
                target_rec_instances = [0] * len(target_ids)
            else:
                target_labels = target_ids
                content = response[turn]["content"] if turn < len(response) else ""
                checked_ids, target_objective, target_metric, target_groups = _content_token_fields(
                    processor, content, task, sample_weight, response_mode
                )
                if checked_ids != target_ids:
                    raise ValueError("assistant target IDs changed during loss-field construction")
                target_rec_groups = [
                    rec_group_index if group in (TASK_IDS["full_sid"], TASK_IDS["r2"]) else 0
                    for group in target_groups
                ]
                target_rec_instances = [
                    rec_instance_index if value else 0 for value in target_rec_groups
                ]
            source_modes = [sample_mode_id] * len(source_ids) if task == "full" else [0] * len(source_ids)
            target_modes = [sample_mode_id] * len(target_ids) if task == "full" else [0] * len(target_ids)
            source_sample_groups = [rec_group_index] * len(source_ids) if task == "full" else [0] * len(source_ids)
            target_sample_groups = [rec_group_index] * len(target_ids) if task == "full" else [0] * len(target_ids)

            if processor.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_labels + target_labels + labels
                objective = source_objective + target_objective + objective
                metric = source_metric + target_metric + metric
                groups = source_groups + target_groups + groups
                rec_group_ids = [0] * len(source_ids) + target_rec_groups + rec_group_ids
                rec_instance_ids = [0] * len(source_ids) + target_rec_instances + rec_instance_ids
                mode_ids = source_modes + target_modes + mode_ids
                sample_group_ids = source_sample_groups + target_sample_groups + sample_group_ids
            else:
                input_ids += source_ids + target_ids
                labels += source_labels + target_labels
                objective += source_objective + target_objective
                metric += source_metric + target_metric
                groups += source_groups + target_groups
                rec_group_ids += [0] * len(source_ids) + target_rec_groups
                rec_instance_ids += [0] * len(source_ids) + target_rec_instances
                mode_ids += source_modes + target_modes
                sample_group_ids += source_sample_groups + target_sample_groups

        if processor.template.efficient_eos:
            input_ids.append(processor.tokenizer.eos_token_id)
            labels.append(processor.tokenizer.eos_token_id)
            if task == "base":
                objective.append(1.0)
                metric.append(1.0)
                groups.append(TASK_IDS["base"])
            else:
                objective.append(0.0)
                metric.append(0.0)
                groups.append(0)
            rec_group_ids.append(0)
            rec_instance_ids.append(0)
            mode_ids.append(sample_mode_id if task == "full" else 0)
            sample_group_ids.append(rec_group_index if task == "full" else 0)
        if not (
            len(input_ids) == len(labels) == len(objective) == len(metric) == len(groups)
            == len(rec_group_ids) == len(rec_instance_ids) == len(mode_ids) == len(sample_group_ids)
        ):
            raise ValueError("packed loss fields are not token-aligned")
        return input_ids, labels, objective, metric, groups, rec_group_ids, rec_instance_ids, mode_ids, sample_group_ids

    def preprocess(processor: Any, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        valid = 0
        batch_ids, batch_labels, batch_objective, batch_metric, batch_groups = [], [], [], [], []
        batch_rec_groups, batch_rec_instances, batch_modes, batch_sample_groups = [], [], [], []
        batch_images, batch_videos, batch_audios = [], [], []
        lengths: list[int] = []
        length_to_indices: defaultdict[int, list[int]] = defaultdict(list)
        tasks = examples.get("_curriculum_task", ["base"] * len(examples["_prompt"]))
        sample_weights = examples.get("_sample_weight", [1.0] * len(examples["_prompt"]))
        rec_group_indices = examples.get("_rec_group_index", [0] * len(examples["_prompt"]))
        response_modes = examples.get("_response_mode", ["none"] * len(examples["_prompt"]))
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0("Dropped malformed SFT example")
                continue
            values = encode(
                processor,
                examples["_prompt"][i],
                examples["_response"][i],
                examples["_system"][i],
                examples["_tools"][i],
                examples["_images"][i] or [],
                examples["_videos"][i] or [],
                examples["_audios"][i] or [],
                str(tasks[i]),
                float(sample_weights[i]),
                int(rec_group_indices[i]),
                valid + 1,
                str(response_modes[i]),
            )
            input_ids, labels, objective, metric, groups, rec_groups, rec_instances, modes, sample_groups = values
            length = len(input_ids)
            if length > processor.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {processor.data_args.cutoff_len}.")
                continue
            lengths.append(length)
            length_to_indices[length].append(valid)
            batch_ids.append(input_ids)
            batch_labels.append(labels)
            batch_objective.append(objective)
            batch_metric.append(metric)
            batch_groups.append(groups)
            batch_rec_groups.append(rec_groups)
            batch_rec_instances.append(rec_instances)
            batch_modes.append(modes)
            batch_sample_groups.append(sample_groups)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            batch_audios.append(examples["_audios"][i] or [])
            valid += 1

        output: defaultdict[str, list[Any]] = defaultdict(list)
        for knapsack in greedy_knapsack(lengths, processor.data_args.cutoff_len):
            ids: list[int] = []
            labels: list[int] = []
            objective: list[float] = []
            metric: list[float] = []
            groups: list[int] = []
            rec_groups: list[int] = []
            rec_instances: list[int] = []
            modes: list[int] = []
            sample_groups: list[int] = []
            attention: list[int] = []
            positions: list[int] = []
            images: list[Any] = []
            videos: list[Any] = []
            audios: list[Any] = []
            if processor.data_args.neat_packing:
                boundaries = [0]
                image_subseq_ids: list[int] = []
                video_subseq_ids: list[int] = []
                audio_subseq_ids: list[int] = []
            for subsequence, length in enumerate(knapsack):
                index = length_to_indices[length].pop()
                ids += batch_ids[index]
                labels += batch_labels[index]
                objective += batch_objective[index]
                metric += batch_metric[index]
                groups += batch_groups[index]
                rec_groups += batch_rec_groups[index]
                # Re-number instances inside each packed sequence so IDs stay unique.
                rec_instances += [subsequence + 1 if value else 0 for value in batch_rec_instances[index]]
                modes += batch_modes[index]
                sample_groups += batch_sample_groups[index]
                positions += list(range(len(batch_ids[index])))
                attention += [subsequence + 1 if processor.data_args.neat_packing else 1] * len(batch_ids[index])
                images += batch_images[index]
                videos += batch_videos[index]
                audios += batch_audios[index]
                if processor.data_args.neat_packing:
                    boundaries.append(boundaries[-1] + len(batch_ids[index]))
                    image_subseq_ids += [subsequence] * len(batch_images[index])
                    video_subseq_ids += [subsequence] * len(batch_videos[index])
                    audio_subseq_ids += [subsequence] * len(batch_audios[index])

            pad = processor.data_args.cutoff_len - len(ids) + 1
            if pad > 0:
                ids += [processor.tokenizer.pad_token_id] * pad
                labels += [IGNORE_INDEX] * pad
                objective += [0.0] * pad
                metric += [0.0] * pad
                groups += [0] * pad
                rec_groups += [0] * pad
                rec_instances += [0] * pad
                modes += [0] * pad
                sample_groups += [0] * pad
                positions += [0] * pad
                attention += [0 if processor.data_args.neat_packing else 1] * pad
                if processor.data_args.neat_packing:
                    boundaries.append(boundaries[-1] + pad)
            if len(ids) != processor.data_args.cutoff_len + 1:
                raise ValueError("packed sequence length mismatch")
            output["input_ids"].append(ids)
            output["labels"].append(labels)
            output["attention_mask"].append(attention)
            output["position_ids"].append(positions)
            output["loss_weights"].append(objective)
            output["metric_weights"].append(metric)
            output["loss_groups"].append(groups)
            output["rec_group_ids"].append(rec_groups)
            output["rec_instance_ids"].append(rec_instances)
            output["response_mode_ids"].append(modes)
            output["sample_group_ids"].append(sample_groups)
            if processor.data_args.neat_packing:
                output["packing_params"].append(
                    vars(PackingParams(
                        sequence_boundaries=boundaries,
                        image_subseq_ids=image_subseq_ids or [MAX_SU_SEQ_IDX],
                        video_subseq_ids=video_subseq_ids or [MAX_SU_SEQ_IDX],
                        audio_subseq_ids=audio_subseq_ids or [MAX_SU_SEQ_IDX],
                        right_padding_length=pad,
                    ))
                )
            output["images"].append(images or None)
            output["videos"].append(videos or None)
            output["audios"].append(audios or None)
        return output

    supervised_processor.PackedSupervisedDatasetProcessor.preprocess_dataset = preprocess


class CurriculumIterableDataset:
    """Factory for a torch IterableDataset, kept import-light until patches are active."""

    @staticmethod
    def build(pools: dict[str, Any], expanded_plan: list[tuple[str, int]]) -> Any:
        import torch

        class Dataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                super().__init__()
                self.expanded_plan = expanded_plan

            def __len__(self) -> int:
                return len(self.expanded_plan)

            def __getitem__(self, item: int) -> dict[str, Any]:
                pool, index = self.expanded_plan[item]
                return dict(pools[pool][index])

        return Dataset()


def largest_remainder(total: int, probabilities: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in probabilities.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    for key in sorted(probabilities, key=lambda item: raw[item] - counts[item], reverse=True)[:remainder]:
        counts[key] += 1
    return counts


def multi_positive_set_loss(log_probs: torch.Tensor, positive_indices: list[int]) -> torch.Tensor:
    if not positive_indices:
        raise ValueError("multi-positive set cannot be empty")
    return -torch.logsumexp(log_probs[positive_indices], dim=0)


def sid_hit32_surrogate(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    topk: int = 32,
    positive_temperature: float = .10,
    hit_temperature: float = .50,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_positive = positive_temperature * torch.logsumexp(
        positive_scores / positive_temperature, dim=0
    )
    k = min(topk, int(negative_scores.numel()))
    if k <= 0:
        raise ValueError("candidate pool has no negatives")
    threshold = torch.topk(negative_scores, k=k).values[-1]
    return torch.nn.functional.softplus((threshold - best_positive) / hit_temperature), best_positive


def semantic_pair_loss(
    scores: torch.Tensor,
    relevance: list[float],
    pairs: list[tuple[int, int]],
    margin: float = .20,
) -> torch.Tensor:
    terms = [
        torch.nn.functional.softplus(
            margin * (relevance[i] - relevance[j]) - (scores[i] - scores[j])
        )
        for i, j in pairs
    ]
    return torch.stack(terms).mean() if terms else scores.sum() * 0.0


def first_divergence_pairs(
    anchor: tuple[int, int, int], positives: list[tuple[int, int, int]],
    inventory: list[tuple[int, int, int]], rng: random.Random,
    quota: dict[str, int], domain: str | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return legal (positive-token, negative-token) pairs by first divergence.

    All group positives are excluded at every hierarchy level, so another true
    future is never treated as an FDR negative.
    """
    if LEGAL_SID_UNIVERSE is not None:
        if domain is None:
            raise ValueError("static legal SID universe sampling requires a domain")
        return LEGAL_SID_UNIVERSE.sample_pairs(domain, anchor, positives, rng, quota)
    pos_a = {item[0] for item in positives}
    pos_b = {item[1] for item in positives if item[0] == anchor[0]}
    pos_c = {item[2] for item in positives if item[:2] == anchor[:2]}
    candidates = {"a": [], "b": [], "c": []}
    for item in inventory:
        if item in positives:
            continue
        if item[0] != anchor[0]:
            if item[0] not in pos_a:
                candidates["a"].append((anchor[0], item[0]))
        elif item[1] != anchor[1]:
            if item[1] not in pos_b:
                candidates["b"].append((anchor[1], item[1]))
        elif item[2] != anchor[2] and item[2] not in pos_c:
            candidates["c"].append((anchor[2], item[2]))
    for values in candidates.values():
        rng.shuffle(values)
    return {level: candidates[level][: int(quota[level])] for level in ("a", "b", "c")}


def first_divergence_rank_loss(
    stage_logits: dict[str, torch.Tensor], pairs: dict[str, list[tuple[int, int]]],
    margin: dict[str, float], temperature: dict[str, float], level_weight: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Strict-prefix local FDR loss and diagnostics; no tuple score is formed."""
    values: dict[str, torch.Tensor] = {}
    active: list[tuple[float, torch.Tensor]] = []
    zero = next(iter(stage_logits.values())).sum() * 0.0
    for level in ("a", "b", "c"):
        local = pairs[level]
        if not local:
            values[f"fd_loss_{level}"] = zero
            values[f"fd_pair_count_{level}"] = zero
            values[f"fd_pair_accuracy_{level}"] = zero
            values[f"fd_margin_{level}"] = zero
            values[f"fd_violation_rate_{level}"] = zero
            continue
        pos = torch.tensor([item[0] for item in local], device=stage_logits[level].device)
        neg = torch.tensor([item[1] for item in local], device=stage_logits[level].device)
        gap = stage_logits[level][pos] - stage_logits[level][neg]
        local_loss = torch.nn.functional.softplus((float(margin[level]) - gap) / float(temperature[level])).mean()
        values[f"fd_loss_{level}"] = local_loss
        values[f"fd_pair_count_{level}"] = torch.tensor(float(len(local)), device=gap.device)
        values[f"fd_pair_accuracy_{level}"] = (gap > 0).float().mean()
        values[f"fd_margin_{level}"] = gap.mean()
        values[f"fd_violation_rate_{level}"] = (gap < float(margin[level])).float().mean()
        active.append((float(level_weight[level]), local_loss))
    total = sum(weight * value for weight, value in active) / sum(weight for weight, _ in active) if active else zero
    values["fd_total"] = total
    return total, values


def _nonpad_tokens(item: dict[str, Any]) -> int:
    return sum(1 for value in item["attention_mask"] if int(value) != 0)


def _packed_nonpad_lengths(dataset: Any) -> list[int]:
    """Read only Arrow packing metadata, avoiding full 32K row decoding."""
    packing = dataset.data.column("packing_params")
    lengths: list[int] = []
    for chunk in packing.chunks:
        paddings = chunk.field("right_padding_length").to_pylist()
        lengths.extend(32768 - int(value or 0) for value in paddings)
    if len(lengths) != len(dataset):
        raise RuntimeError("packed length metadata does not match dataset rows")
    return lengths


def _packed_group_exposures(
    dataset: Any, group_aliases: dict[int, int] | None = None
) -> list[dict[int, int]]:
    """Return the raw prompt groups present in every packed row.

    A group is counted once even when it owns several raw rows in the same
    pack.  The exposure identity is the stable prompt_group_id, never the
    transient packed-row or within-pack instance number.
    """
    result: list[dict[int, int]] = []
    for item in dataset:
        groups = item["rec_group_ids"]
        instances = item["rec_instance_ids"]
        seen: set[int] = set()
        for group, instance in zip(groups, instances):
            group, instance = int(group), int(instance)
            if group > 0 and instance > 0:
                seen.add(group_aliases.get(group, group) if group_aliases else group)
        result.append({group: 1 for group in seen})
    if len(result) != len(dataset):
        raise RuntimeError("packed group exposure metadata does not match dataset rows")
    return result


def _packed_domain_token_exposures(
    dataset: Any, group_domains: dict[int, str]
) -> list[dict[str, int]]:
    """Attribute every non-padding token in a packed Full row to its target domain.

    Neat packing preserves original-example boundaries. Each Full example has
    exactly one non-zero recommendation group, so this gives an exact domain
    token accounting even when one packed row contains several domains.
    """
    result: list[dict[str, int]] = []
    for item in dataset:
        boundaries = [int(value) for value in item["packing_params"]["sequence_boundaries"]]
        rec_groups = item["rec_group_ids"]
        nonpad = int(sum(1 for value in item["attention_mask"] if int(value) != 0))
        totals: defaultdict[str, int] = defaultdict(int)
        for start, end in zip(boundaries, boundaries[1:]):
            if start >= nonpad:
                continue
            end = min(end, nonpad)
            groups = {int(value) for value in rec_groups[start:end] if int(value) > 0}
            if len(groups) != 1:
                raise ValueError(f"packed Full sequence must map to one target domain, got groups={sorted(groups)}")
            group = next(iter(groups))
            domain = group_domains.get(group)
            if domain is None:
                raise ValueError(f"packed Full sequence references unknown group {group}")
            totals[domain] += end - start
        if sum(totals.values()) != nonpad:
            raise ValueError("packed Full domain token accounting does not equal non-padding length")
        result.append(dict(totals))
    return result


def _apply_domain_conditioned_full_c2(
    plan: list[tuple[str, int]], stage_reports: list[dict[str, Any]],
    pool_lengths: dict[str, list[int]], full_domain_tokens: dict[str, list[dict[str, int]]],
    domain_pool_lengths: dict[str, list[int]], config: dict[str, Any], seed: int,
) -> dict[str, Any]:
    """Replace only C2 Full rows with independently packed domain rows.

    The Control plan is built first.  Its exact per-domain Full token totals
    become the frozen branch budget; only Video's internal K mix changes.
    """
    c2 = next((item for item in stage_reports if item["name"] == "stage_c2"), None)
    if c2 is None:
        raise ValueError("stage_c2 is required for domain-conditioned Full curriculum")
    start, end = int(c2["plan_start"]), int(c2["plan_end"])
    domains = [str(value) for value in config["domains"]]
    ratios = {str(domain): {str(k): float(v) for k, v in values.items()} for domain, values in config["entropy"].items()}
    if set(domains) != set(ratios):
        raise ValueError("domain-conditioned C2 ratios must name every configured domain")
    for domain in domains:
        if set(ratios[domain]) != {"k1", "k2", "k3"} or not math.isclose(sum(ratios[domain].values()), 1.0):
            raise ValueError(f"invalid C2 K ratios for {domain}: {ratios[domain]}")

    control_domain_tokens: defaultdict[str, int] = defaultdict(int)
    positions: list[int] = []
    for position in range(start, end):
        pool, index = plan[position]
        if pool in {"full_k1", "full_k2", "full_k3"}:
            positions.append(position)
            for domain, tokens in full_domain_tokens[pool][index].items():
                control_domain_tokens[domain] += int(tokens)
    if set(control_domain_tokens) != set(domains):
        raise ValueError(f"Control C2 Full domain coverage mismatch: {dict(control_domain_tokens)}")

    requested = {
        f"{domain}_{bucket}": int(round(control_domain_tokens[domain] * ratios[domain][bucket]))
        for domain in domains for bucket in ("k1", "k2", "k3")
    }
    # Deterministic independently shuffled streams preserve all pre-C2 plan
    # entries and all R2/R1/base rows. Full pools have no reuse cap.
    rng = random.Random(seed + 5101)
    orders: dict[str, list[int]] = {}
    offsets: defaultdict[str, int] = defaultdict(int)
    for name, lengths in domain_pool_lengths.items():
        order = list(range(len(lengths)))
        rng.shuffle(order)
        orders[name] = order

    actual: defaultdict[str, int] = defaultdict(int)
    selected_counts: defaultdict[str, int] = defaultdict(int)
    for position in positions:
        choices = list(requested)
        choice = max(
            choices,
            key=lambda name: (requested[name] - actual[name]) / max(requested[name], 1),
        )
        domain, bucket = choice.rsplit("_", 1)
        pool = f"full_{domain}_{bucket}"
        if pool not in orders:
            raise ValueError(f"missing independently packed C2 pool {pool}")
        if offsets[pool] >= len(orders[pool]):
            rng.shuffle(orders[pool])
            offsets[pool] = 0
        index = orders[pool][offsets[pool]]
        offsets[pool] += 1
        plan[position] = (pool, index)
        actual[choice] += int(domain_pool_lengths[pool][index])
        selected_counts[choice] += 1

    actual_domain = {domain: sum(actual[f"{domain}_{bucket}"] for bucket in ("k1", "k2", "k3")) for domain in domains}
    domain_ratio_delta_pp = {
        domain: abs(actual_domain[domain] / sum(actual_domain.values()) - control_domain_tokens[domain] / sum(control_domain_tokens.values())) * 100.0
        for domain in domains
    }
    if max(domain_ratio_delta_pp.values(), default=0.0) > float(config.get("max_domain_ratio_delta_pp", 0.5)):
        raise RuntimeError(f"C2 Full domain budget drift exceeds contract: {domain_ratio_delta_pp}")
    return {
        "enabled": True,
        "scope": "stage_c2/full_only",
        "control_full_tokens_by_domain": dict(control_domain_tokens),
        "requested_full_tokens_by_domain_bucket": requested,
        "actual_full_tokens_by_domain_bucket": dict(actual),
        "actual_full_tokens_by_domain": actual_domain,
        "domain_token_ratio_delta_pp_vs_control": domain_ratio_delta_pp,
        "selected_pack_counts_by_domain_bucket": dict(selected_counts),
        "entropy": ratios,
    }


def make_token_plan(
    pools: dict[str, Any], pool_lengths: dict[str, list[int]], total_budget: int,
    world_size: int, accumulation: int, seed: int,
    r2_k1_pack_exposures: list[dict[int, int]] | None = None,
    base_fraction: float = 0.0,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """Greedily satisfy task and entropy quotas using actual packed non-pad tokens."""
    recommendation_stages = [
        ("stage_a", 0.00, 0.20, {"full": .50, "r1": .35, "r2": .15}),
        ("stage_b", 0.20, 0.40, {"full": .50, "r1": .15, "r2": .35}),
        ("stage_c1", 0.40, 0.80, {"full": .80, "r1": .10, "r2": .10}),
        ("stage_c2", 0.80, 1.00, {"full": .75, "r1": .05, "r2": .20}),
    ]
    if not 0.0 <= base_fraction < 1.0:
        raise ValueError(f"base_fraction must be in [0, 1), got {base_fraction}")
    stages = []
    for stage_name, start, end, recommendation_ratio in recommendation_stages:
        task_ratio = {"base": base_fraction}
        task_ratio.update({name: (1.0 - base_fraction) * value for name, value in recommendation_ratio.items()})
        stages.append((stage_name, start, end, task_ratio))
    entropy = {
        "full": {
            "stage_a": {"k1": .20, "k2": .35, "k3": .45},
            "stage_b": {"k1": .35, "k2": .35, "k3": .30},
            "stage_c1": {"k1": .55, "k2": .30, "k3": .15},
            "stage_c2": {"k1": .75, "k2": .20, "k3": .05},
        },
        "r2": {
            "stage_a": {"k1": .15, "k2": .35, "k3": .50},
            "stage_b": {"k1": .20, "k2": .35, "k3": .45},
            "stage_c1": {"k1": .35, "k2": .35, "k3": .30},
            "stage_c2": {"k1": .50, "k2": .35, "k3": .15},
        },
    }
    reuse_caps = {"r1": 2.0, "r2_k1": 1.0, "r2_k2": 4.0, "r2_k3": 4.0}
    r2_k1_reservation = {"stage_a": .10, "stage_b": .15, "stage_c1": .30, "stage_c2": .45}
    if r2_k1_pack_exposures is None or len(r2_k1_pack_exposures) != len(pools["r2_k1"]):
        raise ValueError("R2-K1 raw group exposure metadata is required for every packed shard row")
    # Four independently shuffled shards are merely a repacking mechanism.
    # They never create a new exposure identity: every raw prompt group has a
    # hard total exposure cap of four, including its first observation.
    all_k1_groups = {group for pack in r2_k1_pack_exposures for group in pack}
    group_exposure_cap = {group: 4 for group in all_k1_groups}
    stage_pack_limits = largest_remainder(len(pools["r2_k1"]), r2_k1_reservation)
    rng = random.Random(seed)
    orders = {name: list(range(len(dataset))) for name, dataset in pools.items()}
    for order in orders.values():
        rng.shuffle(order)
    offsets = defaultdict(int)
    uses = defaultdict(int)
    group_exposures: defaultdict[int, int] = defaultdict(int)
    plan: list[tuple[str, int]] = []
    global_used = 0
    stage_reports: list[dict[str, Any]] = []

    def next_index(pool: str, stage_name: str | None = None, stage_k1_uses: int = 0) -> int | None:
        cap = reuse_caps.get(pool, float("inf"))
        if not math.isinf(cap) and uses[pool] >= math.floor(len(orders[pool]) * cap):
            return None
        if pool == "r2_k1" and stage_name is not None:
            if stage_k1_uses >= max(1, stage_pack_limits[stage_name]):
                return None
            while offsets[pool] < len(orders[pool]):
                value = orders[pool][offsets[pool]]
                offsets[pool] += 1
                exposure = r2_k1_pack_exposures[value]
                if all(group_exposures[group] + count <= group_exposure_cap[group] for group, count in exposure.items()):
                    for group, count in exposure.items():
                        group_exposures[group] += count
                    uses[pool] += 1
                    return value
            return None
        if offsets[pool] >= len(orders[pool]):
            rng.shuffle(orders[pool])
            offsets[pool] = 0
        value = orders[pool][offsets[pool]]
        offsets[pool] += 1
        uses[pool] += 1
        return value

    for stage_name, start, end, task_ratio in stages:
        stage_budget = round(total_budget * (end - start))
        stage_used = 0
        task_used = defaultdict(int)
        bucket_used = defaultdict(int)
        stage_k1_uses = 0
        requested_r2_k1_tokens = 0
        actual_r2_k1_tokens = 0
        overflow_r2_k1 = defaultdict(int)
        requested_r2_k1_tokens = round(stage_budget * task_ratio["r2"] * entropy["r2"][stage_name]["k1"])
        stage_plan_start = len(plan)
        while stage_used < stage_budget:
            task = max(
                task_ratio,
                key=lambda key: max(stage_budget * task_ratio[key] - task_used[key], 0)
                / max(stage_budget * task_ratio[key], 1),
            )
            if task in {"base", "r1"}:
                pool = task
            else:
                ratios = entropy[task][stage_name]
                bucket = max(
                    ratios,
                    key=lambda key: max(stage_budget * task_ratio[task] * ratios[key] - bucket_used[f"{task}_{key}"], 0)
                    / max(stage_budget * task_ratio[task] * ratios[key], 1),
                )
                pool = f"{task}_{bucket}"
            block: list[tuple[str, int]] = []
            block_tokens = 0
            requested_block_pool = pool
            for _rank in range(world_size):
                requested_pool = requested_block_pool
                pool = requested_pool
                index = next_index(pool, stage_name, stage_k1_uses)
                if index is None:
                    alternatives = [name for name in pools if name.startswith(task + "_")]
                    if task == "r1":
                        alternatives = ["r1"]
                    available = []
                    for name in alternatives:
                        cap = reuse_caps.get(name, float("inf"))
                        stage_cap = stage_pack_limits[stage_name] if name == "r2_k1" else float("inf")
                        if (math.isinf(cap) or uses[name] < math.floor(len(orders[name]) * cap)) and (name != "r2_k1" or stage_k1_uses < max(1, stage_cap)):
                            available.append(name)
                    if not available:
                        raise RuntimeError(f"reuse cap exhausted for task {task}")
                    if task == "r1":
                        pool = "r1"
                    else:
                        # The requested entropy bucket can exhaust its reuse
                        # cap (especially sparse R2-k1). Continue with the
                        # available bucket having the largest normalized token
                        # deficit instead of always falling back to k2.
                        if requested_pool == "r2_k1":
                            overflow = {"r2_k2": .70, "r2_k3": .30}
                            choices = [name for name in available if name in overflow]
                            overflow_total = sum(overflow_r2_k1.values())
                            pool = max(
                                choices,
                                key=lambda name: overflow[name] * (overflow_total + 1) - overflow_r2_k1[name],
                            ) if choices else max(available)
                            overflow_r2_k1[pool] += 1
                        else:
                            pool = max(available, key=lambda name: -bucket_used[name] / max(ratios[name.rsplit("_", 1)[1]], 1e-12))
                    index = next_index(pool, stage_name, stage_k1_uses)
                    if index is None:
                        raise RuntimeError(f"selected unavailable fallback pool {pool}")
                block.append((pool, int(index)))
                block_tokens += pool_lengths[pool][int(index)]
                if pool == "r2_k1":
                    stage_k1_uses += 1
                    actual_r2_k1_tokens += pool_lengths[pool][int(index)]
            plan.extend(block)
            stage_used += block_tokens
            global_used += block_tokens
            task_used[task] += block_tokens
            if task != "r1":
                for selected_pool, selected_index in block:
                    bucket_used[selected_pool] += pool_lengths[selected_pool][selected_index]
        stage_reports.append(
            {
                "name": stage_name,
                "plan_start": stage_plan_start,
                "plan_end": len(plan),
                "target_tokens": stage_budget,
                "actual_tokens": stage_used,
                "task_tokens": dict(task_used),
                "bucket_tokens": dict(bucket_used),
                "expanded_examples": len(plan) - stage_plan_start,
                "requested_r2_k1_tokens": requested_r2_k1_tokens,
                "feasible_r2_k1_tokens": sum(sorted(pool_lengths["r2_k1"], reverse=True)[:stage_pack_limits[stage_name]]),
                "actual_r2_k1_tokens": actual_r2_k1_tokens,
                "r2_k1_reserved_pack_cap": stage_pack_limits[stage_name],
                "r2_k1_overflow_to_k2_packs": overflow_r2_k1["r2_k2"],
                "r2_k1_overflow_to_k3_packs": overflow_r2_k1["r2_k3"],
            }
        )
    global_microbatches = len(plan) // world_size
    remainder = global_microbatches % accumulation
    if remainder:
        needed_blocks = accumulation - remainder
        fallback = "full_k1"
        for _ in range(needed_blocks):
            for _rank in range(world_size):
                index = next_index(fallback)
                if index is None:
                    raise RuntimeError("cannot pad final gradient accumulation block")
                plan.append((fallback, index))
                global_used += pool_lengths[fallback][index]
    report = {
        "strategy": "max_normalized_actual_token_deficit",
        "target_tokens": total_budget,
        "planned_actual_tokens": global_used,
        "tolerance_ratio": (global_used - total_budget) / total_budget,
        "stages": stage_reports,
        "pool_pack_counts": {name: len(value) for name, value in pools.items()},
        "pool_reuse_counts": dict(uses),
        "base_token_fraction": base_fraction,
        "r2_k1_reuse_unit": "prompt_group_id",
        "r2_k1_max_total_exposure_factor": 4.0,
        "r2_k1_unique_groups_seen": sum(1 for value in group_exposures.values() if value > 0),
        "r2_k1_group_exposure_mean": sum(group_exposures.values()) / max(1, len(group_exposures)),
        "r2_k1_group_exposure_max": max(group_exposures.values(), default=0),
        "r2_k1_group_exposure_cap_max": max(group_exposure_cap.values(), default=0),
        "r2_k1_group_exposure_violations": sum(
            1 for group, value in group_exposures.items() if value > group_exposure_cap[group]
        ),
        "r2_k1_repack_round": 4,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps": len(plan) // world_size // accumulation,
    }
    return plan, report


def _patch_collator_trainer_and_model(
    raw: dict[str, Any], pool_cache_root: Path, base_pool_cache_root: Path
) -> None:
    import torch
    from datasets import concatenate_datasets
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.train.sft import trainer as trainer_module
    import llamafactory.train.sft.workflow as workflow

    joint_cfg = REC_JOINT_CONFIG
    seed = int(raw.get("seed", 19260817))
    stage_token_ids: dict[str, list[int]] = {}
    stage_id_to_index: dict[str, dict[int, int]] = {}
    group_catalog: dict[int, list[tuple[int, int, int]]] = {}
    group_domains: dict[int, str] = {}
    domain_inventory: dict[str, list[tuple[int, int, int]]] = {}
    validation_domain_inventory: dict[str, list[tuple[int, int, int]]] = {}
    train_group_ids: set[int] = set()
    exposure_group_by_rec_group: dict[int, int] = {}
    if joint_cfg.get("enabled"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(raw["model_name_or_path"]), trust_remote_code=bool(raw.get("trust_remote_code", True))
        )
        for stage in ("a", "b", "c"):
            tokens = [f"<s_{stage}_{index}>" for index in range(8192)]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(set(token_ids)) != 8192 or tokenizer.unk_token_id in token_ids:
                raise ValueError(f"SID {stage.upper()} vocabulary is not 8192 distinct tokenizer tokens")
            stage_token_ids[stage] = [int(value) for value in token_ids]
            stage_id_to_index[stage] = {token_id: index for index, token_id in enumerate(token_ids)}
        catalog_path = Path(str(raw["rec_group_catalog_path"]))
        catalog_raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        if int(catalog_raw.get("schema_version", 0)) != 2:
            raise ValueError("V4 requires the split-aware rec_group_catalog schema_version=2")
        prompt_groups = sorted({
            str(item["prompt_group_id"])
            for item in catalog_raw["train_groups"] + catalog_raw["validation_groups"]
        })
        prompt_group_index = {value: index + 1 for index, value in enumerate(prompt_groups)}
        for item in catalog_raw["train_groups"]:
            train_group_ids.add(int(item["group_index"]))
            group_catalog[int(item["group_index"])] = [
                tuple(int(value) for value in re.search(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", sid).groups())
                for sid in item["positive_future_sids"]
            ]
            group_domains[int(item["group_index"])] = str(item["domain"])
            exposure_group_by_rec_group[int(item["group_index"])] = prompt_group_index[str(item["prompt_group_id"])]
        for item in catalog_raw["validation_groups"]:
            group_catalog[int(item["group_index"])] = [
                tuple(int(value) for value in re.search(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", sid).groups())
                for sid in item["positive_future_sids"]
            ]
            group_domains[int(item["group_index"])] = str(item["domain"])
            exposure_group_by_rec_group[int(item["group_index"])] = prompt_group_index[str(item["prompt_group_id"])]
        for domain, values in catalog_raw["train_domain_inventory"].items():
            domain_inventory[domain] = [
                tuple(int(value) for value in re.search(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", sid).groups())
                for sid in values
            ]
        for domain, values in catalog_raw["validation_domain_inventory"].items():
            validation_domain_inventory[domain] = [
                tuple(int(value) for value in re.search(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", sid).groups())
                for sid in values
            ]

    original_unpad = SFTDataCollatorWith4DAttentionMask._unpad_packed_features
    original_collator_call = SFTDataCollatorWith4DAttentionMask.__call__

    def collator_call(self: Any, features: list[dict[str, Any]]) -> Any:
        # Reuse the byte-identical V3 material/user packed cache. It predates
        # recommendation metadata, so inject zero-valued fields only at batch
        # collation instead of duplicating its 9.9 GiB Arrow cache.
        normalized = []
        for feature in features:
            if feature.get("rec_group_ids") is None or feature.get("rec_instance_ids") is None:
                feature = dict(feature)
                length = len(feature["input_ids"])
                feature["rec_group_ids"] = [0] * length
                feature["rec_instance_ids"] = [0] * length
                feature["response_mode_ids"] = [0] * length
                feature["sample_group_ids"] = [0] * length
            elif feature.get("response_mode_ids") is None:
                feature = dict(feature)
                feature["response_mode_ids"] = [0] * len(feature["input_ids"])
                feature["sample_group_ids"] = [0] * len(feature["input_ids"])
            normalized.append(feature)
        return original_collator_call(self, normalized)

    SFTDataCollatorWith4DAttentionMask.__call__ = collator_call

    @staticmethod
    def unpad(features: dict[str, Any]) -> None:
        attention = features.get("attention_mask")
        indices = None
        full_length = None
        if torch.is_tensor(attention) and attention.dim() == 2 and attention.size(0) == 1:
            found = torch.nonzero(attention[0] != 0, as_tuple=False).flatten()
            if found.numel() != attention.size(1):
                indices, full_length = found, attention.size(1)
        original_unpad(features)
        if indices is not None:
            for key in ("loss_weights", "metric_weights", "loss_groups", "rec_group_ids", "rec_instance_ids", "response_mode_ids", "sample_group_ids"):
                value = features.get(key)
                if torch.is_tensor(value) and value.dim() == 2 and value.size(1) == full_length:
                    features[key] = value.index_select(1, indices)

    SFTDataCollatorWith4DAttentionMask._unpad_packed_features = unpad

    original_get_dataset = workflow.get_dataset

    def get_curriculum_dataset(template: Any, model_args: Any, data_args: Any, training_args: Any, **kwargs: Any) -> dict[str, Any]:
        global CURRICULUM_PLAN_REPORT
        kwargs.pop("stage", None)
        pools: dict[str, Any] = {}
        for pool_name, dataset_names in active_pool_datasets().items():
            pool_args = copy.deepcopy(data_args)
            pool_args.dataset = list(dataset_names)
            pool_args.eval_dataset = None
            pool_args.val_size = 0.0
            cache_root = base_pool_cache_root if pool_name in {"base", "val_base"} else pool_cache_root
            pool_args.tokenized_path = str(cache_root / pool_name)
            module = original_get_dataset(
                template, model_args, pool_args, training_args, stage="sft", **kwargs
            )
            pools[pool_name] = module["train_dataset"]
        train_pools = {
            key: pools[key]
            for key in ("base", "full_k1", "full_k2", "full_k3", "r1", "r2_k1", "r2_k2", "r2_k3")
        }
        pool_lengths = {key: _packed_nonpad_lengths(value) for key, value in pools.items()}
        r2_k1_pack_exposures = _packed_group_exposures(pools["r2_k1"], exposure_group_by_rec_group)
        base_budget = sum(pool_lengths["base"])
        full_budget = sum(sum(pool_lengths[key]) for key in ("full_k1", "full_k2", "full_k3"))
        baseline_budget = base_budget + full_budget
        base_fraction = base_budget / baseline_budget
        configured_max = int(raw.get("curriculum_max_steps_override", 0))
        expanded_plan, plan_report = make_token_plan(
            train_pools,
            {key: pool_lengths[key] for key in train_pools},
            baseline_budget,
            training_args.world_size,
            training_args.gradient_accumulation_steps,
            training_args.seed,
            r2_k1_pack_exposures,
            base_fraction,
        )
        if DOMAIN_CONDITIONED_FULL_C2 is not None:
            domains = [str(value) for value in DOMAIN_CONDITIONED_FULL_C2["domains"]]
            domain_pool_names = [
                f"full_{domain}_{bucket}"
                for domain in domains for bucket in ("k1", "k2", "k3")
            ]
            missing = [name for name in domain_pool_names if name not in pools]
            if missing:
                raise ValueError(f"domain-conditioned C2 pools were not loaded: {missing}")
            full_domain_tokens = {
                pool: _packed_domain_token_exposures(pools[pool], group_domains)
                for pool in ("full_k1", "full_k2", "full_k3")
            }
            plan_report["domain_conditioned_full_c2"] = _apply_domain_conditioned_full_c2(
                expanded_plan,
                plan_report["stages"],
                {key: pool_lengths[key] for key in ("full_k1", "full_k2", "full_k3")},
                full_domain_tokens,
                {name: pool_lengths[name] for name in domain_pool_names},
                DOMAIN_CONDITIONED_FULL_C2,
                training_args.seed,
            )
        max_steps = len(expanded_plan) // training_args.world_size // training_args.gradient_accumulation_steps
        if configured_max > 0:
            max_steps = configured_max
            expanded_plan = expanded_plan[: max_steps * training_args.world_size * training_args.gradient_accumulation_steps]
        training_args.max_steps = max_steps
        plan_report.update(
            {
                "baseline_definition": "95% V2 material + 95% V2 user_clean + prompt-group 95% recommendation Full, packed separately at 32K",
                "baseline_actual_nonpad_tokens": baseline_budget,
                "base_actual_nonpad_tokens": base_budget,
                "recommendation_full_actual_nonpad_tokens": full_budget,
                "training_optimizer_steps": max_steps,
            }
        )
        CURRICULUM_PLAN_REPORT = plan_report
        output = Path(training_args.output_dir)
        if training_args.should_save:
            output.mkdir(parents=True, exist_ok=True)
            (output / "curriculum_plan.json").write_text(
                json.dumps(plan_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if training_args.should_log:
            print(json.dumps({"event": "curriculum_plan", **plan_report}, ensure_ascii=False), flush=True)
        all_train_pools = dict(train_pools)
        if DOMAIN_CONDITIONED_FULL_C2 is not None:
            all_train_pools.update({
                name: pools[name]
                for name in (
                    f"full_{domain}_{bucket}"
                    for domain in DOMAIN_CONDITIONED_FULL_C2["domains"]
                    for bucket in ("k1", "k2", "k3")
                )
            })
        train_dataset = CurriculumIterableDataset.build(
            all_train_pools, expanded_plan
        )
        eval_dataset = concatenate_datasets(
            [pools[key] for key in ("val_base", "val_full", "val_r1", "val_r2")]
        )
        return {"train_dataset": train_dataset, "eval_dataset": eval_dataset}

    workflow.get_dataset = get_curriculum_dataset

    original_load_model = workflow.load_model

    def load_weighted_model(*args: Any, **kwargs: Any) -> Any:
        model = original_load_model(*args, **kwargs)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if not hasattr(base, "model") or not hasattr(base, "lm_head"):
            raise RuntimeError("custom Liger loss requires a causal LM with model and lm_head")
        import triton
        from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
        from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
        from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, is_hip
        from transformers.modeling_outputs import CausalLMOutputWithPast

        class WeightedLigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
            """One fused CE call with arbitrary per-token objective weights.

            Upstream Liger supports class weights, but not the per-token weights
            required by TASK V3. Calling it once for every distinct 1/k value
            retains one full LM-head gradient buffer per call (about 2.69 GiB
            for this model). This implementation applies the weights inside a
            single fused pass and therefore retains exactly one such buffer.
            """

            @staticmethod
            @amp_custom_fwd
            def forward(
                ctx: Any,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                target: torch.Tensor,
                token_weights: torch.Tensor,
                bias: torch.Tensor | None = None,
                ignore_index: int = IGNORE_INDEX,
                softcap: float | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                bt, width = hidden.shape
                vocab = weight.shape[0]
                block_size = min(65536 // 2, triton.next_power_of_2(vocab))
                inc_factor = triton.cdiv(vocab, width)
                chunk_size = triton.next_power_of_2(triton.cdiv(bt, inc_factor))
                num_chunks = triton.cdiv(bt, chunk_size)

                grad_hidden = torch.zeros_like(hidden)
                grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
                grad_bias = torch.zeros_like(bias) if bias is not None else None
                token_losses = torch.zeros(bt, dtype=torch.float32, device=hidden.device)
                normalized_weights = token_weights.float() / token_weights.float().sum().clamp_min(1e-8)

                for chunk_id in range(num_chunks):
                    start = chunk_id * chunk_size
                    end = min((chunk_id + 1) * chunk_size, bt)
                    hidden_chunk = hidden[start:end]
                    logits_chunk = hidden_chunk @ weight.t()
                    if bias is not None:
                        logits_chunk = logits_chunk + bias
                    target_chunk = target[start:end].contiguous()
                    loss_slice = token_losses[start:end]
                    rows = logits_chunk.shape[0]
                    logits_chunk = logits_chunk.contiguous()
                    liger_cross_entropy_kernel[(rows,)](
                        X_ptr=logits_chunk,
                        X_stride=logits_chunk.stride(-2),
                        Y_ptr=target_chunk,
                        Y_stride=target_chunk.stride(-1),
                        weight_ptr=None,
                        loss_ptr=loss_slice,
                        z_loss_ptr=None,
                        loss_stride=loss_slice.stride(-1),
                        token_accuracy_ptr=None,
                        token_accuracy_stride=0,
                        predicted_tokens_ptr=None,
                        predicted_tokens_stride=0,
                        n_cols=vocab,
                        n_non_ignore=bt,
                        sum_non_ignore_weight=bt,
                        weight_sum=0.0,
                        ignore_index=ignore_index,
                        lse_square_scale=0.0,
                        label_smoothing=0.0,
                        reduction="none",
                        softcap=softcap,
                        RETURN_Z_LOSS=False,
                        RETURN_TOKEN_ACCURACY=False,
                        RETURN_PREDICTED_TOKENS=False,
                        HAS_WEIGHT=False,
                        HAS_SOFTCAPPING=softcap is not None,
                        HAS_GRADIENTS=hidden.requires_grad,
                        BLOCK_SIZE=block_size,
                        num_warps=32 if not is_hip() else 16,
                    )
                    # The kernel has replaced logits with unnormalised dCE/dlogits.
                    logits_chunk.mul_(normalized_weights[start:end, None])
                    if hidden.requires_grad:
                        grad_hidden[start:end] = logits_chunk @ weight
                    if grad_weight is not None and hidden.requires_grad:
                        # Torch 2.5 has no addmm(out_dtype=fp32). Chunking the
                        # vocab dimension avoids a second 2.69 GiB temporary.
                        vocab_chunk = 8192
                        for vocab_start in range(0, vocab, vocab_chunk):
                            vocab_end = min(vocab_start + vocab_chunk, vocab)
                            partial = torch.mm(
                                logits_chunk[:, vocab_start:vocab_end].t(), hidden_chunk
                            ).float()
                            grad_weight[vocab_start:vocab_end].add_(partial)
                    if grad_bias is not None and hidden.requires_grad:
                        grad_bias.add_(logits_chunk.sum(dim=0).to(grad_bias.dtype))

                loss = torch.sum(token_losses * normalized_weights)
                ctx.save_for_backward(
                    grad_hidden.detach(),
                    grad_weight.detach() if grad_weight is not None else None,
                    grad_bias.detach() if grad_bias is not None else None,
                )
                return loss, token_losses

            @staticmethod
            @amp_custom_bwd
            def backward(ctx: Any, grad_loss: torch.Tensor, grad_token_losses: Any) -> tuple[Any, ...]:
                del grad_token_losses
                saved = ctx.saved_tensors
                grad_hidden, grad_weight = saved[0], saved[1]
                grad_bias = saved[2] if len(saved) > 2 else None
                grad_hidden, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
                    grad_loss, grad_hidden, grad_weight, grad_bias
                )
                return grad_hidden, grad_weight, None, None, grad_bias, None, None

        def forward(
            model_self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            labels: Any = None,
            use_cache: Any = None,
            cache_position: Any = None,
            loss_weights: Any = None,
            metric_weights: Any = None,
            loss_groups: Any = None,
            rec_group_ids: Any = None,
            rec_instance_ids: Any = None,
            response_mode_ids: Any = None,
            sample_group_ids: Any = None,
            return_dict: Any = None,
            **model_kwargs: Any,
        ) -> Any:
            outputs = model_self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **model_kwargs,
            )
            hidden = outputs.last_hidden_state
            if labels is None:
                logits = model_self.lm_head(hidden)
                return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
            if any(value is None for value in (loss_weights, metric_weights, loss_groups, rec_group_ids, rec_instance_ids, response_mode_ids, sample_group_ids)):
                raise RuntimeError("custom loss fields missing")
            shifted_hidden = hidden[..., :-1, :].contiguous().view(-1, model_self.config.hidden_size)
            shifted_labels = labels[..., 1:].contiguous().view(-1).to(shifted_hidden.device)
            objective = loss_weights[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.float32)
            metric = metric_weights[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.float32)
            groups = loss_groups[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.long)
            shifted_rec_groups = rec_group_ids[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.long)
            shifted_rec_instances = rec_instance_ids[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.long)
            shifted_modes = response_mode_ids[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.long)
            shifted_sample_groups = sample_group_ids[..., 1:].contiguous().view(-1).to(shifted_hidden.device, torch.long)
            valid = (shifted_labels != IGNORE_INDEX) & (objective > 0) & (groups > 0)
            objective_den = objective.masked_select(valid).sum()
            component_num: dict[int, torch.Tensor] = {}
            component_den: dict[int, torch.Tensor] = {}
            if bool(valid.any()):
                valid_hidden = shifted_hidden[valid]
                valid_labels = shifted_labels[valid]
                valid_objective = objective[valid]
                valid_metric = metric[valid]
                valid_groups = groups[valid]
                ce_loss, token_losses = WeightedLigerFusedLinearCrossEntropyFunction.apply(
                    valid_hidden,
                    model_self.lm_head.weight,
                    valid_labels,
                    valid_objective,
                    getattr(model_self.lm_head, "bias", None),
                    IGNORE_INDEX,
                    getattr(model_self.config, "final_logit_softcapping", None),
                )
            else:
                ce_loss = shifted_hidden.sum() * 0.0
                token_losses = torch.empty(0, device=shifted_hidden.device, dtype=torch.float32)
                valid_objective = objective[valid]
                valid_metric = metric[valid]
                valid_groups = groups[valid]
            loss = ce_loss
            joint_metrics: dict[str, torch.Tensor] = {}
            rec_groups = (groups == TASK_IDS["full_sid"]) | (groups == TASK_IDS["r2"])
            if joint_cfg.get("enabled") and bool(rec_groups.any()):
                instance_losses: list[torch.Tensor] = []
                instance_weights: list[float] = []
                instance_modes: list[str] = []
                instance_domains: list[str] = []
                metric_lists: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
                is_full_batch = bool((groups == TASK_IDS["full_sid"]).any())
                all_ids = {
                    stage: torch.tensor(stage_token_ids[stage], device=shifted_hidden.device, dtype=torch.long)
                    for stage in ("a", "b", "c")
                }
                # Map target token IDs to their 0..8191 SID index once per
                # packed batch. This avoids repeatedly copying the whole 32K
                # label vector to CPU for every recommendation instance.
                stage_lookup: dict[str, torch.Tensor] = {}
                safe_labels = shifted_labels.clamp_min(0)
                for stage in ("a", "b", "c"):
                    lookup = torch.full(
                        (int(model_self.config.vocab_size),), -1,
                        device=shifted_hidden.device, dtype=torch.long,
                    )
                    lookup[all_ids[stage]] = torch.arange(8192, device=shifted_hidden.device)
                    stage_lookup[stage] = lookup
                for instance in torch.unique(shifted_rec_instances[rec_groups]).tolist():
                    if int(instance) <= 0:
                        continue
                    mask = rec_groups & (shifted_rec_instances == int(instance))
                    group_values = torch.unique(shifted_rec_groups[mask])
                    if group_values.numel() != 1:
                        raise ValueError("one recommendation instance maps to multiple future groups")
                    group_index = int(group_values.item())
                    positives = group_catalog[group_index]
                    mode_values = torch.unique(shifted_modes[mask])
                    mode_id = int(mode_values.max().item()) if mode_values.numel() else 0
                    mode = "cot" if mode_id == MODE_IDS["cot"] else "uncot" if mode_id == MODE_IDS["uncot"] else "none"
                    if is_full_batch and mode == "none":
                        raise ValueError("Full recommendation instance lacks response mode")
                    stage_logits: dict[str, torch.Tensor] = {}
                    truth: dict[str, int] = {}
                    for stage in ("a", "b", "c"):
                        indexed_labels = stage_lookup[stage][safe_labels]
                        stage_mask = mask & (indexed_labels >= 0)
                        if int(stage_mask.sum().item()) != 1:
                            raise ValueError(f"instance {instance} lacks one {stage.upper()} target")
                        stage_hidden = shifted_hidden[stage_mask]
                        # Full already has a fused CoT head gradient. Detach the
                        # head slice there to cap peak memory; R2 updates both
                        # hidden states and SID head rows through this objective.
                        head = model_self.lm_head.weight.index_select(0, all_ids[stage])
                        if is_full_batch:
                            head = head.detach()
                        stage_logits[stage] = torch.nn.functional.log_softmax(
                            torch.nn.functional.linear(stage_hidden, head).float().squeeze(0), dim=0
                        )
                        truth[stage] = int(indexed_labels[stage_mask].item())

                    pos_a = sorted({a for a, _b, _c in positives})
                    pos_b = sorted({b for a, b, _c in positives if a == truth["a"]})
                    pos_c = sorted({c for a, b, c in positives if a == truth["a"] and b == truth["b"]})
                    if not pos_b or not pos_c:
                        raise ValueError("teacher-forced positive hierarchy is empty")
                    set_losses = {
                        "a_set": multi_positive_set_loss(stage_logits["a"], pos_a),
                        "ab_set": multi_positive_set_loss(stage_logits["b"], pos_b),
                        "sid_set": multi_positive_set_loss(stage_logits["c"], pos_c),
                    }
                    anchor_nll = {
                        f"anchor_nll_{stage}": -stage_logits[stage][truth[stage]]
                        for stage in ("a", "b", "c")
                    }
                    anchor_nll["anchor_nll_sid_sum"] = sum(anchor_nll.values())

                    fdr_cfg = joint_cfg["first_divergence_rank"]
                    rng = random.Random(seed + group_index)
                    fdr_pairs = first_divergence_pairs(
                        (truth["a"], truth["b"], truth["c"]), positives,
                        (domain_inventory if group_index in train_group_ids else validation_domain_inventory)[group_domains[group_index]], rng,
                        {key: int(value) for key, value in fdr_cfg["pair_quota"].items()},
                        group_domains[group_index],
                    )
                    fdr_loss, fdr_metrics = first_divergence_rank_loss(
                        stage_logits, fdr_pairs, fdr_cfg["margin"], fdr_cfg["temperature"], fdr_cfg["level_weight"]
                    )
                    parts = []
                    active_sum = 0.0
                    for name, value in set_losses.items():
                        cfg = joint_cfg[name]
                        if cfg["enabled"]:
                            parts.append(float(cfg["weight"]) * value); active_sum += float(cfg["weight"])
                    for name, value in (("first_divergence_rank", fdr_loss),):
                        cfg = joint_cfg[name]
                        if cfg["enabled"]:
                            parts.append(float(cfg["weight"]) * value); active_sum += float(cfg["weight"])
                    rec_loss = torch.stack(parts).sum() / active_sum
                    instance_losses.append(rec_loss)
                    instance_modes.append(mode)
                    instance_domains.append(group_domains[group_index])
                    instance_weights.append((0.5 if mode == "cot" else 1.0) if is_full_batch else 1.0 / len(positives))
                    item_metrics = {**set_losses, **anchor_nll, "fd_total": fdr_loss, "rec": rec_loss, **fdr_metrics}
                    for name, value in item_metrics.items():
                        metric_lists[name].append(value.detach())
                        if mode in {"cot", "uncot"}:
                            metric_lists[f"{mode}_{name}"].append(value.detach())
                            domain = group_domains[group_index]
                            domain_name = {"video": "video", "prod": "prod", "ad": "ad", "living": "living"}[domain]
                            domain_key = f"{domain_name}_{mode}_{name}"
                            if domain_key in DOMAIN_MODE_METRICS:
                                metric_lists[domain_key].append(value.detach())
                weights = torch.tensor(instance_weights, device=shifted_hidden.device)
                if is_full_batch:
                    sample_count = max(1, len(instance_losses))
                    rec_batch_loss = (torch.stack(instance_losses) * weights).sum() / sample_count
                    cot_fraction = instance_modes.count("cot") / sample_count
                    loss = 0.5 * cot_fraction * ce_loss + rec_batch_loss
                else:
                    rec_batch_loss = (torch.stack(instance_losses) * weights).sum() / weights.sum().clamp_min(1e-8)
                    loss = rec_batch_loss
                for name, values in metric_lists.items():
                    joint_metrics[name] = torch.stack(values).mean()
            detached_losses = token_losses.detach()
            objective_num = (detached_losses * valid_objective).sum()
            for group_id in torch.unique(valid_groups).tolist():
                group_mask = valid_groups == int(group_id)
                component_num[int(group_id)] = (
                    detached_losses[group_mask] * valid_metric[group_mask]
                ).sum()
                component_den[int(group_id)] = valid_metric[group_mask].sum()
            result = CausalLMOutputWithPast(
                loss=loss,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            domain_mode_tokens = {
                f"{domain}_{mode}": 0
                for domain in ("video", "prod", "ad", "living") for mode in ("cot", "uncot")
            }
            for group_index in torch.unique(shifted_sample_groups[shifted_sample_groups > 0]).tolist():
                group_mask = shifted_sample_groups == int(group_index)
                for mode in ("cot", "uncot"):
                    count = int((group_mask & (shifted_modes == MODE_IDS[mode])).sum().item())
                    domain_mode_tokens[f"{group_domains[int(group_index)]}_{mode}"] += count
            global LAST_CURRICULUM_METRICS
            LAST_CURRICULUM_METRICS = {
                "objective_num": objective_num.detach(),
                "objective_den": objective_den.detach(),
                "component_num": component_num,
                "component_den": component_den,
                "groups": sorted(int(value) for value in torch.unique(groups[groups > 0]).tolist()),
                "joint": joint_metrics,
                "token_count": int((input_ids != int(model_self.config.pad_token_id or 0)).sum().item()),
                "mode_counts": {name: instance_modes.count(name) for name in ("cot", "uncot")} if joint_cfg.get("enabled") and bool(rec_groups.any()) else {},
                "mode_tokens": {
                    name: int((shifted_modes == mode_id).sum().item())
                    for name, mode_id in (("cot", MODE_IDS["cot"]), ("uncot", MODE_IDS["uncot"]))
                },
                "domain_mode_counts": {
                    f"{domain}_{mode}": sum(1 for item_domain, item_mode in zip(instance_domains, instance_modes) if item_domain == domain and item_mode == mode)
                    for domain in ("video", "prod", "ad", "living") for mode in ("cot", "uncot")
                } if joint_cfg.get("enabled") and bool(rec_groups.any()) else {},
                "domain_mode_tokens": domain_mode_tokens,
                "entropy_buckets": sorted({
                    "k1" if len(group_catalog[int(value)]) == 1 else "k2" if len(group_catalog[int(value)]) <= 4 else "k3"
                    for value in torch.unique(shifted_rec_groups[shifted_rec_groups > 0]).tolist()
                }),
            }
            return result

        base.forward = MethodType(forward, base)
        base._task_v2_candidate_weighted_liger = True
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps({"event": "task_v2_candidate_weighted_liger_enabled", "group_ids": TASK_IDS}), flush=True)
        return model

    workflow.load_model = load_weighted_model

    trainer_class = trainer_module.CustomSeq2SeqTrainer
    original_init = trainer_class.__init__
    original_log = trainer_class.log
    original_evaluate = trainer_class.evaluate
    original_save = trainer_class._save
    original_optimizer_factory = trainer_class.get_optimizer_cls_and_kwargs

    @staticmethod
    def memory_bounded_optimizer_factory(args: Any, model: Any = None) -> tuple[Any, dict[str, Any]]:
        optimizer_cls, optimizer_kwargs = original_optimizer_factory(args, model)
        if optimizer_cls is torch.optim.AdamW and int(os.environ.get("WORLD_SIZE", "1")) <= 2:
            # Two-way FSDP leaves only a few MiB free after Adam states are
            # materialized. The foreach implementation allocates an extra
            # list-sized sqrt buffer; scalar tensor updates implement the same
            # AdamW equations without that transient peak.
            optimizer_kwargs["foreach"] = False
        return optimizer_cls, optimizer_kwargs

    def trainer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Transformers 5.3 infers support for `num_items_in_batch` from this
        # override's **kwargs, then multiplies a custom already-normalized loss
        # by world_size. This objective is independently normalized per packed
        # task batch, so explicitly disable that token-count rescaling.
        self.model_accepts_loss_kwargs = False
        self._task_train_accum = defaultdict(float)
        self._task_eval_accum = defaultdict(float)
        self._task_stage_counts = defaultdict(float)
        self._task_stage = 1
        self._mode_loss_history = {"cot": [], "uncot": []}

    # Metric accumulation is rank-local because each rank receives a different
    # curriculum task.  A collective must nevertheless use the same tensor
    # shape and key order on every rank.  Building keys from the local dict can
    # desynchronise NCCL (one rank may enter the next FSDP all-gather while
    # another is still reducing metrics), so keep one exhaustive schema.
    reduction_keys = metric_reduction_keys()

    def reduce_accum(self: Any, accum: dict[str, float]) -> dict[str, float]:
        unknown = set(accum).difference(reduction_keys)
        if unknown:
            raise RuntimeError(f"metric reduction schema is missing keys: {sorted(unknown)}")
        keys = reduction_keys
        tensor = torch.tensor([accum[key] for key in keys], device=self.accelerator.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}

    def record(self: Any, metrics: dict[str, Any], evaluating: bool) -> None:
        accum = self._task_eval_accum if evaluating else self._task_train_accum
        groups = [int(value) for value in metrics["groups"]]
        task = "base"
        if TASK_IDS["full_cot"] in groups or TASK_IDS["full_sid"] in groups:
            task = "full"
        elif TASK_IDS["r1"] in groups:
            task = "r1"
        elif TASK_IDS["r2"] in groups:
            task = "r2"
        accum[f"objective_num/{task}"] += float(metrics["objective_num"].item())
        accum[f"objective_den/{task}"] += float(metrics["objective_den"].item())
        for group_id, value in metrics["component_num"].items():
            accum[f"component_num/{int(group_id)}"] += float(value.item())
        for group_id, value in metrics["component_den"].items():
            accum[f"component_den/{int(group_id)}"] += float(value.item())
        for name, value in metrics.get("joint", {}).items():
            accum[f"joint_num/{task}/{name}"] += float(value.item())
            accum[f"joint_den/{task}/{name}"] += 1.0
        for mode, value in metrics.get("mode_counts", {}).items():
            accum[f"mode_count/{mode}"] += float(value)
        for mode, value in metrics.get("mode_tokens", {}).items():
            accum[f"mode_tokens/{mode}"] += float(value)
        for key, value in metrics.get("domain_mode_counts", {}).items():
            domain, mode = key.rsplit("_", 1)
            accum[f"domain_mode_count/{domain}/{mode}"] += float(value)
        for key, value in metrics.get("domain_mode_tokens", {}).items():
            domain, mode = key.rsplit("_", 1)
            accum[f"domain_mode_tokens/{domain}/{mode}"] += float(value)
        accum[f"packs/{task}"] += 1.0
        accum[f"tokens/{task}"] += float(metrics.get("token_count", 0))
        for bucket in metrics.get("entropy_buckets", []):
            accum[f"tokens/{task}/{bucket}"] += float(metrics.get("token_count", 0))
        if not evaluating:
            current_stage = stage_for_step(self)
            if current_stage != self._task_stage:
                self._task_stage_counts.clear()
                self._task_stage = current_stage
            self._task_stage_counts[task] += 1.0
            self._task_stage_counts[f"tokens/{task}"] += float(metrics.get("token_count", 0))
            for bucket in metrics.get("entropy_buckets", []):
                self._task_stage_counts[f"tokens/{task}/{bucket}"] += float(metrics.get("token_count", 0))
            for mode, value in metrics.get("mode_counts", {}).items():
                self._task_stage_counts[f"mode_count/{mode}"] += float(value)
            for mode, value in metrics.get("mode_tokens", {}).items():
                self._task_stage_counts[f"mode_tokens/{mode}"] += float(value)
            for key, value in metrics.get("domain_mode_counts", {}).items():
                domain, mode = key.rsplit("_", 1)
                self._task_stage_counts[f"domain_mode_count/{domain}/{mode}"] += float(value)
            for key, value in metrics.get("domain_mode_tokens", {}).items():
                domain, mode = key.rsplit("_", 1)
                self._task_stage_counts[f"domain_mode_tokens/{domain}/{mode}"] += float(value)

    def compute_loss(self: Any, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return_outputs = bool(kwargs.get("return_outputs", False))
        outputs = model(**inputs)
        global LAST_CURRICULUM_METRICS
        metrics = LAST_CURRICULUM_METRICS
        if outputs.loss is None or metrics is None:
            raise RuntimeError("weighted model did not return task metrics")
        record(self, metrics, evaluating=not model.training)
        LAST_CURRICULUM_METRICS = None
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def stage_for_step(self: Any) -> int:
        progress = self.state.global_step / max(1, self.state.max_steps)
        return 1 if progress < 0.2 else 2 if progress < 0.4 else 3 if progress < 0.8 else 4

    def loss_fields(values: dict[str, float], prefix: str) -> dict[str, float]:
        result: dict[str, float] = {}
        group_names = {
            TASK_IDS["base"]: "loss_base_token",
            TASK_IDS["full_cot"]: "loss_full_cot",
            TASK_IDS["r1"]: "loss_r1_cot",
        }
        for group_id, name in group_names.items():
            numerator = values.get(f"component_num/{group_id}", 0.0)
            denominator = values.get(f"component_den/{group_id}", 0.0)
            if denominator > 0:
                result[f"{prefix}/{name}"] = numerator / denominator
        for name in JOINT_METRIC_NAMES:
            numerator = sum(values.get(f"joint_num/{task}/{name}", 0.0) for task in ("full", "r2"))
            denominator = sum(values.get(f"joint_den/{task}/{name}", 0.0) for task in ("full", "r2"))
            if denominator > 0:
                metric_name = f"loss_{name}" if name in {"sid_set", "a_set", "ab_set", "fd_total", "fd_loss_a", "fd_loss_b", "fd_loss_c"} else name
                result[f"{prefix}/{metric_name}"] = numerator / denominator
        for task in ("full", "r2"):
            for stage in ("a", "b", "c"):
                key = f"anchor_nll_{stage}"
                denominator = values.get(f"joint_den/{task}/{key}", 0.0)
                if denominator > 0:
                    result[f"{prefix}/{task}_anchor_nll_{stage}"] = values[f"joint_num/{task}/{key}"] / denominator
            key = "anchor_nll_sid_sum"
            denominator = values.get(f"joint_den/{task}/{key}", 0.0)
            if denominator > 0:
                result[f"{prefix}/{task}_anchor_nll_sid_sum"] = values[f"joint_num/{task}/{key}"] / denominator
        for mode in ("cot", "uncot"):
            for key, suffix in (("rec", "rec_loss"), ("anchor_nll_sid_sum", "anchor_nll_sid_sum"), ("fd_total", "fd_total")):
                name = f"{mode}_{key}"
                denominator = values.get(f"joint_den/full/{name}", 0.0)
                if denominator > 0:
                    result[f"{prefix}/full_{mode}_{suffix}"] = values[f"joint_num/full/{name}"] / denominator
        for domain in ("video", "prod", "ad", "living"):
            for mode in ("cot", "uncot"):
                for key in ("rec", "anchor_nll_sid_sum", "fd_total"):
                    name = f"{domain}_{mode}_{key}"
                    denominator = values.get(f"joint_den/full/{name}", 0.0)
                    if denominator > 0:
                        result[f"{prefix}/{name}"] = values[f"joint_num/full/{name}"] / denominator
        for task in ("full", "r2"):
            for name in ("rec", "sid_set", "a_set", "ab_set", "fd_total", "fd_loss_a", "fd_loss_b", "fd_loss_c"):
                denominator = values.get(f"joint_den/{task}/{name}", 0.0)
                if denominator > 0:
                    result[f"{prefix}/{task}_loss_{name}"] = values[f"joint_num/{task}/{name}"] / denominator
        if f"{prefix}/full_loss_rec" in result and f"{prefix}/loss_full_cot" in result:
            result[f"{prefix}/loss_full_rec"] = result[f"{prefix}/full_loss_rec"]
            result[f"{prefix}/loss_full"] = .5 * result[f"{prefix}/loss_full_cot"] + .5 * result[f"{prefix}/full_loss_rec"]
        if f"{prefix}/r2_loss_rec" in result:
            result[f"{prefix}/loss_r2"] = result[f"{prefix}/r2_loss_rec"]
            result[f"{prefix}/loss_r2_group_weighted"] = result[f"{prefix}/r2_loss_rec"]
        if V42_CONFIG.get("variant") in {"interest_only", "video_interest_only"}:
            if f"{prefix}/loss_full_cot" in result:
                result[f"{prefix}/loss_full_interest"] = result[f"{prefix}/loss_full_cot"]
            if f"{prefix}/loss_r1_cot" in result:
                result[f"{prefix}/loss_r1_interest"] = result[f"{prefix}/loss_r1_cot"]
        return result

    def log(self: Any, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs and self._task_train_accum:
            values = reduce_accum(self, self._task_train_accum)
            logs.update(loss_fields(values, "train"))
            self._task_train_accum.clear()
            stage = stage_for_step(self)
            counts = reduce_accum(self, self._task_stage_counts)
            tasks = ("base", "full", "r1", "r2")
            all_total = sum(counts.get(task, 0.0) for task in tasks)
            rec_total = sum(counts.get(task, 0.0) for task in ("full", "r1", "r2"))
            for task in tasks:
                logs[f"train/sample_count_{task}"] = counts.get(task, 0.0)
                logs[f"train/sample_ratio_{task}"] = counts.get(task, 0.0) / all_total if all_total else 0.0
                if task != "base":
                    logs[f"train/recommendation_sample_ratio_{task}"] = counts.get(task, 0.0) / rec_total if rec_total else 0.0
            token_total = sum(counts.get(f"tokens/{task}", 0.0) for task in tasks)
            rec_token_total = sum(counts.get(f"tokens/{task}", 0.0) for task in ("full", "r1", "r2"))
            for task in tasks:
                task_tokens = counts.get(f"tokens/{task}", 0.0)
                logs[f"train/tokens_{task}"] = task_tokens
                logs[f"train/token_ratio_{task}"] = task_tokens / token_total if token_total else 0.0
                if task != "base":
                    logs[f"train/recommendation_token_ratio_{task}"] = task_tokens / rec_token_total if rec_token_total else 0.0
                if task in {"full", "r2"}:
                    for bucket in ("k1", "k2", "k3"):
                        value = counts.get(f"tokens/{task}/{bucket}", 0.0)
                        logs[f"train/{task}_tokens_{bucket}"] = value
                        logs[f"train/{task}_token_ratio_{bucket}"] = value / task_tokens if task_tokens else 0.0
            logs["train/curriculum_stage"] = float(stage)
            mode_rows = {mode: counts.get(f"mode_count/{mode}", 0.0) for mode in ("cot", "uncot")}
            mode_tokens = {mode: counts.get(f"mode_tokens/{mode}", 0.0) for mode in ("cot", "uncot")}
            mode_row_total = sum(mode_rows.values())
            mode_token_total = sum(mode_tokens.values())
            for mode in ("cot", "uncot"):
                logs[f"train/rows_full_{mode}"] = mode_rows[mode]
                logs[f"train/tokens_full_{mode}"] = mode_tokens[mode]
            logs["train/sample_cot_ratio"] = mode_rows["cot"] / mode_row_total if mode_row_total else 0.0
            logs["train/token_cot_ratio"] = mode_tokens["cot"] / mode_token_total if mode_token_total else 0.0
            for domain in ("video", "prod", "ad", "living"):
                domain_rows = {mode: counts.get(f"domain_mode_count/{domain}/{mode}", 0.0) for mode in ("cot", "uncot")}
                domain_tokens = {mode: counts.get(f"domain_mode_tokens/{domain}/{mode}", 0.0) for mode in ("cot", "uncot")}
                row_total = sum(domain_rows.values())
                token_sum = sum(domain_tokens.values())
                logs[f"train/{domain}_sample_cot_ratio"] = domain_rows["cot"] / row_total if row_total else 0.0
                logs[f"train/{domain}_token_cot_ratio"] = domain_tokens["cot"] / token_sum if token_sum else 0.0
            if CURRICULUM_PLAN_REPORT.get("stages"):
                planned = CURRICULUM_PLAN_REPORT["stages"][stage - 1]
                for name in ("requested_r2_k1_tokens", "feasible_r2_k1_tokens", "actual_r2_k1_tokens"):
                    logs[f"train/{name}"] = float(planned.get(name, 0))
                for name in ("r2_k1_unique_groups_seen", "r2_k1_group_exposure_mean", "r2_k1_group_exposure_max", "r2_k1_repack_round"):
                    logs[f"train/{name}"] = float(CURRICULUM_PLAN_REPORT.get(name, 0))
            for mode in ("cot", "uncot"):
                key = f"train/full_{mode}_rec_loss"
                if key not in logs:
                    continue
                history = self._mode_loss_history[mode]
                history.append(float(logs[key]))
                for window in (20, 50):
                    values_window = history[-window:]
                    ema = values_window[0]
                    alpha = 2.0 / (window + 1.0)
                    for value in values_window[1:]:
                        ema = alpha * value + (1.0 - alpha) * ema
                    mean = sum(values_window) / len(values_window)
                    std = math.sqrt(sum((value - mean) ** 2 for value in values_window) / len(values_window))
                    logs[f"train/{mode}_rec_loss_ema_{window}"] = ema
                    logs[f"train/{mode}_rec_loss_std_{window}"] = std
                    logs[f"train/{mode}_rec_loss_cv_{window}"] = std / (mean + 1e-8)
            for name, value in V42_CONFIG.get("static_metrics", {}).items():
                logs[f"train/{name}"] = float(value)
        original_log(self, logs, *args, **kwargs)

    def evaluate(self: Any, *args: Any, **kwargs: Any) -> dict[str, float]:
        self._task_eval_accum.clear()
        metrics = original_evaluate(self, *args, **kwargs)
        values = reduce_accum(self, self._task_eval_accum)
        extra = loss_fields(values, "val")
        if extra:
            metrics.update(extra)
            self.log(extra)
        self._task_eval_accum.clear()
        return metrics

    def save_bf16(self: Any, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
        """Persist the joint-loss experiment as BF16, never an FP32 full state."""
        if joint_cfg.get("enabled") and state_dict is not None:
            state_dict = {
                name: tensor.to(dtype=torch.bfloat16) if torch.is_tensor(tensor) and tensor.is_floating_point() else tensor
                for name, tensor in state_dict.items()
            }
            if self.args.should_save:
                print(json.dumps({"event": "checkpoint_cast", "dtype": "bfloat16"}), flush=True)
        original_save(self, output_dir=output_dir, state_dict=state_dict)

    trainer_class.__init__ = trainer_init
    trainer_class.compute_loss = compute_loss
    trainer_class.log = log
    trainer_class.evaluate = evaluate
    trainer_class._save = save_bf16
    trainer_class.get_optimizer_cls_and_kwargs = memory_bounded_optimizer_factory


def _allow_trusted_local_resume_state(checkpoint: str | None) -> None:
    """Allow Transformers 5.3 to restore this run's own Torch-2.5 state.

    The security gate rejects every pickle-based optimizer/RNG state on Torch
    <2.6.  This experiment resumes only the locally created, launcher-checked
    checkpoint-405 files; model weights remain safetensors.  Keep the override
    narrowly scoped to this process, after the launcher has verified the exact
    required local files, rather than weakening any global environment policy.
    """
    import transformers.trainer as trainer_module
    import transformers.utils.import_utils as import_utils
    if hasattr(import_utils, "check_torch_load_is_safe"):
        def trusted_checkpoint_only() -> None:
            return None
        import_utils.check_torch_load_is_safe = trusted_checkpoint_only
        trainer_module.check_torch_load_is_safe = trusted_checkpoint_only
    if checkpoint is None:
        return
    # Torch 2.5's weights-only unpickler also rejects the NumPy RNG object in
    # the otherwise valid checkpoint.  Change only loads whose file is in the
    # exact launcher-validated resume directory; all other torch.load calls
    # retain their original safety behavior.
    import torch as _torch
    checkpoint_dir = Path(checkpoint).resolve()
    original_torch_load = _torch.load

    def trusted_torch_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            file_path = Path(os.fspath(file)).resolve()
        except (TypeError, ValueError, OSError):
            file_path = None
        if file_path is not None and file_path.parent == checkpoint_dir and kwargs.get("weights_only") is True:
            kwargs["weights_only"] = False
        return original_torch_load(file, *args, **kwargs)

    _torch.load = trusted_torch_load


def _jsonl_callback(output_dir: Path, smoke: bool = False) -> Any:
    from transformers import TrainerCallback

    class JsonlCallback(TrainerCallback):
        def __init__(self) -> None:
            self.save_milestones: set[int] = set()
            self.eval_milestones: set[int] = set()
            self.stop_step: int | None = None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if not smoke:
                self.save_milestones = {max(1, round(state.max_steps * value)) for value in (.2, .4, .8)}
                self.eval_milestones = self.save_milestones | {max(1, int(state.max_steps))}
                if STOP_AT_PROGRESS is not None:
                    self.stop_step = max(1, round(state.max_steps * STOP_AT_PROGRESS))
                    self.save_milestones.add(self.stop_step)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step in self.save_milestones:
                control.should_save = True
            if state.global_step in self.eval_milestones:
                control.should_evaluate = True
            if self.stop_step is not None and state.global_step >= self.stop_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            del args, control, kwargs
            if state.is_world_process_zero and logs:
                row = {"step": int(state.global_step), **logs}
                with (output_dir / "trainer_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if smoke and state.global_step >= 2 and logs and "loss" in logs:
                raise SmokeComplete("two finite logged optimizer steps completed")

    return JsonlCallback()


def prepare_pool_caches(
    parser_raw: dict[str, Any], pool_cache_root: Path, base_pool_cache_root: Path
) -> None:
    """Build every packed pool once before distributed workers are launched."""
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    prepare_raw = dict(parser_raw)
    prepare_raw["do_train"] = False
    prepare_raw["do_eval"] = False
    prepare_raw["bf16"] = False
    prepare_raw["use_cpu"] = True
    model_args, data_args, training_args, _finetuning_args, _ = _parse_train_args(prepare_raw)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    report: dict[str, int] = {}
    for pool_name, dataset_names in active_pool_datasets().items():
        pool_args = copy.deepcopy(data_args)
        pool_args.dataset = list(dataset_names)
        pool_args.eval_dataset = None
        pool_args.val_size = 0.0
        cache_root = base_pool_cache_root if pool_name in {"base", "val_base"} else pool_cache_root
        pool_args.tokenized_path = str(cache_root / pool_name)
        module = get_dataset(
            template,
            model_args,
            pool_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )
        report[pool_name] = len(module["train_dataset"])
    print(json.dumps({"event": "all_curriculum_pool_caches_ready", "pack_counts": report}, ensure_ascii=False))


def main() -> None:
    global REC_JOINT_CONFIG, LEGAL_SID_UNIVERSE, STOP_AT_PROGRESS, DOMAIN_CONDITIONED_FULL_C2, V42_CONFIG
    cli = parse_args()
    raw: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)  # type: ignore[assignment]
    v42 = raw.pop("v42_experiment", None)
    if not isinstance(v42, dict) or v42.get("variant") not in {
        "interest_only", "video_interest_only", "cot_uncot_domainratio"
    }:
        raise ValueError("v42_experiment.variant must select one V4.2 experiment")
    V42_CONFIG = dict(v42)
    domain_c2 = raw.pop("domain_conditioned_full_c2", None)
    if domain_c2 is not None:
        DOMAIN_CONDITIONED_FULL_C2 = dict(domain_c2)
        if DOMAIN_CONDITIONED_FULL_C2.get("stage") != "stage_c2":
            raise ValueError("domain_conditioned_full_c2 is restricted to stage_c2")
    joint = raw.pop("rec_joint_loss", None)
    if joint:
        REC_JOINT_CONFIG = dict(joint)
        REC_JOINT_CONFIG.setdefault("enabled", False)
        fdr = REC_JOINT_CONFIG.get("first_divergence_rank", {})
        if fdr.get("candidate_source") == "static_legal_sid_universe":
            cache_root = Path(str(fdr["sid_index_cache"]))
            LEGAL_SID_UNIVERSE = LegalSidUniverse(cache_root)
        if REC_JOINT_CONFIG.get("enabled"):
            required = {"sid_set", "a_set", "ab_set", "first_divergence_rank"}
            if not required.issubset(REC_JOINT_CONFIG):
                raise ValueError(f"rec_joint_loss is missing modules: {sorted(required - set(REC_JOINT_CONFIG))}")
            active = [value for key, value in REC_JOINT_CONFIG.items() if key in required and value.get("enabled")]
            if not active or sum(float(value["weight"]) for value in active) <= 0:
                raise ValueError("at least one positive-weight recommendation loss must be active")
    catalog_path = str(raw.pop("rec_group_catalog_path"))
    pool_cache_root = Path(str(raw.pop("curriculum_tokenized_pool_root")))
    base_pool_cache_root = Path(str(raw.pop("base_tokenized_pool_root")))
    raw.pop("curriculum_schedule", None)
    raw.pop("curriculum_scope", None)
    stop_at = raw.pop("stop_at_progress", None)
    if stop_at is not None:
        STOP_AT_PROGRESS = float(stop_at)
        if not 0.0 < STOP_AT_PROGRESS <= 1.0:
            raise ValueError("stop_at_progress must be in (0, 1]")
    if cli.max_steps is not None:
        raw["curriculum_max_steps_override"] = cli.max_steps
    if cli.output_dir is not None:
        raw["output_dir"] = str(cli.output_dir)
    if cli.reuse_cache:
        raw["overwrite_cache"] = False
    if int(raw.get("cutoff_len", 0)) != 32768:
        raise ValueError("TASK V2 is locked to cutoff_len=32768")
    if not raw.get("packing") or not raw.get("neat_packing"):
        raise ValueError("TASK V2 requires packing and neat_packing")
    if raw.get("finetuning_type") != "full":
        raise ValueError("TASK V2 experiment must use full fine-tuning")
    if not raw.get("enable_liger_kernel") or raw.get("flash_attn") != "fa2":
        raise ValueError("TASK V2 requires Liger kernel and FlashAttention 2")

    custom_max = int(raw.pop("curriculum_max_steps_override", 0))
    raw["curriculum_max_steps_override"] = custom_max
    parser_raw = dict(raw)
    parser_raw.pop("curriculum_max_steps_override", None)
    raw["rec_group_catalog_path"] = catalog_path
    if cli.validate_only:
        from llamafactory.hparams.parser import _parse_train_args

        # Parse and validate all dataclass fields without invoking the final
        # LLaMA-Factory guard that requires an already-running process group.
        model_args, data_args, training_args, finetuning_args, _ = _parse_train_args(parser_raw)
        print(
            json.dumps(
                {
                    "event": "configuration_validated",
                    "finetuning_type": raw["finetuning_type"],
                    "configured_cutoff_len": int(raw["cutoff_len"]),
                    "internal_packing_cutoff_len": data_args.cutoff_len,
                    "flash_attention": model_args.flash_attn,
                    "liger": model_args.enable_liger_kernel,
                    "transformers": __import__("transformers").__version__,
                    "validation": "external deterministic 5% files",
                    "curriculum": ["50/35/15", "50/15/35", "80/10/10", "75/5/20"],
                    "rec_joint_loss": REC_JOINT_CONFIG,
                },
                ensure_ascii=False,
            )
        )
        return

    if cli.prepare_data_only:
        _patch_dataset_metadata()
        _patch_packed_processor()
        prepare_pool_caches(parser_raw, pool_cache_root, base_pool_cache_root)
        return

    raw.pop("curriculum_max_steps_override", None)
    if custom_max > 0:
        raw["curriculum_max_steps_override"] = custom_max
    _patch_dataset_metadata()
    _patch_packed_processor()
    _patch_collator_trainer_and_model(raw, pool_cache_root, base_pool_cache_root)
    _allow_trusted_local_resume_state(parser_raw.get("resume_from_checkpoint"))
    from llamafactory.train.tuner import run_exp

    output_dir = Path(str(raw["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_exp(args=parser_raw, callbacks=[_jsonl_callback(output_dir, smoke=cli.smoke)])
    except SmokeComplete as exc:
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps({"event": "smoke_complete", "reason": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
