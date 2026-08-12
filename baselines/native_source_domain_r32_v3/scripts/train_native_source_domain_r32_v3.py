#!/usr/bin/env python3
"""Run LLaMA-Factory SFT with source-aware and material-domain-aware loss weights.

The patches stay in this launcher and do not modify the installed LLaMA-Factory.
Source metadata is retained through conversion, converted to a token-aligned weight
vector before packing, and removed before the model forward pass. Material caption
samples additionally use sample-normalized CE and a four-domain objective weight.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from rec_pu.recommendation_pu_phase2 import PackedSegment, locate_packed_rec_pu_targets
from rec_pu.recommendation_pu_loss import rec_pu_batched_position_loss
from rec_pu.core_recommendation_metrics import candidate_window_active, pack_effective_share_sum_count
from rec_pu.alpha_recommendation_monitor import (
    ALL_METRIC_NAMES as ALPHA_MONITOR_METRIC_NAMES,
    AlphaMonitorConfig,
    collect_alpha_recommendation_monitor,
)
from rec_pu.alpha_validation_monitor import AlphaValidationConfig, AlphaValidationRunner, VALIDATION_METRIC_NAMES
from rec_pu.sid8_rec_pu_integration import (
    RecPUConfig,
    build_sid_component_vocab,
    coerce_packed_target,
    compute_native_sid8_loss,
    probe_rec_pu_position,
    serialise_packed_target,
)
from pack_ratio_sampler import PackRatioConfig, PackRatioSampler, derive_pack_task_id, parse_target_ratios

from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.converter import AlpacaDatasetConverter
from llamafactory.data.processor.processor_utils import greedy_knapsack
from llamafactory.data.processor.supervised import MAX_SU_SEQ_IDX, PackingParams, PackedSupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.tuner import run_exp
from llamafactory.model.model_utils import checkpointing as checkpointing_utils


SOURCE_ITEM_WEIGHTS = {
    "sid_bucket_reverse": 2.0,
    "material_sample": 2.0,
    # Preserve the pre-correction effective weights after fixing source labels.
    "understand_user": 3.0,
    "recommend": 2.0,
}
CANONICAL_SOURCE = "sid_bucket_canonical_no_think"
CANONICAL_TEXT_WEIGHT = 4.0
GLOBAL_ITEM_WEIGHT = float(os.getenv("GLOBAL_ITEM_WEIGHT", "0"))
ITEM_TOKEN_PATTERN = re.compile(r"(?:<s_[abc]_\d+>|<\|(?:ad|video|prod|living|search)_begin\|>)")
MATERIAL_DOMAIN_PATTERN = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
TASK_NAMES = ("material", "recommendation", "user_action", "user_chain")
TASK_ID_BY_NAME = {name: index for index, name in enumerate(TASK_NAMES)}
_REFERENCE_DOMAIN_WEIGHTS = {
    # alpha / p_d, normalized to keep the mean material-sample scale near 1.0
    # for train_bucket_grouped_item8: video=30092, prod=29180, ad=22768, living=17960.
    "video": 1.1791795483099141,
    "prod": 0.7738397930531297,
    "ad": 1.133452723686895,
    "living": 0.8980530210503628,
}


def _load_material_domain_weights() -> dict[str, float]:
    manifest_path = os.getenv("MATERIAL_DOMAIN_MANIFEST")
    if not manifest_path:
        return _REFERENCE_DOMAIN_WEIGHTS
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    weights = manifest.get("material_domain_weights", {})
    required = set(_REFERENCE_DOMAIN_WEIGHTS)
    if set(weights) != required or any(float(weights[key]) <= 0.0 for key in required):
        raise ValueError(f"Invalid material_domain_weights in {manifest_path}")
    return {key: float(weights[key]) for key in required}


MATERIAL_DOMAIN_WEIGHTS = _load_material_domain_weights()


def _load_rec_pu_config() -> RecPUConfig:
    """Read optional REC-PU YAML fields without changing existing NSD configs."""

    raw: dict[str, object] = {}
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.suffix in {".yaml", ".yml"} and config_path.is_file():
            import yaml

            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("REC-PU launch config must be a YAML mapping.")
            raw = loaded
    enabled = raw.get(
        "rec_pu_enabled", os.getenv("REC_PU_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    scale = raw.get("rec_pu_unlabeled_sid_grad_scale", os.getenv("REC_PU_UNLABELED_SID_GRAD_SCALE", "0.05"))
    return RecPUConfig(rec_pu_enabled=bool(enabled), rec_pu_unlabeled_sid_grad_scale=float(scale))


def _load_pack_ratio_config() -> PackRatioConfig:
    """Read strict PackRatio settings without affecting the default sampler path."""

    raw: dict[str, object] = {}
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.suffix in {".yaml", ".yml"} and config_path.is_file():
            import yaml

            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("PackRatio launch config must be a YAML mapping.")
            raw = loaded

    enabled = raw.get("multitask_pack_ratio_enabled", False)
    targets = raw.get(
        "multitask_pack_ratio_targets",
        "material=0.25,recommendation=0.25,user_action=0.25,user_chain=0.25",
    )
    if not isinstance(enabled, bool):
        raise ValueError("`multitask_pack_ratio_enabled` must be boolean.")
    if not isinstance(targets, str):
        raise ValueError("`multitask_pack_ratio_targets` must be a string.")
    # OFF must be exact native behavior. In particular, an ignored targets value
    # must not prevent the original RandomSampler path from starting.
    if not enabled:
        return PackRatioConfig(enabled=False, target_ratios=(0.25, 0.25, 0.25, 0.25))
    return PackRatioConfig(enabled=True, target_ratios=parse_target_ratios(targets))


def _load_rec_candidate_metric_config() -> tuple[bool, int]:
    """Read official low-frequency TF candidate-monitoring fields."""

    raw: dict[str, object] = {}
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.suffix in {".yaml", ".yml"} and config_path.is_file():
            import yaml

            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    enabled = raw.get("rec_candidate_metrics_enabled", False)
    interval = raw.get("rec_candidate_metrics_interval", 50)
    if not isinstance(enabled, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("rec_candidate_metrics_enabled must be bool and interval must be >= 1.")
    return enabled, interval


def _load_alpha_monitor_config() -> AlphaMonitorConfig:
    """Read the monitor-only ablation flags without changing legacy runs."""

    raw: dict[str, object] = {}
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.suffix in {".yaml", ".yml"} and config_path.is_file():
            import yaml

            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    enabled = raw.get("alpha_monitor_enabled", False)
    tf_enabled = raw.get("alpha_train_tf_enabled", True)
    interval = raw.get("alpha_train_tf_interval", 50)
    if not isinstance(enabled, bool) or not isinstance(tf_enabled, bool) or not isinstance(interval, int):
        raise ValueError("alpha monitor fields must be bool/bool/int.")
    return AlphaMonitorConfig(enabled=enabled, train_tf_enabled=tf_enabled, train_tf_interval=interval)


def _load_alpha_validation_config() -> AlphaValidationConfig:
    """Read optional sidecar validation fields; legacy launches remain fully off."""

    raw: dict[str, object] = {}
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if config_path.suffix in {".yaml", ".yml"} and config_path.is_file():
            import yaml

            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    values = {
        "enabled": raw.get("alpha_validation_enabled", False),
        "probe_enabled": raw.get("alpha_dev_probe_enabled", False),
        "probe_interval": raw.get("alpha_dev_probe_interval", 100),
        "full_dev_enabled": raw.get("alpha_full_dev_enabled", False),
        "full_dev_at_epoch_end": raw.get("alpha_full_dev_at_epoch_end", False),
        "split_dir": raw.get("alpha_validation_split_dir", "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1"),
        "dev_cache": raw.get("alpha_validation_dev_cache", "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev2_8k"),
        "probe_cache": raw.get("alpha_validation_probe_cache", "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev_probe_v1_8k"),
        "metrics_path": raw.get("alpha_validation_metrics_path", "/data/logs/baselines/native_source_domain_r32_v3/alpha_validation_metrics.jsonl"),
    }
    if not all(isinstance(values[name], bool) for name in ("enabled", "probe_enabled", "full_dev_enabled", "full_dev_at_epoch_end")):
        raise ValueError("alpha validation enabled fields must be booleans.")
    if not isinstance(values["probe_interval"], int) or values["probe_interval"] < 1:
        raise ValueError("alpha_dev_probe_interval must be an integer >= 1.")
    if not all(isinstance(values[name], str) and values[name] for name in ("split_dir", "dev_cache", "probe_cache", "metrics_path")):
        raise ValueError("alpha validation paths must be non-empty strings.")
    return AlphaValidationConfig(**values)


REC_PU_CONFIG = _load_rec_pu_config()
PACK_RATIO_CONFIG = _load_pack_ratio_config()
REC_CANDIDATE_METRICS_ENABLED, REC_CANDIDATE_METRICS_INTERVAL = _load_rec_candidate_metric_config()
ALPHA_MONITOR_CONFIG = _load_alpha_monitor_config()
ALPHA_VALIDATION_CONFIG = _load_alpha_validation_config()
REC_PU_DEBUG_PROBE = os.getenv("REC_PU_DEBUG_PROBE", "0").strip().lower() in {"1", "true", "yes", "on"}
# Diagnostic-only: values use the same logits produced by the training
# forward. Set-PU is a direct scalar objective, so its logged final-SID value
# has the same forward/backward geometry as the actual replacement loss.
REC_PU_DIAGNOSTICS = os.getenv("REC_PU_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}
REC_PU_DIAGNOSTICS_RANK0_ONLY = os.getenv("REC_PU_DIAGNOSTICS_RANK0_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
REC_PU_GRAD_DIAGNOSTICS = os.getenv("REC_PU_GRAD_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}
REC_PU_GRAD_DIAG_STEPS = frozenset({10, 20, 25, 30, 35, 40})
_original_alpaca_convert = AlpacaDatasetConverter.__call__
_original_get_train_sampler = CustomSeq2SeqTrainer._get_train_sampler
_original_trainer_init = CustomSeq2SeqTrainer.__init__


def _convert_with_source(self, example):
    output = _original_alpaca_convert(self, example)
    output["_data_source"] = example.get("data_source", "")
    output["_source_segment"] = example.get("source_segment", "")
    output["_aux_metadata_json"] = example.get("aux_metadata_json", "")
    rec_pu_fields = (
        "recommendation_group_id",
        "recommendation_group_size",
        "recommendation_all_gold_sids",
        "recommendation_current_gold_sid",
    )
    raw_rec_pu = {field: example[field] for field in rec_pu_fields if field in example}
    if not raw_rec_pu and example.get("aux_metadata_json"):
        try:
            candidate = json.loads(example["aux_metadata_json"])
        except json.JSONDecodeError as error:
            raise ValueError("Recommendation aux_metadata_json is not valid JSON.") from error
        if isinstance(candidate, dict):
            raw_rec_pu = {field: candidate[field] for field in rec_pu_fields if field in candidate}
    if raw_rec_pu and set(raw_rec_pu) != set(rec_pu_fields):
        raise ValueError("Recommendation REC-PU metadata is incomplete before tokenization.")
    output["_rec_pu_metadata_json"] = json.dumps(raw_rec_pu, ensure_ascii=False, sort_keys=True) if raw_rec_pu else ""
    return output


def _task_id(source: str, source_segment: str) -> int:
    if source in {CANONICAL_SOURCE, "sid_bucket_reverse", "material_sample"}:
        return TASK_ID_BY_NAME["material"]
    if source == "recommend":
        return TASK_ID_BY_NAME["recommendation"]
    if source_segment == "user_action":
        return TASK_ID_BY_NAME["user_action"]
    if source_segment in {"user_chain_cot", "user_chain_nocot"}:
        return TASK_ID_BY_NAME["user_chain"]
    return -1


def _item_token_ids(tokenizer) -> set[int]:
    return {token_id for token, token_id in tokenizer.get_vocab().items() if ITEM_TOKEN_PATTERN.fullmatch(token)}


def _build_loss_weights(labels: list[int], source: str, item_ids: set[int]) -> list[float]:
    if source == CANONICAL_SOURCE:
        return [CANONICAL_TEXT_WEIGHT if label != IGNORE_INDEX else 0.0 for label in labels]

    item_weight = GLOBAL_ITEM_WEIGHT if GLOBAL_ITEM_WEIGHT > 0 else SOURCE_ITEM_WEIGHTS.get(source, 1.0)
    return [
        0.0 if label == IGNORE_INDEX else item_weight if label in item_ids else 1.0
        for label in labels
    ]


def _material_domain_weight(source: str, prompt, response) -> float:
    if source != "material_sample":
        return 1.0

    text = "".join(str(turn.get("content", "")) for turn in prompt)
    text += "".join(str(turn.get("content", "")) for turn in response)
    match = MATERIAL_DOMAIN_PATTERN.search(text)
    if match is None:
        return 1.0

    return MATERIAL_DOMAIN_WEIGHTS[match.group(1)]


def _unpad_packed_features_with_weights(features) -> None:
    attention_mask = features.get("attention_mask")
    if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
        return
    seq_len = attention_mask.size(1)
    non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    if non_padding.numel() == seq_len:
        return

    sequence_keys = {
        "input_ids",
        "labels",
        "loss_weights",
        "sample_ids",
        "sample_task_ids",
        "sample_domain_weights",
        "attention_mask",
        "token_type_ids",
    }
    for key, value in list(features.items()):
        if not torch.is_tensor(value):
            continue
        if key == "position_ids" and value.size(-1) == seq_len:
            features[key] = value.index_select(-1, non_padding)
        elif key == "cross_attention_mask" and value.dim() >= 2 and value.size(0) == 1 and value.size(1) == seq_len:
            features[key] = value.index_select(1, non_padding)
        elif key in sequence_keys and value.dim() == 2 and value.size(0) == 1 and value.size(1) == seq_len:
            features[key] = value.index_select(1, non_padding)


def _preprocess_packed_with_weights(self, examples):
    valid_num = 0
    batch_input_ids, batch_labels, batch_loss_weights = [], [], []
    batch_sample_domain_weights, batch_sample_task_ids, batch_rec_pu_metadata = [], [], []
    batch_images, batch_videos, batch_audios = [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    item_ids = _item_token_ids(self.tokenizer)

    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            continue

        input_ids, labels = self._encode_data_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            audios=examples["_audios"][i] or [],
        )
        length = len(input_ids)
        if length > self.data_args.cutoff_len:
            continue

        source = examples["_data_source"][i]
        source_segment = examples["_source_segment"][i]
        lengths.append(length)
        length2indexes[length].append(valid_num)
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_loss_weights.append(_build_loss_weights(labels, source, item_ids))
        batch_sample_domain_weights.append(
            _material_domain_weight(source, examples["_prompt"][i], examples["_response"][i])
        )
        batch_sample_task_ids.append(_task_id(source, source_segment))
        batch_rec_pu_metadata.append(examples["_rec_pu_metadata_json"][i])
        batch_images.append(examples["_images"][i] or [])
        batch_videos.append(examples["_videos"][i] or [])
        batch_audios.append(examples["_audios"][i] or [])
        valid_num += 1

    model_inputs = defaultdict(list)
    requires_packing_params = self.data_args.neat_packing
    for knapsack in greedy_knapsack(lengths, self.data_args.cutoff_len):
        packed_input_ids, packed_attention_masks = [], []
        packed_position_ids, packed_labels, packed_loss_weights = [], [], []
        packed_sample_ids, packed_sample_task_ids, packed_sample_domain_weights = [], [], []
        packed_rec_pu_segments = []
        packed_images, packed_videos, packed_audios = [], [], []
        if requires_packing_params:
            sequence_boundaries = [0]
            image_subseq_ids, video_subseq_ids, audio_subseq_ids = [], [], []

        for subseq_idx, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            sample_ids = batch_input_ids[index]
            segment_source = examples["_source_segment"][index]
            segment_start = len(packed_input_ids)
            packed_input_ids += sample_ids
            packed_position_ids += list(range(len(sample_ids)))
            packed_labels += batch_labels[index]
            packed_loss_weights += batch_loss_weights[index]
            packed_sample_ids += [subseq_idx] * len(sample_ids)
            packed_sample_task_ids += [batch_sample_task_ids[index]] * len(sample_ids)
            packed_sample_domain_weights += [batch_sample_domain_weights[index]] * len(sample_ids)
            raw_rec_pu = batch_rec_pu_metadata[index]
            if ALPHA_MONITOR_CONFIG.enabled and raw_rec_pu and segment_source not in {
                "recommendation_cot", "recommendation_nocot"
            }:
                raise ValueError(
                    "alpha monitor requires recommendation_cot or recommendation_nocot source_segment; "
                    f"got {segment_source!r}."
                )
            if (REC_PU_CONFIG.enabled or ALPHA_MONITOR_CONFIG.enabled) and raw_rec_pu:
                packed_rec_pu_segments.append(
                    PackedSegment(
                        start=segment_start,
                        end=segment_start + len(sample_ids),
                        task_name="recommendation",
                        metadata=json.loads(raw_rec_pu),
                        source_segment=segment_source,
                    )
                )
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            packed_audios += batch_audios[index]
            if requires_packing_params:
                sequence_boundaries.append(sequence_boundaries[-1] + len(sample_ids))
                image_subseq_ids.extend([subseq_idx] * len(batch_images[index]))
                video_subseq_ids.extend([subseq_idx] * len(batch_videos[index]))
                audio_subseq_ids.extend([subseq_idx] * len(batch_audios[index]))
            packed_attention_masks += [subseq_idx + 1 if self.data_args.neat_packing else 1] * len(sample_ids)

        pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
        if pad_length > 0:
            packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
            packed_position_ids += [0] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            packed_loss_weights += [0.0] * pad_length
            packed_sample_ids += [-1] * pad_length
            packed_sample_task_ids += [-1] * pad_length
            packed_sample_domain_weights += [0.0] * pad_length
            packed_attention_masks += [0 if self.data_args.neat_packing else 1] * pad_length
            if requires_packing_params:
                sequence_boundaries.append(sequence_boundaries[-1] + pad_length)

        if len(packed_input_ids) != self.data_args.cutoff_len + 1:
            raise ValueError("Packed example length does not equal cutoff_len + 1.")

        # Scalar CPU metadata for a post-packing sampler. It follows the exact
        # causal shift and valid-token semantics of the native loss.
        pack_task_id = derive_pack_task_id(packed_labels, packed_sample_task_ids, IGNORE_INDEX)

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["position_ids"].append(packed_position_ids)
        model_inputs["labels"].append(packed_labels)
        model_inputs["loss_weights"].append(packed_loss_weights)
        model_inputs["sample_ids"].append(packed_sample_ids)
        model_inputs["sample_task_ids"].append(packed_sample_task_ids)
        model_inputs["sample_domain_weights"].append(packed_sample_domain_weights)
        model_inputs["pack_task_id"].append(pack_task_id)
        if REC_PU_CONFIG.enabled or ALPHA_MONITOR_CONFIG.enabled:
            targets = locate_packed_rec_pu_targets(
                packed_labels, packed_rec_pu_segments, self.tokenizer, final_occurrence=True
            )
            # Arrow requires a stable feature schema. Keep trainer-only targets
            # as one string through dataset caching, then restore them in the
            # collator after tensorization.
            model_inputs["rec_pu_targets_json"].append(
                json.dumps([serialise_packed_target(target) for target in targets], separators=(",", ":"))
            )
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)
        model_inputs["audios"].append(packed_audios or None)
        if requires_packing_params:
            model_inputs["packing_params"].append(
                asdict(
                    PackingParams(
                        sequence_boundaries=sequence_boundaries,
                        image_subseq_ids=image_subseq_ids or [MAX_SU_SEQ_IDX],
                        video_subseq_ids=video_subseq_ids or [MAX_SU_SEQ_IDX],
                        audio_subseq_ids=audio_subseq_ids or [MAX_SU_SEQ_IDX],
                        right_padding_length=pad_length,
                    )
                )
            )

    return model_inputs


_original_collator_call = SFTDataCollatorWith4DAttentionMask.__call__


def _collate_with_rec_pu_metadata(self, features):
    """Keep REC-PU targets trainer-side; never let the base collator tensorize them."""

    targets_json_by_row = [feature.pop("rec_pu_targets_json", "[]") for feature in features]
    for feature in features:
        feature.pop("pack_task_id", None)
    batch = _original_collator_call(self, features)
    if REC_PU_CONFIG.enabled or ALPHA_MONITOR_CONFIG.enabled:
        batch["rec_pu_targets"] = [json.loads(value) for value in targets_json_by_row]
    return batch


def _get_train_sampler_with_pack_ratio(self, train_dataset=None):
    """Use the original sampler byte-for-byte in OFF mode."""

    if not PACK_RATIO_CONFIG.enabled:
        return _original_get_train_sampler(self, train_dataset)

    dataset = self.train_dataset if train_dataset is None else train_dataset
    if dataset is None or not hasattr(dataset, "column_names") or "pack_task_id" not in dataset.column_names:
        raise ValueError("PackRatio is enabled but packed dataset has no `pack_task_id` column.")

    sampler = PackRatioSampler(
        dataset["pack_task_id"],
        PACK_RATIO_CONFIG.target_ratios,
        seed=int(self.args.seed),
    )
    if int(os.getenv("RANK", "0")) == 0:
        ratios = ", ".join(
            f"{name}={ratio:.6f}" for name, ratio in zip(TASK_NAMES, PACK_RATIO_CONFIG.target_ratios)
        )
        print(
            "PackRatio enabled: "
            f"ratios=[{ratios}] pools={[len(pool) for pool in sampler.pools]} "
            f"fingerprint={sampler.fingerprint()}",
            flush=True,
        )
    return sampler


class _AlphaValidationCallback(TrainerCallback):
    """Run probe/full-dev strictly as a sidecar after optimiser steps."""

    def __init__(self, trainer, config: AlphaValidationConfig) -> None:
        self.trainer = trainer
        self.config = config
        self.runner = AlphaValidationRunner(config)
        self._full_steps: set[int] = set()

    def _run(self, state, kwargs, kind: str) -> None:
        model = kwargs.get("model") or self.trainer.model
        result = self.runner.run(
            self.trainer,
            model,
            kind=kind,
            global_step=int(state.global_step),
            epoch=None if state.epoch is None else float(state.epoch),
        )
        if kind == "probe":
            gaps = self.runner.record_probe_gap(self.trainer, result)
            result["gaps"] = gaps
        self.runner.append_record(result)
        # Main training logs get only va~vo, ga~ge, and the conservative flag;
        # all secondary/domain values stay in alpha_validation_metrics.jsonl.
        pending = {
            key: value for key, value in result["metrics"].items()
            if key in VALIDATION_METRIC_NAMES and value is not None
        }
        pending.update({key: value for key, value in result.get("gaps", {}).items() if value is not None})
        self.trainer._alpha_validation_pending_logs = pending

    def on_step_end(self, args, state, control, **kwargs):
        if self.config.probe_enabled and state.global_step and state.global_step % self.config.probe_interval == 0:
            self._run(state, kwargs, "probe")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.config.full_dev_enabled and self.config.full_dev_at_epoch_end and int(state.global_step) not in self._full_steps:
            self._run(state, kwargs, "full")
            self._full_steps.add(int(state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Smoke-only opt-in: a short interrupted train can still exercise one
        # complete dev pass without changing the formal epoch-end cadence.
        if (
            os.getenv("ALPHA_VALIDATION_RUN_FULL_ON_TRAIN_END", "0") == "1"
            and self.config.full_dev_enabled
            and int(state.global_step) not in self._full_steps
        ):
            self._run(state, kwargs, "full")
            self._full_steps.add(int(state.global_step))
        return control


def _trainer_init_with_alpha_validation(self, *args, **kwargs):
    _original_trainer_init(self, *args, **kwargs)
    if ALPHA_VALIDATION_CONFIG.enabled:
        self.add_callback(_AlphaValidationCallback(self, ALPHA_VALIDATION_CONFIG))


def _maybe_run_rec_pu_debug_probe(
    trainer,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    details,
    targets_by_row,
    sid_component_vocab,
) -> None:
    """One real-logit probe for Smoke A, deliberately absent from normal runs."""

    if not (REC_PU_CONFIG.enabled and REC_PU_DEBUG_PROBE) or getattr(trainer, "_rec_pu_debug_probe_done", False):
        return
    if not targets_by_row:
        return
    all_component_ids = set(sid_component_vocab.a) | set(sid_component_vocab.b) | set(sid_component_vocab.c)
    for row, packed_targets in enumerate(targets_by_row):
        for target in packed_targets:
            for level, logit_position, label_position, positives in (
                ("a", target["a_logit_position"], target["a_label_position"], target["positives"]["a"]),
                ("b", target["b_logit_position"], target["b_label_position"], target["positives"]["b"]),
                ("c", target["c_logit_position"], target["c_label_position"], target["positives"]["c"]),
            ):
                level_ids = sid_component_vocab.for_level(level)
                positive_ids = tuple(int(item) for item in positives)
                unlabeled = [item for item in level_ids if item not in set(positive_ids)]
                if not unlabeled:
                    continue
                wrong_level_ids = sid_component_vocab.for_level({"a": "b", "b": "c", "c": "a"}[level])
                wrong_level_id = int(wrong_level_ids[0])
                normal_token_id = next(
                    token_id
                    for token_id in range(logits.size(-1))
                    if token_id not in all_component_ids and token_id not in positive_ids
                )
                probe = probe_rec_pu_position(
                    logits[row, int(logit_position)],
                    target_label_id=int(labels[row, int(label_position)].item()),
                    positive_ids=positive_ids,
                    same_level_ids=level_ids,
                    wrong_level_id=wrong_level_id,
                    normal_token_id=normal_token_id,
                    beta=REC_PU_CONFIG.unlabeled_sid_grad_scale,
                )
                shifted = sample_ids[row, 1:]
                shifted_labels = labels[row, 1:]
                segment_id = int(shifted[int(logit_position)].item())
                segment_mask = (shifted == segment_id) & (shifted_labels != IGNORE_INDEX)
                final_positions = [
                    int(target["a_logit_position"]),
                    int(target["b_logit_position"]),
                    int(target["c_logit_position"]),
                ]
                probe.update(
                    {
                        "rank": int(os.getenv("RANK", "0")),
                        "level": level,
                        "segment_id": segment_id,
                        "baseline_final_sid_numerator": float(
                            details.base_contributions[row, final_positions].sum().detach().cpu().item()
                        ),
                        "rec_pu_final_sid_numerator": float(
                            details.contributions[row, final_positions].sum().detach().cpu().item()
                        ),
                        "baseline_denominator": float(segment_mask.sum().detach().cpu().item()),
                        "rec_pu_denominator": float(segment_mask.sum().detach().cpu().item()),
                        "double_counting": False,
                    }
                )
                trainer._rec_pu_debug_probe_done = True
                print("REC_PU_DEBUG_PROBE=" + json.dumps(probe, sort_keys=True), flush=True)
                return


def _compute_source_weighted_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    weights = inputs.pop("loss_weights").to(dtype=torch.float32)
    sample_ids = inputs.pop("sample_ids").to(dtype=torch.long)
    sample_task_ids = inputs.pop("sample_task_ids").to(dtype=torch.long)
    sample_domain_weights = inputs.pop("sample_domain_weights").to(dtype=torch.float32)
    rec_pu_targets = inputs.pop("rec_pu_targets", None)
    outputs = model(**inputs)
    sid_component_vocab = None
    if REC_PU_CONFIG.enabled or REC_CANDIDATE_METRICS_ENABLED or ALPHA_MONITOR_CONFIG.enabled:
        sid_component_vocab = getattr(self, "_rec_pu_component_vocab", None)
        if sid_component_vocab is None:
            tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            sid_component_vocab = build_sid_component_vocab(tokenizer)
            self._rec_pu_component_vocab = sid_component_vocab
    collect_candidate_metrics = candidate_window_active(
        self.state.global_step, REC_CANDIDATE_METRICS_ENABLED, REC_CANDIDATE_METRICS_INTERVAL
    )
    collect_alpha_tf = candidate_window_active(
        self.state.global_step,
        ALPHA_MONITOR_CONFIG.enabled and ALPHA_MONITOR_CONFIG.train_tf_enabled,
        ALPHA_MONITOR_CONFIG.train_tf_interval,
    )
    loss, details = compute_native_sid8_loss(
        logits=outputs.logits,
        labels=labels,
        loss_weights=weights,
        sample_ids=sample_ids,
        sample_task_ids=sample_task_ids,
        sample_domain_weights=sample_domain_weights,
        rec_pu_targets=rec_pu_targets,
        rec_pu_config=REC_PU_CONFIG,
        sid_component_vocab=sid_component_vocab,
        collect_candidate_metrics=collect_candidate_metrics,
        collect_rec_metrics=REC_CANDIDATE_METRICS_ENABLED,
    )
    _maybe_run_rec_pu_debug_probe(
        self,
        logits=outputs.logits,
        labels=labels,
        sample_ids=sample_ids,
        details=details,
        targets_by_row=rec_pu_targets,
        sid_component_vocab=sid_component_vocab,
    )
    sample_losses = details.sample_numerators / details.sample_token_counts.clamp_min(1.0)
    _accumulate_task_loss_metrics(
        self, sample_losses.detach() * details.sample_domain_weights.detach(), details.sample_task_ids
    )
    _accumulate_rec_pu_metrics(self, details)
    if REC_PU_CONFIG.enabled or REC_CANDIDATE_METRICS_ENABLED:
        _accumulate_core_recommendation_metrics(self, details)
    _accumulate_rec_candidate_metrics(self, details)
    _accumulate_alpha_recommendation_monitor(
        self,
        details=details,
        logits=outputs.logits,
        labels=labels,
        rec_pu_targets=rec_pu_targets,
        sid_component_vocab=sid_component_vocab,
        collect_tf=collect_alpha_tf,
    )
    _accumulate_rec_pu_diagnostics(
        self,
        logits=outputs.logits,
        labels=labels,
        loss_weights=weights,
        sample_task_ids=sample_task_ids,
        rec_pu_targets=rec_pu_targets,
        sid_component_vocab=sid_component_vocab,
        details=details,
    )
    _maybe_accumulate_gradient_diagnostics(
        self,
        model=model,
        logits=outputs.logits,
        labels=labels,
        rec_pu_targets=rec_pu_targets,
        sid_component_vocab=sid_component_vocab,
        details=details,
    )
    return (loss, outputs) if return_outputs else loss


def _accumulate_task_loss_metrics(trainer, sample_losses: torch.Tensor, sample_tasks: torch.Tensor) -> None:
    valid_tasks = (sample_tasks >= 0) & (sample_tasks < len(TASK_NAMES))
    if not valid_tasks.any():
        return
    stats = getattr(trainer, "_native_task_loss_stats", None)
    if stats is None or stats.device != sample_losses.device:
        stats = torch.zeros((len(TASK_NAMES), 2), device=sample_losses.device, dtype=torch.float64)
        trainer._native_task_loss_stats = stats
    task_indices = sample_tasks[valid_tasks]
    stats[:, 0].scatter_add_(0, task_indices, sample_losses[valid_tasks].to(dtype=stats.dtype))
    stats[:, 1].scatter_add_(0, task_indices, torch.ones_like(task_indices, dtype=stats.dtype))


def _accumulate_rec_pu_metrics(trainer, details) -> None:
    if not (REC_PU_CONFIG.enabled or REC_CANDIDATE_METRICS_ENABLED):
        return
    values = torch.tensor(
        [
            details.rec_pu_segments,
            details.rec_pu_positions,
            details.rec_pu_singleton_segments,
            details.rec_pu_segments - details.rec_pu_singleton_segments,
            details.positive_count_a_sum,
            details.positive_count_b_sum,
            details.positive_count_c_sum,
        ],
        device=details.contributions.device,
        dtype=torch.float64,
    )
    stats = getattr(trainer, "_rec_pu_stats", None)
    if stats is None or stats.device != values.device:
        stats = torch.zeros_like(values)
        trainer._rec_pu_stats = stats
    stats.add_(values)


_CORE_RECOMMENDATION_METRIC_NAMES = (
    "a_rec_setpu_a", "a_rec_setpu_b", "a_rec_setpu_c",
    "b_rec_posmass_a", "b_rec_posmass_b", "b_rec_posmass_c",
    "c_rec_goldprob_a", "c_rec_goldprob_b", "c_rec_goldprob_c",
    "d_rec_posentropy_a",
    "e_share_material", "e_share_recommendation", "e_share_user_action", "e_share_user_chain",
)

_REC_CANDIDATE_METRIC_NAMES = (
    "f_rec_tf_a_hit8", "f_rec_tf_a_hit32",
    "g_rec_tf_a_cov8", "g_rec_tf_a_cov32",
    "h_rec_tf_b_hit8", "h_rec_tf_c_hit8",
    "i_rec_tf_chain_32_8_8",
)


def _core_recommendation_metric_stats(trainer, device):
    stats = getattr(trainer, "_core_recommendation_metric_stats", None)
    if stats is None or stats.device != device:
        stats = torch.zeros((len(_CORE_RECOMMENDATION_METRIC_NAMES), 2), device=device, dtype=torch.float64)
        trainer._core_recommendation_metric_stats = stats
    return stats


def _accumulate_core_recommendation_metrics(trainer, details) -> None:
    """Accumulate detached Set-PU quality and pack-mean exposure metrics.

    This is intentionally independent of REC_PU_DIAGNOSTICS.  All scalar
    objective values are produced by the loss itself; the only extra tensor
    work is a positive-set gather for a-level entropy and O(segments) task
    bincounts.  DDP communication happens only from ``Trainer.log``.
    """

    stats = _core_recommendation_metric_stats(trainer, details.contributions.device)
    stats[0:3].add_(details.rec_setpu_sum_count)
    stats[3:6].add_(details.rec_posmass_sum_count)
    stats[6:9].add_(details.rec_goldprob_sum_count)
    stats[9].add_(details.rec_posentropy_a_sum_count)
    stats[10:14].add_(
        pack_effective_share_sum_count(
            details.sample_task_ids, details.sample_row_ids, details.batch_size, len(TASK_NAMES)
        )
    )


def _accumulate_rec_candidate_metrics(trainer, details) -> None:
    """Store only detached sum/counts for the current sampled GA window."""

    if not REC_CANDIDATE_METRICS_ENABLED:
        return
    stats = getattr(trainer, "_candidate_metric_stats", None)
    if stats is None or stats.device != details.contributions.device:
        stats = torch.zeros((len(_REC_CANDIDATE_METRIC_NAMES), 2), device=details.contributions.device, dtype=torch.float64)
        trainer._candidate_metric_stats = stats
    stats.add_(details.rec_candidate_sum_count)


def _accumulate_alpha_recommendation_monitor(
    trainer, *, details, logits, labels, rec_pu_targets, sid_component_vocab, collect_tf: bool
) -> None:
    """Accumulate detached alpha monitor sum/counts; never alter native SID8 loss."""

    if not ALPHA_MONITOR_CONFIG.enabled:
        return
    stats = getattr(trainer, "_alpha_recommendation_monitor_stats", None)
    if stats is None or stats.device != logits.device:
        stats = torch.zeros((len(ALPHA_MONITOR_METRIC_NAMES), 2), device=logits.device, dtype=torch.float64)
        trainer._alpha_recommendation_monitor_stats = stats
    collected = collect_alpha_recommendation_monitor(
            base_per_token_ce=details.base_per_token_ce.detach(),
            logits=logits.detach(),
            labels=labels,
            rec_targets=rec_pu_targets,
            sid_component_vocab=sid_component_vocab,
            collect_tf=collect_tf,
    )
    stats.add_(collected)
    # This rolling accumulator is separate from logging.  It covers every
    # training microbatch since the prior fixed dev probe, so gaps never use a
    # noisy one-logging-step value.
    if ALPHA_VALIDATION_CONFIG.enabled and ALPHA_VALIDATION_CONFIG.probe_enabled:
        gap_stats = getattr(trainer, "_alpha_train_gap_stats", None)
        if gap_stats is None or gap_stats.device != logits.device:
            gap_stats = torch.zeros((6, 2), device=logits.device, dtype=torch.float64)
            trainer._alpha_train_gap_stats = gap_stats
        gap_stats.add_(collected[:6])


_REC_DIAG_NAMES = (
    "rec_text_ce", "rec_think_sid_ce", "rec_final_onehot_ce", "rec_final_setpu_loss",
    "rec_pos_mass", "rec_u_mass", "rec_pos_min_prob", "rec_pos_mean_prob", "rec_pos_max_prob",
    "rec_gold_top1_acc", "rec_positive_top1_acc", "rec_pos_vs_u_margin",
)
_REC_DIAG_LEVEL_NAMES = tuple(
    f"{name}_{level}"
    for name in ("rec_pos_mass", "rec_u_mass", "rec_positive_top1_acc", "rec_pos_vs_u_margin")
    for level in ("a", "b", "c")
)


def _add_metric(stats, index, name, values) -> None:
    """Accumulate a detached finite mean as (sum, count)."""
    if values.numel() == 0:
        return
    values = values.detach().to(device=stats.device, dtype=stats.dtype).reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel():
        stats[index[name], 0] += values.sum()
        stats[index[name], 1] += values.numel()


def _rec_diag_stats(trainer, device):
    names = _REC_DIAG_NAMES + _REC_DIAG_LEVEL_NAMES
    stats = getattr(trainer, "_rec_pu_diag_stats", None)
    if stats is None or stats.device != device:
        stats = torch.zeros((len(names), 2), device=device, dtype=torch.float64)
        trainer._rec_pu_diag_stats = stats
    return stats, {name: i for i, name in enumerate(names)}


def _native_item_ids(trainer, device):
    value = getattr(trainer, "_native_item_ids_for_diagnostics", None)
    if value is None or value.device != device:
        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        value = torch.tensor(sorted(_item_token_ids(tokenizer)), dtype=torch.long, device=device)
        trainer._native_item_ids_for_diagnostics = value
    return value


def _accumulate_rec_pu_diagnostics(
    trainer, *, logits, labels, loss_weights, sample_task_ids, rec_pu_targets, sid_component_vocab, details
) -> None:
    """Collect detached, selected-position diagnostics; no second model forward.

    The routine avoids a full ``softmax([batch, seq, vocab])``. It evaluates
    logsumexp and indexed gathers only for real final a/b/c positions.
    """
    if not (REC_PU_DIAGNOSTICS and REC_PU_CONFIG.enabled):
        return
    with torch.no_grad():
        stats, index = _rec_diag_stats(trainer, logits.device)
        # The requested metrics are sampled at the existing logging cadence.
        # Rank 0's 16 real GA microbatches form the reporting sample; other
        # ranks allocate zero-shaped-compatible stats only, so DDP all-reduce
        # remains safe without paying four copies of selected-logit logsumexp.
        target_step = int(trainer.state.global_step) + 1
        if target_step % int(trainer.args.logging_steps) != 0:
            return
        if REC_PU_DIAGNOSTICS_RANK0_ONLY and int(os.getenv("RANK", "0")) != 0:
            return
        shift_labels = labels[..., 1:]
        shift_weights = loss_weights[..., 1:].float()
        shift_tasks = sample_task_ids[..., 1:]
        valid = (shift_labels != IGNORE_INDEX) & (shift_tasks >= 0)
        rec_mask = valid & (shift_tasks == TASK_ID_BY_NAME["recommendation"])
        final_mask = torch.zeros_like(rec_mask)
        entries = {"a": [], "b": [], "c": []}
        for row, row_targets in enumerate(rec_pu_targets or []):
            for raw_target in row_targets:
                target = coerce_packed_target(raw_target)
                for level, position, positives in (
                    ("a", target.a_logit_position, target.positives.a),
                    ("b", target.b_logit_position, target.positives.b),
                    ("c", target.c_logit_position, target.positives.c),
                ):
                    if position < 0 or position >= logits.size(1) - 1:
                        raise ValueError("REC-PU diagnostic target lies outside shifted logits.")
                    final_mask[row, position] = True
                    entries[level].append((row, position, tuple(int(value) for value in positives)))

        # CE decomposition uses baseline one-hot values; final replacement is
        # reported on the same positions separately.
        base_ce = details.base_contributions / shift_weights.clamp_min(1.0)
        actual_ce = details.contributions / shift_weights.clamp_min(1.0)
        item_ids = _native_item_ids(trainer, logits.device)
        sid_mask = torch.isin(shift_labels, item_ids) if item_ids.numel() else torch.zeros_like(rec_mask)
        _add_metric(stats, index, "rec_text_ce", base_ce[rec_mask & ~sid_mask])
        _add_metric(stats, index, "rec_think_sid_ce", base_ce[rec_mask & sid_mask & ~final_mask])
        _add_metric(stats, index, "rec_final_onehot_ce", base_ce[final_mask])
        _add_metric(stats, index, "rec_final_setpu_loss", actual_ce[final_mask])

        # Actual native numerator split: membership is lexical SID/domain
        # identity, not loss weight, so canonical response-4 text remains text.
        sid_stats = getattr(trainer, "_native_sid_share_stats", None)
        if sid_stats is None or sid_stats.device != logits.device:
            sid_stats = torch.zeros((len(TASK_NAMES), 2), device=logits.device, dtype=torch.float64)
            trainer._native_sid_share_stats = sid_stats
        for task_id in range(len(TASK_NAMES)):
            task_mask = valid & (shift_tasks == task_id)
            sid_stats[task_id, 0] += details.contributions[task_mask & sid_mask].sum().detach().to(torch.float64)
            sid_stats[task_id, 1] += details.contributions[task_mask & ~sid_mask].sum().detach().to(torch.float64)

        # P/U/O mass and teacher-forced top-1 metrics use original detached
        # logits. Critically, index final positions *before* converting to
        # fp32: making ``logits[..., :-1, :].float()`` would clone the whole
        # 8K x vocabulary tensor at a logging step and defeats the low-cost
        # diagnostic contract.
        for level, level_entries in entries.items():
            if not level_entries:
                continue
            level_ids = torch.tensor(sid_component_vocab.for_level(level), dtype=torch.long, device=logits.device)
            rows = torch.tensor([row for row, _, _ in level_entries], dtype=torch.long, device=logits.device)
            positions = torch.tensor([position for _, position, _ in level_entries], dtype=torch.long, device=logits.device)
            selected_vectors = logits[rows, positions].detach().float()
            # The expensive vocabulary-wide reductions are batched by level.
            # A GA16 logging step can contain many final SID positions; doing
            # one tiny CUDA reduction per position made the monitor (not the
            # loss) dominate wall-clock time. These two tensors have shape
            # [selected positions], not [packed sequence positions].
            denominators = torch.logsumexp(selected_vectors, dim=1)
            top1_ids = selected_vectors.argmax(dim=1)
            level_logits_all = selected_vectors.index_select(1, level_ids)
            level_masses = torch.exp(torch.logsumexp(level_logits_all, dim=1) - denominators)
            measures = defaultdict(list)
            for entry_index, (row, position, positives) in enumerate(level_entries):
                vector = selected_vectors[entry_index]
                positive_ids = torch.tensor(positives, dtype=torch.long, device=logits.device)
                denominator = denominators[entry_index]
                positive_logits = vector.index_select(0, positive_ids)
                level_logits = level_logits_all[entry_index]
                pos_mass = torch.exp(torch.logsumexp(positive_logits, dim=0) - denominator)
                pos_probs = torch.exp(positive_logits - denominator)
                top1 = top1_ids[entry_index]
                positive_mask = (level_ids[:, None] == positive_ids[None, :]).any(dim=1)
                measures["rec_pos_mass"].append(pos_mass)
                measures["rec_u_mass"].append((level_masses[entry_index] - pos_mass).clamp_min(0.0))
                measures["rec_pos_min_prob"].append(pos_probs.min())
                measures["rec_pos_mean_prob"].append(pos_probs.mean())
                measures["rec_pos_max_prob"].append(pos_probs.max())
                measures["rec_gold_top1_acc"].append((top1 == shift_labels[row, position]).float())
                measures["rec_positive_top1_acc"].append((top1 == positive_ids).any().float())
                if not bool(positive_mask.all().item()):
                    u_logits = level_logits.masked_fill(positive_mask, float("-inf"))
                    measures["rec_pos_vs_u_margin"].append(positive_logits.max() - u_logits.max())
            for name, values in measures.items():
                stacked = torch.stack(values)
                _add_metric(stats, index, name, stacked)
                if name in {"rec_pos_mass", "rec_u_mass", "rec_positive_top1_acc", "rec_pos_vs_u_margin"}:
                    _add_metric(stats, index, f"{name}_{level}", stacked)


_GRAD_DIAG_NAMES = (
    "cos_material_rec", "cos_user_action_rec", "cos_user_chain_rec",
    "grad_norm_material", "grad_norm_recommendation", "grad_norm_user_action", "grad_norm_user_chain",
    "cos_recpu_vs_onehot", "recpu_to_onehot_grad_norm_ratio",
)


def _grad_diag_stats(trainer, device):
    stats = getattr(trainer, "_rec_pu_grad_diag_stats", None)
    if stats is None or stats.device != device:
        stats = torch.zeros((len(_GRAD_DIAG_NAMES), 2), device=device, dtype=torch.float64)
        trainer._rec_pu_grad_diag_stats = stats
    return stats, {name: i for i, name in enumerate(_GRAD_DIAG_NAMES)}


def _reference_lora_params(trainer, model):
    cached = getattr(trainer, "_rec_pu_reference_lora_params", None)
    if cached is not None:
        return cached
    selected = []
    wanted_layers = {32, 33, 34, 35}
    wanted_modules = ("q_proj", "v_proj", "o_proj", "down_proj")
    for name, parameter in model.named_parameters():
        match = re.search(r"layers\.(\d+)\.", name)
        if (
            parameter.requires_grad
            and "lora_B" in name
            and match is not None
            and int(match.group(1)) in wanted_layers
            and any(module in name for module in wanted_modules)
        ):
            selected.append((name, parameter))
    if not selected:
        # Fail closed to a deterministic small LoRA-B subset if a PEFT naming
        # change removes the Qwen layer pattern. This is diagnostic-only.
        selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_B" in name][-16:]
    if not selected:
        raise ValueError("REC-PU gradient diagnostic found no trainable LoRA-B reference parameters.")
    trainer._rec_pu_reference_lora_params = selected
    if int(os.getenv("RANK", "0")) == 0:
        print("REC_PU_GRAD_REFERENCE=" + json.dumps([name for name, _ in selected]), flush=True)
    return selected


def _vjp_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters], retain_graph=True, allow_unused=True)
    return torch.cat([
        torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
        if gradient is None else gradient.detach().float().reshape(-1)
        for (_, parameter), gradient in zip(parameters, gradients)
    ])


def _add_grad_metric(stats, index, name, value):
    if torch.isfinite(value):
        stats[index[name], 0] += value.detach().to(dtype=stats.dtype)
        stats[index[name], 1] += 1


def _maybe_accumulate_gradient_diagnostics(
    trainer, *, model, logits, labels, rec_pu_targets, sid_component_vocab, details
) -> None:
    """Rare diagnostic VJPs on a small LoRA-B subset, never training grads.

    Triggered at most once at each requested optimizer step on rank 0. It
    reuses the live forward graph and calls ``autograd.grad`` only for the
    reference subset; the normal Trainer backward and optimizer update remain
    unchanged. If the selected packed microbatch lacks one of the four tasks,
    the probe is skipped rather than fabricating a cross-task direction.
    """
    if not (REC_PU_GRAD_DIAGNOSTICS and REC_PU_CONFIG.enabled):
        return
    stats, index = _grad_diag_stats(trainer, logits.device)  # allocate on every rank for DDP all-reduce safety
    target_step = int(trainer.state.global_step) + 1
    if int(os.getenv("RANK", "0")) != 0 or target_step not in REC_PU_GRAD_DIAG_STEPS:
        return
    completed = getattr(trainer, "_rec_pu_grad_diag_completed_steps", set())
    if target_step in completed:
        return
    if not rec_pu_targets or not any(rec_pu_targets):
        return

    sample_tasks = details.sample_task_ids
    present = {int(task.item()) for task in sample_tasks if int(task.item()) >= 0}
    if set(range(len(TASK_NAMES))) - present:
        return
    parameters = _reference_lora_params(trainer, model)
    sample_losses = details.sample_numerators / details.sample_token_counts.clamp_min(1.0)
    task_vectors = {}
    for task_id, task_name in enumerate(TASK_NAMES):
        task_loss = (sample_losses[sample_tasks == task_id] * details.sample_domain_weights[sample_tasks == task_id]).mean()
        task_vectors[task_name] = _vjp_vector(task_loss, parameters)
        _add_grad_metric(stats, index, f"grad_norm_{task_name}", task_vectors[task_name].norm())
    rec_vector = task_vectors["recommendation"]
    for task_name in ("material", "user_action", "user_chain"):
        _add_grad_metric(stats, index, f"cos_{task_name}_rec", F.cosine_similarity(task_vectors[task_name], rec_vector, dim=0))

    # Compare only the same final recommendation positions. This is a pure
    # diagnostic objective, not a second training loss or extra model forward.
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    pu_terms, onehot_terms = [], []
    by_level = {"a": [], "b": [], "c": []}
    for row, row_targets in enumerate(rec_pu_targets):
        for raw_target in row_targets:
            target = coerce_packed_target(raw_target)
            by_level["a"].append((row, target.a_logit_position, target.positives.a))
            by_level["b"].append((row, target.b_logit_position, target.positives.b))
            by_level["c"].append((row, target.c_logit_position, target.positives.c))
    for level, entries in by_level.items():
        if not entries:
            continue
        rows = torch.tensor([row for row, _, _ in entries], device=logits.device, dtype=torch.long)
        positions = torch.tensor([position for _, position, _ in entries], device=logits.device, dtype=torch.long)
        selected_logits = shift_logits[rows, positions]
        pu_terms.append(rec_pu_batched_position_loss(
            selected_logits, [positives for _, _, positives in entries], sid_component_vocab.for_level(level),
            beta=REC_PU_CONFIG.unlabeled_sid_grad_scale,
        ).mean())
        onehot_terms.append(F.cross_entropy(selected_logits.float(), shift_labels[rows, positions]))
    if pu_terms and onehot_terms:
        pu_vector = _vjp_vector(torch.stack(pu_terms).mean(), parameters)
        onehot_vector = _vjp_vector(torch.stack(onehot_terms).mean(), parameters)
        _add_grad_metric(stats, index, "cos_recpu_vs_onehot", F.cosine_similarity(pu_vector, onehot_vector, dim=0))
        _add_grad_metric(stats, index, "recpu_to_onehot_grad_norm_ratio", pu_vector.norm() / onehot_vector.norm().clamp_min(1e-30))
    completed = set(completed)
    completed.add(target_step)
    trainer._rec_pu_grad_diag_completed_steps = completed


_original_trainer_log = CustomSeq2SeqTrainer.log


def _log_with_task_losses(self, logs, *args, **kwargs):
    stats = getattr(self, "_native_task_loss_stats", None)
    if stats is not None and stats[:, 1].sum().item() > 0:
        global_stats = stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM)
        for task_index, task_name in enumerate(TASK_NAMES):
            sample_count = global_stats[task_index, 1]
            if sample_count.item() > 0:
                logs[f"task_loss_{task_name}"] = (global_stats[task_index, 0] / sample_count).item()
        stats.zero_()
    core_stats = getattr(self, "_core_recommendation_metric_stats", None)
    if core_stats is not None:
        global_core_stats = core_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # One compact [14, 2] reduction covers every new always-on metric.
            torch.distributed.all_reduce(global_core_stats, op=torch.distributed.ReduceOp.SUM)
        for metric_index, metric_name in enumerate(_CORE_RECOMMENDATION_METRIC_NAMES):
            count = global_core_stats[metric_index, 1]
            if count.item() > 0:
                logs[metric_name] = (global_core_stats[metric_index, 0] / count).item()
        core_stats.zero_()
    candidate_stats = getattr(self, "_candidate_metric_stats", None)
    if candidate_stats is not None:
        global_candidate_stats = candidate_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_candidate_stats, op=torch.distributed.ReduceOp.SUM)
        # Do not replay old values on non-sampling logging steps.
        if global_candidate_stats[:, 1].sum().item() > 0:
            for metric_index, metric_name in enumerate(_REC_CANDIDATE_METRIC_NAMES):
                count = global_candidate_stats[metric_index, 1]
                if count.item() > 0:
                    logs[metric_name] = (global_candidate_stats[metric_index, 0] / count).item()
        candidate_stats.zero_()
    alpha_stats = getattr(self, "_alpha_recommendation_monitor_stats", None)
    if alpha_stats is not None:
        global_alpha_stats = alpha_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # All alpha monitor values share one compact reduction per log event.
            torch.distributed.all_reduce(global_alpha_stats, op=torch.distributed.ReduceOp.SUM)
        for metric_index, metric_name in enumerate(ALPHA_MONITOR_METRIC_NAMES):
            value, count = global_alpha_stats[metric_index]
            if metric_name.startswith("rec_monitor_"):
                logs[metric_name] = value.item()
            elif count.item() > 0:
                logs[metric_name] = (value / count).item()
        alpha_stats.zero_()
    rec_stats = getattr(self, "_rec_pu_stats", None)
    if rec_stats is not None:
        global_rec_stats = rec_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_rec_stats, op=torch.distributed.ReduceOp.SUM)
        names = (
            "rec_pu_segments",
            "rec_pu_positions",
            "rec_pu_singleton_segments",
            "rec_pu_multi_positive_segments",
            "positive_count_a_sum",
            "positive_count_b_sum",
            "positive_count_c_sum",
        )
        logs.update({name: value.item() for name, value in zip(names, global_rec_stats)})
        if global_rec_stats[0].item() > 0:
            logs["rec_pu_positive_count_a_mean"] = (global_rec_stats[4] / global_rec_stats[0]).item()
            logs["rec_pu_positive_count_b_mean"] = (global_rec_stats[5] / global_rec_stats[0]).item()
            logs["rec_pu_positive_count_c_mean"] = (global_rec_stats[6] / global_rec_stats[0]).item()
        rec_stats.zero_()
    diag_stats = getattr(self, "_rec_pu_diag_stats", None)
    if diag_stats is not None:
        global_diag = diag_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_diag, op=torch.distributed.ReduceOp.SUM)
        for metric_index, metric_name in enumerate(_REC_DIAG_NAMES + _REC_DIAG_LEVEL_NAMES):
            count = global_diag[metric_index, 1]
            if count.item() > 0:
                logs[metric_name] = (global_diag[metric_index, 0] / count).item()
        diag_stats.zero_()
    sid_stats = getattr(self, "_native_sid_share_stats", None)
    if sid_stats is not None:
        global_sid = sid_stats.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_sid, op=torch.distributed.ReduceOp.SUM)
        total_sid = global_sid[:, 0].sum()
        total_text = global_sid[:, 1].sum()
        if (total_sid + total_text).item() > 0:
            logs["sid_weighted_numerator_share"] = (total_sid / (total_sid + total_text)).item()
            logs["text_numerator_share"] = (total_text / (total_sid + total_text)).item()
        for task_index, task_name in enumerate(TASK_NAMES):
            sid_value, text_value = global_sid[task_index]
            if (sid_value + text_value).item() > 0:
                logs[f"{task_name}_sid_numerator_share"] = (sid_value / (sid_value + text_value)).item()
        sid_stats.zero_()
    grad_diag = getattr(self, "_rec_pu_grad_diag_stats", None)
    if grad_diag is not None:
        global_grad_diag = grad_diag.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_grad_diag, op=torch.distributed.ReduceOp.SUM)
        for metric_index, metric_name in enumerate(_GRAD_DIAG_NAMES):
            count = global_grad_diag[metric_index, 1]
            if count.item() > 0:
                logs[metric_name] = (global_grad_diag[metric_index, 0] / count).item()
        grad_diag.zero_()
    pending_validation_logs = getattr(self, "_alpha_validation_pending_logs", None)
    if pending_validation_logs:
        logs.update(pending_validation_logs)
        self._alpha_validation_pending_logs = {}
    return _original_trainer_log(self, logs, *args, **kwargs)


def _install_fractional_gradient_checkpointing() -> None:
    """Checkpoint a deterministic prefix of decoder blocks for an ablation.

    The reference route is unchanged for NATIVE_GC_FRACTION=1.  Fractional
    mode exists only for memory/throughput smoke tests and intentionally keeps
    the loss, model parameters, and optimizer semantics untouched.
    """

    fraction = float(os.getenv("NATIVE_GC_FRACTION", "1.0"))
    if fraction == 1.0:
        return
    if fraction not in (0.4, 0.5):
        raise ValueError("NATIVE_GC_FRACTION currently supports only 1.0, 0.5, or 0.4")

    checkpointed_layers = int(36 * fraction)

    base_factory = checkpointing_utils.get_custom_gradient_checkpointing_func

    def fractional_factory(gradient_checkpointing_func):
        full_checkpoint = base_factory(gradient_checkpointing_func)
        decoder_order_by_id: dict[int, int] = {}

        @wraps(full_checkpoint)
        def fractional_checkpointing_func(func, *args, **kwargs):
            module = func.func.__self__ if hasattr(func, "func") else func.__self__
            # Transformers 5.3 does not persist layer_idx on Qwen3DecoderLayer.
            # The forward order is the ModuleList order, so assign a persistent
            # per-process index on first use and checkpoint a fixed prefix.
            if module.__class__.__name__ == "Qwen3DecoderLayer":
                module_id = id(module)
                if module_id not in decoder_order_by_id:
                    decoder_order_by_id[module_id] = len(decoder_order_by_id)
                    if len(decoder_order_by_id) == 36:
                        print(
                            f"Native fractional GC confirmed: {checkpointed_layers} of 36 "
                            "Qwen3 decoder blocks checkpointed.",
                            flush=True,
                        )
                should_checkpoint = decoder_order_by_id[module_id] < checkpointed_layers
            else:
                should_checkpoint = False
            if should_checkpoint:
                return full_checkpoint(func, *args, **kwargs)
            return func(*args, **kwargs)

        return fractional_checkpointing_func

    checkpointing_utils.get_custom_gradient_checkpointing_func = fractional_factory
    print(
        f"Native fractional GC enabled: checkpoint decoder layers 0-{checkpointed_layers - 1} of 36.",
        flush=True,
    )


def install_native_patches() -> None:
    """Install the native route once for training or integration-only audits."""

    _install_fractional_gradient_checkpointing()
    AlpacaDatasetConverter.__call__ = _convert_with_source
    PackedSupervisedDatasetProcessor.preprocess_dataset = _preprocess_packed_with_weights
    SFTDataCollatorWith4DAttentionMask._unpad_packed_features = staticmethod(_unpad_packed_features_with_weights)
    SFTDataCollatorWith4DAttentionMask.__call__ = _collate_with_rec_pu_metadata
    CustomSeq2SeqTrainer.compute_loss = _compute_source_weighted_loss
    CustomSeq2SeqTrainer._get_train_sampler = _get_train_sampler_with_pack_ratio
    CustomSeq2SeqTrainer.log = _log_with_task_losses
    CustomSeq2SeqTrainer.__init__ = _trainer_init_with_alpha_validation
    if os.getenv("SOURCE_WEIGHT_SKIP_FINAL_SAVE") == "1":
        CustomSeq2SeqTrainer.save_model = lambda self, *args, **kwargs: None


class _StopAfterOptimizerStepsCallback(TrainerCallback):
    """Smoke-only stop hook that leaves the configured scheduler horizon intact."""

    def __init__(self, stop_after_steps: int) -> None:
        self.stop_after_steps = int(stop_after_steps)
        self._started_at: float | None = None
        self._durations: list[float] = []

    def on_step_begin(self, args, state, control, **kwargs):
        import time

        self._started_at = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        import time

        if self._started_at is not None:
            self._durations.append(time.perf_counter() - self._started_at)
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        payload = {
            "rank": int(os.getenv("RANK", "0")),
            "stop_after_steps": self.stop_after_steps,
            "completed_steps": int(state.global_step),
            "step_durations_sec": self._durations,
            "peak_cuda_gib": (
                round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0
            ),
        }
        print("PACK_RATIO_SMOKE_TIMING=" + json.dumps(payload, sort_keys=True), flush=True)
        return control


def main() -> None:
    install_native_patches()
    stop_after_steps = int(os.getenv("PACK_RATIO_STOP_AFTER_STEPS", "0"))
    if stop_after_steps < 0:
        raise ValueError("PACK_RATIO_STOP_AFTER_STEPS must be >= 0.")
    callbacks = [_StopAfterOptimizerStepsCallback(stop_after_steps)] if stop_after_steps else None
    run_exp(callbacks=callbacks)


if __name__ == "__main__":
    main()
