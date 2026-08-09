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
from collections import defaultdict
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as F

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
_original_alpaca_convert = AlpacaDatasetConverter.__call__


def _convert_with_source(self, example):
    output = _original_alpaca_convert(self, example)
    output["_data_source"] = example.get("data_source", "")
    output["_source_segment"] = example.get("source_segment", "")
    # Deliberately model-external. A future recommendation auxiliary loss can
    # parse this field in the processor without changing the base SFT route.
    output["_aux_metadata_json"] = example.get("aux_metadata_json", "")
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
    batch_sample_domain_weights, batch_sample_task_ids = [], []
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
        packed_images, packed_videos, packed_audios = [], [], []
        if requires_packing_params:
            sequence_boundaries = [0]
            image_subseq_ids, video_subseq_ids, audio_subseq_ids = [], [], []

        for subseq_idx, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            sample_ids = batch_input_ids[index]
            packed_input_ids += sample_ids
            packed_position_ids += list(range(len(sample_ids)))
            packed_labels += batch_labels[index]
            packed_loss_weights += batch_loss_weights[index]
            packed_sample_ids += [subseq_idx] * len(sample_ids)
            packed_sample_task_ids += [batch_sample_task_ids[index]] * len(sample_ids)
            packed_sample_domain_weights += [batch_sample_domain_weights[index]] * len(sample_ids)
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

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["position_ids"].append(packed_position_ids)
        model_inputs["labels"].append(packed_labels)
        model_inputs["loss_weights"].append(packed_loss_weights)
        model_inputs["sample_ids"].append(packed_sample_ids)
        model_inputs["sample_task_ids"].append(packed_sample_task_ids)
        model_inputs["sample_domain_weights"].append(packed_sample_domain_weights)
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


def _compute_source_weighted_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    weights = inputs.pop("loss_weights").to(dtype=torch.float32)
    sample_ids = inputs.pop("sample_ids").to(dtype=torch.long)
    sample_task_ids = inputs.pop("sample_task_ids").to(dtype=torch.long)
    sample_domain_weights = inputs.pop("sample_domain_weights").to(dtype=torch.float32)
    outputs = model(**inputs)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = weights[..., 1:].contiguous()
    shift_sample_ids = sample_ids[..., 1:].contiguous()
    shift_sample_task_ids = sample_task_ids[..., 1:].contiguous()
    shift_sample_domain_weights = sample_domain_weights[..., 1:].contiguous()
    valid = (shift_labels != IGNORE_INDEX) & (shift_sample_ids >= 0)

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)

    batch_size = shift_sample_ids.size(0)
    sample_stride = shift_sample_ids.max().clamp_min(0) + 1
    row_offsets = torch.arange(batch_size, device=shift_sample_ids.device).unsqueeze(1) * sample_stride
    global_sample_ids = torch.where(valid, shift_sample_ids + row_offsets, torch.full_like(shift_sample_ids, -1))

    flat_valid = valid.view(-1)
    flat_sample_ids = global_sample_ids.view(-1)[flat_valid]
    if flat_sample_ids.numel() == 0:
        loss = (per_token_loss * 0.0).sum()
        return (loss, outputs) if return_outputs else loss

    flat_loss = (per_token_loss.float() * shift_weights).view(-1)[flat_valid]
    flat_task_ids = shift_sample_task_ids.view(-1)[flat_valid]
    flat_domain_weights = shift_sample_domain_weights.view(-1)[flat_valid]

    unique_ids, inverse = torch.unique(flat_sample_ids, sorted=False, return_inverse=True)
    sample_loss_sums = torch.zeros(unique_ids.numel(), device=flat_loss.device, dtype=flat_loss.dtype)
    sample_token_counts = torch.zeros_like(sample_loss_sums)
    sample_domain_sums = torch.zeros_like(sample_loss_sums)
    sample_task_sums = torch.zeros_like(sample_loss_sums)
    sample_loss_sums.scatter_add_(0, inverse, flat_loss)
    sample_token_counts.scatter_add_(0, inverse, torch.ones_like(flat_loss))
    sample_domain_sums.scatter_add_(0, inverse, flat_domain_weights)
    sample_task_sums.scatter_add_(0, inverse, flat_task_ids.to(dtype=flat_loss.dtype))

    sample_losses = sample_loss_sums / sample_token_counts.clamp_min(1.0)
    sample_domains = sample_domain_sums / sample_token_counts.clamp_min(1.0)
    sample_tasks = (sample_task_sums / sample_token_counts.clamp_min(1.0)).round().to(dtype=torch.long)
    loss = (sample_losses * sample_domains).mean()
    _accumulate_task_loss_metrics(self, sample_losses.detach() * sample_domains.detach(), sample_tasks)
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


def main() -> None:
    _install_fractional_gradient_checkpointing()
    AlpacaDatasetConverter.__call__ = _convert_with_source
    PackedSupervisedDatasetProcessor.preprocess_dataset = _preprocess_packed_with_weights
    SFTDataCollatorWith4DAttentionMask._unpad_packed_features = staticmethod(_unpad_packed_features_with_weights)
    CustomSeq2SeqTrainer.compute_loss = _compute_source_weighted_loss
    CustomSeq2SeqTrainer.log = _log_with_task_losses
    if os.getenv("SOURCE_WEIGHT_SKIP_FINAL_SAVE") == "1":
        CustomSeq2SeqTrainer.save_model = lambda self, *args, **kwargs: None
    run_exp()


if __name__ == "__main__":
    main()
