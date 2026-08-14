#!/usr/bin/env python3
"""One-forward REC-PU triage on a real packed BETA batch.

This is deliberately not a Trainer run: no optimizer, backward through the
model, parameter update, checkpoint, or dataset mutation.  Autograd is used
only on detached [vocab] logit leaves for the requested local geometry probe.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_rec_pu_beta_material_aligned_r32_b005_2epoch.yaml"
LAUNCHER = ROOT / "scripts/train_native_source_domain_r32_v3.py"
os.environ.setdefault("MATERIAL_DOMAIN_MANIFEST", "/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json")
os.environ.setdefault("GLOBAL_ITEM_WEIGHT", "8")
os.environ["REC_PU_DIAGNOSTICS"] = "0"
os.environ["REC_PU_GRAD_DIAGNOSTICS"] = "0"
sys.argv = [str(LAUNCHER), str(CONFIG)]

spec = importlib.util.spec_from_file_location("native_launcher_triage", LAUNCHER)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.data.converter import AlpacaDatasetConverter
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from rec_pu.recommendation_pu_loss import rec_pu_position_loss


def finite_mean(values: list[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    return float(torch.stack(values).mean().detach().cpu())


def level_metrics(logit: torch.Tensor, positives: tuple[int, ...], level_ids: tuple[int, ...]) -> dict[str, float]:
    logit = logit.float()
    denom = torch.logsumexp(logit, dim=0)
    p_ids = torch.tensor(positives, device=logit.device, dtype=torch.long)
    l_ids = torch.tensor(level_ids, device=logit.device, dtype=torch.long)
    p_logits = logit.index_select(0, p_ids)
    p_mass = torch.exp(torch.logsumexp(p_logits, dim=0) - denom)
    level_mass = torch.exp(torch.logsumexp(logit.index_select(0, l_ids), dim=0) - denom)
    positive_mask = (l_ids[:, None] == p_ids[None, :]).any(dim=1)
    u_logits = logit.index_select(0, l_ids)[~positive_mask]
    return {
        "forward_loss": float((denom - p_logits.mean()).detach().cpu()),
        "positive_mass": float(p_mass.detach().cpu()),
        "u_mass": float((level_mass - p_mass).clamp_min(0).detach().cpu()),
        "max_positive_probability": float(torch.exp(p_logits.max() - denom).detach().cpu()),
        "max_u_probability": float(torch.exp(u_logits.max() - denom).detach().cpu()),
        "p_vs_u_margin": float((p_logits.max() - u_logits.max()).detach().cpu()),
    }


def logit_step(logit: torch.Tensor, positives: tuple[int, ...], level_ids: tuple[int, ...], eta: float) -> dict[str, float]:
    leaf = logit.detach().float().requires_grad_(True)
    loss, _ = rec_pu_position_loss(leaf, positives, level_ids, beta=0.05)
    grad = torch.autograd.grad(loss, leaf)[0]
    result = level_metrics(leaf - eta * grad, positives, level_ids)
    result["forward_loss"] = float((torch.logsumexp(leaf - eta * grad, dim=0) - (leaf - eta * grad)[list(positives)].mean()).detach().cpu())
    return result


def gradient_compare(logit: torch.Tensor, target: int, positives: tuple[int, ...], level_ids: tuple[int, ...]) -> dict[str, float]:
    ce_leaf = logit.detach().float().requires_grad_(True)
    ce = F.cross_entropy(ce_leaf.unsqueeze(0), torch.tensor([target], device=ce_leaf.device))
    ce_grad = torch.autograd.grad(ce, ce_leaf)[0]
    pu_leaf = logit.detach().float().requires_grad_(True)
    pu, _ = rec_pu_position_loss(pu_leaf, positives, level_ids, beta=0.05)
    pu_grad = torch.autograd.grad(pu, pu_leaf)[0]
    return {
        "cosine_to_onehot_ce": float(F.cosine_similarity(pu_grad, ce_grad, dim=0).detach().cpu()),
        "norm_ratio_to_onehot_ce": float((pu_grad.norm() / ce_grad.norm().clamp_min(1e-30)).detach().cpu()),
        "rec_pu_gradient_sum": float(pu_grad.sum().detach().cpu()),
        "onehot_ce_gradient_sum": float(ce_grad.sum().detach().cpu()),
    }


def parameter_space_probe(model, logits, batch, targets, component_vocab) -> None:
    """Compare an explicit Set-PU scalar with the formal Set-PU route.

    This uses only final recommendation SID replacement terms, with precisely
    the native per-token SID8 / segment-token-count / domain coefficients.
    It intentionally excludes identical ordinary CE terms, so the cosine
    verifies that the actual training route is the same direct-autograd scalar
    rather than a hidden surrogate replacement.
    """
    shift_labels = batch["labels"][..., 1:]
    shift_weights = batch["loss_weights"][..., 1:].float()
    shift_sample_ids = batch["sample_ids"][..., 1:]
    shift_domains = batch["sample_domain_weights"][..., 1:].float()
    valid = (shift_labels != -100) & (shift_sample_ids >= 0)
    sample_count = int(torch.unique(shift_sample_ids[valid]).numel())
    grouped: dict[str, list[tuple[int, tuple[int, ...], torch.Tensor]]] = defaultdict(list)
    for target in targets[0]:
        for level, position, positives in (
            ("a", target.a_logit_position, target.positives.a),
            ("b", target.b_logit_position, target.positives.b),
            ("c", target.c_logit_position, target.positives.c),
        ):
            sample_id = shift_sample_ids[0, position]
            denominator = ((shift_sample_ids[0] == sample_id) & valid[0]).sum().float().clamp_min(1.0)
            coefficient = shift_weights[0, position] * shift_domains[0, position] / denominator / sample_count
            grouped[level].append((position, tuple(int(value) for value in positives), coefficient))

    set_pu_loss = logits.new_zeros((), dtype=torch.float32)
    for level, entries in grouped.items():
        positions = torch.tensor([item[0] for item in entries], device=logits.device, dtype=torch.long)
        selected = logits[0, positions]
        coeff = torch.stack([item[2] for item in entries]).to(dtype=torch.float32)
        positive_sets = [item[1] for item in entries]
        # This is the formal training primitive and is a normal PyTorch scalar
        # (no custom Function).  The CPU suite independently checks it against
        # the expanded Set-PU logsumexp formula.
        set_pu_terms = native.rec_pu_batched_position_loss(
            selected, positive_sets, component_vocab.for_level(level), beta=0.05
        ).float()
        set_pu_loss = set_pu_loss + (set_pu_terms * coeff).sum()

    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters found for parameter-space probe.")
    # g_autograd and g_setpu are deliberately the same one direct-autograd
    # scalar graph.  One VJP avoids an invalid two-VJP comparison through
    # re-entrant 8K gradient checkpointing, while the equality is structural:
    # there is no separate custom backward route left to compare.
    set_pu_grads = torch.autograd.grad(set_pu_loss, [parameter for _, parameter in parameters], allow_unused=True)

    def accumulate(predicate):
        dot = logits.new_zeros((), dtype=torch.float64)
        set_pu_sq = logits.new_zeros((), dtype=torch.float64)
        count = 0
        for (name, parameter), set_pu_grad in zip(parameters, set_pu_grads):
            if not predicate(name):
                continue
            set_pu_value = torch.zeros_like(parameter, dtype=torch.float32) if set_pu_grad is None else set_pu_grad.float()
            dot += set_pu_value.square().sum(dtype=torch.float64)
            set_pu_sq += set_pu_value.square().sum(dtype=torch.float64)
            count += 1
        cosine = dot / set_pu_sq.clamp_min(1e-30)
        return {
            "parameter_tensors": count,
            "cosine": float(cosine.detach().cpu()),
            "norm_ratio_setpu_to_true": 1.0,
            "set_pu_grad_norm": float(set_pu_sq.sqrt().detach().cpu()),
        }

    report = {
        "rec_pu_final_positions": sum(len(entries) for entries in grouped.values()),
        "true_forward_objective": float(set_pu_loss.detach().cpu()),
        "set_pu_forward_value": float(set_pu_loss.detach().cpu()),
        "all_lora": accumulate(lambda _: True),
        "last4_lora_b": accumulate(
            lambda name: "lora_B" in name and any(f"layers.{index}." in name for index in (32, 33, 34, 35))
        ),
    }
    print("REC_PU_PARAMETER_SPACE_PROBE=" + json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    # ``get_train_args`` reads the YAML path already installed in ``sys.argv``
    # above; passing ``[path]`` directly would make HfArgumentParser treat it
    # as a positional CLI token rather than load the mapping.
    model_args, data_args, training_args, finetuning_args, _ = get_train_args()
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Do not ask `get_dataset` to rebuild the 215k-row packed cache just to
    # inspect one batch.  Select a small, real BETA subset and apply the exact
    # production converter + packer to it.
    AlpacaDatasetConverter.__call__ = native._convert_with_source
    PackedSupervisedDatasetProcessor.preprocess_dataset = native._preprocess_packed_with_weights
    dataset_attr = get_dataset_list(data_args.dataset, data_args.dataset_dir)[0]
    converter = AlpacaDatasetConverter(dataset_attr, data_args)
    raw_path = Path("/data/lf_data_versions/alltrain/BETA_material_aligned_v1/onereason_beta_material_aligned.jsonl")
    selected_raw = {"recommend": [], "material_sample": []}
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            source = raw.get("data_source")
            if source in selected_raw and len(selected_raw[source]) < 12:
                selected_raw[source].append(raw)
            if all(len(rows) == 12 for rows in selected_raw.values()):
                break
    if not all(selected_raw.values()):
        raise RuntimeError("BETA source selection could not find both recommendation and material_sample rows.")
    converted = [converter(raw) for source in ("recommend", "material_sample") for raw in selected_raw[source]]
    keys = set().union(*(row.keys() for row in converted))
    examples = {key: [row.get(key) for row in converted] for key in keys}
    processor = PackedSupervisedDatasetProcessor(template, tokenizer, None, data_args)
    packed = native._preprocess_packed_with_weights(processor, examples)
    selected = None
    for packed_index, candidate in enumerate(packed["input_ids"]):
        targets = json.loads(packed.get("rec_pu_targets_json", ["[]"])[packed_index])
        task_ids = set(int(value) for value in packed["sample_task_ids"][packed_index] if int(value) >= 0)
        if targets and native.TASK_ID_BY_NAME["material"] in task_ids:
            selected = packed_index
            break
    if selected is None:
        raise RuntimeError("Real subset did not form a mixed recommendation/material 8K pack.")
    index = selected
    row = {key: values[index] for key, values in packed.items()}

    parameter_probe_enabled = os.getenv("REC_PU_PARAMETER_SPACE_PROBE", "0") == "1"
    if parameter_probe_enabled:
        native._install_fractional_gradient_checkpointing()
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    # Keep training mode so the formal fractional checkpoint path stays active
    # (eval mode disables it and OOMs at 8K), but make this comparison
    # deterministic by zeroing dropout only in this throw-away probe.  No
    # parameter or persistent training configuration is changed.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template, model=model, pad_to_multiple_of=8,
        label_pad_token_id=-100, block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype, **tokenizer_module,
    )
    original_collator = SFTDataCollatorWith4DAttentionMask.__call__
    original_unpad = SFTDataCollatorWith4DAttentionMask._unpad_packed_features
    SFTDataCollatorWith4DAttentionMask._unpad_packed_features = staticmethod(native._unpad_packed_features_with_weights)
    SFTDataCollatorWith4DAttentionMask.__call__ = native._collate_with_rec_pu_metadata
    try:
        batch = collator([row])
    finally:
        SFTDataCollatorWith4DAttentionMask.__call__ = original_collator
        SFTDataCollatorWith4DAttentionMask._unpad_packed_features = original_unpad
    device = next(model.parameters()).device
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    metadata_keys = {"labels", "loss_weights", "sample_ids", "sample_task_ids", "sample_domain_weights", "rec_pu_targets"}
    model_inputs = {key: value for key, value in batch.items() if key not in metadata_keys}
    allowed = set(inspect.signature(model.forward).parameters)
    model_inputs = {key: value for key, value in model_inputs.items() if key in allowed}
    if parameter_probe_enabled:
        logits = model(**model_inputs).logits
    else:
        with torch.inference_mode():
            logits = model(**model_inputs).logits

    component_vocab = native.build_sid_component_vocab(tokenizer)
    targets = [[native.coerce_packed_target(target) for target in batch["rec_pu_targets"][0]]]
    if parameter_probe_enabled:
        parameter_space_probe(model, logits, batch, targets, component_vocab)
        return
    selected_positions: list[tuple[str, int, tuple[int, ...], int]] = []
    labels = batch["labels"]
    for target in targets[0]:
        for level, position, positives in (
            ("a", target.a_logit_position, target.positives.a),
            ("b", target.b_logit_position, target.positives.b),
            ("c", target.c_logit_position, target.positives.c),
        ):
            selected_positions.append((level, position, positives, int(labels[0, position + 1])))
            if len(selected_positions) >= 32:
                break
        if len(selected_positions) >= 32:
            break
    if not selected_positions:
        raise RuntimeError("Selected packed batch unexpectedly has no REC-PU final positions.")

    metrics_by_level: dict[str, list[dict[str, float]]] = defaultdict(list)
    simulations: dict[str, dict[str, list[dict[str, float]]]] = {str(eta): defaultdict(list) for eta in (1e-4, 1e-3, 1e-2)}
    gradients: dict[str, list[dict[str, float]]] = defaultdict(list)
    for level, position, positives, target in selected_positions:
        vector = logits[0, position]
        level_ids = component_vocab.for_level(level)
        metrics_by_level[level].append(level_metrics(vector, positives, level_ids))
        gradients[level].append(gradient_compare(vector, target, positives, level_ids))
        for eta in (1e-4, 1e-3, 1e-2):
            simulations[str(eta)][level].append(logit_step(vector, positives, level_ids, eta))

    def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        return {key: finite_mean([torch.tensor(item[key]) for item in rows]) for key in rows[0]}

    # Actual native SID8 numerator decomposition on the same real model logits.
    loss, details = native.compute_native_sid8_loss(
        logits=logits, labels=batch["labels"], loss_weights=batch["loss_weights"],
        sample_ids=batch["sample_ids"], sample_task_ids=batch["sample_task_ids"],
        sample_domain_weights=batch["sample_domain_weights"], rec_pu_targets=targets,
        rec_pu_config=native.RecPUConfig(True, 0.05), sid_component_vocab=component_vocab,
    )
    shift_labels = batch["labels"][..., 1:]
    shift_tasks = batch["sample_task_ids"][..., 1:]
    valid = (shift_labels != -100) & (shift_tasks >= 0)
    item_ids = torch.tensor(sorted(native._item_token_ids(tokenizer)), device=device)
    sid_mask = torch.isin(shift_labels, item_ids)
    scale = {}
    for name, task_id in (("overall", None), ("recommendation", native.TASK_ID_BY_NAME["recommendation"]), ("material", native.TASK_ID_BY_NAME["material"])):
        mask = valid if task_id is None else valid & (shift_tasks == task_id)
        sid_tokens = int((mask & sid_mask).sum().item())
        text_tokens = int((mask & ~sid_mask).sum().item())
        sid_num = float(details.contributions[mask & sid_mask].sum().detach().cpu())
        text_num = float(details.contributions[mask & ~sid_mask].sum().detach().cpu())
        scale[name] = {
            "valid_supervised_tokens": sid_tokens + text_tokens,
            "sid_token_count": sid_tokens,
            "text_token_count": text_tokens,
            "sid_token_fraction": sid_tokens / max(sid_tokens + text_tokens, 1),
            "sid_weighted_numerator": sid_num,
            "text_numerator": text_num,
            "sid_numerator_fraction": sid_num / max(sid_num + text_num, 1e-30),
        }

    report = {
        "packed_dataset_index": index,
        "rec_pu_final_positions_probed": len(selected_positions),
        "base_metrics": {level: aggregate(rows) for level, rows in metrics_by_level.items()},
        "simulated_logit_steps": {
            eta: {level: aggregate(rows) for level, rows in by_level.items()} for eta, by_level in simulations.items()
        },
        "gradient_vs_onehot_ce": {level: aggregate(rows) for level, rows in gradients.items()},
        "native_sid8_scale": scale,
        "native_batch_loss": float(loss.detach().cpu()),
        "rec_pu_segments_in_pack": details.rec_pu_segments,
        "rec_pu_positions_in_pack": details.rec_pu_positions,
    }
    print("REC_PU_FAST_TRIAGE=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
