"""CPU-only GA16 equivalence audit using the deployed CustomSeq2SeqTrainer.

This intentionally uses a custom ``compute_loss(..., **kwargs)`` in the same
shape as the native NSD / REC-PU monkeypatch: it removes labels, calls the
model once, and ignores ``num_items_in_batch``.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


torch.manual_seed(20260811)
torch.set_num_threads(1)
VOCAB = 7
GA = 16


class FixedDataset(Dataset):
    def __init__(self):
        self.rows = [
            {
                "input_ids": torch.tensor([idx % 3], dtype=torch.long),
                "labels": torch.tensor([(idx * 3 + 1) % VOCAB], dtype=torch.long),
            }
            for idx in range(GA)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class ToyLM(nn.Module):
    # ``**kwargs`` deliberately models the relevant Qwen forward property.
    def __init__(self):
        super().__init__()
        self.logit_table = nn.Parameter(torch.randn(3, VOCAB) * 0.3)
        self.config = SimpleNamespace(_attn_implementation=None)

    def forward(self, input_ids=None, **kwargs):
        logits = self.logit_table[input_ids]
        return SimpleNamespace(logits=logits)


def native_style_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    self._ga_audit_kwargs.append(
        {
            "num_items_in_batch": None
            if kwargs.get("num_items_in_batch") is None
            else int(kwargs["num_items_in_batch"]),
            "current_gradient_accumulation_steps": int(self.current_gradient_accumulation_steps),
        }
    )
    outputs = model(**inputs)
    loss = F.cross_entropy(outputs.logits.float().reshape(-1, VOCAB), labels.reshape(-1))
    return (loss, outputs) if return_outputs else loss


def flat_grad(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1).float().cpu() for p in model.parameters() if p.grad is not None])


def flat_param_delta(model: nn.Module, initial: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [(p.detach().cpu() - initial[name].cpu()).reshape(-1).float() for name, p in model.named_parameters()]
    )


def metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    return {
        "norm": float(value.norm()),
        "ref_norm": float(reference.norm()),
        "norm_ratio": float(value.norm() / reference.norm().clamp_min(1e-30)),
        "cosine": float(F.cosine_similarity(value, reference, dim=0)),
        "max_abs_diff": float((value - reference).abs().max()),
    }


def manual_reference(initial: dict[str, torch.Tensor], max_grad_norm: float) -> dict[str, torch.Tensor]:
    model = ToyLM()
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    dataset = FixedDataset()
    for row in dataset.rows:
        logits = model(row["input_ids"].unsqueeze(0)).logits
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), row["labels"].reshape(-1)) / GA
        loss.backward()
    pre = flat_grad(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    post = flat_grad(model)
    optimizer.step()
    return {"pre": pre, "post": post, "delta": flat_param_delta(model, initial)}


def run_case(name: str, initial: dict[str, torch.Tensor], force_loss_kwargs: bool | None) -> dict:
    model = ToyLM()
    model.load_state_dict(initial)
    out = tempfile.mkdtemp(prefix=f"ga16-{name}-")
    args = TrainingArguments(
        output_dir=out,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GA,
        max_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=20260811,
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=args,
        finetuning_args=FinetuningArguments(),
        processor=None,
        tokenizer=None,
        train_dataset=FixedDataset(),
    )
    trainer._ga_audit_kwargs = []
    trainer.compute_loss = native_style_compute_loss.__get__(trainer, type(trainer))
    before_property = bool(trainer.model_accepts_loss_kwargs)
    if force_loss_kwargs is not None:
        trainer.model_accepts_loss_kwargs = force_loss_kwargs

    observed = {"pre": None, "post": None}
    original_clip = trainer.accelerator.clip_grad_norm_

    def capture_clip(params, max_norm, *args, **kwargs):
        observed["pre"] = flat_grad(model)
        result = original_clip(params, max_norm, *args, **kwargs)
        observed["post"] = flat_grad(model)
        return result

    trainer.accelerator.clip_grad_norm_ = capture_clip
    trainer.train()
    delta = flat_param_delta(model, initial)
    return {
        "model_accepts_loss_kwargs_before_override": before_property,
        "model_accepts_loss_kwargs_used": bool(trainer.model_accepts_loss_kwargs),
        "compute_loss_func": trainer.compute_loss_func is not None,
        "current_gradient_accumulation_steps": int(trainer.current_gradient_accumulation_steps),
        "args_gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
        "args_max_grad_norm": float(trainer.args.max_grad_norm),
        "args_average_tokens_across_devices": bool(trainer.args.average_tokens_across_devices),
        "loss_kwargs": trainer._ga_audit_kwargs,
        "pre": observed["pre"],
        "post": observed["post"],
        "delta": delta,
    }


def serializable(case: dict, reference: dict) -> dict:
    return {
        key: case[key]
        for key in (
            "model_accepts_loss_kwargs_before_override",
            "model_accepts_loss_kwargs_used",
            "compute_loss_func",
            "current_gradient_accumulation_steps",
            "args_gradient_accumulation_steps",
            "args_max_grad_norm",
            "args_average_tokens_across_devices",
            "loss_kwargs",
        )
    } | {
        "preclip": metrics(case["pre"], reference["pre"]),
        "postclip": metrics(case["post"], reference["post"]),
        "adamw_delta": metrics(case["delta"], reference["delta"]),
    }


def main():
    seed_model = ToyLM()
    initial = {name: p.detach().clone() for name, p in seed_model.named_parameters()}
    reference = manual_reference(initial, max_grad_norm=1.0)
    # None: the exact deployed state with a text-only processor=None. In the
    # formal Transformers 5.3 environment Accelerate performs the GA scaling
    # inside backward when this flag remains true.
    deployed = run_case("deployed", initial, force_loss_kwargs=None)
    # A forced False is an explicitly negative control: it demonstrates why a
    # manual compatibility patch must not be applied to this environment.
    forced_false = run_case("forced-false", initial, force_loss_kwargs=False)
    report = {
        "reference": {key: float(value.norm()) for key, value in reference.items()},
        "deployed": serializable(deployed, reference),
        "forced_false_negative_control": serializable(forced_false, reference),
    }
    print("GA16_AUDIT=" + json.dumps(report, sort_keys=True))
    assert report["deployed"]["model_accepts_loss_kwargs_used"] is True
    assert abs(report["deployed"]["preclip"]["norm_ratio"] - 1.0) < 1e-6
    assert report["deployed"]["preclip"]["cosine"] > 0.999999
    assert abs(report["deployed"]["postclip"]["norm_ratio"] - 1.0) < 1e-6
    assert report["deployed"]["postclip"]["cosine"] > 0.999999
    assert abs(report["deployed"]["adamw_delta"]["norm_ratio"] - 1.0) < 1e-5
    assert report["deployed"]["adamw_delta"]["cosine"] > 0.999999
    assert report["forced_false_negative_control"]["model_accepts_loss_kwargs_used"] is False
    assert abs(report["forced_false_negative_control"]["preclip"]["norm_ratio"] - 1.0 / GA) < 1e-6


if __name__ == "__main__":
    main()
