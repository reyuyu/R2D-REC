"""CPU sanity check for the opt-in, no-training-gradient REC-PU probe."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import torch
from torch import nn


SPEC = importlib.util.spec_from_file_location(
    "native_grad_diag", "/data/baselines/native_source_domain_r32_v3/scripts/train_native_source_domain_r32_v3.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
MOD.REC_PU_GRAD_DIAGNOSTICS = True
MOD.REC_PU_CONFIG = MOD.RecPUConfig(rec_pu_enabled=True, rec_pu_unlabeled_sid_grad_scale=0.05)


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = nn.Parameter(torch.randn(2, 2))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj, self.o_proj, self.down_proj = Leaf(), Leaf(), Leaf(), Leaf()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(36)])
    def make_logits(self):
        scalar = sum(parameter.sum() for parameter in self.parameters())
        return torch.randn(1, 5, 8, requires_grad=True) + scalar * 0.01


class Vocab:
    def for_level(self, level):
        return {"a": (1, 2), "b": (3, 4), "c": (5, 6)}[level]


def main():
    torch.manual_seed(1)
    model = Model()
    logits = model.make_logits()
    labels = torch.tensor([[-100, 0, 1, 3, 5]])
    # Four synthetic segment losses, each task represented in the same pack.
    values = logits.sum(dim=(1, 2))
    details = SimpleNamespace(
        sample_numerators=torch.stack([values[0] * (i + 1) for i in range(4)]),
        sample_token_counts=torch.ones(4), sample_domain_weights=torch.ones(4),
        sample_task_ids=torch.tensor([0, 1, 2, 3]),
    )
    target = {
        "segment_index": 1, "segment_start": 0, "segment_end": 5,
        "a_label_position": 2, "b_label_position": 3, "c_label_position": 4,
        "a_logit_position": 1, "b_logit_position": 2, "c_logit_position": 3,
        "positives": {"a": (1,), "b": (3,), "c": (5,)},
    }
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=9))
    MOD._maybe_accumulate_gradient_diagnostics(
        trainer, model=model, logits=logits, labels=labels, rec_pu_targets=[[target]],
        sid_component_vocab=Vocab(), details=details,
    )
    stats = trainer._rec_pu_grad_diag_stats
    print("REC_PU_GRAD_DIAGNOSTICS_PASS", stats[:, 1].tolist())
    assert torch.isfinite(stats).all()
    assert all(count.item() == 1 for count in stats[:, 1])
    assert all(parameter.grad is None for parameter in model.parameters())


if __name__ == "__main__":
    main()
