"""CPU regression for detached REC-PU instability diagnostics."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import torch


SPEC = importlib.util.spec_from_file_location(
    "native_diag", "/data/baselines/native_source_domain_r32_v3/scripts/train_native_source_domain_r32_v3.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
MOD.REC_PU_DIAGNOSTICS = True
MOD.REC_PU_CONFIG = MOD.RecPUConfig(rec_pu_enabled=True, rec_pu_unlabeled_sid_grad_scale=0.05)


class Tokenizer:
    def get_vocab(self):
        return {
            "text": 0, "<s_a_0>": 1, "<s_a_1>": 2, "<s_b_0>": 3,
            "<s_b_1>": 4, "<s_c_0>": 5, "<s_c_1>": 6, "<|prod_begin|>": 7,
        }


class Trainer:
    processing_class = Tokenizer()
    args = SimpleNamespace(logging_steps=1)
    state = SimpleNamespace(global_step=0)


class Vocab:
    ids = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
    def for_level(self, level):
        return self.ids[level]


def main():
    # Unshifted labels: final a/b/c labels at 2/3/4 therefore logit positions 1/2/3.
    labels = torch.tensor([[-100, 0, 1, 3, 5]], dtype=torch.long)
    weights = torch.tensor([[0.0, 1.0, 8.0, 8.0, 8.0]])
    tasks = torch.tensor([[-1, 1, 1, 1, 1]], dtype=torch.long)
    logits = torch.tensor([[[0.0] * 8, [0.2, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.1, 3.0, 0.0, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.1, 0.0, 0.0, 3.0, 0.0, 0.0],
                            [0.0] * 8]], dtype=torch.float32)
    base_ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), reduction="none").view(1, 4)
    details = SimpleNamespace(base_contributions=base_ce * weights[:, 1:], contributions=base_ce * weights[:, 1:])
    target = {
        "segment_index": 0, "segment_start": 0, "segment_end": 5,
        "a_label_position": 2, "b_label_position": 3, "c_label_position": 4,
        "a_logit_position": 1, "b_logit_position": 2, "c_logit_position": 3,
        "positives": {"a": (1,), "b": (3,), "c": (5,)},
    }
    trainer = Trainer()
    MOD._accumulate_rec_pu_diagnostics(
        trainer, logits=logits, labels=labels, loss_weights=weights,
        sample_task_ids=tasks, rec_pu_targets=[[target]], sid_component_vocab=Vocab(), details=details,
    )
    stats = trainer._rec_pu_diag_stats
    index = {name: i for i, name in enumerate(MOD._REC_DIAG_NAMES + MOD._REC_DIAG_LEVEL_NAMES)}
    report = {name: float(stats[index[name], 0] / stats[index[name], 1]) for name in MOD._REC_DIAG_NAMES if stats[index[name], 1]}
    print("REC_PU_DIAGNOSTICS_PASS", report)
    assert report["rec_pos_mass"] > report["rec_u_mass"]
    assert report["rec_positive_top1_acc"] == 1.0
    assert report["rec_gold_top1_acc"] == 1.0
    assert torch.isfinite(stats).all()
    assert trainer._native_sid_share_stats.shape == (4, 2)


if __name__ == "__main__":
    main()
