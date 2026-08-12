"""CPU regressions for the alpha Phase-3 sidecar validation primitives."""

from __future__ import annotations

import copy
import random

import torch

from rec_pu.alpha_recommendation_monitor import ALL_METRIC_NAMES, collect_alpha_recommendation_monitor
from rec_pu.alpha_validation_monitor import (
    DOMAINS,
    EXTRA_INDEX,
    ExactShardSampler,
    collect_alpha_validation_extras,
    exact_shard_indices,
    preserved_eval_state,
    source_domain_weights,
    validation_metric_parity,
)
from rec_pu.sid8_rec_pu_integration import SIDComponentVocab


VOCAB = SIDComponentVocab(a=(1, 2, 7), b=(3, 4, 8), c=(5, 6, 9))


def target(route: str, start: int, end: int, a: int, b: int, c: int) -> dict:
    return {
        "segment_index": 0, "segment_start": start, "segment_end": end,
        "a_label_position": a, "b_label_position": b, "c_label_position": c,
        "a_logit_position": a - 1, "b_logit_position": b - 1, "c_logit_position": c - 1,
        "positives": {"a": (1,), "b": (3,), "c": (5,)}, "source_segment": route,
    }


def fixture():
    # domain at label position 2, then final a/b/c labels at 3/4/5.
    labels = torch.tensor([[-100, 0, 11, 1, 3, 5, -100]], dtype=torch.long)
    base_ce = torch.tensor([[0.2, 0.3, 1.0, 2.0, 3.0, 0.5]], dtype=torch.float32)
    logits = torch.zeros((1, 7, 16), dtype=torch.float32)
    logits[0, 2, 1], logits[0, 3, 3], logits[0, 4, 5] = 9.0, 9.0, 9.0
    targets = [[target("recommendation_nocot", 1, 6, 3, 4, 5)]]
    return labels, base_ce, logits, targets


def test_train_eval_primary_metric_parity():
    labels, ce, logits, targets = fixture()
    train = collect_alpha_recommendation_monitor(
        base_per_token_ce=ce, logits=logits, labels=labels, rec_targets=targets, sid_component_vocab=VOCAB, collect_tf=True
    )
    validation = validation_metric_parity(
        base_per_token_ce=ce, logits=logits, labels=labels, rec_targets=targets, sid_component_vocab=VOCAB
    )
    assert torch.equal(train, validation)
    extras = collect_alpha_validation_extras(
        base_per_token_ce=ce, logits=logits, labels=labels, rec_targets=targets, sid_component_vocab=VOCAB,
        domain_token_map={11: "video", 12: "prod", 13: "ad", 14: "living"},
    )
    assert abs(extras[EXTRA_INDEX["val_rec_gold_prob_a"], 0].item() - torch.exp(torch.tensor(-1.0)).item()) < 1e-7
    assert extras[EXTRA_INDEX["val_rec_video_gold_sid_ce"], 1].item() == 3
    assert extras[EXTRA_INDEX["val_rec_video_tf_chain"], 0].item() == 1
    assert extras[EXTRA_INDEX["val_rec_segments"], 0].item() == 1


def test_exact_once_sharding_has_no_padding_duplicates():
    shards = [exact_shard_indices(733, rank, 4) for rank in range(4)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(733))
    assert len(flattened) == len(set(flattened)) == 733
    assert [len(shard) for shard in shards] == [184, 183, 183, 183]
    assert list(ExactShardSampler(7, 2, 4)) == [2, 6]


def test_eval_preserves_grad_optimizer_scheduler_rng_and_train_mode():
    torch.manual_seed(123)
    random.seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(p=0.5))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    warmup = model(torch.ones(1, 3)).sum()
    warmup.backward()
    grads_before = [parameter.grad.detach().clone() for parameter in model.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    cpu_before, python_before = torch.get_rng_state().clone(), random.getstate()
    with preserved_eval_state(model), torch.inference_mode():
        _ = model(torch.randn(2, 3))
        _ = torch.rand(3)
        _ = random.random()
    assert model.training
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert random.getstate() == python_before  # Python RNG is untouched by the evaluator.
    for before, parameter in zip(grads_before, model.parameters()):
        assert torch.equal(before, parameter.grad)
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before


def test_source_domain_reweighting_is_not_hardcoded(tmp_path):
    audit = {"recommendation": {"domain_rows": {domain: {"original": index + 1} for index, domain in enumerate(DOMAINS)}}}
    (tmp_path / "split_audit.json").write_text(__import__("json").dumps(audit), encoding="utf-8")
    weights = source_domain_weights(tmp_path)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["living"] == 4 / 10


def test_rolling_train_metric_shape_matches_only_a_to_f():
    from rec_pu.alpha_validation_monitor import ROLLING_TRAIN_METRIC_NAMES, VALIDATION_METRIC_NAMES
    assert len(ROLLING_TRAIN_METRIC_NAMES) == 6
    assert ROLLING_TRAIN_METRIC_NAMES[-1] == "f_rec_gold_c_ce"
    assert VALIDATION_METRIC_NAMES[0] == "va_rec_cot_body_ce"
    assert VALIDATION_METRIC_NAMES[1] == "vb_rec_cot_gold_sid_ce"


def test_main_log_primary_set_excludes_secondary_val_metrics():
    from rec_pu.alpha_validation_monitor import VALIDATION_METRIC_NAMES
    assert all(name.startswith(("va_", "vb_", "vc_", "vd_", "ve_", "vf_", "vg_", "vh_", "vi_", "vj_", "vk_", "vl_", "vm_", "vn_", "vo_")) for name in VALIDATION_METRIC_NAMES)
    assert "val_rec_gold_prob_a" not in VALIDATION_METRIC_NAMES


if __name__ == "__main__":
    test_train_eval_primary_metric_parity()
    test_exact_once_sharding_has_no_padding_duplicates()
    test_eval_preserves_grad_optimizer_scheduler_rng_and_train_mode()
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_source_domain_reweighting_is_not_hardcoded(Path(directory))
    test_rolling_train_metric_shape_matches_only_a_to_f()
    test_main_log_primary_set_excludes_secondary_val_metrics()
    print("alpha validation monitor tests: PASS")
