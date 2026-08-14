"""CPU regression tests for the BETA-SETloss packed-row ratio ablation."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch
from transformers import HfArgumentParser


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
sys.path.insert(0, str(ROOT))
from pack_ratio_sampler import (  # noqa: E402
    PackRatioConfig,
    PackRatioSampler,
    derive_pack_task_id,
    parse_target_ratios,
    target_counts,
    task_stream_window,
)


RATIOS = parse_target_ratios("material=0.20,recommendation=0.45,user_action=0.20,user_chain=0.15")


def test_target_counts_33616() -> None:
    assert target_counts(33616, RATIOS) == (6723, 15127, 6723, 5043)


def test_strict_ratio_parsing() -> None:
    assert parse_target_ratios("material=0.20,recommendation=0.45,user_action=0.20,user_chain=0.15") == RATIOS
    for invalid in (
        "material=0.5,recommendation=0.5,user_action=0.1",
        "material=0.2,recommendation=0.4,user_action=0.2,user_chain=0.1,world=0.1",
        "material=0,recommendation=0.5,user_action=0.25,user_chain=0.25",
        "material=0.2,recommendation=0.4,user_action=0.2,user_chain=0.1",
    ):
        try:
            parse_target_ratios(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid ratio string was accepted: {!r}".format(invalid))


def test_hparams_accept_the_official_fields() -> None:
    from llamafactory.hparams import FinetuningArguments

    (args,) = HfArgumentParser(FinetuningArguments).parse_dict(
        {
            "multitask_pack_ratio_enabled": True,
            "multitask_pack_ratio_targets": "material=0.20,recommendation=0.45,user_action=0.20,user_chain=0.15",
        }
    )
    assert args.multitask_pack_ratio_enabled is True
    assert args.multitask_pack_ratio_targets.startswith("material=0.20")


def test_ratio_exactness_and_plan_length() -> None:
    ids = [task for task in range(4) for _ in range(40)]
    sampler = PackRatioSampler(ids, RATIOS, seed=17)
    plan, tasks = sampler.build_plan()
    assert len(plan) == len(ids) == len(sampler)
    assert Counter(tasks) == Counter({0: 32, 1: 72, 2: 32, 3: 24})


def test_oversample_coverage_before_repeat() -> None:
    stream = task_stream_window([0, 1, 2], base_seed=9, task_id=0, start=0, count=7)
    assert set(stream[:3]) == {0, 1, 2}
    assert set(stream[3:6]) == {0, 1, 2}


def test_undersample_has_no_duplicate() -> None:
    stream = task_stream_window(list(range(100)), base_seed=3, task_id=2, start=0, count=20)
    assert len(set(stream)) == 20


def test_cross_epoch_coverage_advances_stream() -> None:
    epoch0 = task_stream_window(list(range(100)), base_seed=11, task_id=3, start=0, count=20)
    epoch1 = task_stream_window(list(range(100)), base_seed=11, task_id=3, start=20, count=20)
    assert not set(epoch0) & set(epoch1)


def test_determinism_and_epoch_variation() -> None:
    ids = [task for task in range(4) for _ in range(30)]
    first = PackRatioSampler(ids, RATIOS, seed=20260811)
    second = PackRatioSampler(ids, RATIOS, seed=20260811)
    assert first.build_plan() == second.build_plan()
    first.set_epoch(1)
    assert first.build_plan() != second.build_plan()


def test_task_rng_isolation() -> None:
    reference = task_stream_window(list(range(100, 120)), base_seed=5, task_id=1, start=0, count=12)
    # Other task pools are deliberately not involved in task 1's derived seed.
    _ = PackRatioSampler([0] * 3 + [1] * 20 + [2] * 99 + [3] * 5, RATIOS, seed=5).build_plan()
    observed = task_stream_window(list(range(100, 120)), base_seed=5, task_id=1, start=0, count=12)
    assert observed == reference


def test_shift_semantics_and_tie_break() -> None:
    ignore = -100
    # Index 0 must be ignored by causal shift. Valid shifted positions select user_action (task 2).
    assert derive_pack_task_id([7, ignore, 8, 9], [1, 3, 2, 2], ignore) == 2
    # One valid material and one recommendation token: fixed task order selects material.
    assert derive_pack_task_id([ignore, 4, 5], [-1, 0, 1], ignore) == 0
    assert derive_pack_task_id([ignore, 4, 5], [-1, 0, -1], ignore) == 0


def _native_module():
    path = ROOT / "scripts" / "train_native_source_domain_r32_v3.py"
    spec = importlib.util.spec_from_file_location("native_source_pack_ratio_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_off_path_delegates_original_sampler() -> None:
    module = _native_module()
    sentinel = object()
    previous = module._original_get_train_sampler
    previous_config = module.PACK_RATIO_CONFIG
    calls = []
    try:
        module._original_get_train_sampler = lambda trainer, dataset=None: calls.append(dataset) or sentinel
        module.PACK_RATIO_CONFIG = PackRatioConfig(enabled=False, target_ratios=RATIOS)
        dataset = object()
        assert module._get_train_sampler_with_pack_ratio(object(), dataset) is sentinel
        assert calls == [dataset]
    finally:
        module._original_get_train_sampler = previous
        module.PACK_RATIO_CONFIG = previous_config


def test_off_ignores_targets_and_preserves_native_startup() -> None:
    module = _native_module()
    previous_argv = sys.argv[:]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
        handle.write("multitask_pack_ratio_enabled: false\nmultitask_pack_ratio_targets: not-a-ratio\n")
        config_path = handle.name
    try:
        sys.argv = ["native", config_path]
        config = module._load_pack_ratio_config()
        assert config.enabled is False
        assert config.target_ratios == (.25, .25, .25, .25)
    finally:
        sys.argv = previous_argv
        Path(config_path).unlink(missing_ok=True)


def test_collator_strips_pack_task_id() -> None:
    module = _native_module()
    previous = module._original_collator_call
    try:
        module._original_collator_call = lambda _self, features: {"features": features}
        features = [{"pack_task_id": 2, "rec_pu_targets_json": "[]", "labels": [1, 2]}]
        output = module._collate_with_rec_pu_metadata(object(), features)
        assert "pack_task_id" not in output["features"][0]
        assert "pack_task_id" not in output
    finally:
        module._original_collator_call = previous


def test_loss_parity_with_sampler_only_metadata() -> None:
    module = _native_module()
    logits = torch.tensor([[[.1, .2, .3], [.4, .5, .6], [.2, .1, .0]]], dtype=torch.float64)
    labels = torch.tensor([[-100, 1, 2]])
    weights = torch.ones_like(labels, dtype=torch.float64)
    sample_ids = torch.tensor([[0, 0, 0]])
    task_ids = torch.tensor([[0, 0, 0]])
    domain = torch.ones_like(labels, dtype=torch.float64)
    before, _ = module.compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=sample_ids,
        sample_task_ids=task_ids, sample_domain_weights=domain,
        rec_pu_targets=None, rec_pu_config=module.RecPUConfig(False, .05), sid_component_vocab=None,
    )
    pack_task_id = derive_pack_task_id(labels[0].tolist(), task_ids[0].tolist(), -100)
    assert pack_task_id == 0
    after, _ = module.compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=sample_ids,
        sample_task_ids=task_ids, sample_domain_weights=domain,
        rec_pu_targets=None, rec_pu_config=module.RecPUConfig(False, .05), sid_component_vocab=None,
    )
    assert torch.equal(before, after)


if __name__ == "__main__":
    test_target_counts_33616()
    test_strict_ratio_parsing()
    test_hparams_accept_the_official_fields()
    test_ratio_exactness_and_plan_length()
    test_oversample_coverage_before_repeat()
    test_undersample_has_no_duplicate()
    test_cross_epoch_coverage_advances_stream()
    test_determinism_and_epoch_variation()
    test_task_rng_isolation()
    test_shift_semantics_and_tie_break()
    test_off_path_delegates_original_sampler()
    test_off_ignores_targets_and_preserves_native_startup()
    test_collator_strips_pack_task_id()
    test_loss_parity_with_sampler_only_metadata()
    print("PACK_RATIO_UNIT_TESTS=PASS")
