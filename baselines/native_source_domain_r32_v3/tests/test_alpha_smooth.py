"""CPU regressions for the Epoch2-only Alpha final-SID smoothing ablation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from rec_pu.alpha_recommendation_monitor import collect_alpha_recommendation_monitor
from rec_pu.alpha_validation_monitor import AlphaValidationConfig, AlphaValidationRunner, collect_alpha_validation_extras, validation_metric_parity
from rec_pu.recommendation_pu_phase2 import RecPUPackedTarget, TokenPrefixPositiveSets
from rec_pu.sid8_rec_pu_integration import (
    AlphaSmoothConfig,
    RecPUConfig,
    SIDComponentVocab,
    compute_native_sid8_loss,
    final_sid_targets_required,
)


VOCAB_SIZE = 16
COMPONENTS = SIDComponentVocab(a=(1, 2, 3), b=(4, 5, 6), c=(7, 8, 9))


def _target(
    route: str, start: int, segment_index: int = 0, *, segment_start: int | None = None, segment_end: int | None = None
) -> RecPUPackedTarget:
    return RecPUPackedTarget(
        segment_index=segment_index,
        segment_start=start - 1 if segment_start is None else segment_start,
        segment_end=start + 3 if segment_end is None else segment_end,
        a_label_position=start,
        b_label_position=start + 1,
        c_label_position=start + 2,
        a_logit_position=start - 1,
        b_logit_position=start,
        c_logit_position=start + 1,
        positives=TokenPrefixPositiveSets(a=(1,), b=(4,), c=(7,)),
        source_segment=route,
    )


def _targets():
    return [[
        _target("recommendation_cot", 5, segment_start=1, segment_end=8),
        _target("recommendation_nocot", 9, 1),
    ]]


def _inputs():
    # First segment: a visible CoT SID (1/4/7) then final A/B/C (1/4/7).
    # Second segment: NoThink final A/B/C. Third segment is a user task with
    # SID-shaped tokens but no locator target, proving it cannot be smoothed.
    labels = torch.tensor([[-100, 1, 4, 7, 10, 1, 4, 7, 11, 1, 4, 7, 0, 2, 5, 8, -100]], dtype=torch.long)
    weights = torch.where(labels == -100, torch.zeros_like(labels, dtype=torch.float32), torch.ones_like(labels, dtype=torch.float32))
    for position in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15):
        weights[0, position] = 8.0
    sample_ids = torch.tensor([[-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1]], dtype=torch.long)
    tasks = torch.tensor([[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, -1]], dtype=torch.long)
    domains = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3, 0.0]])
    torch.manual_seed(43)
    logits = torch.randn((1, labels.size(1), VOCAB_SIZE), dtype=torch.float64, requires_grad=True)
    return logits, labels, weights, sample_ids, tasks, domains


def _run(logits, *, config=AlphaSmoothConfig(), active=False, targets=None):
    values = _inputs()[1:]
    return compute_native_sid8_loss(
        logits=logits,
        labels=values[0],
        loss_weights=values[1],
        sample_ids=values[2],
        sample_task_ids=values[3],
        sample_domain_weights=values[4],
        rec_pu_targets=targets if targets is not None else _targets(),
        rec_pu_config=RecPUConfig(False, 0.05),
        alpha_smooth_config=config,
        alpha_smooth_active=active,
        sid_component_vocab=COMPONENTS,
    )


def _grad(loss, logits):
    return torch.autograd.grad(loss, logits)[0]


def _load_training_script_with_stubs():
    """Load the real launcher loss path without installing LLaMA-Factory."""

    module_names = (
        "transformers",
        "llamafactory", "llamafactory.data", "llamafactory.data.collator", "llamafactory.data.converter",
        "llamafactory.data.processor", "llamafactory.data.processor.processor_utils",
        "llamafactory.data.processor.supervised", "llamafactory.extras", "llamafactory.extras.constants",
        "llamafactory.train", "llamafactory.train.sft", "llamafactory.train.sft.trainer",
        "llamafactory.train.tuner", "llamafactory.model", "llamafactory.model.model_utils",
        "llamafactory.model.model_utils.checkpointing",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    def module(name):
        value = types.ModuleType(name)
        value.__path__ = []
        sys.modules[name] = value
        return value

    transformers = module("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    module("llamafactory")
    module("llamafactory.data")
    collator = module("llamafactory.data.collator")
    converter = module("llamafactory.data.converter")
    module("llamafactory.data.processor")
    processor_utils = module("llamafactory.data.processor.processor_utils")
    supervised = module("llamafactory.data.processor.supervised")
    module("llamafactory.extras")
    constants = module("llamafactory.extras.constants")
    module("llamafactory.train")
    module("llamafactory.train.sft")
    trainer = module("llamafactory.train.sft.trainer")
    tuner = module("llamafactory.train.tuner")
    module("llamafactory.model")
    module("llamafactory.model.model_utils")
    checkpointing = module("llamafactory.model.model_utils.checkpointing")

    class DummyCollator:
        @staticmethod
        def _unpad_packed_features(*_):
            return None

        def __call__(self, features):
            return features

    class DummyConverter:
        def __call__(self, example):
            return example

    class DummyProcessor:
        def preprocess_dataset(self, *_, **__):
            return {}

    class DummyTrainer:
        def __init__(self, *_, **__):
            pass

        def _get_train_sampler(self, *_, **__):
            return None

        def log(self, logs, *_, **__):
            return logs

    collator.SFTDataCollatorWith4DAttentionMask = DummyCollator
    converter.AlpacaDatasetConverter = DummyConverter
    processor_utils.greedy_knapsack = lambda lengths, _: [[length] for length in lengths]
    supervised.MAX_SU_SEQ_IDX = -1
    supervised.PackingParams = type("PackingParams", (), {})
    supervised.PackedSupervisedDatasetProcessor = DummyProcessor
    constants.IGNORE_INDEX = -100
    trainer.CustomSeq2SeqTrainer = DummyTrainer
    tuner.run_exp = lambda **_: None
    checkpointing.get_custom_gradient_checkpointing_func = lambda value: value

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_native_source_domain_r32_v3.py"
    spec = importlib.util.spec_from_file_location("alpha_smooth_training_path_test", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(loaded)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return loaded


def test_disabled_epsilon_zero_and_epoch1_have_exact_native_parity():
    base_logits = _inputs()[0]
    baseline, baseline_details = _run(base_logits, targets=None)
    baseline_grad = _grad(baseline, base_logits)
    for config, active in (
        (AlphaSmoothConfig(enabled=False), False),
        (AlphaSmoothConfig(enabled=True, epsilon=0.0), True),
        (AlphaSmoothConfig(enabled=True), False),
    ):
        logits = _inputs()[0]
        loss, details = _run(logits, config=config, active=active)
        assert torch.equal(loss, baseline)
        assert torch.equal(_grad(loss, logits), baseline_grad)
        assert torch.equal(details.contributions, baseline_details.contributions)


def test_epoch_boundary_is_stateless_and_resume_safe():
    config = AlphaSmoothConfig(enabled=True, epsilon=0.05, start_epoch=1.0)
    assert not config.active_for_epoch(None)
    assert not config.active_for_epoch(0.999999)
    assert config.active_for_epoch(1.0)
    # A freshly constructed config receives the restored checkpoint epoch.
    assert AlphaSmoothConfig(enabled=True, epsilon=0.05, start_epoch=1.0).active_for_epoch(1.375)


def test_rec_pu_off_alphasmooth_still_requires_and_changes_final_targets():
    assert final_sid_targets_required(
        rec_pu_enabled=False, alpha_monitor_enabled=False, alpha_smooth_enabled=True,
    )
    assert not final_sid_targets_required(
        rec_pu_enabled=False, alpha_monitor_enabled=False, alpha_smooth_enabled=False,
    )
    baseline_logits = _inputs()[0]
    baseline_loss, _ = _run(baseline_logits, config=AlphaSmoothConfig(enabled=True), active=False)
    smooth_logits = _inputs()[0]
    smooth_loss, details = _run(smooth_logits, config=AlphaSmoothConfig(enabled=True), active=True)
    assert details.alpha_smooth_sum_count[2, 1].item() == 6
    assert not torch.equal(smooth_loss, baseline_loss)
    assert not torch.equal(_grad(smooth_loss, smooth_logits), _grad(baseline_loss, baseline_logits))


def test_real_training_target_path_survives_with_rec_pu_and_monitor_off():
    script = _load_training_script_with_stubs()
    script.REC_PU_CONFIG = RecPUConfig(False, 0.05)
    script.REC_CANDIDATE_METRICS_ENABLED = False
    script.ALPHA_MONITOR_CONFIG = types.SimpleNamespace(enabled=False)
    script.ALPHA_SMOOTH_CONFIG = AlphaSmoothConfig(enabled=True)
    assert script._rec_targets_required()


def test_active_alphasmooth_fails_closed_when_target_metadata_is_absent():
    logits, labels, weights, sample_ids, tasks, domains = _inputs()
    try:
        compute_native_sid8_loss(
            logits=logits,
            labels=labels,
            loss_weights=weights,
            sample_ids=sample_ids,
            sample_task_ids=tasks,
            sample_domain_weights=domains,
            rec_pu_targets=None,
            rec_pu_config=RecPUConfig(False, 0.05),
            alpha_smooth_config=AlphaSmoothConfig(enabled=True),
            alpha_smooth_active=True,
            sid_component_vocab=COMPONENTS,
        )
    except ValueError as error:
        assert "rec_pu_targets is absent" in str(error)
    else:
        raise AssertionError("Active AlphaSmooth must not silently run without packed targets")


def test_transformers53_microbatch_epoch_contract():
    # Transformers 5.3 updates state.epoch only after an optimizer step as
    # epoch + (step + 1) / steps_in_epoch. compute_loss therefore sees 1.0
    # throughout Epoch2, including the first microbatch and an Epoch2 resume.
    config = AlphaSmoothConfig(enabled=True, epsilon=0.05, start_epoch=1.0)
    # 33,810 global packs are split across four DDP ranks; the largest local
    # loader has 8,453 batches.  The final GA window has five microbatches.
    steps_in_epoch, gradient_accumulation = 8_453, 16
    assert steps_in_epoch % gradient_accumulation == 5
    state_epoch = 0.0
    for step in range(steps_in_epoch):
        # compute_loss runs before the optimizer state update for every
        # microbatch, including all five batches in the final GA window.
        assert not config.active_for_epoch(state_epoch)
        if (step + 1) % gradient_accumulation == 0 or step + 1 == steps_in_epoch:
            state_epoch = (step + 1) / steps_in_epoch
    assert state_epoch == 1.0
    epoch1_state_after_last_step = state_epoch
    assert config.active_for_epoch(epoch1_state_after_last_step)
    assert config.active_for_epoch(1.0)  # Epoch2 first microbatch / epoch-end checkpoint resume.


def test_formula_and_gradient_match_dense_full_vocab_reference():
    epsilon = 0.05
    level_ids = torch.tensor(COMPONENTS.a)
    gold = torch.tensor([1])
    logits = torch.tensor([[0.2, 1.3, -0.4, 0.8, -0.1, 0.5, -0.3, 0.1, 0.0, 0.2, 4.7, -0.8, 0.4, -0.2, 0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    base = F.cross_entropy(logits, gold, reduction="none")
    selected = logits.float().index_select(1, level_ids)
    gold_logit = logits.float().gather(1, gold.unsqueeze(1)).squeeze(1)
    efficient = base.float() + epsilon * (gold_logit - (selected.sum(1) - gold_logit) / (len(level_ids) - 1))
    dense_q = torch.zeros_like(logits)
    dense_q[0, gold.item()] = 1.0 - epsilon
    dense_q[0, level_ids[level_ids != gold.item()]] = epsilon / (len(level_ids) - 1)
    dense = -(dense_q * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    assert torch.allclose(efficient.double(), dense, rtol=0, atol=2e-7)
    efficient_grad = _grad(efficient.sum(), logits)
    dense_logits = logits.detach().clone().requires_grad_(True)
    dense_q = dense_q.detach()
    dense_grad = _grad((-(dense_q * F.log_softmax(dense_logits, dim=-1)).sum(dim=-1)).sum(), dense_logits)
    assert torch.allclose(efficient_grad, dense_grad, rtol=0, atol=2e-7)
    # A non-SID logit dominates the denominator, which would be invisible to
    # a mistaken local softmax over A-level ids.
    local = -((torch.tensor([1.0 - epsilon, epsilon / 2, epsilon / 2], dtype=torch.float64)) * F.log_softmax(logits[:, level_ids], dim=-1)[0]).sum()
    assert abs(dense.item() - local.item()) > 1.0


def test_epsilon_point05_gradient_sanity_cases():
    """Print three intuitive full-vocabulary regularization cases for audit."""

    epsilon = 0.05
    level_ids = torch.tensor(COMPONENTS.a)
    gold = torch.tensor([1])
    observed = []
    for name, gold_value in (("near", 0.0), ("high", 5.0), ("very_high", 12.0)):
        logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            logits[0, gold.item()] = gold_value
        raw = F.cross_entropy(logits, gold, reduction="none")
        selected = logits.float().index_select(1, level_ids)
        gold_logit = logits.float().gather(1, gold.unsqueeze(1)).squeeze(1)
        smoothed = raw.float() + epsilon * (
            gold_logit - (selected.sum(1) - gold_logit) / (len(level_ids) - 1)
        )
        gradient = _grad(smoothed.sum(), logits)[0]
        observed.append((name, raw.item(), smoothed.item(), (smoothed - raw).item(), gradient[gold].item(), gradient[2].item()))
    for name, raw, smoothed, delta, gold_grad, other_grad in observed:
        print(
            f"ALPHASMOOTH_SANITY {name} raw={raw:.6f} smoothed={smoothed:.6f} "
            f"delta={delta:.6f} gold_grad={gold_grad:.6f} same_level_other_grad={other_grad:.6f}"
        )
    assert abs(observed[0][3]) < 1e-6
    assert 0.0 < observed[1][3] < observed[2][3]
    assert observed[1][5] < 0.0 and observed[2][5] < 0.0
    assert abs(observed[2][3]) < 1.0


def test_only_cot_and_nothink_final_abc_change_and_sid8_denominator_stay_fixed():
    base_logits = _inputs()[0]
    base_loss, base = _run(base_logits)
    labels = _inputs()[1]
    expected_raw_ce = F.cross_entropy(
        base_logits[:, :-1].reshape(-1, VOCAB_SIZE), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none"
    ).view_as(labels[:, 1:])
    assert torch.equal(base.base_per_token_ce, expected_raw_ce)
    smooth_logits = _inputs()[0]
    smooth_loss, smooth = _run(smooth_logits, config=AlphaSmoothConfig(enabled=True), active=True)
    assert not torch.equal(base_loss, smooth_loss)
    assert torch.equal(base.base_per_token_ce, smooth.base_per_token_ce)
    assert torch.equal(base.denominator, smooth.denominator)
    assert torch.equal(base.sample_weight_mass, smooth.sample_weight_mass)
    changed = set(smooth.changed_positions)
    assert changed == {(0, 4), (0, 5), (0, 6), (0, 8), (0, 9), (0, 10)}
    for position in range(base.contributions.size(1)):
        if (0, position) not in changed:
            assert torch.equal(base.contributions[0, position], smooth.contributions[0, position])
    # Causal positions 3/7 predict the final COT/NoThink domain tokens.
    assert torch.equal(base.contributions[0, torch.tensor([3, 7])], smooth.contributions[0, torch.tensor([3, 7])])
    assert torch.equal(base.contributions[0, 11:15], smooth.contributions[0, 11:15])
    # The raw labels at changed shifted positions retain SID8 weights.
    shifted_weights = _inputs()[2][:, 1:]
    assert torch.equal(shifted_weights[0, torch.tensor([4, 5, 6, 8, 9, 10])], torch.full((6,), 8.0))
    assert smooth.alpha_smooth_sum_count[2, 1].item() == 6
    assert smooth.alpha_smooth_sum_count[3, 1].item() == 6
    sid8 = torch.full((6,), 8.0, dtype=smooth.contributions.dtype)
    actual_smoothed_ce = smooth.contributions[0, torch.tensor(sorted(position for _, position in changed))] / sid8
    actual_delta_ce = (
        smooth.contributions[0, torch.tensor(sorted(position for _, position in changed))]
        - base.contributions[0, torch.tensor(sorted(position for _, position in changed))]
    ) / sid8
    assert torch.allclose(smooth.alpha_smooth_sum_count[2, 0] / 6, actual_smoothed_ce.mean().double())
    assert torch.allclose(smooth.alpha_smooth_sum_count[3, 0] / 6, actual_delta_ce.mean().double())


def test_material_and_both_user_tasks_are_untouched_in_a_mixed_pack():
    labels = torch.tensor([[-100, 0, 2, 5, 10, 1, 4, 7, 0, 2, 5, 0, 3, 6]], dtype=torch.long)
    weights = torch.where(labels == -100, torch.zeros_like(labels, dtype=torch.float32), torch.ones_like(labels, dtype=torch.float32))
    for position in (2, 3, 4, 5, 6, 7, 9, 10, 12, 13):
        weights[0, position] = 8.0
    sample_ids = torch.tensor([[-1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.long)
    task_ids = torch.tensor([[-1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.long)
    domain_weights = torch.tensor([[0.0, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    target = [[_target("recommendation_cot", 5, segment_start=4, segment_end=8)]]
    torch.manual_seed(91)
    base_logits = torch.randn((1, labels.size(1), VOCAB_SIZE), dtype=torch.float64, requires_grad=True)
    common = dict(
        labels=labels, loss_weights=weights, sample_ids=sample_ids, sample_task_ids=task_ids,
        sample_domain_weights=domain_weights, rec_pu_targets=target, rec_pu_config=RecPUConfig(False, 0.05),
        sid_component_vocab=COMPONENTS,
    )
    _, base = compute_native_sid8_loss(logits=base_logits, **common)
    smooth_logits = base_logits.detach().clone().requires_grad_(True)
    _, smooth = compute_native_sid8_loss(
        logits=smooth_logits, alpha_smooth_config=AlphaSmoothConfig(enabled=True), alpha_smooth_active=True, **common
    )
    assert set(smooth.changed_positions) == {(0, 4), (0, 5), (0, 6)}
    for position in range(base.contributions.size(1)):
        if position not in {4, 5, 6}:
            assert torch.equal(base.contributions[0, position], smooth.contributions[0, position])


def test_alpha_smooth_diagnostics_are_safe_without_recommendation_targets():
    logits = _inputs()[0]
    loss, details = _run(logits, config=AlphaSmoothConfig(enabled=True), active=True, targets=[[]])
    assert torch.isfinite(loss)
    assert details.alpha_smooth_sum_count[0].tolist() == [1.0, 1.0]
    assert details.alpha_smooth_sum_count[1].tolist() == [0.05, 1.0]
    assert details.alpha_smooth_sum_count[2, 1].item() == 0.0
    assert details.alpha_smooth_sum_count[3, 1].item() == 0.0


def test_monitor_and_validation_continue_to_use_raw_base_ce():
    logits = _inputs()[0]
    _, details = _run(logits, config=AlphaSmoothConfig(enabled=True), active=True)
    labels = _inputs()[1]
    targets = _targets()
    train = collect_alpha_recommendation_monitor(
        base_per_token_ce=details.base_per_token_ce.detach(), logits=logits.detach(), labels=labels,
        rec_targets=targets, sid_component_vocab=COMPONENTS, collect_tf=True,
    )
    validation = validation_metric_parity(
        base_per_token_ce=details.base_per_token_ce.detach(), logits=logits.detach(), labels=labels,
        rec_targets=targets, sid_component_vocab=COMPONENTS,
    )
    assert torch.equal(train, validation)
    assert not torch.equal(details.base_per_token_ce, details.contributions / _inputs()[2][:, 1:])


def test_validation_explicit_off_isolation_under_epoch2_training_config():
    logits = _inputs()[0]
    labels, weights, sample_ids, tasks, domains = _inputs()[1:]
    targets = _targets()
    baseline_loss, baseline = compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=sample_ids,
        sample_task_ids=tasks, sample_domain_weights=domains, rec_pu_targets=targets,
        rec_pu_config=RecPUConfig(False, 0.05), alpha_smooth_config=AlphaSmoothConfig(enabled=False),
        alpha_smooth_active=False, sid_component_vocab=COMPONENTS,
    )
    # This represents an Epoch2 trainer whose global AlphaSmooth config is on;
    # validation still passes a separate explicit-off config to the loss path.
    validation_loss, validation = compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=sample_ids,
        sample_task_ids=tasks, sample_domain_weights=domains, rec_pu_targets=targets,
        rec_pu_config=RecPUConfig(False, 0.05), alpha_smooth_config=AlphaSmoothConfig(enabled=False),
        alpha_smooth_active=False, sid_component_vocab=COMPONENTS,
    )
    assert torch.equal(validation_loss, baseline_loss)
    assert torch.equal(validation.base_per_token_ce, baseline.base_per_token_ce)
    # In this fixture the final COT A label is at index 5, so index 4 is
    # its domain slot for the validation helper's causal target geometry.
    domain_map = {10: "video", 11: "prod"}
    assert torch.equal(
        collect_alpha_validation_extras(
            base_per_token_ce=validation.base_per_token_ce, logits=logits, labels=labels,
            rec_targets=targets, sid_component_vocab=COMPONENTS, domain_token_map=domain_map,
        ),
        collect_alpha_validation_extras(
            base_per_token_ce=baseline.base_per_token_ce, logits=logits, labels=labels,
            rec_targets=targets, sid_component_vocab=COMPONENTS, domain_token_map=domain_map,
        ),
    )


def test_rec_pu_conflict_fails_fast():
    logits = _inputs()[0]
    try:
        compute_native_sid8_loss(
            logits=logits, labels=_inputs()[1], loss_weights=_inputs()[2], sample_ids=_inputs()[3],
            sample_task_ids=_inputs()[4], sample_domain_weights=_inputs()[5],
            rec_pu_config=RecPUConfig(True, 0.05), alpha_smooth_config=AlphaSmoothConfig(enabled=True),
            sid_component_vocab=COMPONENTS,
        )
    except ValueError as error:
        assert "cannot both" in str(error)
    else:
        raise AssertionError("REC-PU and AlphaSmooth must fail fast")


def test_real_training_compute_loss_has_one_forward_in_all_phases():
    script = _load_training_script_with_stubs()

    class Tokenizer:
        def get_vocab(self):
            return {
                "<s_a_1>": 1, "<s_a_2>": 2, "<s_a_3>": 3,
                "<s_b_1>": 4, "<s_b_2>": 5, "<s_b_3>": 6,
                "<s_c_1>": 7, "<s_c_2>": 8, "<s_c_3>": 9,
            }

    class CountingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, **_):
            self.calls += 1
            return types.SimpleNamespace(logits=_inputs()[0])

    for config, active in (
        (AlphaSmoothConfig(enabled=False), False),
        (AlphaSmoothConfig(enabled=True), False),
        (AlphaSmoothConfig(enabled=True), True),
    ):
        script.REC_PU_CONFIG = RecPUConfig(False, 0.05)
        script.REC_CANDIDATE_METRICS_ENABLED = False
        script.ALPHA_MONITOR_CONFIG = types.SimpleNamespace(enabled=False, train_tf_enabled=False, train_tf_interval=50)
        script.ALPHA_VALIDATION_CONFIG = types.SimpleNamespace(enabled=False, probe_enabled=False)
        script.ALPHA_SMOOTH_CONFIG = config
        model = CountingModel()
        epoch = 1.0 if active else 0.999
        trainer = types.SimpleNamespace(
            state=types.SimpleNamespace(global_step=0, epoch=epoch), processing_class=Tokenizer(),
        )
        _, labels, weights, sample_ids, tasks, domains = _inputs()
        inputs = {
            "input_ids": torch.zeros((1, labels.size(1)), dtype=torch.long),
            "labels": labels.clone(), "loss_weights": weights.clone(), "sample_ids": sample_ids.clone(),
            "sample_task_ids": tasks.clone(), "sample_domain_weights": domains.clone(),
            "rec_pu_targets": _targets(),
        }
        loss = script._compute_source_weighted_loss(trainer, model, inputs)
        assert model.calls == 1
        assert torch.isfinite(loss)


def test_real_validation_runner_ignores_epoch2_training_state():
    class Tokenizer:
        tokens = {
            "<|video_begin|>": 10, "<|prod_begin|>": 11, "<|ad_begin|>": 12, "<|living_begin|>": 13,
        }

        def convert_tokens_to_ids(self, token):
            return self.tokens.get(token, -1)

        def convert_ids_to_tokens(self, token_id):
            return next((token for token, value in self.tokens.items() if value == token_id), "<unk>")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.register_buffer("fixed_logits", torch.randn((1, 6, VOCAB_SIZE), dtype=torch.float32))
            self.calls = 0

        def forward(self, **_):
            self.calls += 1
            return types.SimpleNamespace(logits=self.fixed_logits)

    labels = torch.tensor([[-100, 0, 10, 1, 4, 7]], dtype=torch.long)
    row = {
        "input_ids": torch.zeros((6,), dtype=torch.long), "labels": labels[0],
        "loss_weights": torch.tensor([0.0, 1.0, 1.0, 8.0, 8.0, 8.0]),
        "sample_ids": torch.zeros_like(labels[0]), "sample_task_ids": torch.ones_like(labels[0]),
        "sample_domain_weights": torch.ones_like(labels[0], dtype=torch.float32),
        "rec_pu_targets": [[_target("recommendation_nocot", 3)]],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "split_audit.json").write_text(
            json.dumps({"recommendation": {"domain_rows": {domain: {"original": 1} for domain in ("video", "prod", "ad", "living")}}}),
            encoding="utf-8",
        )
        (root / "dev.jsonl").write_text("{}\n", encoding="utf-8")
        config = AlphaValidationConfig(enabled=True, split_dir=str(root), dev_cache="unused", probe_cache="unused")
        model = Model()
        metrics = []
        for trainer_epoch in (0.25, 1.8):
            runner = AlphaValidationRunner(config)
            runner._datasets["probe"] = [row]
            def collate(features):
                return {
                    key: value.unsqueeze(0) if torch.is_tensor(value) else value
                    for key, value in features[0].items()
                }
            trainer = types.SimpleNamespace(
                data_collator=collate, processing_class=Tokenizer(),
                _rec_pu_component_vocab=COMPONENTS, state=types.SimpleNamespace(epoch=trainer_epoch),
            )
            metrics.append(runner.run(trainer, model, kind="probe", global_step=1, epoch=trainer_epoch)["metrics"])
        assert model.calls == 2
        assert metrics[0] == metrics[1]


def main():
    tests = [
        test_disabled_epsilon_zero_and_epoch1_have_exact_native_parity,
        test_epoch_boundary_is_stateless_and_resume_safe,
        test_rec_pu_off_alphasmooth_still_requires_and_changes_final_targets,
        test_real_training_target_path_survives_with_rec_pu_and_monitor_off,
        test_active_alphasmooth_fails_closed_when_target_metadata_is_absent,
        test_transformers53_microbatch_epoch_contract,
        test_formula_and_gradient_match_dense_full_vocab_reference,
        test_epsilon_point05_gradient_sanity_cases,
        test_only_cot_and_nothink_final_abc_change_and_sid8_denominator_stay_fixed,
        test_material_and_both_user_tasks_are_untouched_in_a_mixed_pack,
        test_alpha_smooth_diagnostics_are_safe_without_recommendation_targets,
        test_monitor_and_validation_continue_to_use_raw_base_ce,
        test_validation_explicit_off_isolation_under_epoch2_training_config,
        test_rec_pu_conflict_fails_fast,
        test_real_training_compute_loss_has_one_forward_in_all_phases,
        test_real_validation_runner_ignores_epoch2_training_state,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
