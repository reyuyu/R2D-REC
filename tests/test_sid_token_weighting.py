import copy
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from llamafactory.data.multitask import TaskPackCollator
from llamafactory.train.sft.sid_token_weighting import (
    SID_WEIGHT_STAT_INDEX,
    SID_WEIGHT_STAT_SIZE,
    SidTokenWeightingController,
    sid_weight_statistics_to_metrics,
)
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer


IGNORE_INDEX = -100


class FakeTokenizer:
    def get_added_vocab(self):
        return {
            "<|video_begin|>": 10,
            "<|prod_begin|>": 11,
            "<s_a_0>": 20,
            "<s_a_1>": 21,
            "<s_b_0>": 30,
            "<s_b_1>": 31,
            "<s_c_0>": 40,
            "<s_c_1>": 41,
        }


def make_args(**overrides):
    values = {"sid_token_weight": 8.0, "sid_text_weight": 1.0}
    values.update(overrides)
    return SimpleNamespace(**values)


def controller(**overrides):
    return SidTokenWeightingController(FakeTokenizer(), make_args(**overrides))


def raw_causal_ce(logits, labels):
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def expected_weighted_ce(logits, labels, sid_ids, sid_weight=8.0, text_weight=1.0):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(IGNORE_INDEX)
    lookup = torch.zeros(shift_logits.shape[-1], dtype=torch.bool)
    lookup[torch.tensor(sorted(sid_ids))] = True
    sid = valid & lookup[shift_labels.clamp_min(0)]
    per = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).reshape_as(shift_labels)
    weights = torch.where(sid, torch.full_like(per, sid_weight), torch.full_like(per, text_weight))
    mass = weights * valid.to(per.dtype)
    return (per * mass).sum() / mass.sum().clamp_min(1.0)


def test_sid_tokens_receive_weight_eight_and_text_tokens_one():
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 64, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 7, 20, 30, 8]])
    result = controller().compute(logits, labels)
    expected = expected_weighted_ce(logits, labels, {20, 21, 30, 31, 40, 41})
    torch.testing.assert_close(result.loss, expected)
    assert result.statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] == 2
    assert result.statistics[SID_WEIGHT_STAT_INDEX["supervised_token_count"]] == 4


def test_ignore_labels_do_not_enter_numerator_or_denominator():
    logits = torch.zeros(1, 5, 64)
    labels = torch.tensor([[IGNORE_INDEX, 20, IGNORE_INDEX, 7, IGNORE_INDEX]])
    result = controller().compute(logits, labels)
    assert result.statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] == 1
    assert result.statistics[SID_WEIGHT_STAT_INDEX["supervised_token_count"]] == 2
    assert result.statistics[SID_WEIGHT_STAT_INDEX["total_weighted_mass"]] == 9


def test_prompt_sid_with_ignore_label_is_not_weighted():
    logits = torch.zeros(1, 4, 64)
    # Token 20 appears in the prompt and the answer. Only the supervised answer
    # occurrence contributes after the causal label shift.
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 20, 7]])
    result = controller().compute(logits, labels)
    assert result.statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] == 1
    assert result.statistics[SID_WEIGHT_STAT_INDEX["supervised_token_count"]] == 2


def test_sid_mask_uses_the_same_causal_shift_as_labels():
    logits = torch.zeros(1, 3, 64)
    labels = torch.tensor([[IGNORE_INDEX, 7, 20]])
    logits[0, 0, 7] = 12.0
    logits[0, 0, 20] = -12.0
    logits[0, 1, 20] = 12.0
    result = controller().compute(logits, labels)
    # The SID target at label position 2 must consume logits position 1.
    metrics = sid_weight_statistics_to_metrics(result.statistics)
    assert metrics["sid_token_ce"] < 0.01
    assert metrics["text_token_ce"] < 0.01


def test_no_sid_sample_matches_original_ce_exactly():
    torch.manual_seed(13)
    logits = torch.randn(1, 5, 64, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 5, 6, 7, 8]])
    result = controller().compute(logits, labels)
    torch.testing.assert_close(result.loss, raw_causal_ce(logits, labels))


def test_all_sid_sample_is_finite_and_has_no_nan():
    logits = torch.full((1, 4, 64), -60000.0, dtype=torch.float16)
    logits[0, 0, 20] = 60000.0
    logits[0, 1, 30] = 60000.0
    logits[0, 2, 40] = 60000.0
    labels = torch.tensor([[IGNORE_INDEX, 20, 30, 40]])
    result = controller().compute(logits, labels)
    assert torch.isfinite(result.loss)
    assert result.statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] == 3


def test_compact_metrics_have_expected_normalized_values():
    logits = torch.zeros(1, 4, 64)
    labels = torch.tensor([[IGNORE_INDEX, 20, 7, 30]])
    metrics = sid_weight_statistics_to_metrics(controller().compute(logits, labels).statistics)
    assert metrics["sid_weighted_token_count"] == 2
    assert metrics["sid_supervised_token_ratio"] == 2 / 3
    assert metrics["sid_effective_weight"] == 17 / 3
    assert metrics["sid_weighted_mass_share"] == 16 / 17
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def packed_sample(sample_id, input_ids, labels):
    return {
        "input_ids": input_ids,
        "labels": labels,
        "task_name": "user",
        "task_id": 1,
        "subtask_name": "action_nocot",
        "subtask_id": 2,
        "sample_id": sample_id,
        "seq_len": len(input_ids),
        "supervised_token_count": sum(value != IGNORE_INDEX for value in labels),
        "sample_metadata": {},
    }


def test_packed_segments_keep_sid_mask_positions_local_to_supervised_labels():
    first = packed_sample(0, [1, 20, 7], [IGNORE_INDEX, 20, 7])
    second = packed_sample(1, [2, 30, 8], [IGNORE_INDEX, 30, 8])
    batch = TaskPackCollator()([first, second])
    result = controller().compute(torch.zeros(1, batch["labels"].shape[1], 64), batch["labels"])
    assert result.statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] == 2
    assert result.statistics[SID_WEIGHT_STAT_INDEX["supervised_token_count"]] == 4


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.linspace(-0.2, 0.2, 64))
        self.forward_count = 0

    def forward(self, input_ids, labels=None, **kwargs):
        self.forward_count += 1
        logits = self.logit_bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        loss = raw_causal_ce(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


class CountingAccelerator:
    def __init__(self):
        self.backward_count = 0

    def backward(self, loss):
        self.backward_count += 1
        loss.backward()


class FixedWeightController:
    def __init__(self, weight):
        self.weight = weight
        self.current_task = None

    def task_weight(self, task_name):
        return self.weight

    def set_current_task(self, task_name):
        self.current_task = task_name

    def clear_current_task(self):
        self.current_task = None


class TrainerHarness:
    _MODEL_INPUT_KEYS = MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS
    compute_task_microbatches = MultiTaskMacroSeq2SeqTrainer.compute_task_microbatches

    def __init__(self, sid_weighting=None, gradient_controller=None, divisor=4):
        self.args = SimpleNamespace(device=torch.device("cpu"), n_gpu=1)
        self.state = SimpleNamespace(global_step=0)
        self.user_action_auxiliary = None
        self.sid_token_weighting = sid_weighting
        self.gradient_controller = gradient_controller
        self._loss_scale_denominator = divisor
        self._action_statistics_offset = 5
        self._sid_weight_statistics_offset = 5
        self._task_stat_size = 5 + (SID_WEIGHT_STAT_SIZE if sid_weighting is not None else 0)
        self.accelerator = CountingAccelerator()

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def microbatch(labels):
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([labels]),
        "subtask_name": "chain_nocot",
        "supervised_token_count": sum(value != IGNORE_INDEX for value in labels),
        "num_segments": 1,
        "action_aux_metadata": [],
    }


def test_switch_off_preserves_original_loss_and_gradient():
    labels = [IGNORE_INDEX, 5, 6, 7, 8]
    original, disabled = TinyCausalModel(), TinyCausalModel()
    disabled.load_state_dict(copy.deepcopy(original.state_dict()))
    original_harness = TrainerHarness(None)
    weighted_harness = TrainerHarness(controller())
    original_total, original_stats = original_harness.compute_task_microbatches(original, "user", [microbatch(labels)], True)
    weighted_total, weighted_stats = weighted_harness.compute_task_microbatches(disabled, "user", [microbatch(labels)], True)
    torch.testing.assert_close(original_total, weighted_total)
    torch.testing.assert_close(original.logit_bias.grad, disabled.logit_bias.grad)
    assert original_stats.shape[0] == 5
    assert weighted_stats.shape[0] == 5 + SID_WEIGHT_STAT_SIZE


def test_weighted_base_loss_is_used_before_gradnorm_weight_and_divisor():
    labels = [IGNORE_INDEX, 20, 7, 30, 8]
    model = TinyCausalModel()
    harness = TrainerHarness(controller(), FixedWeightController(1.5), divisor=4)
    total, stats = harness.compute_task_microbatches(model, "user", [microbatch(labels)], True)
    torch.testing.assert_close(total, stats[0] * 1.5 / 4.0)
    assert harness.gradient_controller.current_task is None


def test_enabled_path_still_uses_one_forward_and_one_backward():
    model = TinyCausalModel()
    harness = TrainerHarness(controller())
    harness.compute_task_microbatches(model, "user", [microbatch([IGNORE_INDEX, 20, 7, 30, 8])], True)
    assert model.forward_count == 1
    assert harness.accelerator.backward_count == 1


def test_action_auxiliary_consumes_weighted_base_ce_without_extra_forward():
    # The existing Action test helpers build a real action_nocot packed batch.
    # Import here so this standalone suite remains narrowly scoped.
    from test_user_action_auxiliary import (
        FakeTokenizer as ActionTokenizer,
    )
    from test_user_action_auxiliary import (
        TinyCausalModel as ActionModel,
    )
    from test_user_action_auxiliary import (
        TrainerHarness as ActionHarness,
    )
    from test_user_action_auxiliary import (
        make_args as make_action_args,
    )
    from test_user_action_auxiliary import (
        make_microbatch as make_action_microbatch,
    )

    from llamafactory.train.sft.user_action_auxiliary import (
        ACTION_STAT_INDEX,
        ACTION_STAT_SIZE,
        UserActionAuxiliaryController,
    )

    auxiliary = UserActionAuxiliaryController(ActionTokenizer(), make_action_args())
    harness = ActionHarness(auxiliary=auxiliary)
    harness.sid_token_weighting = controller()
    harness._action_statistics_offset = 5
    harness._sid_weight_statistics_offset = 5 + ACTION_STAT_SIZE
    harness._task_stat_size = 5 + ACTION_STAT_SIZE + SID_WEIGHT_STAT_SIZE
    model = ActionModel()
    _, statistics = harness.compute_task_microbatches(model, "user", [make_action_microbatch()], True)
    action_offset = harness._action_statistics_offset
    sid_offset = harness._sid_weight_statistics_offset
    assert statistics[action_offset + ACTION_STAT_INDEX["user_base_sum"]] > 0
    assert statistics[action_offset + ACTION_STAT_INDEX["user_aux_sum"]] > 0
    assert statistics[sid_offset + SID_WEIGHT_STAT_INDEX["sid_token_count"]] > 0
    assert model.forward_count == 1
    assert harness.accelerator.backward_count == 1


def test_zero_weight_action_topk_diagnostics_do_not_change_auxiliary_loss():
    from test_user_action_auxiliary import FakeTokenizer, make_args, make_logits, make_sample

    from llamafactory.data.action_select import ActionSelectMetadataParser
    from llamafactory.train.sft.user_action_auxiliary import (
        ACTION_STAT_INDEX,
        UserActionAuxiliaryController,
        action_statistics_to_metrics,
    )

    sample = make_sample()
    metadata = ActionSelectMetadataParser(FakeTokenizer()).parse(sample["input_ids"], sample["labels"])
    auxiliary = UserActionAuxiliaryController(
        FakeTokenizer(),
        make_args(
            user_action_history_trie_enabled=False,
            user_action_length_guard_enabled=False,
            user_action_topk_illegal_enabled=True,
            user_action_topk_illegal_weight=0.0,
            user_action_topk_illegal_cap_ratio=0.0,
            user_action_aux_cap_ratio=0.0,
            user_action_aux_warmup_steps=0,
        ),
    )
    result = auxiliary.compute(
        make_logits(len(sample["input_ids"])),
        torch.tensor([sample["labels"]]),
        [metadata],
        torch.tensor(2.0),
        0,
    )
    assert result.loss == 0
    metrics = action_statistics_to_metrics(result.statistics)
    assert set(metrics) == {
        "a_act_legal_sid_mass",
        "a_act_topk_illegal_mass",
        "a_act_top1_illegal_hit_rate",
        "a_act_gold_top5_rate",
    }
    assert result.statistics[ACTION_STAT_INDEX["topk_illegal_mass_count"]] > 0


if __name__ == "__main__":
    tests = [
        test_sid_tokens_receive_weight_eight_and_text_tokens_one,
        test_ignore_labels_do_not_enter_numerator_or_denominator,
        test_prompt_sid_with_ignore_label_is_not_weighted,
        test_sid_mask_uses_the_same_causal_shift_as_labels,
        test_no_sid_sample_matches_original_ce_exactly,
        test_all_sid_sample_is_finite_and_has_no_nan,
        test_compact_metrics_have_expected_normalized_values,
        test_packed_segments_keep_sid_mask_positions_local_to_supervised_labels,
        test_switch_off_preserves_original_loss_and_gradient,
        test_weighted_base_loss_is_used_before_gradnorm_weight_and_divisor,
        test_enabled_path_still_uses_one_forward_and_one_backward,
        test_action_auxiliary_consumes_weighted_base_ce_without_extra_forward,
        test_zero_weight_action_topk_diagnostics_do_not_change_auxiliary_loss,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
