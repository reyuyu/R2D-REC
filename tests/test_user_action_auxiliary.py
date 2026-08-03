import copy
import math
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.data.action_select import ActionSelectMetadataParser, build_action_semantic_token_ids
from llamafactory.data.multitask import TaskPackCollator, TokenizedSubDataset
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer
from llamafactory.train.sft.user_action_auxiliary import (
    ACTION_STAT_INDEX,
    ACTION_STAT_SIZE,
    UserActionAuxiliaryController,
    action_statistics_to_metrics,
)


IGNORE_INDEX = -100
SID_1 = (10, 20, 30, 40)
SID_2 = (10, 20, 30, 41)
SID_3 = (11, 21, 31, 40)


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
    values = {
        "user_action_history_trie_enabled": True,
        "user_action_history_trie_weight": 0.06,
        "user_action_length_guard_enabled": True,
        "user_action_continue_domain_extra": 0.75,
        "user_action_continue_separator_extra": 0.20,
        "user_action_no_early_stop_weight": 0.02,
        "user_action_stop_domain_weight": 0.05,
        "user_action_stop_tail_extra": 1.0,
        "user_action_max_stop_tail_positions": 4,
        "user_action_aux_cap_ratio": 0.08,
        "user_action_aux_warmup_steps": 100,
        "user_action_aux_vectorized_enabled": False,
        "user_action_aux_full_vocab_chunk_size": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_sample(history=(SID_1, SID_2, SID_3), answer=(SID_1, SID_2), tail=(7, 8), separator=(6,)):
    prompt = [1]
    for sid in history:
        prompt.extend(sid)
        prompt.append(9)
    supervised = [5]
    for index, sid in enumerate(answer):
        supervised.extend(sid)
        if index + 1 < len(answer):
            supervised.extend(separator)
    supervised.extend(tail)
    input_ids = prompt + supervised
    labels = [IGNORE_INDEX] * len(prompt) + supervised
    return {"input_ids": input_ids, "labels": labels}


def parsed_sample(**kwargs):
    sample = make_sample(**kwargs)
    metadata = ActionSelectMetadataParser(FakeTokenizer()).parse(sample["input_ids"], sample["labels"])
    return sample, metadata


def make_logits(length, requires_grad=True):
    logits = torch.zeros(1, length, 64, dtype=torch.float32)
    logits[..., 10] = 0.2
    logits[..., 11] = -0.1
    logits[..., 20] = 0.3
    logits[..., 21] = -0.2
    logits[..., 30] = 0.4
    logits[..., 31] = -0.3
    logits[..., 40] = 0.5
    logits[..., 41] = -0.4
    return logits.requires_grad_(requires_grad)


def compute_result(sample, metadata, **args):
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args(**args))
    logits = make_logits(len(sample["input_ids"]))
    labels = torch.tensor([sample["labels"]])
    action_ce = torch.tensor(2.0, requires_grad=True)
    return controller, logits, controller.compute(logits, labels, [metadata], action_ce, 100)


def test_semantic_vocab_is_built_from_active_tokenizer():
    groups = build_action_semantic_token_ids(FakeTokenizer())
    assert groups == {"domain": frozenset({10, 11}), "a": frozenset({20, 21}), "b": frozenset({30, 31}), "c": frozenset({40, 41})}


def test_history_is_parsed_as_complete_sid_tuples():
    _, metadata = parsed_sample()
    assert [tuple(sid) for sid in metadata["history_sids"]] == [SID_1, SID_2, SID_3]


def test_independent_history_columns_cannot_cross_splice():
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    allowed = controller._allowed_sets({SID_1, SID_3}, SID_1)
    assert allowed[1:] == ({20}, {30}, {40})
    assert (10, 21, 31, 40) not in {SID_1, SID_3}


def test_gold_sid_positions_are_exact_and_contiguous():
    sample, metadata = parsed_sample()
    unit = metadata["answer_sid_units"][0]
    positions = [unit[name] for name in ("pos_domain", "pos_a", "pos_b", "pos_c")]
    assert positions == list(range(positions[0], positions[0] + 4))
    assert tuple(sample["input_ids"][position] for position in positions) == SID_1


def test_prompt_sids_are_not_misclassified_as_answer_sids():
    _, metadata = parsed_sample()
    assert len(metadata["answer_sid_units"]) == 2
    assert all(unit["pos_domain"] >= metadata["answer_start"] for unit in metadata["answer_sid_units"])


def test_gold_not_in_history_skips_trie_without_nan():
    sample, metadata = parsed_sample(history=(SID_1,), answer=(SID_3,))
    _, _, result = compute_result(sample, metadata, user_action_length_guard_enabled=False)
    assert result.statistics[ACTION_STAT_INDEX["gold_in_history"]] == 0
    assert result.statistics[ACTION_STAT_INDEX["gold_total"]] == 1
    assert result.loss == 0


def test_duplicate_gold_disables_dynamic_removal():
    sample, metadata = parsed_sample(history=(SID_1,), answer=(SID_1, SID_1))
    assert metadata["gold_duplicate"]
    _, _, result = compute_result(sample, metadata, user_action_length_guard_enabled=False)
    assert result.statistics[ACTION_STAT_INDEX["allowed_c_count"]] == 2
    assert result.statistics[ACTION_STAT_INDEX["removed_sid_sum"]] == 0


def test_parse_failure_safely_returns_no_auxiliary_loss():
    sample = make_sample()
    sample["input_ids"].insert(-2, 21)
    sample["labels"].insert(-2, 21)
    metadata = ActionSelectMetadataParser(FakeTokenizer()).parse(sample["input_ids"], sample["labels"])
    assert not metadata["parse_valid"]
    _, _, result = compute_result(sample, metadata)
    assert result.loss == 0


class OneSampleDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return copy.deepcopy(self.sample)


def test_tokenized_dataset_parses_action_metadata_only_once():
    parser = ActionSelectMetadataParser(FakeTokenizer())
    dataset = TokenizedSubDataset(OneSampleDataset(make_sample()), "user", "action_nocot", 2, parser)
    first, second = dataset[0], dataset[0]
    assert parser.parse_count == 1
    assert first["sample_metadata"] is second["sample_metadata"]


def test_disabled_dataset_path_keeps_empty_metadata():
    dataset = TokenizedSubDataset(OneSampleDataset(make_sample()), "user", "action_nocot", 2)
    assert dataset[0]["sample_metadata"] == {}


def packed_action_batch(samples):
    parser = ActionSelectMetadataParser(FakeTokenizer())
    wrapped = []
    for index, sample in enumerate(samples):
        metadata = parser.parse(sample["input_ids"], sample["labels"])
        wrapped.append(
            {
                **sample,
                "task_name": "user",
                "task_id": 1,
                "subtask_name": "action_nocot",
                "subtask_id": 2,
                "sample_id": index,
                "seq_len": len(sample["input_ids"]),
                "supervised_token_count": sum(value != IGNORE_INDEX for value in sample["labels"]),
                "sample_metadata": metadata,
            }
        )
    return TaskPackCollator()(wrapped)


def test_packing_offsets_two_action_segments_correctly():
    samples = [make_sample(), make_sample(history=(SID_3,), answer=(SID_3,))]
    batch = packed_action_batch(samples)
    first, second = batch["action_aux_metadata"]
    offset = len(samples[0]["input_ids"])
    assert first["answer_start"] < offset
    local_second, _ = parsed_sample(history=(SID_3,), answer=(SID_3,))
    assert second["answer_start"] == offset + next(i for i, value in enumerate(local_second["labels"]) if value != IGNORE_INDEX)


def test_action_metadata_is_not_a_model_forward_key():
    batch = packed_action_batch([make_sample()])
    model_inputs = {key: value for key, value in batch.items() if key in MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS}
    assert "action_aux_metadata" in batch and "action_aux_metadata" not in model_inputs


def test_each_packed_segment_keeps_its_own_history():
    batch = packed_action_batch([make_sample(history=(SID_1,), answer=(SID_1,)), make_sample(history=(SID_3,), answer=(SID_3,))])
    assert batch["action_aux_metadata"][0]["history_sids"] == [list(SID_1)]
    assert batch["action_aux_metadata"][1]["history_sids"] == [list(SID_3)]


def test_one_segment_used_sid_does_not_mutate_another_segment():
    samples = [make_sample(history=(SID_1, SID_2), answer=(SID_1, SID_2))] * 2
    batch = packed_action_batch(samples)
    before = copy.deepcopy(batch["action_aux_metadata"][1])
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    controller.compute(make_logits(batch["input_ids"].shape[1]), batch["labels"], batch["action_aux_metadata"], torch.tensor(2.0), 100)
    assert batch["action_aux_metadata"][1] == before


def test_domain_allowed_set_contains_all_available_domains():
    allowed = UserActionAuxiliaryController._allowed_sets({SID_1, SID_3}, SID_1)
    assert allowed[0] == {10, 11}


def test_a_allowed_set_is_filtered_by_domain():
    allowed = UserActionAuxiliaryController._allowed_sets({SID_1, SID_3}, SID_1)
    assert allowed[1] == {20}


def test_b_allowed_set_is_filtered_by_domain_and_a():
    allowed = UserActionAuxiliaryController._allowed_sets({SID_1, SID_3}, SID_1)
    assert allowed[2] == {30}


def test_c_allowed_set_is_filtered_by_full_prefix():
    allowed = UserActionAuxiliaryController._allowed_sets({SID_1, SID_2, SID_3}, SID_1)
    assert allowed[3] == {40, 41}


def test_complete_sid_removal_deletes_only_one_path():
    available = {SID_1, SID_2, SID_3} - {SID_1}
    assert SID_1 not in available and SID_2 in available and SID_3 in available


def test_shared_prefix_paths_remain_after_complete_sid_removal():
    allowed = UserActionAuxiliaryController._allowed_sets({SID_2, SID_3}, SID_2)
    assert allowed[1] == {20} and allowed[2] == {30} and allowed[3] == {41}


def test_allowed_mass_matches_manual_type_conditional_probability():
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    logits = torch.zeros(1, 3, 64)
    loss, mass = controller._allowed_mass_loss(logits, 2, "domain", {10})
    assert torch.allclose(mass, torch.tensor(0.5))
    assert torch.allclose(loss, torch.tensor(math.log(2.0)))


def test_causal_position_uses_target_minus_one_without_off_by_one():
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    logits = torch.zeros(1, 4, 64)
    logits[0, 1, 10], logits[0, 1, 11] = 8.0, -8.0
    logits[0, 2, 10], logits[0, 2, 11] = -8.0, 8.0
    _, mass = controller._allowed_mass_loss(logits, 2, "domain", {10})
    assert mass > 0.999


def test_nonfinal_boundary_adds_next_domain_loss():
    sample, metadata = parsed_sample()
    _, _, result = compute_result(sample, metadata, user_action_history_trie_enabled=False)
    assert result.statistics[ACTION_STAT_INDEX["continue_loss_sum"]] > 0


def test_nonfinal_boundary_tracks_early_stop_mass():
    sample, metadata = parsed_sample()
    _, _, result = compute_result(sample, metadata)
    assert result.statistics[ACTION_STAT_INDEX["no_early_stop_mass_count"]] == 1
    assert result.statistics[ACTION_STAT_INDEX["no_early_stop_mass_sum"]] > 0


def test_final_tail_adds_closure_loss():
    sample, metadata = parsed_sample(answer=(SID_1,))
    _, _, result = compute_result(sample, metadata, user_action_history_trie_enabled=False)
    assert result.statistics[ACTION_STAT_INDEX["stop_loss_sum"]] > 0


def test_final_tail_tracks_domain_restart_mass():
    sample, metadata = parsed_sample(answer=(SID_1,))
    _, _, result = compute_result(sample, metadata)
    assert result.statistics[ACTION_STAT_INDEX["stop_domain_mass_count"]] == 2


def test_single_sid_has_no_continue_loss():
    sample, metadata = parsed_sample(answer=(SID_1,))
    _, _, result = compute_result(sample, metadata)
    assert result.statistics[ACTION_STAT_INDEX["continue_loss_sum"]] == 0


def test_domain_in_final_tail_skips_all_stop_auxiliary_terms():
    sample, metadata = parsed_sample(answer=(SID_1,), tail=(10, 7))
    assert metadata["parse_valid"] is False
    # Exercise the defensive controller branch with otherwise valid metadata.
    _, valid_metadata = parsed_sample(answer=(SID_1,))
    valid_metadata["final_tail_positions"] = [valid_metadata["answer_sid_units"][0]["pos_domain"]]
    _, _, result = compute_result(sample, valid_metadata)
    assert result.statistics[ACTION_STAT_INDEX["stop_loss_sum"]] == 0


def test_warmup_factor_is_exact_at_zero_fifty_and_one_hundred():
    assert UserActionAuxiliaryController.warmup_factor(0, 100) == 0
    assert UserActionAuxiliaryController.warmup_factor(50, 100) == 0.5
    assert UserActionAuxiliaryController.warmup_factor(100, 100) == 1


def test_cap_limits_final_auxiliary_to_eight_percent_of_action_ce():
    sample, metadata = parsed_sample()
    _, _, result = compute_result(sample, metadata)
    assert result.loss.detach() <= torch.tensor(0.16 + 1e-6)


def test_cap_scale_does_not_backpropagate_into_action_ce():
    sample, metadata = parsed_sample()
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    logits = make_logits(len(sample["input_ids"]))
    action_ce = torch.tensor(2.0, requires_grad=True)
    result = controller.compute(logits, torch.tensor([sample["labels"]]), [metadata], action_ce, 100)
    result.loss.backward()
    assert action_ce.grad is None


def test_empty_metadata_and_zero_valid_segments_stay_finite():
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    result = controller.compute(make_logits(4), torch.full((1, 4), IGNORE_INDEX), [], torch.tensor(0.0), 100)
    assert torch.isfinite(result.loss) and result.loss == 0


def test_fp16_extreme_logits_use_stable_fp32_logsumexp():
    controller = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    logits = torch.full((1, 2, 64), -60000.0, dtype=torch.float16)
    logits[0, 0, 10] = 60000.0
    result = controller._allowed_mass_loss(logits, 1, "domain", {10})
    assert result is not None and all(torch.isfinite(value) for value in result)


def test_all_logged_metrics_are_finite_scalars():
    sample, metadata = parsed_sample()
    _, _, result = compute_result(sample, metadata)
    metrics = action_statistics_to_metrics(result.statistics)
    assert metrics and all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())
    assert set(metrics) == {
        "a_act_segments",
        "a_act_parse_ok_rate",
        "a_act_gold_in_history_rate",
        "a_act_gold_duplicate_rate",
        "b_act_allowed_domain_mass",
        "b_act_allowed_a_mass",
        "b_act_allowed_b_mass",
        "b_act_allowed_c_mass",
        "c_act_seen_sid_removed_avg",
        "c_act_no_early_stop_mass",
        "c_act_stop_domain_mass",
        "d_act_trie_loss",
        "d_act_continue_loss",
        "d_act_stop_loss",
        "d_act_aux_loss",
        "d_act_action_ce",
        "d_act_aux_to_ce_ratio",
        "d_act_cap_active",
        "d_act_warmup_factor",
    }


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.linspace(-0.2, 0.2, 64))
        self.forward_count = 0

    def forward(self, input_ids, labels=None, **kwargs):
        self.forward_count += 1
        logits = self.logit_bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
        loss = F.cross_entropy(shift_logits.reshape(-1, 64), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX)
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

    def task_weight(self, task):
        return self.weight

    def set_current_task(self, task):
        self.current_task = task

    def clear_current_task(self):
        self.current_task = None


class TrainerHarness:
    _MODEL_INPUT_KEYS = MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS
    compute_task_microbatches = MultiTaskMacroSeq2SeqTrainer.compute_task_microbatches

    def __init__(self, auxiliary=None, weight_controller=None, divisor=4, step=100):
        self.args = SimpleNamespace(device=torch.device("cpu"), n_gpu=1)
        self.state = SimpleNamespace(global_step=step)
        self.user_action_auxiliary = auxiliary
        self.gradient_controller = weight_controller
        self._loss_scale_denominator = divisor
        self._task_stat_size = 5 + (ACTION_STAT_SIZE if auxiliary is not None else 0)
        self.accelerator = CountingAccelerator()

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def make_microbatch(subtask="action_nocot"):
    batch = packed_action_batch([make_sample()])
    batch["subtask_name"] = subtask
    if subtask != "action_nocot":
        batch["action_aux_metadata"] = []
    return batch


def test_total_switch_off_matches_original_loss_and_gradient():
    first, second = TinyCausalModel(), TinyCausalModel()
    second.load_state_dict(first.state_dict())
    harness = TrainerHarness(auxiliary=None, weight_controller=None, divisor=4)
    returned, stats = harness.compute_task_microbatches(first, "user", [make_microbatch()], True)
    outputs = second(input_ids=make_microbatch()["input_ids"], labels=make_microbatch()["labels"])
    (outputs.loss / 4).backward()
    assert torch.allclose(returned, outputs.loss.detach() / 4)
    assert torch.allclose(first.logit_bias.grad, second.logit_bias.grad)
    assert stats.shape[0] == 5


def test_auxiliary_only_applies_to_user_action_nocot():
    auxiliary = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    action_model, chain_model = TinyCausalModel(), TinyCausalModel()
    action_harness, chain_harness = TrainerHarness(auxiliary), TrainerHarness(auxiliary)
    _, action_stats = action_harness.compute_task_microbatches(action_model, "user", [make_microbatch()], True)
    _, chain_stats = chain_harness.compute_task_microbatches(chain_model, "user", [make_microbatch("chain_nocot")], True)
    assert action_stats[ACTION_STAT_INDEX["aux_loss_sum"] + 5] > 0
    assert chain_stats[5:].count_nonzero() == 0


def test_material_recommendation_world_paths_are_unchanged():
    auxiliary = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    for task in ("material", "recommendation", "world"):
        model = TinyCausalModel()
        harness = TrainerHarness(auxiliary)
        _, stats = harness.compute_task_microbatches(model, task, [make_microbatch("cot")], True)
        assert stats[5:].count_nonzero() == 0


def test_enabled_action_microbatch_uses_one_forward_and_one_backward():
    model = TinyCausalModel()
    harness = TrainerHarness(UserActionAuxiliaryController(FakeTokenizer(), make_args()))
    harness.compute_task_microbatches(model, "user", [make_microbatch()], True)
    assert model.forward_count == 1
    assert harness.accelerator.backward_count == 1


def test_auxiliary_is_added_before_user_weight_and_original_divisor():
    model = TinyCausalModel()
    harness = TrainerHarness(
        UserActionAuxiliaryController(FakeTokenizer(), make_args()),
        FixedWeightController(2.0),
        divisor=4,
    )
    total, stats = harness.compute_task_microbatches(model, "user", [make_microbatch()], True)
    assert torch.allclose(total, stats[0] * 2.0 / 4.0)


def test_gradnorm_user_raw_loss_receives_base_plus_auxiliary():
    model = TinyCausalModel()
    harness = TrainerHarness(UserActionAuxiliaryController(FakeTokenizer(), make_args()))
    _, stats = harness.compute_task_microbatches(model, "user", [make_microbatch()], True)
    base = stats[5 + ACTION_STAT_INDEX["user_base_sum"]]
    aux = stats[5 + ACTION_STAT_INDEX["user_aux_sum"]]
    total = stats[5 + ACTION_STAT_INDEX["user_total_sum"]]
    assert aux > 0 and torch.allclose(stats[0], base + aux) and torch.allclose(total, base + aux)


def test_loss_divisor_is_not_changed_by_auxiliary_path():
    auxiliary = UserActionAuxiliaryController(FakeTokenizer(), make_args())
    model_a, model_b = TinyCausalModel(), TinyCausalModel()
    model_b.load_state_dict(model_a.state_dict())
    total_a, stats_a = TrainerHarness(auxiliary, divisor=4).compute_task_microbatches(model_a, "user", [make_microbatch()], True)
    total_b, stats_b = TrainerHarness(auxiliary, divisor=8).compute_task_microbatches(model_b, "user", [make_microbatch()], True)
    assert torch.allclose(stats_a[0], stats_b[0])
    assert torch.allclose(total_a, total_b * 2)


def _compute_backend_pair(samples, **overrides):
    batch = packed_action_batch(samples)
    generator = torch.Generator().manual_seed(20260803)
    base_logits = torch.randn(1, batch["input_ids"].shape[1], 64, generator=generator)
    legacy_logits = base_logits.clone().requires_grad_(True)
    vectorized_logits = base_logits.clone().requires_grad_(True)
    common = {
        "user_action_aux_cap_ratio": 1.0,
        "user_action_aux_warmup_steps": 0,
        **overrides,
    }
    legacy = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(**common, user_action_aux_vectorized_enabled=False)
    )
    vectorized = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(**common, user_action_aux_vectorized_enabled=True)
    )
    action_ce = torch.tensor(100.0)
    legacy_result = legacy.compute(
        legacy_logits, batch["labels"], batch["action_aux_metadata"], action_ce, 100
    )
    vectorized_result = vectorized.compute(
        vectorized_logits, batch["labels"], batch["action_aux_metadata"], action_ce, 100
    )
    return legacy_logits, vectorized_logits, legacy_result, vectorized_result


def _assert_backend_equivalent(samples, **overrides):
    legacy_logits, vectorized_logits, legacy_result, vectorized_result = _compute_backend_pair(
        samples, **overrides
    )
    torch.testing.assert_close(vectorized_result.loss, legacy_result.loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        vectorized_result.statistics, legacy_result.statistics, rtol=2e-6, atol=2e-6
    )
    legacy_result.loss.backward()
    vectorized_result.loss.backward()
    torch.testing.assert_close(vectorized_logits.grad, legacy_logits.grad, rtol=3e-6, atol=3e-7)


def test_vectorized_backend_matches_legacy_across_action_shapes():
    cases = (
        [make_sample(answer=(SID_1,))],
        [make_sample(answer=(SID_1, SID_2))],
        [make_sample(answer=(SID_1, SID_2, SID_3), tail=(7, 8, 9, 12, 13))],
        [make_sample(history=(SID_1,), answer=(SID_3,))],
        [make_sample(history=(SID_1,), answer=(SID_1, SID_1))],
        [
            make_sample(history=(SID_1, SID_2), answer=(SID_1, SID_2)),
            make_sample(history=(SID_3,), answer=(SID_3,), tail=(7,)),
        ],
    )
    for samples in cases:
        _assert_backend_equivalent(samples)


def test_vectorized_backend_matches_legacy_for_each_ablation():
    sample = [make_sample()]
    _assert_backend_equivalent(sample, user_action_length_guard_enabled=False)
    _assert_backend_equivalent(sample, user_action_history_trie_enabled=False)
    _assert_backend_equivalent(
        sample,
        user_action_history_trie_enabled=False,
        user_action_length_guard_enabled=False,
    )


def test_vectorized_backend_chunk_size_does_not_change_math_or_gradients():
    samples = [
        make_sample(answer=(SID_1, SID_2, SID_3), tail=(7, 8, 9, 12)),
        make_sample(history=(SID_1, SID_2), answer=(SID_2, SID_1)),
    ]
    for chunk_size in (1, 2, 32):
        _assert_backend_equivalent(samples, user_action_aux_full_vocab_chunk_size=chunk_size)


def test_vectorized_backend_preserves_cap_and_partial_warmup():
    samples = [make_sample(answer=(SID_1, SID_2, SID_3))]
    batch = packed_action_batch(samples)
    logits = make_logits(batch["input_ids"].shape[1])
    labels = batch["labels"]
    metadata = batch["action_aux_metadata"]
    results = []
    for enabled in (False, True):
        controller = UserActionAuxiliaryController(
            FakeTokenizer(),
            make_args(
                user_action_aux_vectorized_enabled=enabled,
                user_action_aux_cap_ratio=0.01,
                user_action_aux_warmup_steps=100,
            ),
        )
        results.append(controller.compute(logits, labels, metadata, torch.tensor(2.0), 50))
    torch.testing.assert_close(results[1].loss, results[0].loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(results[1].statistics, results[0].statistics, rtol=2e-6, atol=2e-6)
    assert results[1].statistics[ACTION_STAT_INDEX["cap_active_sum"]] == 1
    assert results[1].statistics[ACTION_STAT_INDEX["warmup_sum"]] == 0.5


def test_vectorized_backend_handles_parse_failure_and_empty_metadata():
    controller = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=True)
    )
    logits = make_logits(4)
    labels = torch.full((1, 4), IGNORE_INDEX)
    empty = controller.compute(logits, labels, [], torch.tensor(0.0), 100)
    failed = controller.compute(
        logits,
        labels,
        [{"parse_valid": False, "parse_ms": 0.25}],
        torch.tensor(0.0),
        100,
    )
    assert empty.loss == 0 and failed.loss == 0
    assert torch.isfinite(empty.statistics).all() and torch.isfinite(failed.statistics).all()


def test_vectorized_trainer_path_still_uses_one_forward_and_backward():
    auxiliary = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=True)
    )
    model = TinyCausalModel()
    harness = TrainerHarness(auxiliary)
    harness.compute_task_microbatches(model, "user", [make_microbatch()], True)
    assert model.forward_count == 1
    assert harness.accelerator.backward_count == 1


def test_tiny_qwen3_lora_parameter_gradients_match_legacy_backend():
    torch.manual_seed(20260803)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_cache=False,
    )

    def make_model():
        return get_peft_model(
            Qwen3ForCausalLM(config),
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
            ),
        ).to(device).eval()

    legacy_model = make_model()
    vectorized_model = make_model()
    vectorized_model.load_state_dict(legacy_model.state_dict())
    batch = packed_action_batch([make_sample(answer=(SID_1, SID_2, SID_3))])
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    metadata = batch["action_aux_metadata"]
    controllers = (
        UserActionAuxiliaryController(
            FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=False)
        ),
        UserActionAuxiliaryController(
            FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=True)
        ),
    )
    models = (legacy_model, vectorized_model)
    losses = []
    for model, controller in zip(models, controllers):
        outputs = model(input_ids=input_ids, labels=labels)
        auxiliary = controller.compute(outputs.logits, labels, metadata, outputs.loss, 100)
        total = outputs.loss + auxiliary.loss
        total.backward()
        losses.append(total.detach())
    torch.testing.assert_close(losses[1], losses[0], rtol=2e-6, atol=2e-6)
    legacy_gradients = {
        name: parameter.grad
        for name, parameter in legacy_model.named_parameters()
        if parameter.requires_grad
    }
    vectorized_gradients = {
        name: parameter.grad
        for name, parameter in vectorized_model.named_parameters()
        if parameter.requires_grad
    }
    assert legacy_gradients.keys() == vectorized_gradients.keys()
    for name in legacy_gradients:
        assert legacy_gradients[name] is not None, name
        torch.testing.assert_close(
            vectorized_gradients[name], legacy_gradients[name], rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} Action Select auxiliary tests passed.")
