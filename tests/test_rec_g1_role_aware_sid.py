import torch

from llamafactory.train.sft.sid_token_weighting import (
    SID_WEIGHT_STAT_INDEX,
    SidTokenWeightingController,
    sid_weight_statistics_to_metrics,
)


IGNORE_INDEX = -100


class FakeTokenizer:
    def get_added_vocab(self):
        return {
            "<|video_begin|>": 10,
            "<|prod_begin|>": 11,
            "<|ad_begin|>": 12,
            "<|living_begin|>": 13,
            "<think>": 50,
            "</think>": 51,
            "<s_a_0>": 20,
            "<s_a_1>": 21,
            "<s_b_0>": 30,
            "<s_b_1>": 31,
            "<s_c_0>": 40,
            "<s_c_1>": 41,
        }

    def encode(self, text, add_special_tokens=False):
        return {"<think>": [50], "</think>": [51]}[text]


def args(enabled=True):
    return type(
        "Args",
        (),
        {
            "sid_token_weight": 8.0,
            "sid_text_weight": 1.0,
            "recommendation_role_aware_sid_weighting_enabled": enabled,
            "recommendation_think_sid_weight": 1.0,
            "recommendation_final_sid_weight": 8.0,
        },
    )()


def logits(labels, vocab=80):
    sequence = labels.shape[-1] if torch.is_tensor(labels) else len(labels)
    return torch.zeros(1, sequence, vocab)


def test_rec_cot_think_sid_is_one_and_final_sid_is_eight():
    labels = torch.tensor([[IGNORE_INDEX, 50, 20, 51, 30, 7]])
    result = SidTokenWeightingController(FakeTokenizer(), args()).compute(
        logits(labels), labels, task_name="recommendation", subtask_name="cot"
    )
    metrics = sid_weight_statistics_to_metrics(result.statistics)
    assert metrics["rec_think_sid_token_count"] == 1
    assert metrics["rec_final_sid_token_count"] == 1
    assert metrics["rec_think_sid_weighted_mass"] == 1
    assert metrics["rec_final_sid_weighted_mass"] == 8
    assert metrics["sid_effective_weight"] == 12 / 5


def test_gold_same_as_history_is_still_final_weight():
    labels = torch.tensor([[IGNORE_INDEX, 50, 20, 51, 20]])
    result = SidTokenWeightingController(FakeTokenizer(), args()).compute(
        logits(labels), labels, task_name="recommendation", subtask_name="cot"
    )
    assert result.statistics[SID_WEIGHT_STAT_INDEX["rec_think_sid_weighted_mass"]] == 1
    assert result.statistics[SID_WEIGHT_STAT_INDEX["rec_final_sid_weighted_mass"]] == 8


def test_rec_nocot_and_other_tasks_keep_sid_weight_eight():
    labels = torch.tensor([[IGNORE_INDEX, 20, 30]])
    controller = SidTokenWeightingController(FakeTokenizer(), args())
    nocot = controller.compute(logits(labels), labels, task_name="recommendation", subtask_name="nocot")
    material = controller.compute(logits(labels), labels, task_name="material", subtask_name="cot")
    assert sid_weight_statistics_to_metrics(nocot.statistics)["rec_think_sid_token_count"] == 0
    assert sid_weight_statistics_to_metrics(nocot.statistics)["rec_final_sid_token_count"] == 2
    assert nocot.statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] == 16
    assert material.statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] == 16


def test_packed_valid_runs_do_not_cross_segment_roles():
    labels = torch.tensor([[IGNORE_INDEX, 50, 20, 51, 30, IGNORE_INDEX, IGNORE_INDEX, 21, 7]])
    result = SidTokenWeightingController(FakeTokenizer(), args()).compute(
        logits(labels), labels, task_name="recommendation", subtask_name="cot"
    )
    metrics = sid_weight_statistics_to_metrics(result.statistics)
    assert metrics["rec_think_sid_token_count"] == 1
    assert metrics["rec_final_sid_token_count"] == 2
    assert metrics["rec_think_sid_weighted_mass"] == 1
    assert metrics["rec_final_sid_weighted_mass"] == 16


def test_role_aware_disabled_matches_uniform_sid_weighting():
    labels = torch.tensor([[IGNORE_INDEX, 50, 20, 51, 30, 7]])
    g1 = SidTokenWeightingController(FakeTokenizer(), args(True)).compute(
        logits(labels), labels, task_name="recommendation", subtask_name="cot"
    )
    baseline = SidTokenWeightingController(FakeTokenizer(), args(False)).compute(
        logits(labels), labels, task_name="recommendation", subtask_name="cot"
    )
    assert g1.statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] == 9
    assert baseline.statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] == 16


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
