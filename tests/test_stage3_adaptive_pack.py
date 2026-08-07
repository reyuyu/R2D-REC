from torch.utils.data import Dataset

from llamafactory.data.multitask import (
    PACKING_MODE_SAME_SUBTASK_BFD,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL,
    TASK_IDS_BY_LAYOUT,
    TaskDataLoader,
    TokenizedSubDataset,
    LengthBucketPackSampler,
    MultiTaskMacroStepLoader,
    normalize_subtask_pack_lengths,
)


class TinyDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = list(lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        length = self.lengths[index]
        return {
            "input_ids": list(range(length)),
            "labels": [-100] + [1] * (length - 1),
        }


def wrapped(task, subtask, lengths, sid=0):
    return TokenizedSubDataset(
        TinyDataset(lengths), task, subtask, sid,
        task_ids=TASK_IDS_BY_LAYOUT[TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL],
    )


def test_aliases_and_defaults_resolve_per_subtask():
    result = normalize_subtask_pack_lengths(
        {"user/action": 12288, "user/chain_cot": 16384},
        TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL,
        8192,
    )
    assert result["user_action/action_nocot"] == 12288
    assert result["user_chain/cot"] == 16384
    assert result["user_chain/nocot"] == 8192
    assert result["recommendation/cot"] == 8192


def test_subtask_sampler_uses_its_own_pack_limit_and_metadata():
    datasets = {"cot": wrapped("user_chain", "cot", [7000, 5000, 4000])}
    loader = TaskDataLoader(
        "user_chain",
        datasets,
        {"cot": 1.0},
        max_pack_length=8192,
        max_pack_length_by_subtask={"cot": 12288},
        seed=7,
        packing_mode=PACKING_MODE_SAME_SUBTASK_BFD,
        bfd_window_size=32,
    )
    batch = next(loader)
    assert batch["pack_max_length"] == 12288
    assert batch["packed_token_count"] <= 12288
    assert batch["packed_token_count"] > 8192


def test_sampler_resume_rejects_changed_subtask_limit():
    dataset = wrapped("user_action", "action_nocot", [6000, 5000])
    sampler = LengthBucketPackSampler(
        dataset,
        max_pack_length=12288,
        seed=7,
        packing_mode=PACKING_MODE_SAME_SUBTASK_BFD,
    )
    state = sampler.state_dict()
    restored = LengthBucketPackSampler(
        dataset,
        max_pack_length=8192,
        seed=7,
        packing_mode=PACKING_MODE_SAME_SUBTASK_BFD,
    )
    try:
        restored.load_state_dict(state)
    except ValueError as exc:
        assert "Pack length changed" in str(exc)
    else:
        raise AssertionError("changed pack length must invalidate checkpoint resume")


def test_default_limit_remains_8192():
    dataset = wrapped("user_action", "action_nocot", [7000, 5000])
    sampler = LengthBucketPackSampler(
        dataset,
        max_pack_length=8192,
        seed=7,
        packing_mode=PACKING_MODE_SAME_SUBTASK_BFD,
    )
    assert all(sum(dataset[index]["seq_len"] for index in pack) <= 8192 for pack in sampler.global_plan)


def test_coverage_metrics_use_per_pack_limit_for_12k_utilization():
    loader = object.__new__(MultiTaskMacroStepLoader)
    loader.task_loaders = {}
    loader.current_global_allocation = {}
    loader._last_partition_metrics = {}
    loader._last_pack_stats = [(12000, 2, 12288, 9000), (8000, 2, 8192, 7000)]
    loader._last_task_pack_stats = {
        "user_chain": [(12000, 2, 9000)],
        "material": [(8000, 2, 7000)],
    }
    metrics = loader.coverage_metrics()
    assert 0.0 < metrics["pack_utilization_mean"] <= 1.0
    assert metrics["global_tokens_per_macro"] == 20000
    assert metrics["supervised_tokens_per_macro"] == 16000
    assert metrics["task_pack_tokens_mean"]["user_chain"] == 12000


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
    print("All stage3 adaptive pack tests passed.")
