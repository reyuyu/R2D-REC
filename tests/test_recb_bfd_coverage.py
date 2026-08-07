import copy
from torch.utils.data import Dataset

from llamafactory.data.multitask import (
    CoverageDeficitScheduler,
    LengthBucketPackSampler,
    MultiTaskMacroStepLoader,
    PACKING_MODE_SAME_SUBTASK_BFD,
    SUBTASK_RATIOS_BY_LAYOUT,
    TASK_DATASETS_BY_LAYOUT,
    TASK_IDS_BY_LAYOUT,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL,
    TaskDataLoader,
    TokenizedSubDataset,
)


class VariableDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = list(lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        length = self.lengths[index]
        return {"input_ids": list(range(length)), "labels": [-100] + [1] * (length - 1)}


def wrapped(task, subtask, sid, lengths):
    return TokenizedSubDataset(VariableDataset(lengths), task, subtask, sid, task_ids=TASK_IDS_BY_LAYOUT[TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL])


def test_bfd_is_deterministic_bounded_and_nonduplicating():
    dataset = wrapped("recommendation", "cot", 0, [90, 70, 60, 45, 35, 25, 20, 10])
    one = LengthBucketPackSampler(dataset, 100, 17, max_segments=4, packing_mode=PACKING_MODE_SAME_SUBTASK_BFD, bfd_window_size=32)
    two = LengthBucketPackSampler(dataset, 100, 17, max_segments=4, packing_mode=PACKING_MODE_SAME_SUBTASK_BFD, bfd_window_size=32)
    assert one.global_plan == two.global_plan
    flattened = [index for pack in one.global_plan for index in pack]
    assert sorted(flattened) == list(range(len(dataset)))
    assert all(sum(dataset[index]["seq_len"] for index in pack) <= 100 for pack in one.global_plan)
    assert all(len(pack) <= 4 for pack in one.global_plan)


def test_coverage_scheduler_forces_four_tasks_and_tracks_real_queue_ratios():
    planner = CoverageDeficitScheduler({"material": 8, "user_action": 4, "user_chain": 12, "recommendation": 4}, 42, global_slots=8)
    first = planner.allocation_at(0)
    assert sum(first.values()) == 8
    assert all(first[name] >= 1 for name in planner.queue_lengths)
    planner.commit(first)
    assert sum(planner.consumed.values()) == 8
    assert planner.state_dict()["queue_lengths"] == {"material": 8, "user_action": 4, "user_chain": 12, "recommendation": 4}


def make_loaders():
    layout = TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL
    ids = TASK_IDS_BY_LAYOUT[layout]
    lengths = [90, 80, 70, 60, 50, 40, 30, 20]
    result = {}
    sid = 0
    for task, subtasks in TASK_DATASETS_BY_LAYOUT[layout].items():
        wrapped_subtasks = {}
        for subtask in subtasks:
            wrapped_subtasks[subtask] = TokenizedSubDataset(
                VariableDataset(lengths), task, subtask, sid, task_ids=ids
            )
            sid += 1
        result[task] = TaskDataLoader(
            task, wrapped_subtasks, SUBTASK_RATIOS_BY_LAYOUT[layout][task],
            max_pack_length=100, max_segments=1, seed=9,
            packing_mode=PACKING_MODE_SAME_SUBTASK_BFD,
            bfd_window_size=64, coverage_mode=True,
        )
    return result


def test_coverage_macro_is_four_plus_four_and_resume_is_exact():
    allocation = {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
    kwargs = dict(allocation=allocation, max_steps=3, global_allocation=allocation,
                  rank=0, world_size=2, synchronized_global_consumption=True,
                  task_ids=TASK_IDS_BY_LAYOUT[TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL])
    first_loaders = make_loaders()
    queue_lengths = {task: loader.total_pack_count for task, loader in first_loaders.items()}
    first = MultiTaskMacroStepLoader(first_loaders, supercycle=CoverageDeficitScheduler(
        queue_lengths, 9, global_slots=8
    ), **kwargs)
    macro = next(first)
    assert sum(map(len, macro.values())) == 4
    state = copy.deepcopy(first.state_dict())
    expected = next(first)
    restored_loaders = make_loaders()
    restored = MultiTaskMacroStepLoader(restored_loaders, supercycle=CoverageDeficitScheduler(
        queue_lengths, 9, global_slots=8
    ), **kwargs)
    restored.load_state_dict(state)
    actual = next(restored)
    assert {task: [batch["sample_ids"] for batch in batches] for task, batches in actual.items()} == {
        task: [batch["sample_ids"] for batch in batches] for task, batches in expected.items()
    }


def test_coverage_metrics_are_finite():
    allocation = {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
    loaders = make_loaders()
    supercycle = CoverageDeficitScheduler({task: loader.total_pack_count for task, loader in loaders.items()}, 3, 8)
    macro = MultiTaskMacroStepLoader(
        loaders, allocation, 1, global_allocation=allocation, supercycle=supercycle,
        rank=0, world_size=1, synchronized_global_consumption=True,
        task_ids=TASK_IDS_BY_LAYOUT[TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL],
    )
    next(macro)
    metrics = macro.coverage_metrics()
    for key in ("pack_tokens_mean", "pack_utilization_mean", "pack_utilization_p10", "pack_segments_mean"):
        assert metrics[key] == metrics[key]


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
    print("All recC BFD/coverage tests passed.")
