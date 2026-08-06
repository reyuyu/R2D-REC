import copy
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from llamafactory.data.multitask import (
    SUBTASK_RATIOS,
    TASK_DATASETS,
    TASK_IDS,
    TASK_LAYOUT_LEGACY,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD,
    Balanced40SuperCycle,
    MultiTaskMacroStepLoader,
    LengthBucketPackSampler,
    TaskDataLoader,
    TokenizedSubDataset,
    get_action_task_name,
    get_subtask_ratios,
    get_task_datasets,
    get_task_ids,
    resolve_task_layout,
    split_global_microbatch_allocation,
    resolve_multitask_dataset_name,
)
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer


class TinyDataset(Dataset):
    def __init__(self, offset: int):
        self.offset = offset

    def __len__(self):
        return 24

    def __getitem__(self, index):
        length = 5 + ((self.offset + index) % 4)
        return {"input_ids": list(range(length)), "labels": [-100, 1] + [2] * (length - 2)}


def make_task_loaders(rank=0, world_size=1):
    task_loaders = {}
    subtask_id = 0
    for task_name, subtasks in TASK_DATASETS.items():
        wrapped = {}
        for subtask_name in subtasks:
            wrapped[subtask_name] = TokenizedSubDataset(TinyDataset(subtask_id), task_name, subtask_name, subtask_id)
            subtask_id += 1
        task_loaders[task_name] = TaskDataLoader(
            task_name,
            wrapped,
            SUBTASK_RATIOS[task_name],
            max_pack_length=32,
            max_segments=4,
            seed=7,
            rank=rank,
            world_size=world_size,
        )
    return task_loaders


def make_user_split_task_loaders(rank=0, world_size=1):
    task_loaders = {}
    subtask_id = 0
    task_datasets = get_task_datasets(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    ratios = get_subtask_ratios(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    task_ids = get_task_ids(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    for task_name, subtasks in task_datasets.items():
        wrapped = {}
        for subtask_name in subtasks:
            wrapped[subtask_name] = TokenizedSubDataset(
                TinyDataset(subtask_id),
                task_name,
                subtask_name,
                subtask_id,
                task_ids=task_ids,
            )
            subtask_id += 1
        task_loaders[task_name] = TaskDataLoader(
            task_name,
            wrapped,
            ratios[task_name],
            max_pack_length=32,
            max_segments=4,
            seed=7,
            rank=rank,
            world_size=world_size,
        )
    return task_loaders


def test_task_and_subtask_do_not_mix_in_v1_pack():
    for loader in make_task_loaders().values():
        batch = next(loader)
        assert batch["task_id"] == TASK_IDS[batch["task_name"]]
        assert len(set(batch["segment_subtask_ids"])) == 1


def test_pack_offsets_positions_and_supervised_count():
    batch = next(make_task_loaders()["material"])
    offsets = batch["segment_offsets"].tolist()
    assert batch["cu_seqlens"].tolist()[-1] == batch["input_ids"].shape[-1]
    assert batch["supervised_token_count"] == int((batch["labels"] != -100).sum())
    for start, end in offsets:
        assert batch["position_ids"][0, start:end].tolist() == list(range(end - start))
        assert len(set(batch["attention_mask"][0, start:end].tolist())) == 1
    assert all(offsets[index][1] == offsets[index + 1][0] for index in range(len(offsets) - 1))


def test_macro_step_keeps_task_groups_and_has_eight_microbatches():
    allocation = {"material": 2, "user": 2, "recommendation": 3, "world": 1}
    macro = next(MultiTaskMacroStepLoader(make_task_loaders(), allocation, max_steps=2))
    assert [len(macro[name]) for name in allocation] == [2, 2, 3, 1]
    assert sum(map(len, macro.values())) == 8
    for task_name, microbatches in macro.items():
        assert all(batch["task_name"] == task_name for batch in microbatches)


def test_ddp_plans_have_equal_lengths_and_resume_is_reproducible():
    rank0 = make_task_loaders(rank=0, world_size=2)
    rank1 = make_task_loaders(rank=1, world_size=2)
    for task_name in TASK_IDS:
        sampler0 = next(iter(rank0[task_name].samplers.values()))
        sampler1 = next(iter(rank1[task_name].samplers.values()))
        assert len(sampler0.plan) == len(sampler1.plan)
    macro = MultiTaskMacroStepLoader(rank0, {"material": 2, "user": 2, "recommendation": 3, "world": 1}, 4)
    next(macro)
    state = copy.deepcopy(macro.state_dict())
    expected = next(macro)
    restored = MultiTaskMacroStepLoader(make_task_loaders(rank=0, world_size=2), {"material": 2, "user": 2, "recommendation": 3, "world": 1}, 4)
    restored.load_state_dict(state)
    actual = next(restored)
    for task_name in TASK_IDS:
        assert [item["sample_ids"] for item in actual[task_name]] == [item["sample_ids"] for item in expected[task_name]]


def test_ddp_short_subtask_is_padded_instead_of_empty():
    dataset = TokenizedSubDataset(TinyDataset(0), "material", "cot", 0)
    # One large pack split across two ranks must be padded deterministically.
    rank0 = LengthBucketPackSampler(dataset, max_pack_length=1024, seed=7, rank=0, world_size=2)
    rank1 = LengthBucketPackSampler(dataset, max_pack_length=1024, seed=7, rank=1, world_size=2)
    assert len(rank0.plan) == len(rank1.plan) == 1


def test_two_rank_global_macro_keeps_eight_microbatches_total():
    allocation = {"material": 2, "user": 2, "recommendation": 3, "world": 1}
    rank_allocations, active_ranks = split_global_microbatch_allocation(allocation, world_size=2)
    assert rank_allocations == [
        {"material": 2, "user": 2},
        {"recommendation": 3, "world": 1},
    ]
    assert [sum(local.values()) for local in rank_allocations] == [4, 4]
    assert {task: len(ranks) for task, ranks in active_ranks.items()} == {
        "material": 1,
        "user": 1,
        "recommendation": 1,
        "world": 1,
    }


def test_balanced_40_supercycle_has_requested_global_counts():
    planner = Balanced40SuperCycle(seed=42)
    total = {task: 0 for task in TASK_IDS}
    world_macros = 0
    for macro_step in range(40):
        allocation = planner.allocation_at(macro_step)
        assert sum(allocation.values()) == 8
        world_macros += allocation["world"] > 0
        for task, count in allocation.items():
            total[task] += count
    assert total == {"material": 115, "user": 123, "recommendation": 81, "world": 1}
    assert world_macros == 1


def test_balanced_40_two_rank_loader_has_four_local_and_eight_global_microbatches():
    planner = Balanced40SuperCycle(seed=42)
    rank0 = MultiTaskMacroStepLoader(
        make_task_loaders(rank=0, world_size=1),
        {"material": 2, "user": 2, "recommendation": 3, "world": 1},
        max_steps=40,
        supercycle=planner,
        rank=0,
        world_size=2,
        synchronized_global_consumption=True,
    )
    rank1 = MultiTaskMacroStepLoader(
        make_task_loaders(rank=0, world_size=1),
        {"material": 2, "user": 2, "recommendation": 3, "world": 1},
        max_steps=40,
        supercycle=Balanced40SuperCycle(seed=42),
        rank=1,
        world_size=2,
        synchronized_global_consumption=True,
    )
    total = {task: 0 for task in TASK_IDS}
    for _ in range(40):
        first, second = next(rank0), next(rank1)
        assert sum(map(len, first.values())) == sum(map(len, second.values())) == 4
        for task in TASK_IDS:
            total[task] += len(first.get(task, [])) + len(second.get(task, []))
    assert total == {"material": 115, "user": 123, "recommendation": 81, "world": 1}


def test_balanced_40_loader_resume_keeps_next_rank_local_microbatches():
    kwargs = {
        "allocation": {"material": 2, "user": 2, "recommendation": 3, "world": 1},
        "max_steps": 4,
        "rank": 0,
        "world_size": 2,
        "synchronized_global_consumption": True,
    }
    loader = MultiTaskMacroStepLoader(
        make_task_loaders(rank=0, world_size=1), supercycle=Balanced40SuperCycle(seed=42), **kwargs
    )
    next(loader)
    state = copy.deepcopy(loader.state_dict())
    expected = next(loader)
    restored = MultiTaskMacroStepLoader(
        make_task_loaders(rank=0, world_size=1), supercycle=Balanced40SuperCycle(seed=42), **kwargs
    )
    restored.load_state_dict(state)
    actual = next(restored)
    assert actual.keys() == expected.keys()
    for task in actual:
        assert [item["sample_ids"] for item in actual[task]] == [item["sample_ids"] for item in expected[task]]




def test_user_split_layout_maps_and_legacy_defaults_unchanged():
    legacy_ids = get_task_ids(TASK_LAYOUT_LEGACY)
    split_ids = get_task_ids(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    assert legacy_ids == TASK_IDS
    assert split_ids == {"material": 0, "user_action": 1, "user_chain": 2, "recommendation": 3}
    assert "world" not in split_ids
    split_datasets = get_task_datasets(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    assert set(split_datasets) == {"material", "user_action", "user_chain", "recommendation"}
    assert split_datasets["user_action"] == {"action_nocot": "onereason_user_action_nocot"}
    assert set(split_datasets["user_chain"]) == {"cot", "nocot"}
    ratios = get_subtask_ratios(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    assert ratios["user_action"] == {"action_nocot": 1.00}
    assert get_action_task_name(TASK_LAYOUT_LEGACY) == "user"
    assert get_action_task_name(TASK_LAYOUT_USER_SPLIT_NO_WORLD) == "user_action"


def test_user_split_balanced_40_schedule_is_2222_per_step():
    planner = Balanced40SuperCycle(seed=42, layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    total = {task: 0 for task in get_task_ids(TASK_LAYOUT_USER_SPLIT_NO_WORLD)}
    for macro_step in range(40):
        allocation = planner.allocation_at(macro_step)
        assert allocation == {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
        for task, count in allocation.items():
            total[task] += count
    assert total == {"material": 80, "user_action": 80, "user_chain": 80, "recommendation": 80}


def test_user_split_two_rank_loader_keeps_four_local_and_eight_global_microbatches():
    task_ids = get_task_ids(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    allocation = {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
    rank0 = MultiTaskMacroStepLoader(
        make_user_split_task_loaders(rank=0, world_size=1),
        allocation,
        max_steps=4,
        supercycle=Balanced40SuperCycle(seed=42, layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD),
        rank=0,
        world_size=2,
        synchronized_global_consumption=True,
        task_ids=task_ids,
    )
    rank1 = MultiTaskMacroStepLoader(
        make_user_split_task_loaders(rank=0, world_size=1),
        allocation,
        max_steps=4,
        supercycle=Balanced40SuperCycle(seed=42, layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD),
        rank=1,
        world_size=2,
        synchronized_global_consumption=True,
        task_ids=task_ids,
    )
    total = {task: 0 for task in task_ids}
    for _ in range(4):
        first, second = next(rank0), next(rank1)
        assert sum(map(len, first.values())) == sum(map(len, second.values())) == 4
        for task in task_ids:
            total[task] += len(first.get(task, [])) + len(second.get(task, []))
    # Four macro-steps x two global packs per task => eight packs per task overall.
    assert total == {"material": 8, "user_action": 8, "user_chain": 8, "recommendation": 8}


def test_data_args_validates_layout_specific_allocation_and_gradnorm_tasks():
    from llamafactory.hparams.data_args import DataArguments

    base = dict(
        multitask_macro_training=True,
        multitask_global_microbatch_ddp=True,
        multitask_supercycle_mode="balanced_40",
        multitask_max_pack_length=1024,
    )
    args = DataArguments(
        **base,
        multitask_task_layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD,
        multitask_microbatch_allocation={"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2},
        multitask_gradient_control_enabled=True,
        multitask_gradient_monitor_enabled=True,
        multitask_gradnorm_enabled=True,
        multitask_gradnorm_tasks=["material", "user_action", "user_chain", "recommendation"],
    )
    assert args.multitask_task_layout == TASK_LAYOUT_USER_SPLIT_NO_WORLD

    # Legacy layout must reject four-task allocation and four-task GradNorm.
    try:
        DataArguments(
            **base,
            multitask_task_layout=TASK_LAYOUT_LEGACY,
            multitask_microbatch_allocation={"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy layout accepted user_split allocation")

    try:
        DataArguments(
            **base,
            multitask_task_layout=TASK_LAYOUT_LEGACY,
            multitask_microbatch_allocation={"material": 2, "user": 2, "recommendation": 2, "world": 2},
            multitask_gradient_control_enabled=True,
            multitask_gradient_monitor_enabled=True,
            multitask_gradnorm_enabled=True,
            multitask_gradnorm_tasks=["material", "user_action", "user_chain", "recommendation"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy layout accepted four-task GradNorm tasks")


def test_user_split_layout_rejects_world_in_allocation():
    from llamafactory.hparams.data_args import DataArguments

    try:
        DataArguments(
            multitask_macro_training=True,
            multitask_task_layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD,
            multitask_microbatch_allocation={"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 1, "world": 1},
            multitask_max_pack_length=1024,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("user_split layout accepted world task")
if __name__ == "__main__":
    tests = [
        test_task_and_subtask_do_not_mix_in_v1_pack,
        test_pack_offsets_positions_and_supervised_count,
        test_macro_step_keeps_task_groups_and_has_eight_microbatches,
        test_ddp_plans_have_equal_lengths_and_resume_is_reproducible,
        test_ddp_short_subtask_is_padded_instead_of_empty,
        test_two_rank_global_macro_keeps_eight_microbatches_total,
        test_balanced_40_supercycle_has_requested_global_counts,
        test_balanced_40_two_rank_loader_has_four_local_and_eight_global_microbatches,
        test_balanced_40_loader_resume_keeps_next_rank_local_microbatches,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    for test in [
        test_user_split_layout_maps_and_legacy_defaults_unchanged,
        test_user_split_balanced_40_schedule_is_2222_per_step,
        test_user_split_two_rank_loader_keeps_four_local_and_eight_global_microbatches,
        test_data_args_validates_layout_specific_allocation_and_gradnorm_tasks,
        test_user_split_layout_rejects_world_in_allocation,
    ]:
        test()
        print(f"PASS {test.__name__}")



def test_loss_monitoring_fields_keep_raw_means_and_match_gradnorm_scaled_global_loss():
    class FixedWeights:
        def task_weight(self, task_name):
            return {"material": 1.5, "user": 0.5, "recommendation": 2.0, "world": 1.0}[task_name]

    trainer = object.__new__(MultiTaskMacroSeq2SeqTrainer)
    trainer.gradient_controller = FixedWeights()
    statistics = torch.zeros((len(TASK_IDS), 5))
    # (loss_sum, global microbatch_count): raw means are 3, 1, 5, and 2.
    statistics[TASK_IDS["material"], :2] = torch.tensor([6.0, 2.0])
    statistics[TASK_IDS["user"], :2] = torch.tensor([4.0, 4.0])
    statistics[TASK_IDS["recommendation"], :2] = torch.tensor([10.0, 2.0])
    statistics[TASK_IDS["world"], :2] = torch.tensor([2.0, 1.0])

    fields = trainer._loss_monitoring_fields(statistics)
    assert fields["loss_raw_material"] == 3.0
    assert fields["loss_raw_user"] == 1.0
    assert fields["loss_raw_recommendation"] == 5.0
    assert fields["loss_raw_world"] == 2.0
    # (1.5*6 + 0.5*4 + 2*10 + 1*2) / (2+4+2+1) = 11 / 3.
    assert fields["loss_gradnorm_total"] == 11.0 / 3.0


def test_multitask_dataset_version_resolution_supports_one_subtask_override():
    args = type(
        "Args",
        (),
        {
            "multitask_train_dataset_suffix": "_train98",
            "multitask_dataset_version": "v1_thought_prompt",
            "multitask_dataset_version_overrides": {"onereason_user_action_nocot": "raw"},
        },
    )()
    assert resolve_multitask_dataset_name("onereason_material_cot", args) == "onereason_material_cot_v1_thought_prompt_train98"
    assert resolve_multitask_dataset_name("onereason_user_action_nocot", args) == "onereason_user_action_nocot_train98"
