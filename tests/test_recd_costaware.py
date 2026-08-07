import copy
from types import SimpleNamespace

from llamafactory.data.multitask import CoverageDeficitScheduler, MultiTaskMacroStepLoader


def fake_macro_loader(cost_aware):
    loader = MultiTaskMacroStepLoader(
        {},
        {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2},
        max_steps=1,
        global_allocation={"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2},
        rank=0,
        world_size=2,
        cost_aware_partition=cost_aware,
    )
    packs = [
        ("material",  {"packed_token_count": 8000, "segment_lengths": [8000], "sample_ids": [0]}),
        ("material",  {"packed_token_count": 4000, "segment_lengths": [4000, 4000], "sample_ids": [1, 2]}),
        ("user_action", {"packed_token_count": 7000, "segment_lengths": [7000], "sample_ids": [3]}),
        ("user_action", {"packed_token_count": 3000, "segment_lengths": [3000, 3000], "sample_ids": [4, 5]}),
        ("user_chain", {"packed_token_count": 6000, "segment_lengths": [6000], "sample_ids": [6]}),
        ("user_chain", {"packed_token_count": 3500, "segment_lengths": [3500, 3500], "sample_ids": [7, 8]}),
        ("recommendation", {"packed_token_count": 5000, "segment_lengths": [5000], "sample_ids": [9]}),
        ("recommendation", {"packed_token_count": 2500, "segment_lengths": [2500, 2500], "sample_ids": [10, 11]}),
    ]
    return loader, packs


def test_cost_proxy_distinguishes_segment_shapes():
    loader, _ = fake_macro_loader(True)
    one = loader._pack_cost({"packed_token_count": 8000, "segment_lengths": [8000]})
    two = loader._pack_cost({"packed_token_count": 8000, "segment_lengths": [4000, 4000]})
    assert one > two
    assert abs(one - (8000 + 8000 * 8000 / 8192)) < 1e-6


def test_cost_aware_exhaustive_partition_is_four_plus_four_and_balanced():
    baseline_loader, packs = fake_macro_loader(False)
    cost_loader, _ = fake_macro_loader(True)
    baseline = baseline_loader._assign_global_microbatches(packs)
    assigned = cost_loader._assign_global_microbatches(packs)
    assert len(assigned[0]) == len(assigned[1]) == 4
    assert sorted(item[1]["sample_ids"] for item in assigned[0] + assigned[1]) == sorted(
        item[1]["sample_ids"] for item in packs
    )
    assert cost_loader._last_partition_metrics["rank_cost_gap_ratio"] <= baseline_loader._last_partition_metrics["rank_cost_gap_ratio"]
    assert cost_loader._last_partition_metrics["partition_changed"] in (0, 1)


def test_cost_aware_keeps_global_order_and_task_allocation():
    _, packs = fake_macro_loader(True)
    loader, _ = fake_macro_loader(True)
    assigned = loader._assign_global_microbatches(packs)
    assert [task for task, _ in packs] == [
        "material", "material", "user_action", "user_action",
        "user_chain", "user_chain", "recommendation", "recommendation",
    ]
    assert sum(1 for task, _ in assigned[0] + assigned[1] if task == "material") == 2
    assert sum(1 for task, _ in assigned[0] + assigned[1] if task == "user_action") == 2
    assert sum(1 for task, _ in assigned[0] + assigned[1] if task == "user_chain") == 2
    assert sum(1 for task, _ in assigned[0] + assigned[1] if task == "recommendation") == 2


def test_partition_state_and_scheduler_are_resume_deterministic():
    loader, packs = fake_macro_loader(True)
    loader.macro_step = 7
    state = copy.deepcopy(loader.state_dict())
    restored, _ = fake_macro_loader(True)
    restored.load_state_dict(state)
    assert restored.macro_step == 7
    assert restored.cost_aware_partition is True
    assert restored.attention_cost_weight == 1.0
    assert restored._assign_global_microbatches(packs)[0][0][1]["sample_ids"] == loader._assign_global_microbatches(packs)[0][0][1]["sample_ids"]


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
    print("All recD cost-aware partition tests passed.")
