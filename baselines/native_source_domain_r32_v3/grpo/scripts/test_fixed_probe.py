"""CPU-only tests for fixed held-out Probe selection and scheduling."""
import json
import tempfile
from pathlib import Path

from grpo_probe import (
    load_probe_records,
    probe_due,
    select_probe_group_ids,
    validate_probe_schedule,
)
from grpo_trl_trainer import build_route_dataset


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError from {fn.__name__}")


with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "train.jsonl"
    rows = []
    for index in range(12):
        gid = f"group-{index:02d}"
        for route in ("think", "no_think"):
            rows.append({
                "recommendation_group_id": gid,
                "route": route,
                "prompt": f"{route} prompt {index}",
                "target_domain": "video",
                "all_gold_sids": ["<|video_begin|><s_a_1><s_b_2><s_c_3>"],
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    probes = select_probe_group_ids(path, 12, 7, count=4)
    assert probes == select_probe_group_ids(path, 12, 7, count=4)
    assert len(set(probes)) == 4
    records = load_probe_records(path, probes)
    assert all(set(records[gid]) == {"think", "no_think"} for gid in probes)

    dataset = build_route_dataset(path, n_groups=12, seed=7, exclude_group_ids=probes)
    assert not set(probes).intersection(dataset["recommendation_group_id"])
    assert len(set(dataset["recommendation_group_id"])) == 8

    assert select_probe_group_ids(path, 12, 7, explicit_ids=probes) == probes
    expect_value_error(select_probe_group_ids, path, 12, 7, count=3)
    expect_value_error(select_probe_group_ids, path, 12, 7, explicit_ids=probes[:3])
    validate_probe_schedule(probes, 200)
    expect_value_error(validate_probe_schedule, probes, 199)
    assert probe_due(0, 200) and probe_due(200, 200) and not probe_due(198, 200)

print("FIXED PROBE CPU TESTS PASSED")
