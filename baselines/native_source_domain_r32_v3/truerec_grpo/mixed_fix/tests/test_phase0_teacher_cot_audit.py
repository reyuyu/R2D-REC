import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "phase0_teacher_cot_audit.py"
SPEC = importlib.util.spec_from_file_location("phase0_teacher_cot_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


SID1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
SID2 = "<|video_begin|><s_a_4><s_b_5><s_c_6>"
HIST = "<|video_begin|><s_a_9><s_b_8><s_c_7>"


def record(gid="g1", **changes):
    value = {
        "recommendation_group_id": gid, "target_domain": "video", "stage": "stage1",
        "hierarchy_class": "A_RICH", "system": "nothink-system",
        "user_content_nothink": f"history {HIST}\n/no_think", "fixed_domain_token": "<|video_begin|>",
        "all_gold_sids": [SID1], "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"], "history_sids": [HIST],
    }
    value.update(changes)
    return value


def source(gid="g1", output="<think>reason</think> suffix", **changes):
    metadata = {"recommendation_group_id": gid, "recommendation_all_gold_sids": [SID1], "recommendation_current_gold_sid": SID1}
    value = {
        "source_segment": "recommendation_cot", "system": "think-system",
        "instruction": f"history {HIST}\n/think", "input": "", "output": output,
        "aux_metadata_json": json.dumps(metadata),
    }
    value.update(changes)
    return value


def run(records=None, rows=None, splits=None):
    records = records or [record()]
    ids = {x["recommendation_group_id"] for x in records}
    split_map = splits or {"train": ids, "dev": set(), "final": set(), "probe": set()}
    return audit.audit_records(records, rows if rows is not None else [source()], split_map)


def test_unique_join_is_ready():
    assert run()["counts"]["eligible"] == 1


def test_duplicate_identical_is_unique():
    result = run(rows=[source(), source()])
    assert result["counts"]["teacher_cot_unique"] == 1


def test_distinct_duplicates_are_ambiguous():
    result = run(rows=[source(output="<think>a</think>"), source(output="<think>b</think>")])
    assert result["counts"]["teacher_cot_ambiguous"] == 1


def test_missing_is_reported():
    assert run(rows=[])["counts"]["teacher_cot_missing"] == 1


def test_missing_close_is_invalid():
    assert run(rows=[source(output="<think>unfinished")])["counts"]["teacher_cot_invalid_no_think_close"] == 1


def test_suffix_is_excluded():
    assert audit.extract_teacher_cot("<think>x</think> bridge SID") == "<think>x</think>"


def test_gold_sid_in_cot_is_leak():
    result = run(rows=[source(output=f"<think>{SID1}</think> suffix")])
    assert result["cot_gold_sid_leak_suspect_count"] == 1 and not result["curriculum2048_direct_reuse"]


def test_history_sid_alone_is_not_leak():
    result = run(rows=[source(output=f"<think>{HIST}</think> suffix")])
    assert result["cot_gold_sid_leak_suspect_count"] == 0


def test_split_overlap_fails_closed():
    result = run(splits={"train": {"g1"}, "dev": {"g1"}, "final": set(), "probe": set()})
    assert result["split_audit"]["train_dev_overlap"] == 1 and not result["curriculum2048_direct_reuse"]


@pytest.mark.parametrize("kind", ["domain", "gold", "history", "route"])
def test_contract_conflicts_fail(kind):
    row = source()
    if kind == "route": row["instruction"] = f"history {HIST}\n/no_think"
    elif kind == "history": row["instruction"] = "history absent\n/think"
    else:
        meta = json.loads(row["aux_metadata_json"])
        if kind == "gold": meta["recommendation_current_gold_sid"] = SID2
        if kind == "domain": meta["recommendation_all_gold_sids"] = ["<|prod_begin|><s_a_1><s_b_2><s_c_3>"]
        row["aux_metadata_json"] = json.dumps(meta)
    result = run(rows=[row])
    assert result["conflicts"][kind] == 1 and not result["curriculum2048_direct_reuse"]


def test_no_gpu_or_model_operations_are_declared():
    contract = run()["contract"]
    assert contract["gpu_started"] is contract["model_loaded"] is contract["generation_started"] is False
    assert contract["backward_started"] is False and contract["optimizer_steps"] == 0


def test_lineage_rejects_history_mismatch(tmp_path):
    raw = tmp_path / "raw.jsonl"
    lineage = tmp_path / "lineage.jsonl"
    raw.write_text(json.dumps({"instruction": "history absent", "input": "", "output": f"<think>x</think>{SID1}"}) + "\n")
    lineage.write_text(json.dumps(source()) + "\n")
    with pytest.raises(audit.AuditError, match="HISTORY_MISMATCH"):
        list(audit.iter_lineage_teacher_rows(raw, lineage))


def test_statuses_are_exclusive_and_exhaustive():
    records = [record(f"g{i}") for i in range(4)]
    rows = [source("g0"), source("g1", "<think>x"), source("g2", "<think>a</think>"), source("g2", "<think>b</think>")]
    result = run(records, rows)
    assert sum(result["counts"][f"teacher_cot_{s.lower()}"] for s in audit.STATUSES) == 4
