from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from build_paired_curriculum_v1 import (  # noqa: E402
    IMMUTABLE_FIELDS, assert_no_context_overflow, audit_identity, build_paired_record,
)


def source(gid="g1"):
    return {
        "recommendation_group_id": gid, "stage": "stage1", "stage_index": 0,
        "epoch1_index": 0, "target_domain": "video", "fixed_domain_token": "<|video_begin|>",
        "K_A": 2, "K_AB": 2, "K_ABC": 3, "hierarchy_class": "A_RICH",
        "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"], "all_gold_sids": ["sid"],
        "history_sids": ["history"], "system": "system", "user_content_nothink": "user/no_think",
    }


def paired(row=None):
    row = row or source()
    return build_paired_record(row, "user/think", "<think>x</think>", 10, 20)


def test_paired_record_preserves_all_immutable_metadata():
    row, result = source(), paired()
    assert all(result[field] == row[field] for field in IMMUTABLE_FIELDS)
    assert result["think_minus_nothink_token_count"] == 10


def test_identity_and_order_pass():
    rows = [source("g1"), source("g2")]
    rows[1]["epoch1_index"] = 1
    audit_identity(rows, [paired(row) for row in rows], ["g1", "g2"])


def test_identity_or_order_change_fails():
    rows = [source("g1"), source("g2")]
    with pytest.raises(ValueError, match="identity/order"):
        audit_identity(rows, [paired(rows[1]), paired(rows[0])], ["g1", "g2"])


def test_metadata_change_fails():
    row = source(); result = paired(); result["K_A"] = 99
    with pytest.raises(ValueError, match="metadata"):
        audit_identity([row], [result], ["g1"])


def test_context_overflow_fails_closed():
    result = paired(); result["think_context_token_count"] = 98
    with pytest.raises(ValueError, match="truncation"):
        assert_no_context_overflow([result], 100)


def test_context_at_boundary_passes():
    result = paired(); result["think_context_token_count"] = 97
    assert_no_context_overflow([result], 100)
