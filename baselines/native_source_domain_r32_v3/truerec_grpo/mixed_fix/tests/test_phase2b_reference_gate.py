import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from build_paired_curriculum_v1 import publish_if_pass  # noqa: E402
from phase2_renderer_audit import external_reference_summary, phase2b_hard_gate  # noqa: E402
from teacher_cot_renderer_v1 import CANONICAL_SEPARATOR  # noqa: E402


def records():
    return [
        {"recommendation_group_id": "g1", "target_domain": "video", "stage": "stage1", "hierarchy_class": "A_RICH"},
        {"recommendation_group_id": "g2", "target_domain": "prod", "stage": "stage2", "hierarchy_class": "B_RICH"},
        {"recommendation_group_id": "g3", "target_domain": "video", "stage": "stage3", "hierarchy_class": "SINGLETON"},
    ]


def summary(prefix_fail=None, token_fail=None):
    reference = {"g1", "g2"}; prefix_fail = prefix_fail or set(); token_fail = token_fail or set()
    return external_reference_summary(
        records(), reference, reference - prefix_fail, prefix_fail,
        reference - token_fail, token_fail,
    )


def gate(external=None, **changes):
    kwargs = dict(
        constructed_context_pass=2048, constructed_context_fail=0,
        self_consistency_pass=2048, self_consistency_fail=0,
        fixed_domain_pass=2048, route_pass=2048, action_contract=True,
        overflow_count=0, split_failure_count=0,
    ); kwargs.update(changes)
    return phase2b_hard_gate(external or summary(), CANONICAL_SEPARATOR, **kwargs)


def test_missing_external_reference_is_not_parity_failure():
    result = summary()
    assert result["reference_groups"] == 2 and result["reference_missing"] == 1
    assert result["observed_prefix_fail"] == result["observed_token_fail"] == 0
    assert gate(result)


@pytest.mark.parametrize("kind", ["prefix", "token"])
def test_observed_external_mismatch_fails(kind):
    result = summary(prefix_fail={"g1"} if kind == "prefix" else set(), token_fail={"g1"} if kind == "token" else set())
    assert not gate(result)


def test_observed_denominator_is_only_covered_groups():
    with pytest.raises(ValueError, match="denominator"):
        external_reference_summary(records(), {"g1", "g2"}, {"g1"}, set(), {"g1", "g2"}, set())


def test_constructed_metrics_are_independent_hard_gates():
    assert not gate(constructed_context_pass=2047, constructed_context_fail=1)
    assert not gate(self_consistency_pass=2047, self_consistency_fail=1)


def test_publish_only_on_pass(tmp_path):
    row = {"recommendation_group_id": "g"}
    published, sha = publish_if_pass(False, tmp_path, [row] * 2048, {})
    assert not published and sha is None and not (tmp_path / "paired_curriculum2048.jsonl").exists()


def test_published_sha_matches_file(tmp_path):
    rows = [{"recommendation_group_id": f"g{i}"} for i in range(2048)]
    published, sha = publish_if_pass(True, tmp_path, rows, {"separator_source": "real audit"})
    assert published
    assert sha == hashlib.sha256((tmp_path / "paired_curriculum2048.jsonl").read_bytes()).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text())["paired_records"]["sha256"] == sha


def test_canonical_separator_provenance_constant():
    assert CANONICAL_SEPARATOR == "\n"


def test_reference_missing_distribution_is_exact():
    result = summary()
    assert result["covered_distribution"]["domain"] == {"prod": 1, "video": 1}
    assert result["missing_distribution"]["domain"] == {"video": 1}
    assert result["missing_distribution"]["stage"] == {"stage3": 1}
    assert result["missing_distribution"]["hierarchy_class"] == {"SINGLETON": 1}
