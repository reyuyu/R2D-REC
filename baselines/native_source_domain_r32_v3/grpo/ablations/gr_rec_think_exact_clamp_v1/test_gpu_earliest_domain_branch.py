"""CPU contract tests for the earliest Domain branch diagnostic."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_gpu_earliest_domain_branch as audit


class TinyTokenizer:
    declarations = {
        audit.DECLARATIONS["video"]: [10, 11, 20, 30],
        audit.DECLARATIONS["prod"]: [10, 11, 21, 31],
        audit.DECLARATIONS["ad"]: [10, 11, 22, 32],
        audit.DECLARATIONS["living"]: [10, 11, 23, 33],
    }
    singles = {
        "</think>": [1],
        "<|video_begin|>": [40],
        "<|prod_begin|>": [41],
        "<|ad_begin|>": [42],
        "<|living_begin|>": [43],
        "<s_a_1>": [51],
        "<s_b_2>": [52],
        "<s_c_3>": [53],
    }

    def encode(self, text, add_special_tokens=False):
        if text in self.declarations:
            return self.declarations[text]
        return self.singles[text]

    def convert_ids_to_tokens(self, token_id):
        return f"tok-{token_id}"

    def decode(self, token_ids):
        return f"decoded-{token_ids[0]}"


tokenizer = TinyTokenizer()
spec = audit.build_template_spec(tokenizer)
assert spec["longest_common_prefix_token_ids"] == [10, 11]
assert spec["earliest_branch_declaration_token_index"] == 2
assert {
    domain: value["token_id"]
    for domain, value in spec["earliest_branch_tokens"].items()
} == {"video": 20, "prod": 21, "ad": 22, "living": 23}

completion = [1, 10, 11, 20, 30, 40, 51, 52, 53]
alignment = audit.locate_earliest_branch(
    completion, ("video", 1, 2, 3), tokenizer, spec
)
assert alignment.valid is True
assert alignment.declaration_start == 1
assert alignment.branch_position == 3
assert alignment.branch_token_id == 20

missing = audit.locate_earliest_branch(
    [1, 40, 51, 52, 53], ("video", 1, 2, 3), tokenizer, spec
)
assert missing.valid is False
assert missing.failure == "missing_exact_declaration_template"

mismatch = audit.locate_earliest_branch(
    [1, 10, 11, 21, 31, 40, 51, 52, 53],
    ("video", 1, 2, 3), tokenizer, spec,
)
assert mismatch.valid is False
assert mismatch.failure == "declaration_sid_domain_mismatch"


def paired_payload(branch_probability, branch_norm, aligned=True):
    reference_groups = []
    current_groups = []
    parity = []
    for index in range(8):
        domain_only = index in {0, 1, 3, 4, 5}
        reference_groups.append({
            "legacy_grad_norm": 1.0,
            "old_sid_domain_hier_grad_norm": 0.01,
            "new_text_domain_hier_grad_norm": 0.01,
            "old_stage_active": {
                "domain": domain_only, "a": False, "b": False, "c": False,
            },
        })
        current_groups.append({
            "audit_index": index,
            "group_id": str(index),
            "rollout_fingerprint": str(index),
            "rewards": [-0.25, 0.0] * 4,
            "domain_only": domain_only,
            "branch_alignment_valid": aligned,
            "branch_hier_grad_norm": branch_norm if domain_only else None,
            "legacy_branch_hier_cosine": 1.0 if domain_only else None,
            "candidates": [{
                "earliest_branch": {"old_probability": branch_probability},
                "text_domain_noun": {"old_probability": 0.9999},
                "sid_domain": {"old_probability": 0.9999},
            }] * 8,
        })
        parity.append({
            "valid": True, "rollout_fingerprint": True, "rewards": True,
        })
    return audit.build_paired_comparison(
        {"groups": reference_groups},
        {
            "groups": current_groups,
            "trainable_parameter_checksum_before": "same",
            "trainable_parameter_checksum_after": "same",
            "parameter_change": False,
        },
        parity,
    )


assert paired_payload(0.5, 0.5)["summary"]["verdict"] == "EARLIEST_BRANCH_RECOVERS_SIGNAL"
assert paired_payload(0.9999, 0.001)["summary"]["verdict"] == "DOMAIN_DECISION_ALREADY_SATURATED"
assert paired_payload(0.5, 0.5, aligned=False)["summary"]["verdict"] == "BRANCH_POINT_AMBIGUOUS"
assert audit.reference_group_is_domain_only({
    "old_stage_active": {"domain": True, "a": False, "b": False, "c": False}
}) is True
assert audit.reference_group_is_domain_only({
    "old_stage_active": {"domain": True, "a": True, "b": False, "c": False}
}) is False

tree = ast.parse(inspect.getsource(audit))
assert not [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "step"
]

print("EARLIEST DOMAIN BRANCH AUDIT CPU TESTS PASSED")
