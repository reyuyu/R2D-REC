from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "data" / "build_fixed_domain_abc_records.py"
SPEC = importlib.util.spec_from_file_location("build_fixed_domain_abc_records", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def sid(domain: str, a: int, b: int, c: int):
    return domain, a, b, c


def source_group(group_id="g", domain="video", gold=None):
    gold = gold or (sid(domain, 1, 2, 3),)
    return {
        "recommendation_group_id": group_id,
        "system": "system",
        "user_content_nothink": "history /no_think",
        "target_domain": domain,
        "gold": gold,
        "history": (),
        "novelty": "N0",
        "K": len(gold),
        "K_bucket": "K=1" if len(gold) == 1 else "K=2",
    }


class FakeEncoder:
    token_re = module.re.compile(r"<\|(?:video|prod|ad|living)_begin\|>|<s_[abc]_\d+>")

    def __call__(self, text):
        tokens = self.token_re.findall(text)
        if "".join(tokens) == text:
            return [int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) for token in tokens]
        return [ord(char) for char in text]


class FixedDomainABCRecordTest(unittest.TestCase):
    def test_group_level_one_record_contract(self):
        groups = {}
        gold = (sid("video", 1, 2, 3),)
        module.merge_source_row(groups, "g", "s", "u /no_think", "/think", "video", gold, ())
        module.merge_source_row(groups, "g", "s", "u /no_think", "/no_think", "video", gold, ())
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups["g"]["source_rows"], 2)

    def test_route_marker_only_transformation(self):
        canonical, marker = module.canonicalize_route("unchanged history /think")
        self.assertEqual(canonical, "unchanged history /no_think")
        self.assertEqual(marker, "/think")
        with self.assertRaises(module.AuditError):
            module.canonicalize_route("/think extra")

    def test_canonical_prompt_conflict_detection(self):
        groups = {}
        gold = (sid("video", 1, 2, 3),)
        module.merge_source_row(groups, "g", "s", "u1", "/think", "video", gold, ())
        module.merge_source_row(groups, "g", "s", "u2", "/no_think", "video", gold, ())
        self.assertEqual(len(groups["g"]["users"]), 2)

    def test_fixed_domain_mapping(self):
        self.assertEqual(module.DOMAIN_TOKEN, {
            "video": "<|video_begin|>", "prod": "<|prod_begin|>",
            "ad": "<|ad_begin|>", "living": "<|living_begin|>",
        })

    def test_domain_token_belongs_to_context(self):
        record = module.build_record(source_group(), "train_pool")
        self.assertEqual(record["fixed_domain_token"], "<|video_begin|>")
        self.assertNotIn(record["fixed_domain_token"], record["all_gold_abc"][0])

    def test_abc_excludes_domain(self):
        record = module.build_record(source_group(), "train_pool")
        self.assertEqual(record["all_gold_abc"], ["<s_a_1><s_b_2><s_c_3>"])

    def test_multi_positive_gold_preserved(self):
        group = source_group(gold=(sid("video", 1, 2, 3), sid("video", 4, 5, 6)))
        record = module.build_record(group, "dev")
        self.assertEqual(len(record["all_gold_sids"]), 2)
        self.assertEqual(len(record["all_gold_abc"]), 2)

    def test_bridge_contamination_detection(self):
        self.assertTrue(module.has_bridge("s", "该用户最近喜欢的视频有: x"))
        self.assertFalse(module.has_bridge("s", "clean /no_think"))

    def test_abc_exactly_three_tokens(self):
        audit = module.audit_token_contract([source_group()], FakeEncoder())
        self.assertEqual(audit["domain_token_single_token"], "PASS")
        self.assertEqual(audit["abc_action_exactly_3_tokens"], "PASS")

    def test_sft_prefix_before_a_token_parity(self):
        prompt = [1, 2]
        response = [3, 4, 5, 6, 7, 8]
        context = [1, 2, 3, 4]
        passed, position = module.prefix_parity(prompt, response, context, [4, 5, 6, 7])
        self.assertTrue(passed)
        self.assertEqual(position, 3)

    def test_frozen_split_manifest_matching(self):
        groups = [source_group("train"), source_group("dev"), source_group("final")]
        splits = {"train_pool": {"train"}, "dev": {"dev"}, "final": {"final"}, "probe": {"dev"}}
        records = module.assign_records(groups, splits)
        self.assertEqual({name: len(records[name]) for name in records}, {
            "train_pool": 1, "dev": 1, "final": 1, "probe": 1,
        })

    def test_probe_subset_of_dev(self):
        dev, probe = {"a", "b"}, {"b"}
        self.assertTrue(probe <= dev)
        self.assertFalse({"c"} <= dev)


if __name__ == "__main__":
    unittest.main()
