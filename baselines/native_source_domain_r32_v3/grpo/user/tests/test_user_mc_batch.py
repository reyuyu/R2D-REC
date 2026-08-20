import copy
import json
import os
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_batch import MCBatchError, make_mc_policy_batch  # noqa: E402
from user_mc_rollout import prepare_mc_scored_rollout  # noqa: E402


SID_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
A = "<|video_begin|><s_a_101><s_b_201><s_c_301>"
B = "<|prod_begin|><s_a_102><s_b_202><s_c_302>"
X = "<|ad_begin|><s_a_103><s_b_203><s_c_303>"


class BatchTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.piece_to_id = {}
        self.id_to_piece = {}

    def _id(self, piece):
        if piece not in self.piece_to_id:
            token_id = len(self.piece_to_id) + 1
            self.piece_to_id[piece] = token_id
            self.id_to_piece[token_id] = piece
        return self.piece_to_id[piece]

    @staticmethod
    def _sid_pieces(sid, base):
        boundaries = [
            0,
            sid.index("><s_a_") + 1,
            sid.index("><s_b_") + 1,
            sid.index("><s_c_") + 1,
            len(sid),
        ]
        return [
            (
                sid[boundaries[index] : boundaries[index + 1]],
                base + boundaries[index],
                base + boundaries[index + 1],
            )
            for index in range(4)
        ]

    def _pieces(self, text):
        pieces = []
        cursor = 0
        for match in SID_RE.finditer(text):
            pieces.extend(
                (text[index], index, index + 1)
                for index in range(cursor, match.start())
            )
            pieces.extend(self._sid_pieces(match.group(), match.start()))
            cursor = match.end()
        pieces.extend(
            (text[index], index, index + 1) for index in range(cursor, len(text))
        )
        return pieces

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = self._pieces(text)
        result = {"input_ids": [self._id(piece) for piece, _, _ in pieces]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_piece[int(token_id)] for token_id in ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        return f"<chat>{messages[0]['content']}</chat>"


def fake_render_prompt(tokenizer, row):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(tokenizer.encode(rendered, add_special_tokens=False)) != row["prompt_token_count"]:
        raise AssertionError("prompt renderer drift")
    return rendered


def renderer_module():
    module = ModuleType("run_user_grpo_smoke")
    module.render_prompt = fake_render_prompt
    return module


def with_prompt_count(tokenizer, row):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    row["prompt_token_count"] = len(
        tokenizer.encode(rendered, add_special_tokens=False)
    )
    return row


def action_row(tokenizer, sample_id="A", prompt="short"):
    return with_prompt_count(
        tokenizer,
        {
            "sample_id": sample_id,
            "route": "action",
            "prompt": prompt,
            "gold_sids": [A, B],
            "history_sids": [A, B, X],
        },
    )


def event(date, action, logic):
    return {"date": date, "action": action, "logic": logic}


E1 = event("2026-01-01", "watch alpha", "likes alpha for a documented reason")
E2 = event("2026-01-02", "buy beta", "needs beta after comparing alternatives")


def chain_row(tokenizer, sample_id="C", prompt="chain prompt"):
    return with_prompt_count(
        tokenizer,
        {
            "sample_id": sample_id,
            "route": "chain",
            "prompt": prompt,
            "gold_events": [E1, E2],
            "history_events": [],
            "history_sids": [],
        },
    )


def chain_json(events):
    return json.dumps(
        {"logic_chain": {"name": "test", "events": events}}, separators=(",", ":")
    )


def make_rollout(tokenizer, rows, completions):
    ids = [tokenizer.encode(completion, add_special_tokens=False) for completion in completions]
    return prepare_mc_scored_rollout(rows, ids, tokenizer)


def build(tokenizer, rollout):
    with patch.dict(sys.modules, {"run_user_grpo_smoke": renderer_module()}):
        return make_mc_policy_batch(tokenizer, rollout)


class MCPolicyBatchTests(unittest.TestCase):
    def test_one_prompt_k2_produces_two_candidates(self):
        tokenizer = BatchTokenizer()
        rollout = make_rollout(
            tokenizer,
            [action_row(tokenizer)],
            [json.dumps([A]), json.dumps([A, X])],
        )
        result = build(tokenizer, rollout)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["prompt_ids"].shape[0], 2)
        self.assertEqual(result["sample_ids"], ["A", "A"])
        self.assertEqual(result["candidate_indices"], [0, 1])

    def test_two_prompt_order_and_padding_sides(self):
        tokenizer = BatchTokenizer()
        rows = [
            action_row(tokenizer, "A", "x"),
            action_row(tokenizer, "B", "a much longer prompt"),
        ]
        completions = [
            json.dumps([A]),
            json.dumps([A, X]),
            json.dumps([B, A]),
            json.dumps([B]),
        ]
        rollout = make_rollout(tokenizer, rows, completions)
        result = build(tokenizer, rollout)
        self.assertEqual(result["sample_ids"], ["A", "A", "B", "B"])
        self.assertEqual(result["candidate_indices"], [0, 1, 0, 1])
        for index, row in enumerate(rollout["expanded_rows"]):
            expected = tokenizer.encode(fake_render_prompt(tokenizer, row))
            self.assertEqual(result["prompt_ids"][index, -len(expected) :].tolist(), expected)
            self.assertTrue(bool(torch.all(result["prompt_mask"][index, -len(expected) :] == 1)))
            self.assertTrue(bool(torch.all(result["prompt_mask"][index, : -len(expected)] == 0)))
        for index, ids in enumerate(rollout["completion_ids_list"]):
            self.assertEqual(result["completion_ids"][index, : len(ids)].tolist(), ids)
            self.assertTrue(bool(torch.all(result["completion_mask"][index, : len(ids)] == 1)))
            self.assertTrue(bool(torch.all(result["completion_mask"][index, len(ids) :] == 0)))

    def test_action_sid_four_token_unit_matches_completion(self):
        tokenizer = BatchTokenizer()
        rollout = make_rollout(
            tokenizer,
            [action_row(tokenizer)],
            [json.dumps([A, X]), json.dumps([A])],
        )
        result = build(tokenizer, rollout)
        self.assertEqual(
            result["credit_units_per_candidate"],
            rollout["credit_units_per_candidate"],
        )
        action_unit = next(
            unit for unit in result["credit_units_per_candidate"][0] if unit["sid"] == A
        )
        self.assertEqual(len(action_unit["generated_token_indices"]), 4)
        actual = result["completion_ids"][0, action_unit["generated_token_indices"]].tolist()
        self.assertEqual(actual, action_unit["generated_token_ids"])

    def test_chain_long_event_unit_stays_in_real_completion(self):
        tokenizer = BatchTokenizer()
        clean = chain_json([E1, E2])
        rollout = make_rollout(
            tokenizer, [chain_row(tokenizer)], [clean, chain_json([E1])]
        )
        result = build(tokenizer, rollout)
        event_unit = result["credit_units_per_candidate"][0][0]
        self.assertGreater(len(event_unit["generated_token_indices"]), 40)
        self.assertTrue(
            all(
                index < result["completion_lengths"][0]
                for index in event_unit["generated_token_indices"]
            )
        )
        actual = result["completion_ids"][0, event_unit["generated_token_indices"]].tolist()
        self.assertEqual(actual, event_unit["generated_token_ids"])

    def test_empty_credit_candidates_are_never_filtered(self):
        tokenizer = BatchTokenizer()
        valid = json.dumps([A])
        one_empty = make_rollout(
            tokenizer, [action_row(tokenizer)], [valid, "not-json"]
        )
        one_result = build(tokenizer, one_empty)
        self.assertEqual(one_result["candidate_count"], 2)
        self.assertEqual(one_result["credit_units_per_candidate"][1], [])

        both_empty = make_rollout(
            tokenizer, [action_row(tokenizer)], ["bad-json-1", "bad-json-2"]
        )
        both_result = build(tokenizer, both_empty)
        self.assertEqual(both_result["candidate_count"], 2)
        self.assertEqual(both_result["credit_units_per_candidate"], [[], []])

    def test_credit_index_touching_padding_fails_closed(self):
        tokenizer = BatchTokenizer()
        rollout = make_rollout(
            tokenizer,
            [action_row(tokenizer)],
            [json.dumps([A]), json.dumps([A, X])],
        )
        broken = copy.deepcopy(rollout)
        unit = broken["credit_units_per_candidate"][0][0]
        unit["generated_token_indices"][0] = len(broken["completion_ids_list"][0])
        with self.assertRaisesRegex(MCBatchError, "padding"):
            build(tokenizer, broken)

    def test_credit_token_id_mismatch_fails_closed(self):
        tokenizer = BatchTokenizer()
        rollout = make_rollout(
            tokenizer,
            [action_row(tokenizer)],
            [json.dumps([A]), json.dumps([A, X])],
        )
        broken = copy.deepcopy(rollout)
        broken["credit_units_per_candidate"][0][0]["generated_token_ids"][0] += 999
        with self.assertRaisesRegex(MCBatchError, "token mismatch"):
            build(tokenizer, broken)

    def test_grpo_fields_are_absent(self):
        tokenizer = BatchTokenizer()
        rollout = make_rollout(
            tokenizer,
            [action_row(tokenizer)],
            [json.dumps([A]), json.dumps([A, X])],
        )
        result = build(tokenizer, rollout)
        forbidden = {
            "sequence_advantages",
            "token_advantages",
            "old_per_token_logps",
            "local_penalty_mask",
            "group_reward_means",
            "group_reward_stds",
        }
        self.assertTrue(set(result).isdisjoint(forbidden))


@unittest.skipUnless(
    os.environ.get("MC_PARENT_TOKENIZER") and os.environ.get("MC_DATA_DIR"),
    "real tokenizer parity paths are not configured",
)
class MCRealTokenizerParityTests(unittest.TestCase):
    def test_action_and_chain_prompt_token_counts(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["MC_PARENT_TOKENIZER"], local_files_only=True
        )
        rows = [
            json.loads(line)
            for line in (Path(os.environ["MC_DATA_DIR"]) / "train_3000.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        completion = [int(tokenizer.eos_token_id)]
        for route in ("action", "chain"):
            row = next(item for item in rows if item["route"] == route)
            rollout = {
                "K": 2,
                "token_span_space": "generated_completion_ids",
                "expanded_rows": [row, row],
                "completion_ids_list": [completion, completion],
                "credit_units_per_candidate": [[], []],
                "candidate_count": 2,
            }
            result = make_mc_policy_batch(tokenizer, rollout)
            actual = [int(value) for value in result["prompt_mask"].sum(dim=1)]
            self.assertEqual(actual, [int(row["prompt_token_count"])] * 2)


if __name__ == "__main__":
    unittest.main()
