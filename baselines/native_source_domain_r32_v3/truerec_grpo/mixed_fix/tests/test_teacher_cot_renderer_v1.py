from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from teacher_cot_renderer_v1 import (  # noqa: E402
    assert_action_boundary, audit_sft_prefix_parity, discover_canonical_separator,
    think_rl_context_ids, transform_terminal_route, validate_teacher_cot,
)


SPECIAL = re.compile(r"<\|\w+_begin\|>|<s_[abc]_\d+>")


class FakeRenderer:
    def encode(self, text):
        ids, cursor = [], 0
        for match in SPECIAL.finditer(text):
            ids.extend(1000 + ord(ch) for ch in text[cursor:match.start()])
            ids.append(abs(hash(match.group(0))) % 100000 + 200000)
            cursor = match.end()
        ids.extend(1000 + ord(ch) for ch in text[cursor:])
        return ids

    def prompt_ids(self, system, user):
        return [1] + self.encode(system) + [2] + self.encode(user) + [3]

    def full_sft_ids(self, system, user, response):
        return self.prompt_ids(system, user) + self.encode(response)


def test_terminal_route_transformation_only():
    before = "history\n/no_think"
    after = transform_terminal_route(before)
    assert after == "history\n/think"
    assert before[:-9] == after[:-6]


@pytest.mark.parametrize("value", ["/no_think trailing", "prefix/no_think\n", "prefix/think"])
def test_nonterminal_or_wrong_route_fails(value):
    with pytest.raises(ValueError, match="terminal"):
        transform_terminal_route(value)


@pytest.mark.parametrize("cot", ["<think>missing", "", "reason</think>", "<think>x</think> bridge"])
def test_invalid_teacher_boundary_fails(cot):
    with pytest.raises(ValueError):
        validate_teacher_cot(cot)


def test_single_separator_is_canonical():
    cot = "<think>x</think>"
    assert discover_canonical_separator([(cot + "\n<|video_begin|><s_a_1>", cot, "<|video_begin|>")]) == "\n"


def test_multiple_separator_variants_fail_closed():
    cot, domain = "<think>x</think>", "<|video_begin|>"
    with pytest.raises(ValueError, match="variants"):
        discover_canonical_separator([(cot + "\n" + domain, cot, domain), (cot + domain, cot, domain)])


def test_context_domain_boundary_action_and_sft_prefix_parity():
    renderer = FakeRenderer(); cot = "<think>x</think>"; domain = "<|video_begin|>"
    context = think_rl_context_ids(renderer, "sys", "user/think", cot, domain, "\n")
    boundary = assert_action_boundary(renderer, context, domain, "<s_a_1><s_b_2><s_c_3>")
    assert len(boundary["action_token_ids"]) == 3
    assert context[-1] == boundary["domain_token_id"]
    response = cot + "\n" + domain + "<s_a_1><s_b_2><s_c_3>"
    parity = audit_sft_prefix_parity(renderer, "sys", "user/think", response, context)
    assert parity.passed and parity.first_mismatch_index is None


def test_context_does_not_append_gold_action():
    renderer = FakeRenderer(); gold = "<s_a_1><s_b_2><s_c_3>"; domain = "<|video_begin|>"
    context = think_rl_context_ids(renderer, "sys", "user/think", "<think>x</think>", domain, "\n")
    assert context[-3:] != renderer.encode(gold)


def test_domain_must_be_single_token():
    class Bad(FakeRenderer):
        def encode(self, text):
            return [1, 2] if text == "domain" else super().encode(text)
    with pytest.raises(ValueError, match="one token"):
        think_rl_context_ids(Bad(), "s", "u/think", "<think>x</think>", "domain", "\n")
