# -*- coding: utf-8 -*-
"""CPU tests for the deterministic grounded-interest parser."""
import json

from gr_rec_dsr_v1.dsr_objectives import cot_score
from gr_rec_dsr_v1.dsr_parser import parse_interest_section


SID1 = "<|prod_begin|><s_a_1><s_b_11><s_c_111>"
SID2 = "<|prod_begin|><s_a_2><s_b_22><s_c_222>"
SID3 = "<|video_begin|><s_a_3><s_b_33><s_c_333>"
SID4 = "<|living_begin|><s_a_4><s_b_44><s_c_444>"
FAKE = "<|ad_begin|><s_a_999><s_b_999><s_c_999>"
PROMPT = f"历史：{SID1}，{SID2}，{SID3}，{SID4}。"


def render(items, heading="### **【兴趣归纳】**", close="</think>", suffix=""):
    lines = ["<think>", heading, "该用户主要有："]
    lines.extend(f"{number}. **{title}**：{text}" for number, title, text in items)
    lines.extend([close, suffix])
    return "\n".join(lines)


def test_standard_four():
    parsed = parse_interest_section(render([
        (1, "服饰", SID1), (2, "家居", SID2), (3, "影视", SID3), (4, "生活", SID4),
    ]), PROMPT)
    assert parsed.parser_success and parsed.bullet_count == 4 and parsed.grounded_count == 4


def test_two_one_and_more_than_four():
    assert parse_interest_section(render([(1, "A", SID1), (2, "B", SID2)]), PROMPT).grounded_count == 2
    assert parse_interest_section(render([(1, "A", SID1)]), PROMPT).grounded_count == 1
    items = [(index, str(index), SID1 if index % 2 else SID2) for index in range(1, 6)]
    parsed = parse_interest_section(render(items), PROMPT)
    assert parsed.grounded_count == 5 and cot_score(parsed)["count_gate"] == 0.5


def test_missing_and_malformed_numbering():
    missing = parse_interest_section("<think>没有对应 section</think>", PROMPT)
    assert not missing.section_found and not missing.parser_success
    malformed = parse_interest_section(render([(1, "A", SID1), (3, "B", SID2)]), PROMPT)
    assert malformed.parser_success and malformed.numbering == (1, 3)
    assert not malformed.numbering_sequential


def test_heading_variants():
    for heading in ("【兴趣归纳】", "**【兴趣归纳】**", "### **【兴趣归纳】**", "### 【兴趣归纳】"):
        assert parse_interest_section(render([(1, "A", SID1), (2, "B", SID2)], heading), PROMPT).parser_success


def test_ungrounded_and_fake_evidence():
    no_sid = parse_interest_section(render([(1, "A", "只有文字")]), PROMPT)
    fake = parse_interest_section(render([(1, "A", FAKE)]), PROMPT)
    assert no_sid.grounded_count == 0
    assert fake.bullets[0].evidence and not fake.bullets[0].grounded_evidence


def test_repeated_and_disjoint_evidence_diversity():
    repeated = parse_interest_section(render([(1, "A", SID1), (2, "B", SID1)]), PROMPT)
    disjoint = parse_interest_section(render([(1, "A", SID1), (2, "B", SID2)]), PROMPT)
    assert cot_score(repeated)["evidence_diversity"] == 0.0
    assert cot_score(disjoint)["evidence_diversity"] == 1.0
    assert cot_score(disjoint)["s_cot"] > cot_score(repeated)["s_cot"]


def test_think_close_and_next_section_truncation():
    cot = render([(1, "A", SID1), (2, "B", SID2)], suffix=f"1. 后续伪 bullet {SID3}")
    assert parse_interest_section(cot, PROMPT).bullet_count == 2
    cot2 = render([(1, "A", SID1), (2, "B", SID2)], close="### **【行为模式】**", suffix=f"1. 后续 {SID3}")
    assert parse_interest_section(cot2, PROMPT).bullet_count == 2


def test_real_json_output_example():
    parsed = parse_interest_section(render([
        (1, "服饰穿搭", f"关注秋冬服饰 {SID1}"),
        (2, "家居生活", f"浏览家居用品 {SID2}"),
        (3, "虚构兴趣", FAKE),
    ]), PROMPT)
    payload = parsed.to_dict() | cot_score(parsed)
    assert payload["bullet_count"] == 3
    assert payload["grounded_count"] == 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from gr_rec_dsr_v1.run_cpu_tests import main
    raise SystemExit(main([__name__]))
