#!/usr/bin/env python3
"""Unit tests for Mini-Short-CoT section extraction."""

import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "scripts/build_mini_short_cot.py"
spec = importlib.util.spec_from_file_location("build_mini_short_cot", MODULE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_three_section_cot():
    old = "<think>套话\n#### 【兴趣归纳】\n兴趣正文\n#### 【行为模式】\n行为正文\n#### 【预测总结】\n预测正文</think>最终答案"
    new, stats = module.short_cot_output(old)
    assert new == "<think>\n【兴趣归纳】\n兴趣正文\n</think>最终答案"
    assert stats["had_behavior"] and stats["had_prediction"]


def test_bold_heading_and_colon():
    old = "<think>开场\n### **【兴趣归纳】**：\n兴趣正文\n### **【行为模式】**\n行为正文</think>答案"
    new, _ = module.short_cot_output(old)
    assert new == "<think>\n【兴趣归纳】\n兴趣正文\n</think>答案"


def test_already_short_interest_only():
    old = "<think>开场\n【兴趣归纳】：\n唯一正文</think>答案"
    new, stats = module.short_cot_output(old)
    assert new == "<think>\n【兴趣归纳】\n唯一正文\n</think>答案"
    assert not stats["had_behavior"] and not stats["had_prediction"]


def test_missing_interest_fails_closed():
    try:
        module.short_cot_output("<think>没有目标段</think>答案")
    except ValueError as error:
        assert "interest-summary" in str(error)
    else:
        raise AssertionError("missing interest marker must fail")


def test_numbered_interest_heading():
    old = "<think>开场\n### 报告\n#### 1. 兴趣归纳\n兴趣正文</think>答案"
    new, stats = module.short_cot_output(old)
    assert new == "<think>\n【兴趣归纳】\n兴趣正文\n</think>答案"
    assert stats["marker_variant"] == "numbered_heading"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} tests")
