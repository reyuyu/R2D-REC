#!/usr/bin/env python3
"""CPU-only tests for the mini-cot manifest and shared weighting primitive."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_mini_cot_v1_4gpu_gc04_2epoch.yaml"


class TinyTokenizer:
    def get_vocab(self):
        return {"<think>": 10, "</think>": 11}


def main() -> None:
    os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
    sys.argv = ["test_mini_cot_repeat_weighting.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("mini_cot_test_native", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = TinyTokenizer()
    labels = [-100, 10, 101, 102, 11, 201, 301, 302, 303, 401]
    base = [0.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0, 8.0, 8.0, 1.0]
    updated = module._apply_alpha_cot_repeat_weights(labels, base, "recommendation_cot", 4, tokenizer)
    assert updated[1:5] == [0.125] * 4
    assert updated[5:] == base[5:]
    assert module._apply_alpha_cot_repeat_weights(labels, base, "recommendation_nocot", 0, tokenizer) == base
    original = module.ALPHA_COT_REPEAT_CONFIG
    module.ALPHA_COT_REPEAT_CONFIG = module.AlphaCotRepeatConfig(enabled=False)
    assert module._apply_alpha_cot_repeat_weights(labels, base, "recommendation_cot", 4, tokenizer) == base
    module.ALPHA_COT_REPEAT_CONFIG = original
    for bad in (0, -1):
        try:
            module._apply_alpha_cot_repeat_weights(labels, base, "recommendation_cot", bad, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive N must fail")
    print("MINI_COT_REPEAT_WEIGHTING_UNIT_PASS")


if __name__ == "__main__":
    main()
