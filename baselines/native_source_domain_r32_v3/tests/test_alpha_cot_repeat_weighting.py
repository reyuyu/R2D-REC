#!/usr/bin/env python3
"""Small CPU regression for the exact CoT repeat-weight span primitive."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_cot_repeat05n_4gpu_gc04_2epoch.yaml"


class TinyTokenizer:
    def get_vocab(self):
        return {"<think>": 10, "</think>": 11}


def load_native():
    os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
    sys.argv = ["test_alpha_cot_repeat_weighting.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("alpha_cot_repeat_test_native", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    native = load_native()
    tokenizer = TinyTokenizer()
    labels = [-100, 10, 101, 102, 11, 201, 301, 302, 303, 304, 401]
    base = [0.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0, 8.0, 8.0, 8.0, 1.0]
    updated = native._apply_alpha_cot_repeat_weights(labels, base, "recommendation_cot", 2, tokenizer)
    assert updated[:1] == [0.0]
    assert updated[1:5] == [0.25] * 4, updated
    assert updated[5:] == base[5:], updated
    unchanged = native._apply_alpha_cot_repeat_weights(labels, base, "recommendation_nocot", 0, tokenizer)
    assert unchanged == base
    original_config = native.ALPHA_COT_REPEAT_CONFIG
    native.ALPHA_COT_REPEAT_CONFIG = native.AlphaCotRepeatConfig(enabled=False)
    assert native._apply_alpha_cot_repeat_weights(labels, base, "recommendation_cot", 2, tokenizer) == base
    native.ALPHA_COT_REPEAT_CONFIG = original_config
    try:
        native._apply_alpha_cot_repeat_weights([-100, 10, 1], [0.0, 1.0, 1.0], "recommendation_cot", 1, tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("missing </think> must fail closed")
    print("ALPHA_COT_REPEAT_WEIGHTING_UNIT_PASS")


if __name__ == "__main__":
    main()
