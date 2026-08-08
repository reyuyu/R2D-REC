#!/usr/bin/env python3
"""CPU-only tests for Recommendation V3 data and provenance plumbing."""
from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset  # noqa: E402

from llamafactory.data.converter import align_dataset  # noqa: E402
from llamafactory.data.data_utils import Role  # noqa: E402
from llamafactory.data.multitask import TaskPackCollator, TokenizedSubDataset  # noqa: E402
from llamafactory.data.parser import DatasetAttr  # noqa: E402
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from create_recommendation_v3_multi_positive import generate  # noqa: E402


META = {
    "recommendation_group_id": "g1",
    "recommendation_group_size": 2,
    "recommendation_all_gold_sids": ["<|prod_begin|><s_a_1><s_b_2><s_c_3>", "<|prod_begin|><s_a_4><s_b_5><s_c_6>"],
    "recommendation_current_gold_sid": "<|prod_begin|><s_a_1><s_b_2><s_c_3>",
}


def make_row(prompt: str, sid: str, mode: str, cot: str = "reasoning") -> dict:
    output = sid if mode == "nocot" else f"<think>{cot}</think>answer: {sid}"
    return {"instruction": "rec", "input": prompt + "\n/" + ("think" if mode == "cot" else "no_think"), "output": output, "history": []}


def test_generation_and_text_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sid1 = "<|prod_begin|><s_a_1><s_b_2><s_c_3>"
        sid2 = "<|prod_begin|><s_a_4><s_b_5><s_c_6>"
        cot = root / "cot.jsonl"
        nocot = root / "nocot.jsonl"
        cot_rows = [make_row("history-A", sid1, "cot"), make_row("history-B", sid2, "cot")]
        nocot_rows = [make_row("history-A", sid2, "nocot"), make_row("history-B", sid1, "nocot")]
        for path, rows in ((cot, cot_rows), (nocot, nocot_rows)):
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        dataset_info = root / "dataset_info.json"
        dataset_info.write_text(json.dumps({"onereason_recommendation_cot_v2_dual": {"file_name": str(cot)}, "onereason_recommendation_nocot_v2_dual": {"file_name": str(nocot)}}), encoding="utf-8")
        versions = root / "versions.json"
        versions.write_text(json.dumps({"versions": {}}), encoding="utf-8")
        output = root / "v3"
        generate(Namespace(cot_source=cot, nocot_source=nocot, output_root=output, dataset_info=dataset_info, version_manifest=versions, version="v3", parent="v2"))
        out_rows = []
        for path in sorted(output.glob("*.jsonl")):
            out_rows.extend(json.loads(line) for line in path.open(encoding="utf-8"))
        assert len(out_rows) == 4
        assert {row["recommendation_current_gold_sid"] for row in out_rows} == {sid1, sid2}
        assert all(row["recommendation_group_size"] == 2 for row in out_rows)
        source = cot_rows + nocot_rows
        strip = lambda row: {key: value for key, value in row.items() if not key.startswith("recommendation_")}
        assert sorted(json.dumps(strip(row), ensure_ascii=False, sort_keys=True) for row in out_rows) == sorted(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in source)


def test_converter_processor_and_pack_metadata() -> None:
    row = {"instruction": "rec", "input": "history\n/think", "output": "<think>x</think><|prod_begin|><s_a_1><s_b_2><s_c_3>", "history": [], **META}
    dataset = Dataset.from_list([row])
    attr = DatasetAttr(load_from="file", dataset_name="fixture", formatting="alpaca", prompt="instruction", query="input", response="output", history="history")
    data_args = SimpleNamespace(media_dir="", streaming=False, preprocessing_num_workers=1, overwrite_cache=True)
    training_args = SimpleNamespace(local_process_index=0)
    aligned = align_dataset(dataset, attr, data_args, training_args)
    assert aligned[0]["_recommendation_metadata"] == META

    processor = object.__new__(SupervisedDatasetProcessor)
    processor._encode_data_example = lambda **_: ([1, 2, 3], [IGNORE_INDEX, 2, 3])
    examples = {
        "_prompt": [[{"role": Role.USER.value, "content": "x"}]],
        "_response": [[{"role": Role.ASSISTANT.value, "content": "y"}]],
        "_system": [""], "_tools": [""], "_images": [None], "_videos": [None], "_audios": [None],
        "_recommendation_metadata": [META],
    }
    tokenized = processor.preprocess_dataset(examples)
    assert tokenized["recommendation_metadata"] == [META]
    plain_examples = {key: value for key, value in examples.items() if key != "_recommendation_metadata"}
    plain_tokenized = processor.preprocess_dataset(plain_examples)
    assert {key: value for key, value in tokenized.items() if key != "recommendation_metadata"} == plain_tokenized

    token_dataset = Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3], "recommendation_metadata": META}])
    wrapped = TokenizedSubDataset(token_dataset, "recommendation", "cot", 0)
    sample = wrapped[0]
    assert sample["sample_metadata"]["recommendation_multi_positive"] == META
    packed = TaskPackCollator()([sample])
    assert packed["sample_metadata"][0]["recommendation_multi_positive"] == META
    meta2 = dict(META, recommendation_group_id="g2", recommendation_current_gold_sid=META["recommendation_all_gold_sids"][1])
    sample2 = dict(sample, sample_metadata={"recommendation_multi_positive": meta2}, sample_id=1)
    packed_two = TaskPackCollator()([sample, sample2])
    assert [item["recommendation_multi_positive"]["recommendation_group_id"] for item in packed_two["sample_metadata"]] == ["g1", "g2"]
    model_keys = set(MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS)
    assert "recommendation_metadata" not in model_keys and "sample_metadata" not in model_keys


if __name__ == "__main__":
    test_generation_and_text_identity()
    test_converter_processor_and_pack_metadata()
    print("PASS recommendation V3 multi-positive CPU tests")
