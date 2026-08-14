"""One-row real Arrow/collator integration check; CPU only, no model forward."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from datasets import load_from_disk


BASELINE_ROOT = Path("/data/baselines/native_source_domain_r32_v3")
REFERENCE_SRC = Path("/data/reference/llamafactory-01398eb/src")
VERIFIED_TOKENIZED_PATH = "/data/tokenized/onereason_beta_material_aligned_packratio_verified_v1"
CONFIG = BASELINE_ROOT / "config" / "train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015.yaml"

os.environ["MATERIAL_DOMAIN_MANIFEST"] = "/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json"
os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REFERENCE_SRC))
sys.argv = ["pack_ratio_collator_real", str(CONFIG)]

native_path = BASELINE_ROOT / "scripts" / "train_native_source_domain_r32_v3.py"
spec = importlib.util.spec_from_file_location("native_pack_ratio_collator_real", native_path)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import DataArguments, ModelArguments  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402


def main() -> None:
    dataset = load_from_disk(VERIFIED_TOKENIZED_PATH)["train"]
    assert len(dataset) == 33616
    feature = dataset[0]
    assert "pack_task_id" in feature
    sampler = native._get_train_sampler_with_pack_ratio(
        SimpleNamespace(train_dataset=dataset, args=SimpleNamespace(seed=20260806))
    )
    assert len(sampler) == len(dataset)
    assert len(sampler.build_plan()[0]) == len(dataset)

    model_args = ModelArguments(model_name_or_path="/data/models/onereason-8b-pretrain-competition")
    data_args = DataArguments(template="qwen3_nothink")
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    previous_unpad = SFTDataCollatorWith4DAttentionMask._unpad_packed_features
    try:
        SFTDataCollatorWith4DAttentionMask._unpad_packed_features = staticmethod(native._unpad_packed_features_with_weights)
        collator = SFTDataCollatorWith4DAttentionMask(
            template=template,
            model=None,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX,
            block_diag_attn=False,
            neat_packing=True,
            attn_implementation="flash_attention_2",
            compute_dtype=model_args.compute_dtype,
            **tokenizer_module,
        )
        batch = native._collate_with_rec_pu_metadata(collator, [feature])
    finally:
        SFTDataCollatorWith4DAttentionMask._unpad_packed_features = previous_unpad

    assert "pack_task_id" not in batch
    assert "input_ids" in batch and "labels" in batch and "loss_weights" in batch
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1] == batch["loss_weights"].shape[1]
    print("PACK_RATIO_REAL_COLLATOR=PASS cache=" + VERIFIED_TOKENIZED_PATH)


if __name__ == "__main__":
    main()
