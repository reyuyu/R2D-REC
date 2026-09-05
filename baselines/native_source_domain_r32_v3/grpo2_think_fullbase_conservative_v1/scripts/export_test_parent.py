#!/usr/bin/env python3
"""Merge the GRPO-1 checkpoint-300 adapter into the full SFT test parent."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from contracts import (
    GRPO1_ADAPTER_SHA256,
    SFT_MODEL_SHA256,
    canonical_model_identity,
    file_sha256,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_base(path: Path, device: str):
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device},
        attn_implementation="flash_attention_2" if device.startswith("cuda") else None,
    )


def _probe(model, tokenizer, prompts: list[str], fixed_selected_ids=None) -> list[dict]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value[:, -512:].to(model.device) for key, value in encoded.items()}
            logits = model(**encoded).logits[0, -1].float().cpu()
            top32 = torch.topk(logits, 32).indices.sort().values
            selected = (
                torch.tensor(fixed_selected_ids[index], dtype=torch.long)
                if fixed_selected_ids is not None else top32
            )
            generated = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            rows.append({
                "prompt_ids": encoded["input_ids"][0].cpu().tolist(),
                "greedy_ids": generated[0, encoded["input_ids"].shape[1]:].cpu().tolist(),
                "selected_ids": selected.tolist(),
                "selected_logits": logits[selected].tolist(),
                "top32_ids": top32.tolist(),
            })
    return rows


def _render_prompt(tokenizer, prompt):
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def _compare(
    left: list[dict], right: list[dict], max_tolerance: float,
    mean_tolerance: float, relative_tolerance: float,
) -> dict:
    tokenization_exact = all(a["prompt_ids"] == b["prompt_ids"] for a, b in zip(left, right))
    greedy_exact = all(a["greedy_ids"] == b["greedy_ids"] for a, b in zip(left, right))
    selected_ids_exact = all(a["selected_ids"] == b["selected_ids"] for a, b in zip(left, right))
    top32_overlaps = [
        len(set(a["top32_ids"]) & set(b["top32_ids"])) for a, b in zip(left, right)
    ]
    differences = [
        abs(x - y)
        for a, b in zip(left, right)
        for x, y in zip(a["selected_logits"], b["selected_logits"])
    ]
    max_abs = max(differences)
    mean_abs = sum(differences) / len(differences)
    relative_differences = [
        abs(x - y) / max(abs(x), 1.0)
        for a, b in zip(left, right)
        for x, y in zip(a["selected_logits"], b["selected_logits"])
    ]
    max_relative = max(relative_differences)
    passed = (
        tokenization_exact and greedy_exact and selected_ids_exact and min(top32_overlaps) >= 31
        and max_abs <= max_tolerance and mean_abs <= mean_tolerance
        and max_relative <= relative_tolerance
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "tokenization_exact": tokenization_exact,
        "greedy_ids_exact": greedy_exact,
        "selected_logit_ids_exact": selected_ids_exact,
        "top32_overlap_by_prompt": top32_overlaps,
        "top32_minimum_overlap_required": 31,
        "selected_logits_max_abs": max_abs,
        "selected_logits_mean_abs": mean_abs,
        "selected_logits_max_relative": max_relative,
        "selected_logits_max_abs_tolerance": max_tolerance,
        "selected_logits_mean_abs_tolerance": mean_tolerance,
        "selected_logits_max_relative_tolerance": relative_tolerance,
        "tolerance_basis": "BF16 PEFT runtime-vs-merged path; exact greedy IDs and Top-32 IDs remain mandatory",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", type=Path, required=True)
    parser.add_argument("--grpo1-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if file_sha256(args.sft_model / "model.safetensors") != SFT_MODEL_SHA256:
        raise RuntimeError("GRPO2_SOURCE_SFT_SHA_MISMATCH")
    if file_sha256(args.grpo1_adapter / "adapter_model.safetensors") != GRPO1_ADAPTER_SHA256:
        raise RuntimeError("GRPO2_SOURCE_GRPO1_ADAPTER_SHA_MISMATCH")
    lineage = json.loads((args.grpo1_adapter / "lineage.json").read_text(encoding="utf-8"))
    if lineage.get("step") != 300 or lineage.get("parent_base_sha256") != SFT_MODEL_SHA256:
        raise RuntimeError("GRPO2_SOURCE_GRPO1_LINEAGE_MISMATCH")
    rows = [json.loads(line) for line in args.data_path.read_text(encoding="utf-8").splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model, local_files_only=True, trust_remote_code=True,
    )
    indices = (0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1)
    prompts = [_render_prompt(tokenizer, rows[index]["prompt"]) for index in indices]
    source = _load_base(args.sft_model, args.device)
    source = PeftModel.from_pretrained(source, args.grpo1_adapter, is_trainable=False)
    source_probe = _probe(source, tokenizer, prompts)
    merged = source.merge_and_unload(safe_merge=True, progressbar=True)
    args.output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output)
    del source, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        args.output, local_files_only=True, trust_remote_code=True,
    )
    reloaded = _load_base(args.output, args.device)
    merged_probe = _probe(
        reloaded, reloaded_tokenizer, prompts,
        fixed_selected_ids=[row["selected_ids"] for row in source_probe],
    )
    parity = _compare(
        source_probe, merged_probe,
        max_tolerance=0.5, mean_tolerance=0.2, relative_tolerance=0.02,
    )
    write_json(args.output / "functional_parity_details.json", {
        "dataset_row_indices": list(indices),
        "source": source_probe,
        "merged": merged_probe,
        "result": parity,
    })
    del reloaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if parity["status"] != "PASS":
        write_json(args.output / "PARENT_EXPORT_FAILED.json", parity)
        raise RuntimeError(f"GRPO2_PARENT_FUNCTIONAL_PARITY_FAILED: {parity}")
    identity, weight_files = canonical_model_identity(args.output)
    auxiliary_names = [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ]
    auxiliary_files = [
        {"name": name, "size": (args.output / name).stat().st_size,
         "sha256": file_sha256(args.output / name)}
        for name in auxiliary_names if (args.output / name).is_file()
    ]
    manifest = {
        "schema": "grpo2_test_parent_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "GRPO1_REC_BILATERAL",
        "purpose": "GRPO2_DETERMINISM_TEST_PARENT",
        "canonical": False,
        "test_parent_only": True,
        "canonical_grpo1_parent": False,
        "parent_sft_sha256": SFT_MODEL_SHA256,
        "source_grpo1_step": 300,
        "source_grpo1_checkpoint": 300,
        "source_sft_model_sha256": SFT_MODEL_SHA256,
        "source_grpo1_adapter_sha256": GRPO1_ADAPTER_SHA256,
        "merged_full_model_sha256": identity,
        "canonical_model_identity": identity,
        "weight_files": weight_files,
        "auxiliary_files": auxiliary_files,
        "single_model_safetensors_sha256": (
            weight_files[0]["sha256"]
            if len(weight_files) == 1 and weight_files[0]["name"] == "model.safetensors"
            else None
        ),
        "next_stage": "GRPO2_REC_THINK",
        "merge_method": "PEFT_merge_and_unload",
        "dtype": "bfloat16",
        "standalone_reload": "PASS",
        "functional_parity": parity,
    }
    canonical_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["canonical_manifest_sha256"] = hashlib.sha256(canonical_payload).hexdigest()
    write_json(args.output / "full_model_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
