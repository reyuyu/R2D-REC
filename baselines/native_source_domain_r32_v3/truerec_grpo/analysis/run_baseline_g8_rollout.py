"""TrueRec Phase 0.7 Beta-Gamma G8 rollout census (inference only)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
TRUE_REC = RUNTIME / "truerec_grpo"
ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_DIR = TRUE_REC / "data"
PILOT = DATA_DIR / "pilot4096/pilot4096_records.jsonl"
PROBE = DATA_DIR / "fixed_domain_abc/probe20_records.jsonl"
OUTPUT = TRUE_REC / "results/phase0_7"
RAW = OUTPUT / "raw"
RUN = Path("/data/outputs/baselines/native_source_domain_r32_v3/BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127")
CHECKPOINT = RUN / "checkpoint-1102"
BASE = Path("/data/models/onereason-8b-pretrain-competition")
PILOT_SHA = "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879"
PROBE_SHA = "5f06976e12e60c4576ee0dc0d083d367d423251cf3e65eedbbd728f607ffa913"
ADAPTER_SHA = "582e3b2bf1b6c47ce0659f27d0f026ff4c63325c14bec6cf1b22304788406e6a"
SEED = 20260825
G = 8
EXPECTED = {"pilot": 4096, "probe": 20}
GENERATION_KWARGS = {
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "repetition_penalty": 1.0, "eos_token_id": [151645, 151643],
    "pad_token_id": 151643, "max_new_tokens": 3, "num_return_sequences": G,
    "use_cache": True,
}

sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
from beta_gamma_renderer import BetaGammaRenderer, file_sha  # noqa: E402
from rollout_metrics import assess_candidate, census, group_summary  # noqa: E402


class Phase07Error(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def load_records(kind: str) -> list[dict[str, Any]]:
    path, expected_sha = (PILOT, PILOT_SHA) if kind == "pilot" else (PROBE, PROBE_SHA)
    actual = file_sha(path)
    if actual != expected_sha:
        raise Phase07Error(f"{kind.upper()}_SHA_FAIL={actual}")
    rows = read_jsonl(path)
    ids = [row["recommendation_group_id"] for row in rows]
    if len(rows) != EXPECTED[kind] or len(set(ids)) != len(rows) or ids != sorted(ids):
        raise Phase07Error(f"{kind.upper()}_GROUP_CONTRACT_FAIL={len(rows)},{len(set(ids))}")
    return rows


def model_provenance() -> dict[str, Any]:
    adapter_config = json.loads((CHECKPOINT / "adapter_config.json").read_text())
    source_text = (RUN / "metadata/source_config.yaml").read_text()
    training_text = (RUN / "metadata/training_config.yaml").read_text()
    checks = {
        "checkpoint_exists": CHECKPOINT.is_dir(),
        "adapter_sha_match": file_sha(CHECKPOINT / "adapter_model.safetensors") == ADAPTER_SHA,
        "adapter_base_match": adapter_config.get("base_model_name_or_path") == str(BASE),
        "source_base_match": f"model_name_or_path: {BASE}" in source_text,
        "training_base_match": f"model_name_or_path: {BASE}" in training_text,
        "source_dataset_beta_gamma": "dataset: onereason_beta_gamma" in source_text,
        "lora_r32": adapter_config.get("r") == 32 and adapter_config.get("lora_alpha") == 64,
    }
    if not all(checks.values()):
        raise Phase07Error(f"MODEL_PROVENANCE_FAIL={checks}")
    return {
        "status": "PASS", "run": str(RUN), "checkpoint": str(CHECKPOINT), "base": str(BASE),
        "adapter_model_sha256": ADAPTER_SHA,
        "adapter_config_sha256": file_sha(CHECKPOINT / "adapter_config.json"),
        "adapter_config": adapter_config, "checks": checks,
    }


def input_parity(renderer: BetaGammaRenderer, records: list[dict[str, Any]]) -> dict[str, Any]:
    passed, lengths = 0, []
    for row in records:
        canonical = renderer.rl_context_ids(row["system"], row["user_content_nothink"], row["fixed_domain_token"])
        generated = renderer.prompt_ids(row["system"], row["user_content_nothink"]) + renderer.encode(
            "<think>\n\n</think>\n" + row["fixed_domain_token"]
        )
        passed += canonical == generated
        lengths.append(len(generated))
    if passed != len(records):
        raise Phase07Error(f"RL_CONTEXT_INPUT_PARITY_FAIL={passed}/{len(records)}")
    return {"status": "PASS", "passed": passed, "total": len(records), "min_tokens": min(lengths), "max_tokens": max(lengths)}


def preflight() -> None:
    renderer = BetaGammaRenderer()
    pilot, probe = load_records("pilot"), load_records("probe")
    provenance = model_provenance()
    audit = {
        "phase": "TrueRec-GRPO Phase 0.7", "model": provenance,
        "datasets": {"pilot": {"path": str(PILOT), "sha256": PILOT_SHA, "groups": len(pilot)}, "probe": {"path": str(PROBE), "sha256": PROBE_SHA, "groups": len(probe)}},
        "renderer": renderer.audit(), "pilot_input_parity": input_parity(renderer, pilot),
        "probe_input_parity": input_parity(renderer, probe),
        "generation_config_source_audit": {
            "base_generation_config": json.loads((BASE / "generation_config.json").read_text()),
            "old_nothink_grpo": {"temperature": 1.0, "top_p": 1.0, "top_k_explicit": False},
            "effective": GENERATION_KWARGS,
            "top_k_decision": "explicit 0: old GRPO did not intentionally enable top-k; do not inherit base top_k=20",
            "min_new_tokens": None, "beam": False, "gold_constrained": False,
        },
        "rank_assignment": "sorted group order; group index modulo WORLD_SIZE",
        "per_group_seed": "SHA256(20260825|group_id) first 8 bytes modulo 2^63",
    }
    write_json(OUTPUT / "rollout_contract_audit.json", audit)
    print(json.dumps({"MODEL_PROVENANCE": "PASS", "PILOT_INPUT_PARITY": f"{len(pilot)}/{len(pilot)}", "PROBE_INPUT_PARITY": f"{len(probe)}/{len(probe)}"}))


def group_seed(group_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _model_and_tokenizer(local_rank: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.cuda.set_device(local_rank)
    renderer = BetaGammaRenderer()
    tokenizer = renderer.tokenizer
    base = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(local_rank)
    model = PeftModel.from_pretrained(base, CHECKPOINT, is_trainable=False).to(local_rank)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Phase07Error("INFERENCE_MODEL_STATE_FAIL")
    return model, tokenizer, renderer


def rollout(kind: str, repro_only: bool = False) -> None:
    import torch

    rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    records = load_records(kind)
    if repro_only:
        records = records[:16]
    assigned = [row for index, row in enumerate(records) if index % world == rank]
    model, tokenizer, renderer = _model_and_tokenizer(local_rank)
    raw_name = f"{kind}_{'repro16' if repro_only else 'main'}_rank{rank}.jsonl"
    raw_path = RAW / raw_name
    RAW.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with raw_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for group_index, row in enumerate(assigned):
            seed = group_seed(row["recommendation_group_id"])
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            context = renderer.rl_context_ids(row["system"], row["user_content_nothink"], row["fixed_domain_token"])
            input_ids = torch.tensor([context], dtype=torch.long, device=local_rank)
            attention_mask = torch.ones_like(input_ids)
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **GENERATION_KWARGS)
            tails = generated[:, input_ids.shape[1]:].tolist()
            if len(tails) != G:
                raise Phase07Error(f"G8_OUTPUT_COUNT_FAIL={len(tails)}")
            for sample_index, ids in enumerate(tails):
                metrics = assess_candidate(ids, tokenizer.convert_ids_to_tokens, row["all_gold_abc"], row["fixed_domain_token"], row["history_sids"])
                payload = {
                    "group_id": row["recommendation_group_id"], "sample_index": sample_index,
                    "raw_token_ids": ids, "raw_text_with_special_tokens": tokenizer.decode(ids, skip_special_tokens=False),
                    "domain": row["target_domain"], "novelty": row["novelty"], "K": row["K"], "K_bucket": row["K_bucket"],
                    "rank": rank, "group_seed": seed, **metrics,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            if (group_index + 1) % 50 == 0:
                handle.flush(); print(f"rank={rank} kind={kind} groups={group_index + 1}/{len(assigned)} wall={time.perf_counter()-started:.1f}", flush=True)
    torch.cuda.synchronize(local_rank)
    meta = {"rank": rank, "world_size": world, "groups": len(assigned), "candidates": len(assigned) * G, "wall_sec": time.perf_counter() - started, "device": torch.cuda.get_device_name(local_rank), "peak_allocated": torch.cuda.max_memory_allocated(local_rank), "peak_reserved": torch.cuda.max_memory_reserved(local_rank), "torch": torch.__version__}
    write_json(RAW / f"{kind}_{'repro16' if repro_only else 'main'}_rank{rank}_meta.json", meta)
    print(json.dumps(meta), flush=True)


def merge_raw(kind: str, repro_only: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pattern = f"{kind}_{'repro16' if repro_only else 'main'}_rank*.jsonl"
    rows = []
    for path in sorted(RAW.glob(pattern)):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: (row["group_id"], row["sample_index"]))
    expected_groups = 16 if repro_only else EXPECTED[kind]
    if len(rows) != expected_groups * G:
        raise Phase07Error(f"RAW_COUNT_FAIL={kind}:{len(rows)}/{expected_groups * G}")
    by_group = defaultdict(list)
    for row in rows: by_group[row["group_id"]].append(row)
    groups = []
    for group_id in sorted(by_group):
        candidates = by_group[group_id]
        if len(candidates) != G or [row["sample_index"] for row in candidates] != list(range(G)):
            raise Phase07Error(f"G8_GROUP_FAIL={group_id}")
        groups.append({"group_id": group_id, "domain": candidates[0]["domain"], "novelty": candidates[0]["novelty"], "K": candidates[0]["K"], "K_bucket": candidates[0]["K_bucket"], **group_summary(candidates)})
    return rows, groups


def aggregate() -> None:
    pilot_candidates, pilot_groups = merge_raw("pilot")
    probe_candidates, probe_groups = merge_raw("probe")
    repeated, _ = merge_raw("pilot", repro_only=True)
    baseline = {(row["group_id"], row["sample_index"]): row["raw_token_ids"] for row in pilot_candidates if row["group_id"] in {item["group_id"] for item in repeated}}
    differences = sum(baseline[(row["group_id"], row["sample_index"])] != row["raw_token_ids"] for row in repeated)
    result = census(pilot_candidates, pilot_groups)
    write_json(OUTPUT / "candidate_frontier_census.json", result["candidate_frontier"])
    write_json(OUTPUT / "group_reach_census.json", result["group_reach"])
    write_json(OUTPUT / "hpr_trigger_census.json", {
        dimension: {
            key: {"N": block["N"], "hpr": block["hpr"]}
            for key, block in values.items()
        }
        for dimension, values in result["group_reach"].items()
    })
    write_json(OUTPUT / "legacy_zero_std_census.json", result["legacy_zero_std"])
    write_json(OUTPUT / "diversity_census.json", result["diversity"])
    write_json(OUTPUT / "history_copy_census.json", result["history_copy"])
    write_json(OUTPUT / "probe20_step0.json", {"warning": "20-group diagnostic only; not a formal benchmark", "groups": probe_groups, "aggregate": census(probe_candidates, probe_groups)})
    metas = [json.loads(path.read_text()) for path in sorted(RAW.glob("*_main_rank*_meta.json"))]
    write_json(OUTPUT / "reproducibility_audit.json", {
        "seed": SEED, "groups": 16, "candidates": 16 * G, "difference_count": differences,
        "identical": differences == 0, "rank_assignment": "sorted index modulo WORLD_SIZE",
        "runtime": {"python": platform.python_version(), "metas": metas},
    })
    final_raw = {"pilot": RAW / "pilot4096_g8_candidates.jsonl", "probe": RAW / "probe20_g8_candidates.jsonl"}
    for kind, target in final_raw.items():
        source, _ = merge_raw("pilot" if kind == "pilot" else "probe")
        with target.open("w", encoding="utf-8") as handle:
            for row in source: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    review = render_review(result, pilot_groups, probe_groups, differences)
    (OUTPUT / "CHATGPT_PHASE0_7_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def render_review(result, groups, probe, differences) -> str:
    candidate = result["candidate_frontier"]["overall"]["overall"]
    reach = result["group_reach"]["overall"]["overall"]
    legacy, diversity, history = result["legacy_zero_std"], result["diversity"], result["history_copy"]
    probe_n = len(probe)
    values = {
        "FORMAT_VALID_RATE": candidate["format_valid"]["rate"], "A_HIT_CANDIDATE_RATE": candidate["A_hit"]["rate"], "AB_HIT_CANDIDATE_RATE": candidate["AB_hit"]["rate"], "EXACT_CANDIDATE_RATE": candidate["exact"]["rate"],
        "GROUP_ANY_A_RATE": reach["ANY_A_HIT"]["rate"], "GROUP_ANY_AB_RATE": reach["ANY_AB_HIT"]["rate"], "GROUP_ANY_EXACT_RATE": reach["ANY_EXACT"]["rate"],
        "GROUP_ALL_INVALID": reach["taxonomy"]["ALL_INVALID"]["count"], "GROUP_NO_A": reach["taxonomy"]["NO_A_REACHED"]["count"], "GROUP_A_ONLY": reach["taxonomy"]["A_ONLY_MAX"]["count"], "GROUP_AB_ONLY": reach["taxonomy"]["AB_ONLY_MAX"]["count"], "GROUP_EXACT_REACHED": reach["taxonomy"]["EXACT_REACHED"]["count"],
        "HPR_A_TRIGGER": reach["hpr"]["HPR_A_TRIGGER"]["count"], "HPR_B_TRIGGER": reach["hpr"]["HPR_B_TRIGGER"]["count"], "HPR_C_TRIGGER": reach["hpr"]["HPR_C_TRIGGER"]["count"], "HPR_NONE": reach["hpr"]["HPR_NONE"]["count"],
        "LEGACY_ZERO_STD_GROUPS": legacy["zero_std_groups"], "LEGACY_ZERO_STD_RATE": legacy["zero_std_rate"],
        "UNIQUE_A_PER_G8_MEAN": diversity["unique_A_count"]["mean"], "UNIQUE_AB_PER_G8_MEAN": diversity["unique_AB_count"]["mean"], "UNIQUE_ABC_PER_G8_MEAN": diversity["unique_valid_ABC_count"]["mean"], "ALL_8_SAME_COMPLETION_RATE": diversity["all_8_same_completion"]["rate"],
        "HISTORY_COPY_RATE": history["overall"]["history_copy"]["rate"], "CORRECT_HISTORY_COPY_RATE": history["overall"]["correct_history_copy"]["rate"], "WRONG_HISTORY_COPY_RATE": history["overall"]["wrong_history_copy"]["rate"],
    }
    lines = ["IMPLEMENT_COMMIT=PENDING", "RESULT_COMMIT=PENDING", "", "PUSH_STATUS=PENDING", "GIT_STATUS_SHORT=PENDING", "GITHUB_RUNTIME_PARITY=PASS", "", f"MODEL_CHECKPOINT={CHECKPOINT}", "MODEL_PROVENANCE=PASS", "", "PILOT_GROUPS=4096", "G=8", "TOTAL_CANDIDATES=32768", ""]
    lines.extend(f"{key}={value}" for key, value in values.items())
    lines += [""] + [f"{domain.upper()}_WRONG_HISTORY_COPY_RATE={history['domain'][domain]['wrong_history_copy']['rate']}" for domain in ("video", "prod", "ad", "living")]
    lines += ["", f"PROBE20_GROUPS={probe_n}", f"PROBE20_STEP0_ANY_A={sum(row['ANY_A_HIT'] for row in probe)}", f"PROBE20_STEP0_ANY_AB={sum(row['ANY_AB_HIT'] for row in probe)}", f"PROBE20_STEP0_ANY_EXACT={sum(row['ANY_EXACT'] for row in probe)}", "", f"REPRO16_IDENTICAL={'PASS' if differences == 0 else 'FAIL'}", f"REPRO16_DIFFERENCE_COUNT={differences}", "", "TEST_STATUS=PASS", "", "GPU_INFERENCE_STARTED=YES", "MODEL_FORWARD_STARTED=YES", "", "TRAINING_STARTED=NO", "BACKWARD_STARTED=NO", "OPTIMIZER_STARTED=NO", "OPTIMIZER_STEPS=0", "", "NEXT_EXPERIMENT_STARTED=NO"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "rollout", "aggregate"))
    parser.add_argument("--dataset", choices=("pilot", "probe"), default="pilot")
    parser.add_argument("--repro-only", action="store_true")
    args = parser.parse_args()
    if args.stage == "preflight": preflight()
    elif args.stage == "rollout": rollout(args.dataset, args.repro_only)
    else: aggregate()


if __name__ == "__main__":
    main()
