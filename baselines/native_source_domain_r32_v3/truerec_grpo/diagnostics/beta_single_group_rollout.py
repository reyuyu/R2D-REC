"""Phase 1.2B: one real Beta G8 rollout, without loss or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer, file_sha  # noqa: E402
from rollout_runtime_v1 import G, build_rollout_artifacts  # noqa: E402


PILOT = Path("/data/GRPO/truerec_grpo/data/pilot4096/pilot4096_records.jsonl")
PILOT_SHA = "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879"
SELECTOR_PREFIX = "phase1.2b|20260825|"
EOS_TOKEN_IDS = (151645, 151643)
PAD_TOKEN_ID = 151643
GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 3,
    "num_return_sequences": G,
    "eos_token_id": list(EOS_TOKEN_IDS),
    "pad_token_id": PAD_TOKEN_ID,
    "return_dict_in_generate": True,
    "output_scores": True,
    "use_cache": True,
}


class Phase12BError(RuntimeError):
    pass


def selector_digest(group_id: str) -> str:
    return hashlib.sha256((SELECTOR_PREFIX + str(group_id)).encode("utf-8")).hexdigest()


def select_single_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4096 or len({str(row["recommendation_group_id"]) for row in rows}) != 4096:
        raise Phase12BError("Pilot4096 count/unique contract failed")
    return min(rows, key=lambda row: (selector_digest(str(row["recommendation_group_id"])), str(row["recommendation_group_id"])))


def load_selected_group(path: Path = PILOT) -> dict[str, Any]:
    if file_sha(path) != PILOT_SHA:
        raise Phase12BError("Pilot4096 SHA mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return select_single_group(rows)


def independently_reconstruct_old_logps(scores, completion_ids: Sequence[Sequence[int]]):
    """Independent reference: per-step log_softmax then gather sampled ID."""
    import torch

    if scores.ndim != 3 or scores.shape[0] != len(completion_ids):
        raise Phase12BError("scores must be [G,T,V]")
    rows = []
    for sample_index, ids in enumerate(completion_ids):
        values = []
        for step, token_id in enumerate(ids):
            values.append(torch.log_softmax(scores[sample_index, step], dim=-1)[int(token_id)])
        rows.append(torch.stack(values).detach().cpu() if values else torch.empty(0))
    return [tuple(float(value) for value in row) for row in rows]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, physical_gpu_id: int) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    if contract.init_family != INIT_FAMILY or contract.beta_gamma_used_as_init:
        raise Phase12BError("Phase1.2A initialization contract mismatch")
    row = load_selected_group()
    seed = int.from_bytes(hashlib.sha256((SELECTOR_PREFIX + row["recommendation_group_id"]).encode()).digest()[:8], "big") % (2**63)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)

    renderer = BetaGammaRenderer()
    context_ids = renderer.rl_context_ids(row["system"], row["user_content_nothink"], row["fixed_domain_token"])
    domain_ids = renderer.encode(row["fixed_domain_token"])
    if len(domain_ids) != 1 or context_ids[-1] != domain_ids[0]:
        raise Phase12BError("fixed domain is not terminal context token")
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=False, local_files_only=True).to(device)
    model.eval(); model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Phase12BError("inference-only model gate failed")
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **GENERATION_KWARGS)
    torch.cuda.synchronize(device)
    generation_wall = time.perf_counter() - started
    artifacts = build_rollout_artifacts(
        row, context_ids, output, renderer.tokenizer.convert_ids_to_tokens,
        PAD_TOKEN_ID, EOS_TOKEN_IDS,
    )
    reference = independently_reconstruct_old_logps(artifacts.generation_scores, artifacts.completion_ids)
    candidates = []
    differences = []
    for candidate, expected in zip(artifacts.group.candidates, reference):
        if len(candidate.old_logps) != len(candidate.completion_ids) or len(expected) != len(candidate.completion_ids):
            raise Phase12BError("old-logp length mismatch")
        differences.extend(abs(actual - target) for actual, target in zip(candidate.old_logps, expected))
        candidates.append({
            "sample_index": candidate.sample_index,
            "raw_token_ids": list(candidate.completion_ids),
            "raw_text_with_special_tokens": renderer.tokenizer.decode(candidate.completion_ids, skip_special_tokens=False),
            "actual_completion_length": len(candidate.completion_ids),
            "old_logps": list(candidate.old_logps),
            **{key: candidate.metrics[key] for key in ("format_valid", "parsed_abc", "A_hit", "AB_hit", "exact", "frontier")},
        })
    max_diff = max(differences, default=0.0)
    if max_diff > 1e-7:
        raise Phase12BError(f"old-logp reconstruction failed: {max_diff}")
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    rollout_audit = {
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "base": str(PRETRAINED_BASE), "provenance": provenance},
        "group": {key: row[key] for key in ("recommendation_group_id", "target_domain", "novelty", "K")},
        "selector": {"formula": "SHA256(phase1.2b|20260825|recommendation_group_id)", "digest": selector_digest(row["recommendation_group_id"]), "pilot_groups": 4096},
        "prefix": {"train_bridge": False, "fixed_domain_terminal_token": True, "domain_generated_by_model": False, "context_tokens": len(context_ids), "fixed_domain_token": row["fixed_domain_token"], "fixed_domain_token_id": domain_ids[0]},
        "generation_kwargs": GENERATION_KWARGS,
        "seed": seed,
        "candidates": candidates,
        "counts": {"format_valid": sum(item["format_valid"] for item in candidates), "A_hit": sum(item["A_hit"] for item in candidates), "AB_hit": sum(item["AB_hit"] for item in candidates), "exact": sum(item["exact"] for item in candidates)},
    }
    old_audit = {"policy_family": INIT_FAMILY, "token_count": len(differences), "reconstruction": "PASS", "max_abs_diff": max_diff, "independent_method": "per-step log_softmax(output.scores[t]) then gather sampled token"}
    resource_audit = {"physical_gpu_id": physical_gpu_id, "visible_cuda_device": 0, "gpu_name": gpu_name, "free_memory_at_start_gb": free_bytes / (1024 ** 3), "total_memory_gb": total_bytes / (1024 ** 3), "peak_reserved_gb": peak_reserved, "generation_wall_sec": generation_wall}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "single_group_beta_g8_rollout.json", rollout_audit)
    write_json(output_dir / "old_logp_reconstruction_audit.json", old_audit)
    write_json(output_dir / "resource_audit.json", resource_audit)
    (output_dir / "CHATGPT_PHASE1_2B_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2B single-group Beta rollout audit: PASS\n"
        "One deterministically selected Pilot4096 group used the frozen no-bridge fixed-domain context and a real stochastic G8 Beta-Baseline generation.\n"
        "Raw IDs/text and rollout-policy old logps were retained; independent reconstruction passed. No loss, Frontier, HPR, backward, optimizer, or training ran.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PHASE1_2B": "PASS", "group_id": row["recommendation_group_id"], "gpu": physical_gpu_id, "peak_reserved_gb": peak_reserved, "old_logp_tokens": len(differences), "max_abs_diff": max_diff}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
