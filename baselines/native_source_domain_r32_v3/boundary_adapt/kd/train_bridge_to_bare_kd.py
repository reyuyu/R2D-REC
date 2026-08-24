"""Four-GPU Bridge-to-Bare SID-family KD from the Step900 adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BOUNDARY = HERE.parent
_runtime_scripts = Path("/data/GRPO/scripts")
_source_scripts = HERE.parents[1] / "grpo" / "scripts"
sys.path.insert(0, str(_runtime_scripts if _runtime_scripts.is_dir() else _source_scripts))
sys.path.insert(0, str(BOUNDARY)); sys.path.insert(0, str(HERE))
from grpo_model import render_prompt
from boundary_adapt_loss import load_and_validate_provenance
from bridge_to_bare_kd_loss import per_path_kd_losses, row_uniform_kd_objective, scan_sid_families

BASE = "/data/models/onereason-8b-pretrain-competition"
STEP900 = "/data/outputs/boundary_adapt/continuation_from300_to1500/checkpoint-900"
DOMAIN = {"video": "<|video_begin|>", "prod": "<|prod_begin|>", "ad": "<|ad_begin|>", "living": "<|living_begin|>"}
SAVE_STEPS = (50, 100, 200, 300)
LAMBDA_KD = .30
TEMPERATURE = 1.0
LR = 1e-7
SEED = 20260824


def setup() -> tuple[int, int, str]:
    rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world != 4:
        raise RuntimeError(f"this experiment requires exactly four ranks, got {world}")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo"); os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank); dist.init_process_group("nccl")
    torch.manual_seed(SEED + rank); torch.cuda.manual_seed_all(SEED + rank)
    return rank, world, f"cuda:{rank}"


def load_model(device: str, trainable: bool):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, STEP900, is_trainable=trainable)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = bool(trainable and "lora" in name.lower())
    if len(model.peft_config) != 1:
        raise RuntimeError(f"expected exactly one adapter, got {list(model.peft_config)}")
    model.train(trainable)
    return model, tokenizer


def response_pair(row: dict) -> tuple[str, str]:
    marker = DOMAIN[row["target_domain"]]
    bare = row["adapted_response"]
    suffix = marker + row["boundary_gold_sid"]
    if not bare.endswith(suffix):
        raise ValueError("malformed adapted response")
    bridge = row["exact_bridge"]
    if hashlib.sha256(bridge.encode("utf-8")).hexdigest() != row["bridge_sha256"]:
        raise ValueError("bridge digest mismatch")
    # Anchor at the validated boundary suffix; the immutable CoT may mention a
    # domain marker earlier and must never be modified.
    return bare, bare[:-len(suffix)] + bridge + suffix


def encode_pair(tokenizer, row: dict, device: str):
    bare_response, teacher_response = response_pair(row)
    prefix = render_prompt(tokenizer, row["prompt"])
    bare = tokenizer.encode(prefix + bare_response, add_special_tokens=False)
    teacher = tokenizer.encode(prefix + teacher_response, add_special_tokens=False)
    gold = tokenizer.encode(row["boundary_gold_sid"], add_special_tokens=False)
    domain = tokenizer.encode(DOMAIN[row["target_domain"]], add_special_tokens=False)
    if len(gold) != 3 or len(domain) != 1 or bare[-3:] != gold or teacher[-3:] != gold:
        raise ValueError("ABC/domain token contract failed")
    return (
        torch.tensor([bare], device=device), torch.tensor([teacher], device=device),
        torch.tensor([gold], device=device),
        torch.tensor([row["boundary_sample_weight"]], device=device, dtype=torch.float32),
    )


def last_abc_logits(model, ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=4).logits
    if output.shape[1] != 4:
        raise RuntimeError(f"logits_to_keep contract failed: {tuple(output.shape)}")
    return output[:, -4:-1].float()


def compute(student, teacher, tokenizer, row, families, device):
    bare, bridge, gold, weights = encode_pair(tokenizer, row, device)
    with torch.no_grad():
        teacher_logits = last_abc_logits(teacher, bridge)
    student_logits = last_abc_logits(student, bare)
    losses = per_path_kd_losses(student_logits, teacher_logits, gold, families, TEMPERATURE)
    total = row_uniform_kd_objective(losses, weights, lambda_kd=LAMBDA_KD, **PROVENANCE)
    return total, losses, (bare, bridge, gold, weights), (student_logits, teacher_logits)


def read_rows(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def usefulness(rows, model, tokenizer, families, device, rank) -> dict:
    selected, counts, seen = [], defaultdict(int), set()
    for row in rows:
        group, domain = row["boundary_group_id"], row["target_domain"]
        if group in seen or counts[domain] >= 64:
            continue
        seen.add(group); counts[domain] += 1
        shard = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 4
        if shard == rank:
            selected.append(group)
        if all(counts[key] == 64 for key in DOMAIN):
            break
    chosen = set(selected)
    # [groups, bare NLL, bridge NLL, KL_A, KL_B, KL_C], weighted rows sum to groups.
    totals = {key: torch.zeros(6, device=device, dtype=torch.float64) for key in ("overall", *DOMAIN)}
    with torch.inference_mode():
        for row in rows:
            if row["boundary_group_id"] not in chosen:
                continue
            bare, bridge, gold, weights = encode_pair(tokenizer, row, device)
            bare_logits, bridge_logits = last_abc_logits(model, bare), last_abc_logits(model, bridge)
            bare_losses = per_path_kd_losses(bare_logits, bridge_logits, gold, families)
            bridge_nll = torch.nn.functional.cross_entropy(bridge_logits.reshape(-1, bridge_logits.size(-1)), gold.reshape(-1), reduction="none").view(1, 3).mean(-1)
            values = torch.stack((weights, weights * bare_losses["gold"], weights * bridge_nll,
                                  weights * bare_losses["kl_A"], weights * bare_losses["kl_B"], weights * bare_losses["kl_C"]))[:, 0].double()
            totals["overall"] += values; totals[row["target_domain"]] += values
    for value in totals.values():
        dist.all_reduce(value)
    result = {}
    for key, value in totals.items():
        n = float(value[0])
        result[key] = {"groups": n, "bare_gold_nll": float(value[1] / n), "bridge_gold_nll": float(value[2] / n),
                       "kl_A": float(value[3] / n), "kl_B": float(value[4] / n), "kl_C": float(value[5] / n)}
    overall = result["overall"]
    result["pass"] = abs(overall["groups"] - 256) < 1e-3 and overall["bridge_gold_nll"] <= overall["bare_gold_nll"] + .01 and sum(overall[f"kl_{key}"] for key in "ABC") > 1e-8
    return result


def tensor_digest(model, lora: bool) -> str:
    """Deterministic state fingerprint; optimizer exclusion supplies the full base invariant."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ("lora" in name.lower()) == lora:
            flat = parameter.detach().reshape(-1)
            if flat.numel() > 96:
                indices = torch.arange(96, device=flat.device, dtype=torch.int64)
                indices = indices * (flat.numel() - 1) // 95
                flat = flat.index_select(0, indices)
            digest.update(name.encode("utf-8")); digest.update(flat.cpu().float().numpy().tobytes())
    return digest.hexdigest()


def aggregate_scalar(value: float, device: str) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64); dist.all_reduce(tensor)
    return float(tensor / dist.get_world_size())


def write_json(path: str, payload: dict, rank: int) -> None:
    if rank == 0:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def historical_bare_gate(model, tokenizer, device: str, rank: int) -> dict:
    diagnostics = BOUNDARY / "diagnostics"
    if str(diagnostics) not in sys.path:
        sys.path.insert(0, str(diagnostics))
    import fixed_cot_checkpoint_sweep as fixed
    import controlled_generator_decoder_crossover as crossover
    items = json.loads(fixed.MANIFEST.read_text(encoding="utf-8"))["items"]
    local = []
    model.eval()
    for index, item in enumerate(items):
        if index % 4 != rank:
            continue
        prompt = crossover.encode_prompt(tokenizer, item["think_prompt"])
        cot = tokenizer.encode(item["fixed_cot"], add_special_tokens=False)
        domain = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
        history = crossover.history_from_prompt(tokenizer, prompt)
        gold = {crossover.parse_gold(value) for value in item["gold_sids"]}
        context = prompt + cot + domain
        local.append({
            "group_id": item["recommendation_group_id"],
            "beam": crossover.strict_beam(model, tokenizer, context, item["target_domain"], gold, history),
            "teacher_forced": crossover.teacher_forced_gold(model, tokenizer, context, item["gold_sids"]),
        })
    part = Path(f"/data/GRPO/boundary_adapt/results/bridge_kd_training_gate_rank{rank}.json")
    part.write_text(json.dumps(local, ensure_ascii=False), encoding="utf-8"); dist.barrier()
    if rank == 0:
        records = [record for shard in range(4) for record in json.loads(Path(str(part).replace("rank0", f"rank{shard}")).read_text(encoding="utf-8"))]
        merged = {**crossover.aggregate_beam(records), **crossover.aggregate_teacher(records)}
        Path("/data/GRPO/boundary_adapt/results/bridge_kd_training_gate_latest.json").write_text(json.dumps(merged, indent=2) + "\n")
    dist.barrier()
    merged = json.loads(Path("/data/GRPO/boundary_adapt/results/bridge_kd_training_gate_latest.json").read_text())
    model.train(); torch.cuda.empty_cache()
    return merged


def main(args) -> None:
    global PROVENANCE
    rank, world, device = setup()
    PROVENANCE = load_and_validate_provenance(args.rows, args.stats)
    rows = read_rows(args.rows)
    if len(rows) != PROVENANCE["total_paths"]:
        raise RuntimeError("prepared row count changed")
    student, tokenizer = load_model(device, True)
    teacher, teacher_tokenizer = load_model(device, False)
    only_lora_trainable = all(("lora" in name.lower()) == parameter.requires_grad for name, parameter in student.named_parameters())
    if not only_lora_trainable:
        raise RuntimeError("only the existing student LoRA may be trainable")
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise RuntimeError("teacher/student tokenizer mismatch")
    sid = scan_sid_families(tokenizer); families = sid.tensors(device)
    family_shapes = {key: len(getattr(sid, key)) for key in "ABC"}
    teacher.eval()
    usefulness_result = usefulness(rows, teacher, tokenizer, families, device, rank)
    if not usefulness_result["pass"]:
        raise RuntimeError(f"TEACHER_USEFULNESS_FAILED: {usefulness_result}")

    row = rows[rank]
    base_before, lora_before = tensor_digest(student, False), tensor_digest(student, True)
    encoded = encode_pair(tokenizer, row, device)
    student.eval()
    with torch.no_grad():
        init_student = last_abc_logits(student, encoded[0])
        init_reference = last_abc_logits(teacher, encoded[0])
    init_diff = float((init_student - init_reference).abs().max())
    student.train()
    total, losses, encoded, logits = compute(student, teacher, tokenizer, row, families, device)
    total.backward()
    teacher_grad = any(parameter.grad is not None for parameter in teacher.parameters())
    base_grad = any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in student.named_parameters() if "lora" not in name.lower())
    lora_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for name, parameter in student.named_parameters() if "lora" in name.lower())
    zero_update = {
        "optimizer_steps": 0, "init_student_step900_max_abs_diff": init_diff,
        "teacher_grad_nonzero": teacher_grad, "student_base_grad_nonzero": bool(base_grad), "student_lora_grad_nonzero": bool(lora_grad),
        "abc_labels_per_path": int(encoded[2].numel()), "gold_finite": bool(torch.isfinite(losses["gold"]).all()),
        "kd_finite": bool(torch.isfinite(losses["kd"]).all()), "total_finite": bool(torch.isfinite(total)),
        "family_shapes": family_shapes, "base_changed": tensor_digest(student, False) != base_before,
        "lora_changed": tensor_digest(student, True) != lora_before, "second_student_lora_created": len(student.peft_config) != 1,
        "only_lora_trainable": only_lora_trainable,
    }
    zero_update["pass"] = init_diff == 0 and not teacher_grad and not base_grad and lora_grad and zero_update["abc_labels_per_path"] == 3 and not zero_update["base_changed"] and not zero_update["lora_changed"] and not zero_update["second_student_lora_created"]
    dist.barrier()
    preflight = {"teacher_usefulness": usefulness_result, "zero_update": zero_update, "provenance": PROVENANCE}
    if args.mode == "preflight":
        write_json(args.result, preflight, rank)
        if not zero_update["pass"]:
            raise RuntimeError(f"ZERO_UPDATE_PREFLIGHT_FAILED: {zero_update}")
        dist.destroy_process_group(); return
    if not zero_update["pass"]:
        raise RuntimeError(f"ZERO_UPDATE_PREFLIGHT_FAILED: {zero_update}")

    student.zero_grad(set_to_none=True)
    gates = {"0": historical_bare_gate(student, tokenizer, device, rank)}
    wrapped = DDP(student, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW([parameter for parameter in student.parameters() if parameter.requires_grad], lr=LR)
    sampler = DistributedSampler(rows, num_replicas=world, rank=rank, shuffle=True, seed=SEED); sampler.set_epoch(0)
    indices = iter(sampler); metrics = []
    for step in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        train_row = rows[next(indices)]
        total, parts, _, _ = compute(wrapped, teacher, tokenizer, train_row, families, device)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("teacher received gradient")
        optimizer.step()
        values = {"step": step, "L_total": aggregate_scalar(float(total.detach()), device)}
        for key in ("gold", "kd", "kl_A", "kl_B", "kl_C"):
            values["L_" + key] = aggregate_scalar(float(parts[key].mean().detach()), device)
        values["lr"] = LR; metrics.append(values)
        if rank == 0 and (step == 1 or step % 10 == 0):
            print(json.dumps(values), flush=True)
        if step in SAVE_STEPS:
            if rank == 0:
                checkpoint = Path(args.output_dir) / f"checkpoint-{step}"; checkpoint.mkdir(parents=True, exist_ok=True)
                student.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
            dist.barrier()
        if step in (50, 100):
            gates[str(step)] = historical_bare_gate(student, tokenizer, device, rank)
            baseline, current = gates["0"], gates[str(step)]
            clearly_declined = current["beam_raw"] < baseline["beam_raw"] - .10
            nll_worse = current["mean_gold_abc_nll"] > baseline["mean_gold_abc_nll"] + .01
            gates[str(step)]["hard_stop"] = clearly_declined and nll_worse
            if gates[str(step)]["hard_stop"]:
                raise RuntimeError(f"KD_EARLY_HARD_STOP step={step} gates={gates}")
    final = {
        "preflight": preflight, "optimizer_steps": 300, "effective_global_batch_rows": 4,
        "learning_rate": LR, "lambda_kd": LAMBDA_KD, "temperature": TEMPERATURE,
        "teacher_adapter": STEP900, "student_init_adapter": STEP900,
        "old_optimizer_resumed": False, "second_student_lora_created": len(student.peft_config) != 1,
        "base_changed": tensor_digest(student, False) != base_before,
        "lora_changed": tensor_digest(student, True) != lora_before,
        "teacher_grad_nonzero": any(parameter.grad is not None for parameter in teacher.parameters()),
        "checkpoints": [str(Path(args.output_dir) / f"checkpoint-{step}") for step in SAVE_STEPS], "metrics": metrics,
        "historical_training_gates": gates,
    }
    if final["base_changed"] or not final["lora_changed"] or final["teacher_grad_nonzero"] or final["second_student_lora_created"]:
        raise RuntimeError(f"post-training invariant failed: {final}")
    write_json(args.result, final, rank); dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--rows", required=True); parser.add_argument("--stats", required=True); parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.mode == "formal" and (os.environ.get("BRIDGE_KD_CONFIRM_FORMAL") != "1" or not args.output_dir):
        raise SystemExit("formal run requires BRIDGE_KD_CONFIRM_FORMAL=1 and --output-dir")
    main(args)
