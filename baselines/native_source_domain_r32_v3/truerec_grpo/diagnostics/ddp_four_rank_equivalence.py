"""CPU/gloo formal equivalence for four-rank local-G2 TrueRec training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "credit", ROOT / "diagnostics", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from distributed_trainer_v1 import DDP_WORLD_SIZE, DistributedTrueRecGRPOTrainerV1  # noqa: E402
from trainer_g8_microbatch_equivalence import CASES, TOKEN_IDS, TinyPolicy, _group, full_g8_reference  # noqa: E402


AUDIT_LR = 1e-3


def tensor_inventory(module: torch.nn.Module, *, gradients: bool) -> dict[str, torch.Tensor]:
    output = {}
    for name, parameter in module.named_parameters():
        value = parameter.grad if gradients else parameter.detach()
        if value is None:
            raise AssertionError(f"missing gradient for {name}")
        output[name] = value.detach().cpu().clone()
    return output


def _worker(rank: int, init_file: str, output_file: str) -> None:
    if sys.platform != "win32":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=DDP_WORLD_SIZE)
    cases = {}
    try:
        for case_index, (case_name, completions) in enumerate(CASES.items()):
            torch.manual_seed(4400 + case_index)
            policy = TinyPolicy()
            ddp = DistributedDataParallel(policy)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=AUDIT_LR, weight_decay=0.0)
            optimizer.zero_grad(set_to_none=True)
            trainer = DistributedTrueRecGRPOTrainerV1(
                ddp, TOKEN_IDS.__getitem__, 0, device=torch.device("cpu"), streaming_microbatch_size=1,
            )
            result = trainer.backward_global_group(_group(case_name, completions))
            gradients = tensor_inventory(policy, gradients=True)
            optimizer.step()
            if rank == 0:
                cases[case_name] = {
                    "values": {
                        "frontier": result.global_frontier_value,
                        "hpr": result.global_hpr_value_raw,
                        "total": result.global_total_value,
                    },
                    "gradients": gradients,
                    "state": tensor_inventory(policy, gradients=False),
                    "runtime_plan_hash": result.runtime_plan_hash,
                    "physical_forward_calls": result.physical_forward_calls,
                    "physical_backward_calls": result.physical_backward_calls,
                }
            dist.barrier()
        if rank == 0:
            torch.save(cases, output_file)
    finally:
        dist.destroy_process_group()


def run_audit(output_dir: Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "gloo-init")
        output_file = str(Path(directory) / "rank0.pt")
        mp.spawn(_worker, args=(init_file, output_file), nprocs=DDP_WORLD_SIZE, join=True)
        distributed = torch.load(output_file, map_location="cpu", weights_only=False)

    case_results = {}
    maxima = {"frontier": 0.0, "hpr": 0.0, "total": 0.0, "gradient": 0.0, "state": 0.0}
    scaling_ratios = []
    for case_index, (case_name, completions) in enumerate(CASES.items()):
        torch.manual_seed(4400 + case_index)
        reference_policy = TinyPolicy()
        optimizer = torch.optim.AdamW(reference_policy.parameters(), lr=AUDIT_LR, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        reference, runtime = full_g8_reference(reference_policy, _group(case_name, completions))
        reference.total_loss.backward()
        reference_gradients = tensor_inventory(reference_policy, gradients=True)
        optimizer.step()
        reference_state = tensor_inventory(reference_policy, gradients=False)
        actual = distributed[case_name]
        differences = {
            "frontier": abs(float(reference.frontier_loss.detach()) - actual["values"]["frontier"]),
            "hpr": abs(float(reference.hpr_loss_raw.detach()) - actual["values"]["hpr"]),
            "total": abs(float(reference.total_loss.detach()) - actual["values"]["total"]),
            "gradient": 0.0,
            "state": 0.0,
        }
        for name, expected in reference_gradients.items():
            observed = actual["gradients"][name]
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
            differences["gradient"] = max(differences["gradient"], float((observed - expected).abs().max()))
            wrong = observed / DDP_WORLD_SIZE
            if float(expected.abs().max()) > 0:
                scaling_ratios.append(float(wrong.norm() / expected.norm()))
                if torch.allclose(wrong, expected, rtol=1e-5, atol=1e-6):
                    raise AssertionError("missing DDP world-size scaling was not detected")
        for name, expected in reference_state.items():
            observed = actual["state"][name]
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
            differences["state"] = max(differences["state"], float((observed - expected).abs().max()))
        torch.testing.assert_close(
            torch.tensor(actual["values"]["frontier"], dtype=reference.frontier_loss.dtype),
            reference.frontier_loss.detach(),
        )
        torch.testing.assert_close(
            torch.tensor(actual["values"]["hpr"], dtype=reference.hpr_loss_raw.dtype),
            reference.hpr_loss_raw.detach(),
        )
        torch.testing.assert_close(
            torch.tensor(actual["values"]["total"], dtype=reference.total_loss.dtype),
            reference.total_loss.detach(),
        )
        for key, value in differences.items():
            maxima[key] = max(maxima[key], value)
        case_results[case_name] = {
            "status": "PASS", "runtime_trigger": runtime.hpr.trigger,
            "differences": differences,
            "local_physical_forward_calls": actual["physical_forward_calls"],
            "local_physical_backward_calls": actual["physical_backward_calls"],
        }
    if not scaling_ratios or not all(abs(value - 0.25) <= 1e-5 for value in scaling_ratios):
        raise AssertionError(f"wrong-scaling counterexample ratio failed: {scaling_ratios}")
    audit = {
        "status": "PASS", "backend": "gloo", "world_size": DDP_WORLD_SIZE,
        "global_G": 8, "local_G_per_rank": 2, "business_groups_per_optimizer_step": 1,
        "ddp_world_size_scaling_required": True,
        "wrong_scaling_gradient_ratio": sum(scaling_ratios) / len(scaling_ratios),
        "global_frontier_value_equivalent": True,
        "global_hpr_value_equivalent": True,
        "global_total_value_equivalent": True,
        "pre_step_gradient_equivalent": True,
        "post_step_model_state_equivalent": True,
        "cases": case_results, "max_abs_differences": maxima,
        "gpu_inference_started": False, "optimizer_steps_per_case": 1,
        "formal_pilot4096_started": False, "next_experiment_started": False,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cpu_gloo_equivalence.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"DDP_CPU_GLOO": run_audit(args.output_dir)["status"]}))


if __name__ == "__main__":
    main()
