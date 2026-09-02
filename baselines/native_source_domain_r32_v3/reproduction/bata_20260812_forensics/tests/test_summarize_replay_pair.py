import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_replay_pair.py"
SPEC = importlib.util.spec_from_file_location("bata_summarize_replay_pair_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_run(root: Path, label: str, changes: dict | None = None) -> Path:
    changes = changes or {}
    run = root / label
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    for rank in range(4):
        batch_suffix = changes.get("batch", "same")
        rng_suffix = changes.get("rng", "same")
        runtime_order = changes.get("runtime_order", ["shape-0", "shape-1"])
        observed_count = changes.get("observed_count", 2)
        loss_delta = changes.get("loss", 0.0)
        pre_suffix = changes.get("pre", "same")
        post_suffix = changes.get("post", "same")
        preclip_suffix = changes.get("preclip", "same")
        postclip_suffix = changes.get("postclip", "same")
        grad_norm = changes.get("grad_norm", 0.5)
        final_suffix = changes.get("final", "same")
        bucket = {
            "bucket_call_index": 0,
            "bucket_index": 0,
            "layout_sha256": "layout",
            "dtype": "torch.float32",
            "numel": 8,
            "shape": [8],
            "is_last": True,
            "parameter_count": 2,
            "parameter_numel": 8,
        }
        rows = [
            {
                "event": "rng_restored",
                "rank": rank,
                "rng": {"torch_cuda": f"rng-{rank}-{rng_suffix}"},
            },
            {
                "event": "ddp_pre_allreduce",
                "rank": rank,
                "fingerprint": f"pre-{rank}-{pre_suffix}",
                **bucket,
            },
            {
                "event": "ddp_post_allreduce",
                "rank": rank,
                "fingerprint": f"post-{rank}-{post_suffix}",
                **bucket,
            },
            {
                "event": "gradient_clip",
                "rank": rank,
                "stage": "PRE_CLIP",
                "gradients": {
                    "sha256": f"preclip-{rank}-{preclip_suffix}",
                    "tensor_count": 2,
                },
            },
            {
                "event": "gradient_clip",
                "rank": rank,
                "stage": "POST_CLIP",
                "returned_grad_norm": grad_norm,
                "gradients": {
                    "sha256": f"postclip-{rank}-{postclip_suffix}",
                    "tensor_count": 2,
                },
            },
            {
                "event": "optimizer_step",
                "rank": rank,
                "microbatch_count": 2,
                "ordered_batch_fingerprint": f"frozen-{rank}-{batch_suffix}",
                "ordered_microbatch_sha256": [f"micro-{rank}-0-{batch_suffix}", f"micro-{rank}-1-{batch_suffix}"],
                "batch_contract": {
                    "source": "frozen_step554_contract",
                    "contract_sha256": f"contract-{batch_suffix}",
                    "rank_ordered_batch_fingerprint": f"frozen-{rank}-{batch_suffix}",
                    "actual_cpu_ordered_batch_fingerprint": f"frozen-{rank}-{batch_suffix}",
                    "actual_cpu_ordered_microbatch_sha256": [
                        f"micro-{rank}-0-{batch_suffix}",
                        f"micro-{rank}-1-{batch_suffix}",
                    ],
                    "actual_cpu_value_and_order_exact": True,
                    "actual_value_source": "cpu_collator_before_accelerator",
                    "expected_microbatch_count": 2,
                    "observed_microbatch_count": observed_count,
                    "runtime_ordered_metadata_sha256": runtime_order,
                    "runtime_metadata_fingerprint": MODULE.canonical_hash(runtime_order),
                },
                "rng": {"unused": True},
                "rank_local_micro_losses": [1.0 + rank + loss_delta, 2.0 + rank],
                "rank_local_loss_mean": 1.5 + rank + loss_delta / 2,
                "grad_norm": grad_norm,
            },
        ]
        if changes.get("fadet"):
            rows.extend(
                [
                    {
                        "event": "fa2_runtime_probe_installed",
                        "rank": rank,
                        "environment_value": "1",
                    },
                    *(
                        [
                            {
                                "event": "fa2_runtime_call",
                                "rank": rank,
                                "api": "flash_attn_varlen_func",
                                "deterministic": True,
                                "deterministic_confirmed": True,
                                "environment_value": "1",
                            }
                        ]
                        if not changes.get("missing_fa_runtime")
                        else []
                    ),
                ]
            )
        if changes.get("parameter_divergence"):
            rows.append(
                {
                    "event": "parameter_gradient_divergence",
                    "rank": rank,
                    "bucket_call_index": 0,
                    "layout_sha256": "layout",
                    "parameter_count": 2,
                    "parameters": [
                        {
                            "name": "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                            "layer": 0,
                            "category": "q",
                            "numel": 4,
                            "equal": False,
                            "relative_l2": 0.1,
                            "cosine": 0.99,
                            "max_abs_delta": 0.01,
                            "difference_l2": 0.2,
                            "norm_a": 2.0,
                            "norm_b": 2.0,
                            "dot": 3.96,
                        },
                        {
                            "name": "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight",
                            "layer": 0,
                            "category": "up",
                            "numel": 4,
                            "equal": True,
                            "relative_l2": 0.0,
                            "cosine": 1.0,
                            "max_abs_delta": 0.0,
                            "difference_l2": 0.0,
                            "norm_a": 1.0,
                            "norm_b": 1.0,
                            "dot": 1.0,
                        },
                    ],
                }
            )
        if rank == 0:
            rows.extend(
                [
                    {
                        "event": "heavy_fingerprint",
                        "label": "initial553",
                        "canonical_lora_sha256": f"initial-lora-{changes.get('initial_lora', 'same')}",
                        "optimizer": {
                            "sha256": f"initial-optimizer-{changes.get('initial_optimizer', 'same')}"
                        },
                        "checkpoint_sha": {
                            "exact": True,
                            "actual": {
                                "adapter_model.safetensors": f"checkpoint-{changes.get('checkpoint', 'same')}"
                            },
                        },
                    },
                    {
                        "event": "heavy_fingerprint",
                        "label": "step554",
                        "canonical_lora_sha256": f"lora-{final_suffix}",
                        "effective_ba": {"sha256": f"ba-{final_suffix}"},
                        "optimizer": {"sha256": f"optimizer-{final_suffix}"},
                    },
                ]
            )
        (evidence / f"rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return run


def paired(tmp_path: Path, changes_b: dict | None = None) -> dict:
    a = write_run(tmp_path, "A")
    b = write_run(tmp_path, "B", changes_b)
    return MODULE.summarize_pair(a, b)


def paired_fadet(tmp_path: Path, changes_b: dict | None = None, *, missing_runtime: bool = False) -> dict:
    a = write_run(tmp_path, "FADET-A", {"fadet": True, "missing_fa_runtime": missing_runtime})
    changes = {"fadet": True, **(changes_b or {})}
    b = write_run(tmp_path, "FADET-B", changes)
    return MODULE.summarize_pair(a, b)


@pytest.mark.parametrize(
    "change",
    [
        {"batch": "different"},
        {"rng": "different"},
        {"runtime_order": ["shape-1", "shape-0"]},
        {"observed_count": 1},
        {"checkpoint": "different"},
        {"initial_lora": "different"},
        {"initial_optimizer": "different"},
    ],
)
def test_contract_mismatch_is_fail_closed(tmp_path: Path, change: dict) -> None:
    result = paired(tmp_path, change)
    assert result["repeatability_classification"] == "CONTRACT_MISMATCH"
    assert result["verdict"] == "CONTRACT_MISMATCH"


def test_d1_when_contract_matches_but_local_loss_differs(tmp_path: Path) -> None:
    result = paired(tmp_path, {"loss": 0.25, "final": "different"})
    assert result["repeatability_classification"] == "D1_PRE_BACKWARD"
    assert result["verdict"] == "UNRESOLVED"


def test_d2_when_pre_and_post_update_all_match(tmp_path: Path) -> None:
    result = paired(tmp_path)
    assert result["repeatability_classification"] == "D2_REPEATABLE"
    assert result["verdict"] == "STEP554_FULLY_REPEATABLE"


def test_local_backward_is_first_divergence(tmp_path: Path) -> None:
    result = paired(tmp_path, {"pre": "different", "post": "different", "final": "different"})
    assert result["repeatability_classification"] == "D3_POST_BACKWARD"
    assert result["verdict"] == "LOCAL_BACKWARD"


def test_ddp_is_first_divergence(tmp_path: Path) -> None:
    result = paired(tmp_path, {"post": "different", "preclip": "different", "final": "different"})
    assert result["verdict"] == "DDP_NCCL"


def test_clipping_is_first_divergence(tmp_path: Path) -> None:
    result = paired(tmp_path, {"postclip": "different", "grad_norm": 0.6, "final": "different"})
    assert result["verdict"] == "GRAD_CLIPPING"


def test_optimizer_is_first_divergence(tmp_path: Path) -> None:
    result = paired(tmp_path, {"final": "different"})
    assert result["verdict"] == "OPTIMIZER_UPDATE"


def test_fadet_missing_runtime_confirmation_cannot_be_stable(tmp_path: Path) -> None:
    result = paired_fadet(tmp_path, missing_runtime=True)
    assert result["fa2_deterministic_runtime"]["all_runs_all_ranks_confirmed"] is False
    assert result["verdict"] == "UNRESOLVED"


def test_fadet_pre_allreduce_mismatch_is_local_backward(tmp_path: Path) -> None:
    result = paired_fadet(
        tmp_path,
        {
            "pre": "different",
            "post": "different",
            "final": "different",
            "parameter_divergence": True,
        },
    )
    assert result["pre_backward_local_loss_repeatable"] is True
    assert result["verdict"] == "FADET_LOCAL_BACKWARD"
    divergence = result["parameter_gradient_divergence"]
    assert divergence["parameter_count"] == 2
    assert divergence["divergent_parameter_count"] == 1
    assert divergence["attention_projections"]["divergent_parameter_count"] == 1
    assert divergence["mlp_projections"]["divergent_parameter_count"] == 0


def test_fadet_full_equality_is_stable(tmp_path: Path) -> None:
    result = paired_fadet(tmp_path)
    assert result["fa2_deterministic_runtime"]["all_runs_all_ranks_confirmed"] is True
    assert result["verdict"] == "FADET_STEP554_FULLY_REPEATABLE"
