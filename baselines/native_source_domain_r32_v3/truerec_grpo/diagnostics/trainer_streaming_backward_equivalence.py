"""CPU-only formal audit for immediate per-chunk backward of one logical G8 group."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
from typing import Any
import weakref

import torch


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "credit", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from rollout_runtime_v1 import G, PPO_OLD_LOGP_SOURCE  # noqa: E402
from trainer_g8_microbatch_equivalence import CASES, TOKEN_IDS, TinyPolicy, _group, full_g8_reference  # noqa: E402
from truerec_grpo_trainer_v1 import (  # noqa: E402
    PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP,
    TRAINER_MICROBATCH_SIZE,
    TrueRecGRPOTrainerV1,
)
from truerec_loss_v1 import HPR_LAMBDA  # noqa: E402


class GraphLifetimeTinyPolicy(TinyPolicy):
    """Weakly observes whether a previous chunk output survives into the next forward."""
    def __init__(self) -> None:
        super().__init__()
        self._last_logits_ref = None
        self.graph_bearing_output_seen_before_next_forward = False

    def forward(self, input_ids, attention_mask):
        if self._last_logits_ref is not None and self._last_logits_ref() is not None:
            self.graph_bearing_output_seen_before_next_forward = True
        output = super().forward(input_ids, attention_mask)
        self._last_logits_ref = weakref.ref(output.logits)
        return output


def _gradient_max_abs(reference: TinyPolicy, streaming: TinyPolicy) -> float:
    maximum = 0.0
    for (reference_name, reference_parameter), (stream_name, stream_parameter) in zip(
        reference.named_parameters(), streaming.named_parameters(),
    ):
        if reference_name != stream_name or reference_parameter.grad is None or stream_parameter.grad is None:
            raise AssertionError("gradient inventory mismatch")
        torch.testing.assert_close(reference_parameter.grad, stream_parameter.grad)
        maximum = max(maximum, float((reference_parameter.grad - stream_parameter.grad).abs().max()))
    return maximum


def run_audit(output_dir: Path | None = None) -> dict[str, Any]:
    torch.manual_seed(12031)
    initial = TinyPolicy()
    cases = {}
    maxima = {"frontier": 0.0, "hpr": 0.0, "total": 0.0, "gradient": 0.0}
    for case_name, completions in CASES.items():
        group = _group(case_name, completions)
        reference_policy = TinyPolicy()
        reference_policy.load_state_dict(initial.state_dict())
        streaming_policy = GraphLifetimeTinyPolicy()
        streaming_policy.load_state_dict(initial.state_dict())
        reference, runtime = full_g8_reference(reference_policy, group)
        reference.total_loss.backward()
        trainer = TrueRecGRPOTrainerV1(streaming_policy, TOKEN_IDS.__getitem__, 0)
        streamed = trainer.backward_group_streaming(group)
        reference_values = {
            "frontier": float(reference.frontier_loss.detach()),
            "hpr": float(reference.hpr_loss_raw.detach()),
            "total": float(reference.total_loss.detach()),
        }
        streamed_values = {
            "frontier": streamed.frontier_value,
            "hpr": streamed.hpr_value_raw,
            "total": streamed.total_value,
        }
        for key in reference_values:
            torch.testing.assert_close(
                torch.tensor(streamed_values[key], dtype=torch.float64),
                torch.tensor(reference_values[key], dtype=torch.float64),
            )
        gradient_difference = _gradient_max_abs(reference_policy, streaming_policy)
        gc.collect()
        graph_retained = (
            streaming_policy.graph_bearing_output_seen_before_next_forward
            or (streaming_policy._last_logits_ref is not None and streaming_policy._last_logits_ref() is not None)
        )
        if graph_retained:
            raise AssertionError("graph-bearing chunk output survived its backward boundary")
        differences = {
            key: abs(reference_values[key] - streamed_values[key]) for key in reference_values
        }
        differences["gradient"] = gradient_difference
        for key, value in differences.items():
            maxima[key] = max(maxima[key], value)
        if (
            trainer.logical_policy_scoring_passes != 1
            or trainer.physical_policy_forward_calls != PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP
            or trainer.streaming_backward_calls != PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP
            or trainer.streaming_full_g8_plan_builds != 1
            or trainer.hpr_extra_forward_calls != 0
        ):
            raise AssertionError("streaming call-count contract failed")
        cases[case_name] = {
            "status": "PASS",
            "runtime_trigger": runtime.hpr.trigger,
            "differences": differences,
            "logical_policy_scoring_passes": trainer.logical_policy_scoring_passes,
            "physical_policy_forward_calls": trainer.physical_policy_forward_calls,
            "physical_backward_calls": trainer.streaming_backward_calls,
            "full_g8_plan_builds": trainer.streaming_full_g8_plan_builds,
            "graph_bearing_state_retained_across_chunks": False,
        }

    method_source = inspect.getsource(TrueRecGRPOTrainerV1.backward_group_streaming)
    forbidden_accumulators = ("frontier_parts", "hpr_position_losses", ".append(")
    if any(token in method_source for token in forbidden_accumulators):
        raise AssertionError("streaming method contains a graph-bearing chunk accumulator")
    if "zero_grad" in method_source or "optimizer" in method_source:
        raise AssertionError("streaming method controls caller-owned gradient/optimizer state")
    audit = {
        "status": "PASS",
        "G": G,
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "streaming_backward_implemented": True,
        "physical_forward_calls_per_group": PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP,
        "physical_backward_calls_per_group": PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP,
        "full_g8_plan_built_once": True,
        "frontier_value_equivalent": True,
        "hpr_value_equivalent": True,
        "total_value_equivalent": True,
        "gradient_equivalent": True,
        "cases": cases,
        "max_abs_differences": maxima,
        "graph_bearing_state_retained_across_chunks": False,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "hpr_lambda": HPR_LAMBDA,
        "execution": {
            "gpu_inference_started": False,
            "real_model_backward_started": False,
            "optimizer_steps": 0,
            "next_experiment_started": False,
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "streaming_backward_equivalence.json").write_text(
            json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (output_dir / "CHATGPT_PHASE1_2C3_REVIEW.txt").write_text(
            "TrueRec-GRPO Phase 1.2C3 CPU streaming-backward equivalence: PASS\n"
            "One full G8 runtime plan is preserved while four microbatch=2 contributions are backpropagated immediately. Frontier values, globally weighted HPR values, totals, and tiny-policy gradients match the full-G8 reference across all required cases.\n"
            "Weak-reference and source audits confirm that no graph-bearing chunk output or loss accumulator survives across forwards. No GPU, real-model backward, optimizer, or training ran.\n",
            encoding="utf-8",
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = run_audit(args.output_dir)
    print(f"PHASE1_2C3_CPU_AUDIT={audit['status']}")


if __name__ == "__main__":
    main()
