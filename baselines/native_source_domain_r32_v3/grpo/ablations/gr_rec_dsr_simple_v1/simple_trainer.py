"""Isolated DSR-Simple trainer; baseline and DSR v1 remain untouched."""
from __future__ import annotations

import os
import time

import torch

from gr_rec_dsr_v1.dsr_monitor import summarize_nothink_records
from gr_rec_dsr_v1.dsr_trainer import DsrGRPOTrainer

from .simple_monitor import summarize_simple_think_records


class SimpleDsrGRPOTrainer(DsrGRPOTrainer):
    """Reuse the verified PPO/NoThink implementation with Simple Think scores."""

    def __init__(self, *args, **kwargs):
        kwargs["dsr_think_lambda"] = float(
            kwargs.pop(
                "simple_think_lambda",
                os.environ.get("DSR_SIMPLE_THINK_LAMBDA", "0.10"),
            )
        )
        kwargs["dsr_nothink_scale"] = float(
            kwargs.pop(
                "simple_nothink_scale",
                os.environ.get("DSR_SIMPLE_NOTHINK_SCALE", "1.0"),
            )
        )
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        from trl.trainer.grpo_trainer import gather_object

        self._dsr_capture.reset()
        # Skip DsrGRPOTrainer's old Think aggregation, but retain baseline generation/scoring.
        output = super(DsrGRPOTrainer, self)._generate_and_score_completions(inputs)
        local_records = self._dsr_capture.records
        if len(local_records) != len(inputs):
            raise RuntimeError(
                f"DSR-Simple captured {len(local_records)} records for {len(inputs)} local inputs"
            )
        gather_started = time.perf_counter()
        records = gather_object(local_records)
        gather_wall_sec = time.perf_counter() - gather_started
        route = inputs[0]["route"]
        group_size = 4 if route == "think" else 8
        self._validate_groups(records, group_size)
        local_count = len(local_records)
        process_slice = slice(
            self.accelerator.process_index * local_count,
            (self.accelerator.process_index + 1) * local_count,
        )
        device = output["completion_ids"].device
        payload = {
            "route": route,
            "objective": "DSR-Simple",
            "tokenizer_audit": self._dsr_capture.tokenizer_audit,
            "dsr_record_gather_wall_sec": gather_wall_sec,
        }
        traces = []
        if route == "think":
            summary, _scores, advantage_values = summarize_simple_think_records(records)
            output["dsr_aux_advantages"] = torch.tensor(
                advantage_values[process_slice], dtype=torch.float32, device=device
            )
            output["dsr_sa_positions"] = torch.full(
                (local_count,), -1, dtype=torch.long, device=device
            )
            output["dsr_frequency_weights"] = torch.zeros(
                local_count, dtype=torch.float32, device=device
            )
            output["dsr_rescue_coefficients"] = torch.zeros(
                local_count, dtype=torch.float32, device=device
            )
            payload.update(summary)
            traces = [{
                "group_id": item["group_id"],
                "primary_reward": item["primary_reward"],
                "raw_interest_n": item["raw_interest_n"],
                "s_n": item["s_n"],
                "unique_valid_target_a": item["unique_valid_target_a"],
                "d_a": item["d_a"],
                "simple_s_aux": item["simple_s_aux"],
                "simple_a_aux": item["simple_a_aux"],
                "simple_branch": item["simple_branch"],
                "diagnostic_only": item["diagnostic_only"],
            } for item in records[:8]]
        else:
            positions = []
            weights = []
            coefficients = []
            summary, plans = summarize_nothink_records(records, self.dsr_nothink_scale)
            for plan in plans:
                positions.extend(plan.positions if plan.active else [-1] * group_size)
                weights.extend(plan.frequency_weights if plan.active else [0.0] * group_size)
                coefficients.extend([plan.coefficient if plan.active else 0.0] * group_size)
            output["dsr_aux_advantages"] = torch.zeros(
                local_count, dtype=torch.float32, device=device
            )
            output["dsr_sa_positions"] = torch.tensor(
                positions[process_slice], dtype=torch.long, device=device
            )
            output["dsr_frequency_weights"] = torch.tensor(
                weights[process_slice], dtype=torch.float32, device=device
            )
            output["dsr_rescue_coefficients"] = torch.tensor(
                coefficients[process_slice], dtype=torch.float32, device=device
            )
            payload.update(summary)
            payload["nothink_implementation"] = "gr_rec_dsr_v1 (reused)"
        self._smoke_log[-1]["dsr_simple"] = payload
        self._write_dsr_monitor(payload, traces)
        return output

    def log(self, logs, start_time=None):
        result = super().log(logs, start_time)
        return result
