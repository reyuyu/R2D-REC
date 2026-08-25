"""Thin production trainer for Think-only Composite Interest GRPO."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import sys
import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = GRPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import trl_import_fix  # noqa: F401
from grpo_model import render_prompt
from grpo_trl_trainer import RecGRPOTrainer
from trl.trainer.grpo_trainer import gather_object

from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units
from .interest_metric import (
    INTEREST_TIEBREAK_SCALE, beam_primary_composite_rewards,
    display_interest_unit, diversity_monitor,
    matching_diagnostics, monitor_record, population_advantages,
    score_parsed_interests,
)

GROUP_SIZE = 4


def decode_reward_completion(tokenizer, candidate_ids):
    """Decode reward-side CoT without stripping SID or </think> tokens."""
    return tokenizer.decode(candidate_ids, skip_special_tokens=False)



def assert_gold_isolation(item, rendered_prompt=None, decoded_input=None):
    """Fail closed if reward-only Gold CoT reaches model-visible text."""
    gold = item.get("gold_cot")
    if not isinstance(gold, str) or not gold:
        raise RuntimeError("Think Composite record is missing reward-only gold_cot")
    visible = (item.get("prompt"), rendered_prompt, decoded_input)
    if any(gold in value for value in visible if isinstance(value, str)):
        raise RuntimeError("GOLD_COT_LEAKAGE_DETECTED")


def score_candidate(
    completion, gold_cot, prompt, beam_raw,
    tiebreak_scale=INTEREST_TIEBREAK_SCALE,
):
    gold = extract_interest_units(gold_cot, prompt)
    if not gold.parser_success or not gold.units:
        raise RuntimeError("parser-invalid Gold CoT reached training")
    candidate = extract_interest_units(completion, prompt)
    score = score_parsed_interests(candidate, gold)
    raw_n = len(candidate.units)
    grounded_n = sum(bool(unit.grounded_evidence_sids) for unit in candidate.units)
    record = monitor_record(float(beam_raw), score, raw_n, grounded_n, tiebreak_scale)
    record.update({
        "parser_success": candidate.parser_success,
        "parser_failure_reason": candidate.failure_reason,
        "gold_parser_success": gold.parser_success,
        "Q": score.match_quality,
        "pair_matches": [asdict(match) for match in score.matches],
        "interest_set": [unit.normalized_text for unit in candidate.units],
        **matching_diagnostics(candidate, gold, score.matches),
    })
    for key in ("beam_utility", "cot_utility"):
        if not math.isfinite(record[key]) or not 0.0 <= record[key] <= 1.0:
            raise RuntimeError(f"non-finite or out-of-range {key}")
    if not math.isfinite(record["composite_reward"]) or record["composite_reward"] < 0.0:
        raise RuntimeError("non-finite or negative composite_reward")
    return record


def classify_winner(beam_values, composite_values):
    beam_best = max(beam_values)
    top = [index for index, value in enumerate(beam_values) if value == beam_best]
    winner = max(range(len(composite_values)), key=composite_values.__getitem__)
    return {
        "beam_top_set": top,
        "beam_stable_winner": top[0],
        "composite_winner": winner,
        "top_set_tie_break": winner in top and winner != top[0],
        "strict_beam_reversal": winner not in top,
    }


def build_global_runtime(records, beam_rewards, group_size=GROUP_SIZE):
    if len(records) != len(beam_rewards) or len(records) % group_size:
        raise RuntimeError("global Composite metadata/reward alignment failure")
    candidates, groups, advantages = [], [], []
    for start in range(0, len(records), group_size):
        chunk = records[start:start + group_size]
        group_id = chunk[0]["group_id"]
        if len({row["group_id"] for row in chunk}) != 1:
            raise RuntimeError("global G4 recommendation-group contiguity failure")
        raw = beam_rewards[start:start + group_size]
        preliminary = [score_candidate(row["completion"], row["gold_cot"], row["prompt"], reward)
                       for row, reward in zip(chunk, raw)]
        cot_utilities = [row["cot_utility"] for row in preliminary]
        totals, tiebreak_scale = beam_primary_composite_rewards(raw, cot_utilities)
        scored = preliminary
        for row, total in zip(scored, totals):
            row["beam_contribution"] = row["beam_raw"]
            row["cot_contribution"] = tiebreak_scale * row["cot_utility"]
            row["interest_tiebreak_scale"] = tiebreak_scale
            row["composite_reward"] = total
        group_advantages = population_advantages(totals)
        parsed = [extract_interest_units(row["completion"], row["prompt"]) for row in chunk]
        gold_parsed = extract_interest_units(chunk[0]["gold_cot"], chunk[0]["prompt"])
        beam_utilities = [row["beam_utility"] for row in scored]
        cot_utilities = [row["cot_utility"] for row in scored]
        mean_total = sum(totals) / group_size
        composite_std = math.sqrt(sum((value - mean_total) ** 2 for value in totals) / group_size)
        winner = classify_winner(raw, totals)
        if winner["strict_beam_reversal"]:
            raise RuntimeError("STRICT_BEAM_REVERSAL_DETECTED")
        groups.append({
            "group_id": group_id,
            "gold_interest_units": [display_interest_unit(unit) for unit in gold_parsed.units],
            "beam_raw_vector": raw,
            "beam_utility_vector": beam_utilities,
            "cot_utility_vector": cot_utilities,
            "Q_vector": [row["Q"] for row in scored],
            "composite_reward_vector": totals,
            "interest_tiebreak_scale": tiebreak_scale,
            "final_advantage_vector": group_advantages,
            "beam_all_equal": len(set(raw)) == 1,
            "cot_all_equal": len(set(cot_utilities)) == 1,
            "composite_all_equal": len(set(totals)) == 1,
            "beam_active": len(set(raw)) > 1,
            "cot_active": len(set(cot_utilities)) > 1,
            "composite_active": composite_std > 0.0,
            "composite_reward_mean": mean_total,
            "composite_reward_population_std": composite_std,
            "composite_zero_std": composite_std == 0.0,
            "matched_candidate_count": sum(row["matched_interest_count"] > 0 for row in scored),
            "quality_active_candidate_count": sum(row["Q"] > 0 for row in scored),
            "strict_reversal_count": 0,
            **winner,
            **diversity_monitor(parsed),
        })
        for offset, (source, score, advantage) in enumerate(zip(chunk, scored, group_advantages)):
            monitor_source = {key: value for key, value in source.items()
                              if key != "gold_cot"}
            candidates.append({**monitor_source, "candidate_id": offset, **score,
                               "final_sequence_advantage": advantage})
        advantages.extend(group_advantages)
    if not all(math.isfinite(value) for value in advantages):
        raise RuntimeError("non-finite Composite advantage")
    return {
        "candidates": candidates,
        "groups": groups,
        "advantages": advantages,
        "strict_reversal_count": 0,
    }


def slice_global(values, rank, local_count, world_size):
    if len(values) != local_count * world_size:
        raise RuntimeError("global Composite advantage does not match DDP layout")
    start = rank * local_count
    return values[start:start + local_count]


def replace_log_tail(log, replacement):
    replacement = list(replacement)
    if len(log) < len(replacement):
        raise RuntimeError("advantage log is shorter than Composite replacement")
    for _ in replacement:
        log.pop()
    log.extend(replacement)


class ThinkCompositeInterestRecGRPOTrainer(RecGRPOTrainer):
    """Preserve production Think rollout/Beam and replace only final advantages."""

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids):
        if any(item.get("route") != "think" for item in inputs):
            raise RuntimeError("Composite Interest trainer accepts Think records only")
        rendered = [render_prompt(self.processing_class, prompt) for prompt in prompts]
        for item, visible in zip(inputs, rendered):
            token_ids = self.processing_class(visible, add_special_tokens=False)["input_ids"]
            decoded = self.processing_class.decode(token_ids, skip_special_tokens=False)
            assert_gold_isolation(item, visible, decoded)
        if self._monitor_enabled():
            self._monitor.set_beam_context(
                step=int(self.state.global_step),
                rollout_id=int(self._smoke_rollout_id),
                source="training",
            )
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids)
        beam_call = self._current_beam_call() or {}
        beam_results = {
            int(result["task_id"][1]): result
            for result in beam_call.get("local_results", [])
        }
        weighted = (rewards_per_func * self.reward_weights.to(
            self.accelerator.device).unsqueeze(0)).nansum(dim=1).tolist()
        raw_completions = [decode_reward_completion(self.processing_class, candidate_ids)
                           for candidate_ids in completion_ids]
        local_records = [{
            "group_id": item["recommendation_group_id"],
            "rank": self.accelerator.process_index,
            "local_index": index,
            "target_domain": item["target_domain"],
            "prompt": item["prompt"],
            "gold_cot": item["gold_cot"],
            "completion": raw_completion,
            "completion_length": len(candidate_ids),
            "beam_invalid_count": beam_results.get(index, {}).get("invalid"),
            "beam_exact_count": beam_results.get(index, {}).get("exact"),
            "beam_ab_count": beam_results.get(index, {}).get("ab"),
            "beam_a_count": beam_results.get(index, {}).get("a"),
            "beam_fixed_domain_prefix": beam_results.get(index, {}).get(
                "beam_fixed_domain_prefix"
            ),
            "domain_prefix": beam_results.get(index, {}).get("domain_prefix"),
            "beam_search_space": beam_results.get(index, {}).get(
                "beam_search_space"
            ),
        } for index, (item, raw_completion, candidate_ids) in enumerate(
            zip(inputs, raw_completions, completion_ids)
        )]
        self._composite_runtime = build_global_runtime(gather_object(local_records), weighted)
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        runtime = self._composite_runtime
        local_values = slice_global(runtime["advantages"], self.accelerator.process_index,
                                    int(output["advantages"].numel()), self.accelerator.num_processes)
        output["advantages"] = torch.tensor(local_values, dtype=output["advantages"].dtype,
                                             device=output["advantages"].device)
        if self._detailed_monitor:
            replace_log_tail(self._logs["advantages"], runtime["advantages"])
        if self._parity_audit and self._parity_log:
            self._parity_log[-1]["advantages"] = local_values
        self._smoke_log[-1].update({
            "reward_definition": "beam_raw+adaptive_tiebreak_scale*cot_utility",
            "composite_reward_mean": sum(row["composite_reward"] for row in runtime["candidates"]) / len(runtime["candidates"]),
            "composite_zero_std_group_rate": sum(row["composite_all_equal"] for row in runtime["groups"]) / len(runtime["groups"]),
        })
        if self._monitor_enabled():
            self._monitor.write_composite({
                "rollout_id": self._smoke_rollout_id,
                "step": self.state.global_step,
                "route": "think",
                "g": GROUP_SIZE,
                "beam_fixed_domain_prefix": True,
                "beam_search_space": "ABC_CONTINUATION_AFTER_FIXED_DOMAIN",
                "formula": {
                    "beam_primary": True,
                    "interest_tiebreak_scale_max": 0.25,
                    "normalization": "population_std_plus_1e-4",
                    "correction": 0,
                },
                **runtime,
            })
        del self._composite_runtime
        return output
