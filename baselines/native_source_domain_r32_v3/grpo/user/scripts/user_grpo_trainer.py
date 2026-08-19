"""TRL-compatible User GRPO trainer with token-local advantages."""

from __future__ import annotations

import collections
import inspect
import statistics
from types import SimpleNamespace

import torch


def _patch_trl_optional_dependency_flags():
    import trl.import_utils as import_utils

    for name in dir(import_utils):
        if name.startswith("_") and name.endswith("_available"):
            value = getattr(import_utils, name)
            if isinstance(value, tuple) and value and value[0] is False:
                setattr(import_utils, name, False)


_patch_trl_optional_dependency_flags()
from trl import GRPOTrainer  # noqa: E402

from user_action_reward import score_action  # noqa: E402
from user_chain_reward import score_chain  # noqa: E402
from user_generated_token_projection import project_compiled_mask_to_generated  # noqa: E402
from user_penalty_mask import compile_penalty_mask  # noqa: E402
from user_training_objective import (  # noqa: E402
    ADVANTAGE_EPSILON,
    BASE_LAMBDA,
    EPSILON,
    G,
    build_token_advantages,
    clipped_grpo_loss_from_logps,
    group_population_advantages,
)


def prepare_scored_rollout(rows, completion_ids_list, tokenizer):
    """Score one route-homogeneous list of unique prompts, each with G completions."""
    if len(completion_ids_list) != len(rows) * G:
        raise ValueError("rollout must contain exactly G=4 completions per prompt")
    routes = {row["route"] for row in rows}
    if len(routes) != 1:
        raise ValueError("Action and Chain must run in separate route-homogeneous batches")
    route = next(iter(routes))
    expanded_rows = [row for row in rows for _ in range(G)]
    sample_ids = [row["sample_id"] for row in expanded_rows]
    scores = []
    compiled_penalties = []
    completions = []
    rewards = []
    for row, completion_ids in zip(expanded_rows, completion_ids_list):
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        scorer = score_action if route == "action" else score_chain
        score = scorer(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, score.violations, tokenizer, route)
        compiled = project_compiled_mask_to_generated(
            compiled,
            completion_ids,
            tokenizer=tokenizer,
            completion=completion,
        )
        if compiled["token_count"] != len(completion_ids):
            raise AssertionError("generated completion and projected mask shape differ")
        completions.append(completion)
        scores.append(score)
        compiled_penalties.append(compiled)
        rewards.append(float(score.reward))

    sequence_advantages, group_means, group_stds = group_population_advantages(
        rewards, sample_ids, group_size=G, epsilon=ADVANTAGE_EPSILON
    )
    completion_lengths = [len(ids) for ids in completion_ids_list]
    token_advantages, objective = build_token_advantages(
        sequence_advantages,
        compiled_penalties,
        [route] * len(expanded_rows),
        completion_lengths,
        base_lambda=BASE_LAMBDA,
    )
    completion_mask = (
        torch.arange(token_advantages.size(1))[None, :]
        < torch.tensor(completion_lengths)[:, None]
    )
    local_mask = objective["local_mask"] & completion_mask
    violation_counts = collections.Counter(
        violation.kind for score in scores for violation in score.violations
    )
    metrics = {
        "route": route,
        "prompt_count": len(rows),
        "candidate_count": len(expanded_rows),
        "task_reward_mean": statistics.fmean(rewards),
        "task_reward_std": statistics.pstdev(rewards),
        "zero_std_ratio": float((group_stds == 0).float().mean()),
        "sequence_advantage_mean": float(sequence_advantages.mean()),
        "sequence_advantage_std": float(sequence_advantages.std(correction=0)),
        "token_advantage_mean": float(token_advantages[completion_mask].mean()),
        "token_advantage_std": float(token_advantages[completion_mask].std(correction=0)),
        "masked_candidate_rate": float(local_mask.any(dim=1).float().mean()),
        "masked_token_rate": float(local_mask.sum() / completion_mask.sum()),
        "per_kind_masked_token_count": objective["per_kind_masked_token_count"],
        "per_kind_incremental_negative_mass": objective["per_kind_incremental_negative_mass"],
        "positive_sequence_masked_token_flip_count": objective[
            "positive_sequence_masked_token_flip_count"
        ],
        "violation_counts": dict(violation_counts),
    }
    if route == "action":
        metrics.update(
            {
                "f1_mean": statistics.fmean(score.f1 for score in scores),
                "precision_mean": statistics.fmean(score.precision for score in scores),
                "recall_mean": statistics.fmean(score.recall for score in scores),
                "hallucination_candidate_rate": sum(
                    any(item.kind == "hallucinated_sid" for item in score.violations)
                    for score in scores
                )
                / len(scores),
                "duplicate_candidate_rate": sum(
                    any(item.kind == "duplicate_sid" for item in score.violations) for score in scores
                )
                / len(scores),
                "wrong_selection_candidate_rate": sum(
                    any(item.kind == "wrong_selection_sid" for item in score.violations)
                    for score in scores
                )
                / len(scores),
            }
        )
    else:
        metrics.update(
            {
                "total_reward_mean": statistics.fmean(score.reward for score in scores),
                "action_alignment_mean": statistics.fmean(score.action_f1 for score in scores),
                "logic_alignment_mean": statistics.fmean(score.logic_f1 for score in scores),
            }
        )
    return {
        "route": route,
        "expanded_rows": expanded_rows,
        "completion_ids_list": [list(ids) for ids in completion_ids_list],
        "completion_lengths": completion_lengths,
        "completions": completions,
        "scores": scores,
        "compiled_penalties": compiled_penalties,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "group_reward_means": group_means,
        "group_reward_stds": group_stds,
        "sequence_advantages": sequence_advantages,
        "token_advantages": token_advantages,
        "completion_mask": completion_mask,
        "local_penalty_mask": local_mask,
        "metrics": metrics,
    }


class UserGRPOTrainer(GRPOTrainer):
    """GRPO clipped loss whose policy signal is a [B,T] token advantage tensor."""

    @classmethod
    def for_correctness_smoke(cls, model, *, temperature=0.9, forward_batch_size=1):
        trainer = object.__new__(cls)
        trainer.temperature = temperature
        trainer.epsilon_low = EPSILON
        trainer.epsilon_high = EPSILON
        trainer.loss_type = "grpo"
        trainer.current_gradient_accumulation_steps = 1
        trainer.args = SimpleNamespace(delta=None)
        trainer.forward_batch_size = forward_batch_size
        trainer._supports_logits_to_keep = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in inspect.signature(model.forward).parameters.values()
            )
        )
        trainer.last_loss_metrics = {}
        return trainer

    def _get_user_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        outputs = []
        batch_size = self.forward_batch_size
        for start in range(0, input_ids.size(0), batch_size):
            batch_attention = attention_mask[start : start + batch_size]
            position_ids = batch_attention.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(batch_attention == 0, 0)
            model_inputs = {
                "input_ids": input_ids[start : start + batch_size],
                "attention_mask": batch_attention,
                "position_ids": position_ids,
                "use_cache": False,
            }
            if self._supports_logits_to_keep:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :] / self.temperature
            target = input_ids[start : start + batch_size, -logits_to_keep:]
            logps = torch.gather(torch.log_softmax(logits, dim=-1), 2, target.unsqueeze(-1)).squeeze(-1)
            outputs.append(logps)
        return torch.cat(outputs, dim=0)

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("UserGRPOTrainer does not return model outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        token_advantages = inputs["token_advantages"]
        if token_advantages.shape != completion_ids.shape:
            raise ValueError("token_advantages and completion_ids must have identical shape")
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps = self._get_user_per_token_logps(
            model, input_ids, attention_mask, completion_ids.size(1)
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        loss, per_token_loss, log_ratio = clipped_grpo_loss_from_logps(
            per_token_logps,
            old_per_token_logps,
            token_advantages,
            completion_mask,
            epsilon=self.epsilon_low,
            delta=self.args.delta,
        )
        loss = loss / self.current_gradient_accumulation_steps
        with torch.no_grad():
            valid = completion_mask.bool()
            ratios = torch.exp(log_ratio[valid].float())
            self.last_loss_metrics = {
                "loss": float(loss),
                "ratio_mean": float(ratios.mean()),
                "clip_fraction": float(
                    ((ratios - 1.0).abs() > self.epsilon_low).float().mean()
                ),
                "per_token_loss_mean": float(per_token_loss[valid].mean()),
                "finite": bool(
                    torch.isfinite(loss)
                    and torch.isfinite(per_token_logps).all()
                    and torch.isfinite(token_advantages[valid]).all()
                ),
            }
        return loss
