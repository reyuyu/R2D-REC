"""TrueRec fixed-domain G8 rollout records and rollout-policy logp capture."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from rollout_metrics import assess_candidate  # noqa: E402
from policy_scoring_v1 import POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE, parameter_versions, score_full_sequences  # noqa: E402


G = 8
FORMAL_EOS_TOKEN_IDS = (151645, 151643)
FORMAL_PAD_TOKEN_ID = 151643
GENERATION_KWARGS = {
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "repetition_penalty": 1.0, "max_new_tokens": 3, "num_return_sequences": G,
}
OLD_LOGPS_FROM_ROLLOUT_POLICY = True
PPO_OLD_LOGP_SOURCE = "FULL_FORWARD_RESCORE"
OLD_LOGPS_FROM_FULL_FORWARD_RESCORE = True
GENERATION_SCORES_USED_FOR_PPO = False
GENERATION_SCORE_LOGPS_ROLE = "DIAGNOSTIC_ONLY"


@dataclass(frozen=True)
class RolloutCandidate:
    sample_index: int
    completion_ids: tuple[int, ...]
    old_logps: tuple[float, ...]
    metrics: dict[str, Any]
    generation_score_logps: tuple[float, ...] = ()


@dataclass(frozen=True)
class BusinessGroupRollout:
    recommendation_group_id: str
    context_ids: tuple[int, ...]
    fixed_domain_token: str
    all_gold_abc: tuple[str, ...]
    candidates: tuple[RolloutCandidate, ...]

    def __post_init__(self):
        if len(self.candidates) != G or tuple(item.sample_index for item in self.candidates) != tuple(range(G)):
            raise ValueError("one business group must preserve exactly sample_index 0..7")


@dataclass(frozen=True)
class RolloutGenerationArtifacts:
    group: BusinessGroupRollout
    completion_ids: tuple[tuple[int, ...], ...]
    generation_scores: torch.Tensor


@dataclass(frozen=True)
class RawGenerationArtifacts:
    completion_ids: tuple[tuple[int, ...], ...]
    generation_scores: torch.Tensor
    generation_score_logps: tuple[tuple[float, ...], ...]


def generation_contract() -> dict[str, Any]:
    return dict(GENERATION_KWARGS)


def normalize_eos_token_ids(eos_token_id: int | Sequence[int] | None) -> list[int]:
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    return [int(value) for value in (eos_token_id or ())]


def render_rollout_context(record: dict[str, Any], renderer) -> list[int]:
    return renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])


def capture_sampled_logps(
    generation_logits: torch.Tensor, completion_ids: Sequence[Sequence[int]],
) -> list[tuple[float, ...]]:
    """Capture rollout-policy logp at each actually sampled generation step."""
    if generation_logits.ndim != 3 or generation_logits.shape[0] != len(completion_ids):
        raise ValueError("generation_logits must be [G,T,V]")
    log_probs = torch.log_softmax(generation_logits, dim=-1)
    captured_tensors = []
    for sample_index, token_ids in enumerate(completion_ids):
        if not 0 < len(token_ids) <= 3 or len(token_ids) > generation_logits.shape[1]:
            raise ValueError("completion length must be 1..3 and covered by generation logits")
        ids = torch.tensor(token_ids, device=log_probs.device, dtype=torch.long)
        captured_tensors.append(log_probs[sample_index, :len(token_ids)].gather(-1, ids.unsqueeze(-1)).squeeze(-1))
    lengths = [values.numel() for values in captured_tensors]
    flat = torch.cat(captured_tensors).detach().cpu()
    return [tuple(values.tolist()) for values in flat.split(lengths)]


def trim_generated_completion(
    ids: Sequence[int], eos_token_id: int | Sequence[int] | None, pad_token_id: int | None,
) -> list[int]:
    values = [int(value) for value in ids]
    eos_values = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(value) for value in (eos_token_id or ())}
    eos_positions = [index for index, value in enumerate(values) if value in eos_values]
    if eos_positions:
        return values[:eos_positions[0] + 1]
    if pad_token_id is not None:
        while len(values) > 1 and values[-1] == pad_token_id:
            values.pop()
    return values


def rollout_business_group(
    model,
    record: dict[str, Any],
    renderer,
    id_to_token: Callable[[int], str],
    pad_token_id: int,
    eos_token_id: int | Sequence[int] | None,
    device: torch.device | str | None = None,
) -> BusinessGroupRollout:
    """Generate IDs, then attach unchanged-policy full-forward PPO old logps."""
    normalized_eos = normalize_eos_token_ids(eos_token_id)
    if normalized_eos != list(FORMAL_EOS_TOKEN_IDS) or int(pad_token_id) != FORMAL_PAD_TOKEN_ID:
        raise ValueError("formal TrueRec rollout requires frozen EOS/PAD token ids")
    model.eval()
    context_ids = render_rollout_context(record, renderer)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=normalized_eos,
        pad_token_id=int(pad_token_id),
        return_dict_in_generate=True,
        output_scores=True,
        **GENERATION_KWARGS,
    )
    versions_after_generate = parameter_versions(model)
    artifacts = extract_generation_artifacts(context_ids, output, pad_token_id, eos_token_id)
    return rescore_business_group_from_completions(
        model, record, context_ids, artifacts.completion_ids, artifacts.generation_score_logps,
        id_to_token, pad_token_id, device, expected_parameter_versions=versions_after_generate,
    )


def extract_generation_artifacts(
    context_ids: Sequence[int], generate_output, pad_token_id: int,
    eos_token_id: int | Sequence[int] | None,
) -> RawGenerationArtifacts:
    """Retain generate scores strictly as diagnostics alongside sampled IDs."""
    if not getattr(generate_output, "scores", None):
        raise ValueError("generate must return per-step diagnostic scores")
    generation_scores = torch.stack(tuple(generate_output.scores), dim=1)
    raw = generate_output.sequences[:, len(context_ids):]
    completion_ids = tuple(tuple(trim_generated_completion(row.tolist(), eos_token_id, pad_token_id)) for row in raw)
    generation_logps = tuple(capture_sampled_logps(generation_scores, completion_ids))
    return RawGenerationArtifacts(completion_ids, generation_scores, generation_logps)


def build_rescored_rollout_group(
    record: dict[str, Any], context_ids: Sequence[int], completion_ids: Sequence[Sequence[int]],
    rescored_old_logps: Sequence[Sequence[float]], generation_score_logps: Sequence[Sequence[float]],
    id_to_token: Callable[[int], str],
) -> BusinessGroupRollout:
    if not (len(completion_ids) == len(rescored_old_logps) == len(generation_score_logps) == G):
        raise ValueError("rescored rollout group must preserve G=8")
    candidates = []
    for index, (ids, old_logps, diagnostic_logps) in enumerate(zip(completion_ids, rescored_old_logps, generation_score_logps)):
        ids = tuple(int(value) for value in ids)
        old_logps = tuple(float(value) for value in old_logps)
        diagnostic_logps = tuple(float(value) for value in diagnostic_logps)
        if len(ids) != len(old_logps) or len(ids) != len(diagnostic_logps):
            raise ValueError("completion/rescore/generation diagnostic lengths differ")
        metrics = assess_candidate(list(ids), id_to_token, record["all_gold_abc"], record["fixed_domain_token"], record["history_sids"])
        candidates.append(RolloutCandidate(index, ids, old_logps, metrics, diagnostic_logps))
    return BusinessGroupRollout(str(record["recommendation_group_id"]), tuple(int(value) for value in context_ids), str(record["fixed_domain_token"]), tuple(record["all_gold_abc"]), tuple(candidates))


def rescore_business_group_from_completions(
    model, record: dict[str, Any], context_ids: Sequence[int], completion_ids: Sequence[Sequence[int]],
    generation_score_logps: Sequence[Sequence[float]], id_to_token: Callable[[int], str],
    pad_token_id: int, device: torch.device | str | None = None,
    expected_parameter_versions: tuple[int, ...] | None = None,
) -> BusinessGroupRollout:
    """Formal PPO old-logp path: eval + no-grad full-forward rescore."""
    model.eval()
    before = parameter_versions(model)
    if expected_parameter_versions is not None and before != expected_parameter_versions:
        raise RuntimeError("policy parameters changed between generation and old-logp rescore")
    score = score_full_sequences(
        model, context_ids, completion_ids, pad_token_id, device,
        grad_enabled=False, trainable_parameters=(),
    )
    if any(score.requires_grad_by_microbatch) or any(score.graph_connected_by_microbatch):
        raise RuntimeError("PPO old-logp rescore retained an autograd graph")
    if parameter_versions(model) != before:
        raise RuntimeError("policy parameters changed during old-logp rescore")
    return build_rescored_rollout_group(
        record, context_ids, completion_ids, score.logps, generation_score_logps, id_to_token,
    )


def build_rollout_artifacts(
    record: dict[str, Any],
    context_ids: Sequence[int],
    generate_output,
    id_to_token: Callable[[int], str],
    pad_token_id: int,
    eos_token_id: int | Sequence[int] | None,
) -> RolloutGenerationArtifacts:
    """Legacy Phase1.2B generation-score diagnostic artifact builder."""
    raw = extract_generation_artifacts(context_ids, generate_output, pad_token_id, eos_token_id)
    group = build_rollout_group(record, context_ids, raw.completion_ids, raw.generation_scores, id_to_token)
    return RolloutGenerationArtifacts(group, raw.completion_ids, raw.generation_scores)


def build_rollout_group(
    record: dict[str, Any], context_ids: Sequence[int], completion_ids: Sequence[Sequence[int]],
    generation_logits: torch.Tensor, id_to_token: Callable[[int], str],
) -> BusinessGroupRollout:
    if len(completion_ids) != G:
        raise ValueError("rollout group must contain G=8 candidates")
    old_logps = capture_sampled_logps(generation_logits, completion_ids)
    candidates = []
    for index, (ids, logps) in enumerate(zip(completion_ids, old_logps)):
        metrics = assess_candidate(list(ids), id_to_token, record["all_gold_abc"], record["fixed_domain_token"], record["history_sids"])
        candidates.append(RolloutCandidate(index, tuple(int(value) for value in ids), logps, metrics))
    return BusinessGroupRollout(str(record["recommendation_group_id"]), tuple(int(value) for value in context_ids), str(record["fixed_domain_token"]), tuple(record["all_gold_abc"]), tuple(candidates))
