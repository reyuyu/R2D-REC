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


G = 8
GENERATION_KWARGS = {
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "repetition_penalty": 1.0, "max_new_tokens": 3, "num_return_sequences": G,
}
OLD_LOGPS_FROM_ROLLOUT_POLICY = True


@dataclass(frozen=True)
class RolloutCandidate:
    sample_index: int
    completion_ids: tuple[int, ...]
    old_logps: tuple[float, ...]
    metrics: dict[str, Any]


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


def generation_contract() -> dict[str, Any]:
    return dict(GENERATION_KWARGS)


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
    """Execute the fixed G8 generation contract and retain its sampled-policy scores."""
    context_ids = render_rollout_context(record, renderer)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=int(pad_token_id),
        return_dict_in_generate=True,
        output_scores=True,
        **GENERATION_KWARGS,
    )
    return build_rollout_artifacts(
        record, context_ids, output, id_to_token, pad_token_id, eos_token_id,
    ).group


def build_rollout_artifacts(
    record: dict[str, Any],
    context_ids: Sequence[int],
    generate_output,
    id_to_token: Callable[[int], str],
    pad_token_id: int,
    eos_token_id: int | Sequence[int] | None,
) -> RolloutGenerationArtifacts:
    if not getattr(generate_output, "scores", None):
        raise ValueError("generate must return per-step rollout-policy scores")
    generation_scores = torch.stack(tuple(generate_output.scores), dim=1)
    raw = generate_output.sequences[:, len(context_ids):]
    completion_ids = tuple(
        tuple(trim_generated_completion(row.tolist(), eos_token_id, pad_token_id)) for row in raw
    )
    group = build_rollout_group(record, context_ids, completion_ids, generation_scores, id_to_token)
    return RolloutGenerationArtifacts(group, completion_ids, generation_scores)


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
