"""REC-PU Phase 3 replacement for the NSD-R32-V3 SID8 loss path.

The reference Native-SFT loss is segment-normalised: it sums weighted token
cross entropy inside each packed segment, divides by its number of valid
supervised tokens, applies the segment domain factor, then averages segments.
REC-PU replaces only the three selected final recommendation component terms
inside that existing numerator.  It never adds an auxiliary loss or changes a
denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from rec_pu.recommendation_pu_loss import rec_pu_batched_position_loss, rec_pu_position_loss
from rec_pu.recommendation_pu_phase2 import RecPUPackedTarget, TokenPrefixPositiveSets


IGNORE_INDEX = -100
_SID_ID_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


@dataclass(frozen=True)
class RecPUConfig:
    """Set-PU configuration; the old field name remains launcher-compatible."""

    rec_pu_enabled: bool = True
    rec_pu_unlabeled_sid_grad_scale: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.rec_pu_unlabeled_sid_grad_scale <= 1.0:
            raise ValueError("rec_pu_unlabeled_sid_grad_scale must be in [0, 1].")

    @property
    def enabled(self) -> bool:
        return self.rec_pu_enabled

    @property
    def unlabeled_sid_grad_scale(self) -> float:
        """Compatibility name: this is now the Set-PU U weight (alpha)."""
        return self.rec_pu_unlabeled_sid_grad_scale


@dataclass(frozen=True)
class SIDComponentVocab:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: tuple[int, ...]

    def for_level(self, level: str) -> tuple[int, ...]:
        if level not in ("a", "b", "c"):
            raise ValueError(f"Unknown SID component level: {level!r}")
        return getattr(self, level)


@dataclass
class RecPULossDetails:
    """Small diagnostics used by CPU regressions and low-cost runtime logging."""

    base_contributions: torch.Tensor
    contributions: torch.Tensor
    valid_mask: torch.Tensor
    sample_token_counts: torch.Tensor
    sample_weight_mass: torch.Tensor
    base_sample_numerators: torch.Tensor
    sample_numerators: torch.Tensor
    sample_domain_weights: torch.Tensor
    sample_task_ids: torch.Tensor
    rec_pu_segments: int
    rec_pu_positions: int
    rec_pu_singleton_segments: int
    positive_count_a_sum: int
    positive_count_b_sum: int
    positive_count_c_sum: int
    changed_positions: tuple[tuple[int, int], ...]

    @property
    def denominator(self) -> torch.Tensor:
        """Actual NSD denominator: valid supervised token count per segment."""

        return self.sample_token_counts


def build_sid_component_vocab(tokenizer: Any) -> SIDComponentVocab:
    """Scan the current training tokenizer once for exact a/b/c added tokens."""

    import re

    groups: dict[str, list[int]] = {"a": [], "b": [], "c": []}
    pattern = re.compile(r"^<s_([abc])_\d+>$")
    for token, token_id in tokenizer.get_vocab().items():
        match = pattern.fullmatch(token)
        if match:
            groups[match.group(1)].append(int(token_id))
    result = SIDComponentVocab(
        a=tuple(sorted(groups["a"])), b=tuple(sorted(groups["b"])), c=tuple(sorted(groups["c"]))
    )
    if not result.a or not result.b or not result.c:
        raise ValueError("Tokenizer does not expose all <s_a_*>, <s_b_*>, <s_c_*> vocabularies.")
    return result


def _coerce_positives(value: Any) -> TokenPrefixPositiveSets:
    if isinstance(value, TokenPrefixPositiveSets):
        return value
    if isinstance(value, Mapping):
        return TokenPrefixPositiveSets(
            a=tuple(int(item) for item in value["a"]),
            b=tuple(int(item) for item in value["b"]),
            c=tuple(int(item) for item in value["c"]),
        )
    raise TypeError(f"Unsupported REC-PU positives type: {type(value)!r}")


def coerce_packed_target(value: Any) -> RecPUPackedTarget:
    if isinstance(value, RecPUPackedTarget):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported REC-PU target type: {type(value)!r}")
    return RecPUPackedTarget(
        segment_index=int(value["segment_index"]),
        segment_start=int(value["segment_start"]),
        segment_end=int(value["segment_end"]),
        a_label_position=int(value["a_label_position"]),
        b_label_position=int(value["b_label_position"]),
        c_label_position=int(value["c_label_position"]),
        a_logit_position=int(value["a_logit_position"]),
        b_logit_position=int(value["b_logit_position"]),
        c_logit_position=int(value["c_logit_position"]),
        positives=_coerce_positives(value["positives"]),
    )


def serialise_packed_target(target: RecPUPackedTarget) -> dict[str, Any]:
    return {
        "segment_index": target.segment_index,
        "segment_start": target.segment_start,
        "segment_end": target.segment_end,
        "a_label_position": target.a_label_position,
        "b_label_position": target.b_label_position,
        "c_label_position": target.c_label_position,
        "a_logit_position": target.a_logit_position,
        "b_logit_position": target.b_logit_position,
        "c_logit_position": target.c_logit_position,
        "positives": {"a": target.positives.a, "b": target.positives.b, "c": target.positives.c},
    }


def _normalise_targets(
    targets_by_row: Sequence[Sequence[RecPUPackedTarget | Mapping[str, Any]]] | None,
    batch_size: int,
) -> list[list[RecPUPackedTarget]]:
    if targets_by_row is None:
        return [[] for _ in range(batch_size)]
    if len(targets_by_row) != batch_size:
        raise ValueError("rec_pu_targets must have one target list per packed batch row.")
    return [[coerce_packed_target(target) for target in row] for row in targets_by_row]


def _component_positions(target: RecPUPackedTarget) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    return (
        ("a", target.a_logit_position, target.positives.a),
        ("b", target.b_logit_position, target.positives.b),
        ("c", target.c_logit_position, target.positives.c),
    )


def _cached_sid_ids(token_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    key = (str(device), token_ids)
    value = _SID_ID_CACHE.get(key)
    if value is None:
        value = torch.tensor(token_ids, dtype=torch.long, device=device)
        _SID_ID_CACHE[key] = value
    return value


def compute_native_sid8_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    sample_ids: torch.Tensor,
    sample_task_ids: torch.Tensor,
    sample_domain_weights: torch.Tensor,
    rec_pu_targets: Sequence[Sequence[RecPUPackedTarget | Mapping[str, Any]]] | None = None,
    rec_pu_config: RecPUConfig | None = None,
    sid_component_vocab: SIDComponentVocab | None = None,
) -> tuple[torch.Tensor, RecPULossDetails]:
    """Compute the exact NSD-R32-V3 SID8 loss with optional REC-PU replacement.

    `labels`, `loss_weights`, `sample_ids`, task ids and domain weights use the
    original, unshifted causal-LM positions.  Targets use the Phase-2 label and
    logit positions; only logits at `label_position - 1` are substituted.
    """

    config = rec_pu_config or RecPUConfig(rec_pu_enabled=False)
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, length, vocab] and 2-D label tensors.")
    if labels.shape != loss_weights.shape or labels.shape != sample_ids.shape:
        raise ValueError("labels, loss_weights and sample_ids must have identical shapes.")
    if labels.shape != sample_task_ids.shape or labels.shape != sample_domain_weights.shape:
        raise ValueError("All token-aligned metadata tensors must have identical shapes.")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence shapes must agree.")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].to(dtype=torch.float32).contiguous()
    shift_sample_ids = sample_ids[..., 1:].to(dtype=torch.long).contiguous()
    shift_sample_task_ids = sample_task_ids[..., 1:].to(dtype=torch.long).contiguous()
    shift_sample_domain_weights = sample_domain_weights[..., 1:].to(dtype=torch.float32).contiguous()
    valid = (shift_labels != IGNORE_INDEX) & (shift_sample_ids >= 0)

    normalised_targets = _normalise_targets(rec_pu_targets, logits.size(0))
    rec_segments = 0
    rec_positions = 0
    singleton_segments = 0
    positive_sums = {"a": 0, "b": 0, "c": 0}
    changed: list[tuple[int, int]] = []
    if config.enabled:
        if sid_component_vocab is None:
            raise ValueError("sid_component_vocab is required when REC-PU is enabled.")
        grouped_positions: dict[str, list[tuple[int, int, tuple[int, ...]]]] = {"a": [], "b": [], "c": []}
        for row_index, row_targets in enumerate(normalised_targets):
            rec_segments += len(row_targets)
            for target in row_targets:
                target_sizes = (len(target.positives.a), len(target.positives.b), len(target.positives.c))
                if target_sizes == (1, 1, 1):
                    singleton_segments += 1
                for level, logit_position, positive_ids in _component_positions(target):
                    if logit_position < 0 or logit_position >= shift_logits.size(1):
                        raise ValueError(f"REC-PU {level} target lies outside shifted logits.")
                    grouped_positions[level].append((row_index, logit_position, positive_ids))
                    rec_positions += 1
                    positive_sums[level] += len(positive_ids)

        # At most three full-vocabulary computations (a/b/c) replace the old
        # one-per-final-SID-position loop. Runtime validation is likewise
        # batched, avoiding a device-to-host scalar sync for every position.
        for level, entries in grouped_positions.items():
            if not entries:
                continue
            row_indices = torch.tensor([entry[0] for entry in entries], device=logits.device, dtype=torch.long)
            position_indices = torch.tensor([entry[1] for entry in entries], device=logits.device, dtype=torch.long)
            positive_sets = [entry[2] for entry in entries]
            selected_valid = valid[row_indices, position_indices]
            selected_weights = shift_weights[row_indices, position_indices]
            selected_labels = shift_labels[row_indices, position_indices]
            if not bool(selected_valid.all().item()):
                raise ValueError("REC-PU target must be a valid supervised token.")
            if not bool((selected_weights == 8.0).all().item()):
                raise ValueError("REC-PU final SID component must retain the NSD SID8 weight of 8.0.")

            flat_rows = torch.cat(
                [torch.full((len(positives),), index, dtype=torch.long, device=logits.device)
                 for index, positives in enumerate(positive_sets)]
            )
            flat_positive_ids = torch.tensor(
                [token_id for positives in positive_sets for token_id in positives],
                dtype=torch.long,
                device=logits.device,
            )
            label_matches = selected_labels[flat_rows] == flat_positive_ids
            label_match_counts = torch.zeros(len(entries), dtype=torch.long, device=logits.device)
            label_match_counts.scatter_add_(0, flat_rows, label_matches.to(dtype=torch.long))
            if not bool((label_match_counts > 0).all().item()):
                raise ValueError("Teacher-forced final SID component is absent from its positive set.")

            changed.extend((int(row), int(position)) for row, position, _ in entries)

        # Set-PU is a true scalar objective: no detached logits and no custom
        # backward.  Build baseline CE once, then replace exactly the selected
        # final a/b/c values with direct-autograd Set-PU values.
        base_per_token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="none",
        ).view_as(shift_labels)
        per_token_ce = base_per_token_ce
        for level, entries in grouped_positions.items():
            if not entries:
                continue
            rows = torch.tensor([entry[0] for entry in entries], dtype=torch.long, device=logits.device)
            positions = torch.tensor([entry[1] for entry in entries], dtype=torch.long, device=logits.device)
            set_pu = rec_pu_batched_position_loss(
                shift_logits[rows, positions],
                [entry[2] for entry in entries],
                sid_component_vocab.for_level(level),
                beta=config.unlabeled_sid_grad_scale,
            )
            per_token_ce = per_token_ce.index_put((rows, positions), set_pu)
    else:
        base_per_token_ce = per_token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="none",
        ).view_as(shift_labels)

    base_contributions = base_per_token_ce.float() * shift_weights
    contributions = per_token_ce.float() * shift_weights

    batch_size = shift_sample_ids.size(0)
    sample_stride = shift_sample_ids.max().clamp_min(0) + 1
    row_offsets = torch.arange(batch_size, device=shift_sample_ids.device).unsqueeze(1) * sample_stride
    global_sample_ids = torch.where(valid, shift_sample_ids + row_offsets, torch.full_like(shift_sample_ids, -1))
    flat_valid = valid.reshape(-1)
    flat_sample_ids = global_sample_ids.reshape(-1)[flat_valid]
    if flat_sample_ids.numel() == 0:
        zero = (per_token_ce * 0.0).sum()
        empty = torch.zeros(0, dtype=zero.dtype, device=zero.device)
        return zero, RecPULossDetails(
            base_contributions=base_contributions, contributions=contributions, valid_mask=valid,
            sample_token_counts=empty, sample_weight_mass=empty, base_sample_numerators=empty,
            sample_numerators=empty, sample_domain_weights=empty,
            sample_task_ids=torch.zeros(0, dtype=torch.long, device=zero.device), rec_pu_segments=0,
            rec_pu_positions=0, rec_pu_singleton_segments=0, positive_count_a_sum=0,
            positive_count_b_sum=0, positive_count_c_sum=0, changed_positions=(),
        )

    unique_ids, inverse = torch.unique(flat_sample_ids, sorted=False, return_inverse=True)
    count = unique_ids.numel()
    dtype = contributions.dtype
    flat_base = base_contributions.reshape(-1)[flat_valid]
    flat_new = contributions.reshape(-1)[flat_valid]
    flat_weights = shift_weights.reshape(-1)[flat_valid]
    flat_domains = shift_sample_domain_weights.reshape(-1)[flat_valid]
    flat_tasks = shift_sample_task_ids.reshape(-1)[flat_valid]
    base_numerators = torch.zeros(count, device=logits.device, dtype=dtype)
    numerators = torch.zeros_like(base_numerators)
    token_counts = torch.zeros_like(base_numerators)
    weight_mass = torch.zeros_like(base_numerators)
    domain_sums = torch.zeros_like(base_numerators)
    task_sums = torch.zeros_like(base_numerators)
    base_numerators.scatter_add_(0, inverse, flat_base)
    numerators.scatter_add_(0, inverse, flat_new)
    token_counts.scatter_add_(0, inverse, torch.ones_like(flat_new))
    weight_mass.scatter_add_(0, inverse, flat_weights)
    domain_sums.scatter_add_(0, inverse, flat_domains)
    task_sums.scatter_add_(0, inverse, flat_tasks.to(dtype=dtype))
    sample_losses = numerators / token_counts.clamp_min(1.0)
    sample_domains = domain_sums / token_counts.clamp_min(1.0)
    sample_tasks = (task_sums / token_counts.clamp_min(1.0)).round().to(dtype=torch.long)
    loss = (sample_losses * sample_domains).mean()
    details = RecPULossDetails(
        base_contributions=base_contributions, contributions=contributions, valid_mask=valid,
        sample_token_counts=token_counts, sample_weight_mass=weight_mass,
        base_sample_numerators=base_numerators, sample_numerators=numerators,
        sample_domain_weights=sample_domains, sample_task_ids=sample_tasks,
        rec_pu_segments=rec_segments, rec_pu_positions=rec_positions,
        rec_pu_singleton_segments=singleton_segments,
        positive_count_a_sum=positive_sums["a"], positive_count_b_sum=positive_sums["b"],
        positive_count_c_sum=positive_sums["c"], changed_positions=tuple(changed),
    )
    return loss, details


def pop_rec_pu_metadata(inputs: dict[str, Any]) -> tuple[Any, Any]:
    """Remove only trainer-side REC-PU fields before `model(**inputs)`."""

    return inputs.pop("rec_pu_targets", None), inputs.pop("rec_pu_config", None)


def probe_rec_pu_position(
    logits_at_position: torch.Tensor,
    *,
    target_label_id: int,
    positive_ids: Iterable[int],
    same_level_ids: Iterable[int],
    wrong_level_id: int,
    normal_token_id: int,
    beta: float,
) -> dict[str, float | int | bool]:
    """Return a one-position autograd probe without another model forward.

    The caller passes the direct 1-D logits view from the real model output.
    The debug leaf is detached from that already-computed position and never
    connects backwards through the full packed model graph.  It therefore
    probes the exact real logits with ``autograd.grad`` without allocating a
    full [packed_length, vocab] model-output gradient.  The normal training
    loss still owns the original graph and its single later backward.
    """

    positives = tuple(sorted({int(item) for item in positive_ids}))
    level = tuple(sorted({int(item) for item in same_level_ids}))
    if target_label_id not in positives:
        raise ValueError("Probe target label must be an observed positive.")
    unlabeled = tuple(item for item in level if item not in positives)
    if not unlabeled:
        raise ValueError("Probe requires at least one unlabeled same-level SID.")
    if wrong_level_id in level or normal_token_id in level:
        raise ValueError("Probe O tokens must not be same-level SID ids.")
    if logits_at_position.ndim != 1:
        raise ValueError("Probe expects a direct one-dimensional logits view.")

    probe_logits = logits_at_position.detach().float().requires_grad_(True)
    baseline = torch.nn.functional.cross_entropy(
        probe_logits.unsqueeze(0),
        torch.tensor([target_label_id], device=probe_logits.device),
    )
    pu, _ = rec_pu_position_loss(probe_logits, positives, level, beta=beta)
    baseline_grad = torch.autograd.grad(baseline, probe_logits, retain_graph=True)[0]
    pu_grad = torch.autograd.grad(pu, probe_logits)[0]
    u_id = unlabeled[0]
    def ratio(index: int) -> float:
        base = baseline_grad[index]
        return float((pu_grad[index] / base).detach().cpu().item())

    return {
        "positive_id": target_label_id,
        "unlabeled_id": u_id,
        "wrong_level_id": int(wrong_level_id),
        "normal_token_id": int(normal_token_id),
        "u_baseline_grad": float(baseline_grad[u_id].detach().cpu().item()),
        "u_rec_pu_grad": float(pu_grad[u_id].detach().cpu().item()),
        "u_ratio": ratio(u_id),
        "wrong_level_baseline_grad": float(baseline_grad[wrong_level_id].detach().cpu().item()),
        "wrong_level_rec_pu_grad": float(pu_grad[wrong_level_id].detach().cpu().item()),
        "wrong_level_ratio": ratio(wrong_level_id),
        "normal_baseline_grad": float(baseline_grad[normal_token_id].detach().cpu().item()),
        "normal_rec_pu_grad": float(pu_grad[normal_token_id].detach().cpu().item()),
        "normal_ratio": ratio(normal_token_id),
        "positive_rec_pu_grad": float(pu_grad[target_label_id].detach().cpu().item()),
        "positive_gradient_descent_raises_logit": bool(pu_grad[target_label_id].detach().item() < 0),
    }
