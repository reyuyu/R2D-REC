"""SID-family Bridge-to-Bare KD loss with Boundary's group-uniform estimator."""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SID_PATTERNS = {
    "A": re.compile(r"^<s_a_\d+>$"),
    "B": re.compile(r"^<s_b_\d+>$"),
    "C": re.compile(r"^<s_c_\d+>$"),
}


@dataclass(frozen=True)
class SidFamilies:
    A: tuple[int, ...]
    B: tuple[int, ...]
    C: tuple[int, ...]

    def tensors(self, device: torch.device | str) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor(getattr(self, key), device=device) for key in "ABC")


def scan_sid_families(tokenizer) -> SidFamilies:
    vocab = tokenizer.get_vocab()
    found = {key: [] for key in "ABC"}
    for token, token_id in vocab.items():
        for key, pattern in SID_PATTERNS.items():
            if pattern.fullmatch(token):
                if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                    raise ValueError(f"SID token is not atomic: {token}")
                found[key].append(token_id)
    if any(not found[key] for key in "ABC"):
        raise ValueError(f"empty SID family: { {key: len(value) for key, value in found.items()} }")
    if len(set().union(*(set(value) for value in found.values()))) != sum(map(len, found.values())):
        raise ValueError("SID families overlap")
    return SidFamilies(*(tuple(sorted(found[key])) for key in "ABC"))


def family_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
              family_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher_logits.detach().index_select(-1, family_ids) / temperature
    student = student_logits.index_select(-1, family_ids) / temperature
    teacher_logp = F.log_softmax(teacher, dim=-1)
    teacher_p = teacher_logp.exp()
    return (teacher_p * (teacher_logp - F.log_softmax(student, dim=-1))).sum(-1) * temperature ** 2


def per_path_kd_losses(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                       gold_ids: torch.Tensor, family_ids: tuple[torch.Tensor, ...],
                       temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Inputs are the three next-token logits conditioned on Gold prefixes."""
    if student_logits.ndim != 3 or student_logits.shape[1] != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("expected matching [B,3,V] logits")
    if gold_ids.shape != student_logits.shape[:2] or len(family_ids) != 3:
        raise ValueError("invalid Gold/family shapes")
    gold = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), gold_ids.reshape(-1), reduction="none").view_as(gold_ids).mean(-1)
    levels = [family_kl(teacher_logits[:, i], student_logits[:, i], family_ids[i], temperature) for i in range(3)]
    return {"gold": gold, "kd": torch.stack(levels).mean(0), **{f"kl_{key}": levels[i] for i, key in enumerate("ABC")}}


def row_uniform_kd_objective(losses: dict[str, torch.Tensor], sample_weights: torch.Tensor,
                             *, lambda_kd: float, total_paths: int, total_groups: int) -> torch.Tensor:
    if sample_weights.shape != losses["gold"].shape:
        raise ValueError("sample_weights must match per-path losses")
    if total_paths <= 0 or total_groups <= 0 or lambda_kd < 0:
        raise ValueError("invalid objective provenance/configuration")
    per_path = losses["gold"] + lambda_kd * losses["kd"]
    return (per_path * sample_weights).mean() * (float(total_paths) / float(total_groups))
