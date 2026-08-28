"""Pure contracts for Think CoT -> SID suffix GRPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from grpo_sid import SID_RE, parse_sid, q_reward


THINK_CLOSE = "</think>"
REWARD_LEVELS = (-1.0, -0.25, 0.0, 0.5, 2.0, 8.0)
GROUP_SIZE = 8
MAX_RESAMPLE_ROUNDS = 3
MAX_GENERATED_CANDIDATES = GROUP_SIZE * (1 + MAX_RESAMPLE_ROUNDS)


@dataclass(frozen=True)
class SuffixParse:
    closed: bool
    closure_index: int | None
    suffix_text: str
    parsed_sid: tuple[str, int, int, int] | None
    all_parsed_sids: tuple[tuple[str, int, int, int], ...]
    sid_count: int
    multi_sid_output: bool
    parser_status: str

    def monitor_dict(self) -> dict:
        payload = asdict(self)
        payload["parsed_sid"] = list(self.parsed_sid) if self.parsed_sid else None
        payload["all_parsed_sids"] = [list(sid) for sid in self.all_parsed_sids]
        return payload


def parse_suffix_completion(text: str) -> SuffixParse:
    """Parse only the answer suffix after the first complete </think> marker."""
    text = text or ""
    closure_index = text.find(THINK_CLOSE)
    if closure_index < 0:
        return SuffixParse(
            closed=False,
            closure_index=None,
            suffix_text="",
            parsed_sid=None,
            all_parsed_sids=(),
            sid_count=0,
            multi_sid_output=False,
            parser_status="missing_think_close",
        )
    suffix = text[closure_index + len(THINK_CLOSE):]
    occurrences = tuple(
        (match.group("domain"), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        for match in SID_RE.finditer(suffix)
    )
    if not occurrences:
        status = "missing_suffix_sid"
    elif len(occurrences) > 1:
        status = "multiple_suffix_sids"
    else:
        status = "ok"
    return SuffixParse(
        closed=True,
        closure_index=closure_index,
        suffix_text=suffix,
        parsed_sid=occurrences[-1] if occurrences else None,
        all_parsed_sids=occurrences,
        sid_count=len(occurrences),
        multi_sid_output=len(occurrences) > 1,
        parser_status=status,
    )


def suffix_reward(text: str, gold_sids) -> tuple[float, SuffixParse]:
    parsed = parse_suffix_completion(text)
    gold_set = {sid for raw in gold_sids for sid in [parse_sid(raw)] if sid is not None}
    reward = q_reward(parsed.parsed_sid, gold_set)
    if reward not in REWARD_LEVELS:
        raise RuntimeError(f"unexpected suffix reward: {reward}")
    return reward, parsed


def population_std(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("reward vector must not be empty")
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def resample_decision(rewards, attempt: int) -> dict:
    """Return the synchronized pre-backward zero-std rescue decision."""
    rewards = [float(value) for value in rewards]
    if len(rewards) != GROUP_SIZE:
        raise ValueError(f"expected G{GROUP_SIZE}, got {len(rewards)} rewards")
    if not 0 <= attempt <= MAX_RESAMPLE_ROUNDS:
        raise ValueError(f"attempt must be in [0, {MAX_RESAMPLE_ROUNDS}]")
    zero_std = population_std(rewards) == 0.0
    retry = zero_std and attempt < MAX_RESAMPLE_ROUNDS
    exhausted = zero_std and attempt == MAX_RESAMPLE_ROUNDS
    return {
        "attempt": attempt,
        "resample_round": attempt,
        "zero_std": zero_std,
        "retry": retry,
        "exhausted": exhausted,
        "accepted": not retry,
        "generated_candidate_total": GROUP_SIZE * (attempt + 1),
    }


def suffix_mask_for_ids(token_ids, close_token_ids, valid_length=None):
    """Mask tokens strictly after the first full </think> token sequence."""
    ids = [int(token) for token in token_ids]
    close = [int(token) for token in close_token_ids]
    if not close:
        raise ValueError("close_token_ids must not be empty")
    valid = len(ids) if valid_length is None else int(valid_length)
    if not 0 <= valid <= len(ids):
        raise ValueError("valid_length is outside token_ids")
    mask = [0] * len(ids)
    stop = valid - len(close) + 1
    for start in range(max(stop, 0)):
        if ids[start:start + len(close)] == close:
            suffix_start = start + len(close)
            for index in range(suffix_start, valid):
                mask[index] = 1
            break
    return mask
